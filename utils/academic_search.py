import requests
import time
import random
import threading
from typing import List, Dict, Optional, Tuple, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import re
import logging
import json
import os
import torch
import faiss
from sentence_transformers import util
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import pickle
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import redis
except Exception:
    redis = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalEmbeddingModel:
    """Deterministic lightweight embedding fallback for offline environments."""

    def __init__(self, dimension: int = 384):
        self.dimension = int(dimension)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def _text_to_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not text:
            return vec
        tokens = re.findall(r"\w+", text.lower())
        for tok in tokens:
            idx = abs(hash(tok)) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts, convert_to_tensor: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = np.vstack([self._text_to_vector(t) for t in texts])
        if convert_to_tensor:
            try:
                import torch
                return torch.from_numpy(embeddings)
            except Exception:
                return embeddings
        return embeddings


class AcademicSearch:
    """Academic search + local vector search with robust fallbacks."""

    _shared_instance = None
    _shared_lock = threading.Lock()

    @classmethod
    def get_shared_instance(cls):
        with cls._shared_lock:
            if cls._shared_instance is None:
                cls._shared_instance = cls()
            return cls._shared_instance

    def __init__(self):
        self.session = requests.Session()
        retry_strategy = Retry(
            total=1,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.last_request_time = {}
        self.min_interval = 0.15
        self.max_jitter = 0.2
        self.request_timeouts = {
            "semantic_scholar": 3,
            "crossref": 3,
            "arxiv": 4,
            "openalex": 3,
        }

        self.model = self._initialize_model()
        self.cross_encoder = self._initialize_cross_encoder()
        self.vector_db = self._initialize_vector_db()

        self.redis_client = self._init_redis()
        self.local_cache: Dict[str, Any] = {}
        self.cache_file = "academic_cache.json"
        self._ensure_cache_file()
        self._load_cache()
        self.similarity_cache: Dict[str, float] = {}
        self.search_result_cache: Dict[str, List[Dict]] = {}
        self.cache_lock = threading.Lock()

        self.executor = ThreadPoolExecutor(max_workers=8)
        self.stats = {"api_calls": 0, "cache_hits": 0, "vector_searches": 0}

        self.search_apis = [
            {"name": "semantic_scholar", "weight": 0.4, "enabled": True},
            {"name": "crossref", "weight": 0.3, "enabled": True},
            {"name": "arxiv", "weight": 0.1, "enabled": True},
            {"name": "openalex", "weight": 0.2, "enabled": True},
        ]
        self.cosine_micro_batch_size = 8 if torch.cuda.is_available() else 32
        self.cross_encoder_micro_batch_size = 16 if torch.cuda.is_available() else 32
        self.search_parallel_timeout_sec = 6.0

    def _init_redis(self):
        if redis is None:
            logger.info("Redis package not installed; continuing with in-memory cache.")
            return None
        try:
            return redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
        except Exception:
            logger.info("Redis not available; continuing with in-memory cache.")
            return None

    def _initialize_model(self):
        models_to_try = ["all-mpnet-base-v2", "all-MiniLM-L12-v2", "paraphrase-multilingual-mpnet-base-v2"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from sentence_transformers import SentenceTransformer, models as st_models
            for m in models_to_try:
                try:
                    logger.info(f"Attempting to load embedding model '{m}' on device '{device}'")
                    model = SentenceTransformer(m, device=device)
                    logger.info(f"Loaded embedding model '{m}'")
                    return model
                except Exception as e:
                    logger.warning(f"Direct load for '{m}' failed: {e}")
                    try:
                        transformer = st_models.Transformer(m, model_args={"low_cpu_mem_usage": False})
                        pooling = st_models.Pooling(transformer.get_word_embedding_dimension())
                        model = SentenceTransformer(modules=[transformer, pooling], device=device)
                        return model
                    except Exception:
                        continue
        except Exception:
            pass

        logger.warning("Falling back to LocalEmbeddingModel.")
        return LocalEmbeddingModel(dimension=384)

    def _initialize_cross_encoder(self):
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            return None

    def _initialize_vector_db(self):
        try:
            dim = self.model.get_sentence_embedding_dimension()
            idx = faiss.IndexFlatIP(dim)
            idmap = faiss.IndexIDMap(idx)
            if os.path.exists("academic_index.faiss"):
                idmap = faiss.read_index("academic_index.faiss")
            return {"index": idmap, "id_to_metadata": {}, "dimension": dim}
        except Exception:
            return None

    def _ensure_cache_file(self):
        if not os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "w") as f:
                    json.dump({}, f)
            except Exception:
                pass

    def _load_cache(self):
        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                self.local_cache = data if isinstance(data, dict) else {}
        except Exception:
            self.local_cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.local_cache, f)
        except Exception:
            pass

    async def search_parallel(self, query: str, max_results: int = 10) -> List[Dict]:
        """Parallel search across multiple academic backends with ranking."""
        query = self._clean_query(query)
        if not query:
            return []
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
        cached = self._get_cached(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            return cached[:max_results]

        futures = []
        for api in self.search_apis:
            if api["enabled"]:
                method = getattr(self, f"_search_{api['name']}", None)
                if callable(method):
                    futures.append(self.executor.submit(method, query, max_results))

        results = []
        pending_futures = set(futures)
        deadline = time.monotonic() + self.search_parallel_timeout_sec
        while pending_futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "Academic parallel search timed out after %.1fs for query '%s'",
                    self.search_parallel_timeout_sec,
                    query[:80],
                )
                break
            done, pending_futures = wait(
                pending_futures,
                timeout=min(1.5, remaining),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                try:
                    res = future.result()
                    if res:
                        results.extend(res)
                except Exception:
                    pass

        unique = self._deduplicate_results(results)
        ranked = self._rank_results(unique, query) # Fixed: This method now exists
        self._set_cached(cache_key, ranked[:max_results], ttl=3600)
        return ranked[:max_results]

    def _rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Calculates semantic similarity scores and sorts results by relevance."""
        if not results:
            return []
        
        # Extract content for comparison (Abstract preferred over Title)
        content_list = [f"{r.get('title', '')} {r.get('abstract', '')}" for r in results]
        
        # Vectorized similarity calculation
        query_emb = self.model.encode([query], convert_to_tensor=True)
        doc_embs = self.model.encode(content_list, convert_to_tensor=True)
        
        # Compute cosine similarity
        scores = util.cos_sim(query_emb, doc_embs)[0].cpu().tolist()
        
        # Attach scores to results
        for i, score in enumerate(scores):
            results[i]["relevance_score"] = round(float(score), 4)
            
        # Sort descending by score
        return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=1, max=2))
    def _search_semantic_scholar(self, query: str, limit: int = 5) -> List[Dict]:
        cache_key = self._backend_cache_key("semantic_scholar", query, limit)
        cached = self._get_backend_cached(cache_key)
        if cached is not None:
            return cached
        self.stats["api_calls"] += 1
        self._wait_for_api("semantic_scholar")
        try:
            query = self._prepare_backend_query(query, backend="semantic_scholar")
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {"query": query, "limit": limit, "fields": "title,abstract,authors,year,externalIds,citationCount,venue,url"}
            r = self.session.get(url, params=params, timeout=self.request_timeouts["semantic_scholar"])
            if r.status_code == 200:
                data = r.json().get("data", [])
                results = [self._format_semantic_scholar_result(p) for p in data[:limit]]
                self._set_backend_cached(cache_key, results)
                return results
        except Exception:
            pass
        self._set_backend_cached(cache_key, [])
        return []

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=1, max=2))
    def _search_crossref(self, query: str, limit: int = 5) -> List[Dict]:
        cache_key = self._backend_cache_key("crossref", query, limit)
        cached = self._get_backend_cached(cache_key)
        if cached is not None:
            return cached
        self.stats["api_calls"] += 1
        self._wait_for_api("crossref")
        try:
            query = self._prepare_backend_query(query, backend="crossref")
            url = "https://api.crossref.org/works"
            params = {"query": query, "rows": limit}
            r = self.session.get(url, params=params, timeout=self.request_timeouts["crossref"])
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", [])
                results = [self._format_crossref_result(i) for i in items[:limit]]
                self._set_backend_cached(cache_key, results)
                return results
        except Exception:
            pass
        self._set_backend_cached(cache_key, [])
        return []

    def _search_arxiv(self, query: str, limit: int = 3) -> List[Dict]:
        cache_key = self._backend_cache_key("arxiv", query, limit)
        cached = self._get_backend_cached(cache_key)
        if cached is not None:
            return cached
        self.stats["api_calls"] += 1
        self._wait_for_api("arxiv")
        try:
            query = self._prepare_backend_query(query, backend="arxiv")
            url = "http://export.arxiv.org/api/query"
            params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}
            r = self.session.get(url, params=params, timeout=self.request_timeouts["arxiv"])
            if r.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(r.content)
                ns = {"a": "http://www.w3.org/2005/Atom"}
                entries = root.findall("a:entry", ns)
                results = []
                for e in entries[:limit]:
                    title = e.find("a:title", ns)
                    summary = e.find("a:summary", ns)
                    published = e.find("a:published", ns)
                    id_el = e.find("a:id", ns)
                    results.append({
                        "title": title.text if title is not None else "",
                        "abstract": summary.text if summary is not None else "",
                        "authors": [a.text for a in e.findall("a:author/a:name", ns)],
                        "year": published.text[:4] if published is not None and published.text else None,
                        "paperId": id_el.text if id_el is not None else None,
                        "source": "arxiv"
                    })
                self._set_backend_cached(cache_key, results)
                return results
        except Exception:
            pass
        self._set_backend_cached(cache_key, [])
        return []

    def _search_openalex(self, query: str, limit: int = 5) -> List[Dict]:
        cache_key = self._backend_cache_key("openalex", query, limit)
        cached = self._get_backend_cached(cache_key)
        if cached is not None:
            return cached
        self.stats["api_calls"] += 1
        self._wait_for_api("openalex")
        try:
            query = self._prepare_backend_query(query, backend="openalex")
            url = "https://api.openalex.org/works"
            params = {"search": query, "per_page": limit}
            r = self.session.get(url, params=params, timeout=self.request_timeouts["openalex"])
            if r.status_code == 200:
                data = r.json().get("results", [])
                results = [self._format_openalex_result(p) for p in data[:limit]]
                self._set_backend_cached(cache_key, results)
                return results
        except Exception:
            pass
        self._set_backend_cached(cache_key, [])
        return []

    def search_vector_db(self, query: str, k: int = 10, threshold: float = 0.6) -> List[Dict]:
        self.stats["vector_searches"] += 1
        if not self.vector_db or self.vector_db["index"].ntotal == 0:
            return []
        enc = self.model.encode([query], convert_to_tensor=True)
        qemb = enc.cpu().numpy() if hasattr(enc, "cpu") else np.array(enc)
        try:
            distances, indices = self.vector_db["index"].search(qemb, k)
            results = []
            for d, idx in zip(distances[0], indices[0]):
                if int(idx) in self.vector_db["id_to_metadata"] and float(d) >= threshold:
                    md = dict(self.vector_db["id_to_metadata"][int(idx)])
                    md["similarity"] = float(d)
                    results.append(md)
            return results
        except Exception:
            return []

    def add_to_vector_db(self, papers: List[Dict]):
        if not self.vector_db:
            return
        texts = [p.get("abstract") or p.get("title") or "" for p in papers]
        enc = self.model.encode(texts, convert_to_tensor=True)
        embeddings = enc.cpu().numpy().astype(np.float32) if hasattr(enc, "cpu") else np.array(enc).astype(np.float32)
        start = int(self.vector_db["index"].ntotal)
        ids = np.arange(start, start + len(papers)).astype("int64")
        try:
            self.vector_db["index"].add_with_ids(embeddings, ids)
            for i, p in zip(ids, papers):
                self.vector_db["id_to_metadata"][int(i)] = p
            if (start + len(papers)) % 1000 == 0:
                faiss.write_index(self.vector_db["index"], "academic_index.faiss")
        except Exception:
            pass

    def calculate_similarity(self, text1: str, text2: str, method: str = "cosine") -> float:
        if not text1 or not text2:
            return 0.0
        key = self._build_similarity_cache_key(text1, text2, method)
        with self.cache_lock:
            cached = self.similarity_cache.get(key)
        if cached is not None:
            return cached
        if method == "cosine":
            value = self._cosine_similarity(text1, text2)
        elif method == "jaccard":
            value = self._jaccard_similarity(text1, text2)
        elif method == "cross_encoder":
            value = self._cross_encoder_similarity(text1, text2)
        else:
            value = self._ensemble_similarity(text1, text2)
        with self.cache_lock:
            if len(self.similarity_cache) > 10000:
                self.similarity_cache.clear()
            self.similarity_cache[key] = value
        return value

    def batch_calculate_similarity(self, pairs: List[Tuple[str, str]], method: str = "cosine") -> List[float]:
        if not pairs:
            return []

        results: List[Optional[float]] = [None] * len(pairs)
        pending_indices: List[int] = []
        pending_pairs: List[Tuple[str, str]] = []

        for idx, (text1, text2) in enumerate(pairs):
            if not text1 or not text2:
                results[idx] = 0.0
                continue
            key = self._build_similarity_cache_key(text1, text2, method)
            with self.cache_lock:
                cached = self.similarity_cache.get(key)
            if cached is not None:
                results[idx] = cached
            else:
                pending_indices.append(idx)
                pending_pairs.append((text1, text2))

        if pending_pairs:
            if method == "cosine":
                computed = self._batch_cosine_similarity(pending_pairs)
            elif method == "cross_encoder":
                computed = self._batch_cross_encoder_similarity(pending_pairs)
            elif method == "jaccard":
                computed = [self._jaccard_similarity(t1, t2) for t1, t2 in pending_pairs]
            else:
                computed = [self._ensemble_similarity(t1, t2) for t1, t2 in pending_pairs]

            with self.cache_lock:
                if len(self.similarity_cache) > 10000:
                    self.similarity_cache.clear()
                for idx, (text1, text2), value in zip(pending_indices, pending_pairs, computed):
                    results[idx] = value
                    self.similarity_cache[self._build_similarity_cache_key(text1, text2, method)] = value

        return [float(value or 0.0) for value in results]

    def _cosine_similarity(self, t1: str, t2: str) -> float:
        enc = self.model.encode([t1[:1000], t2[:1000]], convert_to_tensor=True)
        try:
            a = enc[0].cpu().numpy() if hasattr(enc, "cpu") else np.array(enc[0])
            b = enc[1].cpu().numpy() if hasattr(enc, "cpu") else np.array(enc[1])
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / denom) if denom != 0 else 0.0
        except Exception:
            return 0.0

    def _cross_encoder_similarity(self, t1: str, t2: str) -> float:
        if not self.cross_encoder:
            return self._cosine_similarity(t1, t2)
        try:
            score = self.cross_encoder.predict(
                [[t1[:500], t2[:500]]],
                batch_size=1,
                show_progress_bar=False,
            )
            return float(score[0])
        except Exception:
            return self._cosine_similarity(t1, t2)

    def _batch_cosine_similarity(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        scores: List[float] = []
        batch_size = max(1, int(self.cosine_micro_batch_size))
        start = 0
        while start < len(pairs):
            current_batch_size = min(batch_size, len(pairs) - start)
            batch_pairs = pairs[start:start + current_batch_size]
            try:
                left_texts = [t1[:1000] for t1, _ in batch_pairs]
                right_texts = [t2[:1000] for _, t2 in batch_pairs]
                left_embs = self.model.encode(left_texts, convert_to_tensor=True)
                right_embs = self.model.encode(right_texts, convert_to_tensor=True)
                batch_scores = util.cos_sim(left_embs, right_embs)
                scores.extend(float(batch_scores[i][i]) for i in range(len(batch_pairs)))
                start += current_batch_size
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise
                batch_size = max(1, current_batch_size // 2)
                logger.warning("Reducing cosine micro-batch size to %s after CUDA OOM", batch_size)
        return scores

    def _batch_cross_encoder_similarity(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        if not self.cross_encoder:
            return self._batch_cosine_similarity(pairs)
        scores: List[float] = []
        batch_size = max(1, int(self.cross_encoder_micro_batch_size))
        start = 0
        while start < len(pairs):
            current_batch_size = min(batch_size, len(pairs) - start)
            batch_pairs = pairs[start:start + current_batch_size]
            try:
                payload = [[t1[:500], t2[:500]] for t1, t2 in batch_pairs]
                batch_scores = self.cross_encoder.predict(
                    payload,
                    batch_size=current_batch_size,
                    show_progress_bar=False,
                )
                scores.extend(float(score) for score in batch_scores)
                start += current_batch_size
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    logger.warning("Cross-encoder batch failed, falling back to cosine similarity: %s", exc)
                    scores.extend(self._batch_cosine_similarity(batch_pairs))
                    start += current_batch_size
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_batch_size == 1:
                    logger.warning("Cross-encoder single-item batch OOM, falling back to cosine similarity")
                    scores.extend(self._batch_cosine_similarity(batch_pairs))
                    start += 1
                    continue
                batch_size = max(1, current_batch_size // 2)
                logger.warning("Reducing cross-encoder micro-batch size to %s after CUDA OOM", batch_size)
            except Exception:
                scores.extend(self._batch_cosine_similarity(batch_pairs))
                start += current_batch_size
        return scores

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        a = set(re.findall(r"\w+", s1.lower()))
        b = set(re.findall(r"\w+", s2.lower()))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _ensemble_similarity(self, t1: str, t2: str) -> float:
        c = self._cosine_similarity(t1, t2)
        j = self._jaccard_similarity(t1, t2)
        ce = self._cross_encoder_similarity(t1, t2) if self.cross_encoder else c
        return 0.5 * c + 0.3 * ce + 0.2 * j

    def _wait_for_api(self, api_name: str):
        cur = time.time()
        last = self.last_request_time.get(api_name, 0)
        elapsed = cur - last
        wait = max(0, self.min_interval - elapsed + random.uniform(0, self.max_jitter))
        if wait > 0:
            time.sleep(wait)
        self.last_request_time[api_name] = time.time()

    def _clean_query(self, q: str) -> str:
        q = re.sub(r"\[\d+\]", "", q)
        q = re.sub(r"[^\w\s]", " ", q)
        words = [w for w in q.split() if len(w) > 2][:15]
        return " ".join(words)

    def _prepare_backend_query(self, query: str, backend: str) -> str:
        cleaned = self._clean_query(query)
        words = cleaned.split()
        if backend == "crossref":
            return " ".join(words[:10])
        if backend == "arxiv":
            return " ".join(words[:12])
        if backend == "semantic_scholar":
            return " ".join(words[:15])
        if backend == "openalex":
            return " ".join(words[:12])
        return cleaned

    def _format_semantic_scholar_result(self, p: Dict) -> Dict:
        return {
            "title": p.get("title", ""),
            "abstract": p.get("abstract", "")[:1000],
            "authors": [a.get("name", "") for a in p.get("authors", [])],
            "year": p.get("year"),
            "doi": (p.get("externalIds") or {}).get("DOI"),
            "citation_count": p.get("citationCount", 0),
            "venue": p.get("venue", ""),
            "url": p.get("url", ""),
            "source": "semantic_scholar",
        }

    def _format_crossref_result(self, p: Dict) -> Dict:
        return {
            "title": " ".join(p.get("title", [""])),
            "abstract": p.get("abstract", "")[:1000],
            "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in p.get("author", [])],
            "year": (p.get("created") or {}).get("date-parts", [[None]])[0][0],
            "doi": p.get("DOI"),
            "citation_count": p.get("is-referenced-by-count", 0),
            "venue": (p.get("container-title") or [""])[0] if p.get("container-title") else "",
            "url": f"https://doi.org/{p.get('DOI','')}",
            "source": "crossref",
        }

    def _format_openalex_result(self, p: Dict) -> Dict:
        return {
            "title": p.get("title", ""),
            "abstract": (p.get("abstract") or "")[:1000],
            "authors": [a.get("author", {}).get("display_name", "") for a in p.get("authorships", [])],
            "year": p.get("publication_year"),
            "doi": (p.get("doi") or "").replace("https://doi.org/", ""),
            "citation_count": p.get("cited_by_count", 0),
            "venue": (p.get("host_venue") or {}).get("display_name", ""),
            "url": p.get("id", ""),
            "source": "openalex",
        }

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        seen_titles, seen_dois = set(), set()
        unique = []
        for r in results:
            doi = (r.get("doi") or "").lower()
            title = (r.get("title") or "").lower()
            if doi and doi in seen_dois: continue
            hash_t = hashlib.md5(title.encode()).hexdigest()[:8]
            if hash_t in seen_titles: continue
            if doi: seen_dois.add(doi)
            seen_titles.add(hash_t)
            unique.append(r)
        return unique

    def _get_cached(self, key: str):
        if self.redis_client:
            try:
                raw = self.redis_client.get(key)
                if raw: return pickle.loads(raw)
            except Exception: pass
        with self.cache_lock:
            return self.local_cache.get(key)

    def _get_backend_cached(self, key: str):
        with self.cache_lock:
            cached = self.search_result_cache.get(key)
        if cached is None:
            return None
        self.stats["cache_hits"] += 1
        return [dict(result) for result in cached]

    def _set_backend_cached(self, key: str, value: List[Dict]):
        snapshot = [dict(result) for result in value]
        with self.cache_lock:
            if len(self.search_result_cache) > 4000:
                self.search_result_cache.clear()
            self.search_result_cache[key] = snapshot

    def _set_cached(self, key: str, value, ttl: int = 3600):
        if self.redis_client:
            try: self.redis_client.setex(key, ttl, pickle.dumps(value))
            except Exception: pass
        with self.cache_lock:
            self.local_cache[key] = value
            if len(self.local_cache) > 5000:
                self._save_cache()
                self.local_cache.clear()

    @staticmethod
    def _normalize_cache_text(text: str, limit: int = 1000) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        return normalized[:limit]

    @classmethod
    def _build_similarity_cache_key(cls, text1: str, text2: str, method: str) -> str:
        left = cls._normalize_cache_text(text1)
        right = cls._normalize_cache_text(text2)
        return f"{method}:{hashlib.md5(left.encode()).hexdigest()}:{hashlib.md5(right.encode()).hexdigest()}"

    @classmethod
    def _backend_cache_key(cls, source_name: str, query: str, limit: int) -> str:
        normalized_query = cls._normalize_cache_text(query, limit=400)
        return f"{source_name}:{limit}:{hashlib.md5(normalized_query.encode()).hexdigest()}"

    def get_stats(self) -> Dict:
        return self.stats
