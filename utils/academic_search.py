import requests
import time
import random
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def __init__(self):
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.last_request_time = {}
        self.min_interval = 0.5
        self.max_jitter = 1.0

        self.model = self._initialize_model()
        self.cross_encoder = self._initialize_cross_encoder()
        self.vector_db = self._initialize_vector_db()

        self.redis_client = self._init_redis()
        self.local_cache: Dict[str, Any] = {}
        self.cache_file = "academic_cache.json"
        self._ensure_cache_file()
        self._load_cache()

        self.executor = ThreadPoolExecutor(max_workers=8)
        self.stats = {"api_calls": 0, "cache_hits": 0, "vector_searches": 0}

        self.search_apis = [
            {"name": "semantic_scholar", "weight": 0.4, "enabled": True},
            {"name": "crossref", "weight": 0.3, "enabled": True},
            {"name": "arxiv", "weight": 0.1, "enabled": True},
            {"name": "openalex", "weight": 0.2, "enabled": True},
        ]

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
        for future in as_completed(futures):
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _search_semantic_scholar(self, query: str, limit: int = 5) -> List[Dict]:
        self.stats["api_calls"] += 1
        self._wait_for_api("semantic_scholar")
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {"query": query, "limit": limit, "fields": "title,abstract,authors,year,externalIds,citationCount,venue,url"}
            r = self.session.get(url, params=params, timeout=12)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return [self._format_semantic_scholar_result(p) for p in data[:limit]]
        except Exception:
            pass
        return []

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def _search_crossref(self, query: str, limit: int = 5) -> List[Dict]:
        self.stats["api_calls"] += 1
        self._wait_for_api("crossref")
        try:
            url = "https://api.crossref.org/works"
            params = {"query": query, "rows": limit}
            r = self.session.get(url, params=params, timeout=12)
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", [])
                return [self._format_crossref_result(i) for i in items[:limit]]
        except Exception:
            pass
        return []

    def _search_arxiv(self, query: str, limit: int = 3) -> List[Dict]:
        self.stats["api_calls"] += 1
        self._wait_for_api("arxiv")
        try:
            url = "http://export.arxiv.org/api/query"
            params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}
            r = self.session.get(url, params=params, timeout=25)
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
                return results
        except Exception:
            pass
        return []

    def _search_openalex(self, query: str, limit: int = 5) -> List[Dict]:
        self.stats["api_calls"] += 1
        self._wait_for_api("openalex")
        try:
            url = "https://api.openalex.org/works"
            params = {"search": query, "per_page": limit}
            r = self.session.get(url, params=params, timeout=12)
            if r.status_code == 200:
                data = r.json().get("results", [])
                return [self._format_openalex_result(p) for p in data[:limit]]
        except Exception:
            pass
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
        if method == "cosine":
            return self._cosine_similarity(text1, text2)
        if method == "jaccard":
            return self._jaccard_similarity(text1, text2)
        if method == "cross_encoder":
            return self._cross_encoder_similarity(text1, text2)
        return self._ensemble_similarity(text1, text2)

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
            score = self.cross_encoder.predict([[t1[:500], t2[:500]]])
            return float(score[0])
        except Exception:
            return self._cosine_similarity(t1, t2)

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
        return self.local_cache.get(key)

    def _set_cached(self, key: str, value, ttl: int = 3600):
        if self.redis_client:
            try: self.redis_client.setex(key, ttl, pickle.dumps(value))
            except Exception: pass
        self.local_cache[key] = value
        if len(self.local_cache) > 5000:
            self._save_cache()
            self.local_cache.clear()

    def get_stats(self) -> Dict:
        return self.stats
