# plagiarism.py
"""
Next-Generation Plagiarism Detector using Siamese SBERT Manifold Alignment.
Detects verbatim, paraphrase, and structural 'Idea Theft'.
"""
import logging
import hashlib
import torch
import numpy as np
import re
import importlib
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import asyncio
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed, wait
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with", "within", "without", "which", "while",
}

DOMAIN_VOCABULARIES = {
    "computer_science": {
        "algorithm", "algorithms", "neural", "network", "networks", "dataset", "datasets",
        "classification", "learning", "transformer", "embedding", "embeddings", "token",
        "tokens", "optimization", "inference", "model", "models", "architecture", "accuracy",
    },
    "medical": {
        "patient", "patients", "clinical", "therapy", "treatment", "diagnosis", "diagnostic",
        "symptom", "symptoms", "hospital", "disease", "diseases", "medical", "medicine",
        "trial", "trials", "biomarker", "epidemic", "prognosis", "pathology",
    },
    "legal": {
        "statute", "statutes", "plaintiff", "defendant", "contract", "contracts", "court",
        "courts", "jurisdiction", "liability", "evidence", "regulation", "regulations",
        "compliance", "clause", "clauses", "law", "legal", "appeal", "tribunal",
    },
    "physics": {
        "quantum", "particle", "particles", "momentum", "wave", "waves", "relativity",
        "thermodynamics", "entropy", "photon", "photons", "field", "fields", "energy",
        "energies", "simulation", "simulations", "mechanics", "spectra", "spectrum",
    },
    "business": {
        "market", "markets", "finance", "financial", "consumer", "consumers", "management",
        "strategy", "strategic", "revenue", "investment", "investments", "business",
        "leadership", "organizational", "productivity", "sales", "profit", "demand", "supply",
    },
}

def _dynamic_import(name: str):
    """Safely import module by trying multiple locations."""
    try:
        return importlib.import_module(name)
    except Exception:
        pass
    try:
        return importlib.import_module(f"utils.{name}")
    except Exception:
        pass
    candidates = [
        Path(__file__).resolve().parent / f"{name}.py",
        Path(__file__).resolve().parent.parent / "utils" / f"{name}.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location(name, str(p))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(f"Cannot import module {name}")

class PlagiarismDetector:
    def __init__(self, searcher=None):
        # 1. Advanced Manifold Alignment Model
        # Multilingual MPNet is the gold standard for semantic similarity
        try:
            self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info("SBERT Manifold Alignment initialized on %s", self.device)
        except Exception as e:
            logger.error("Failed to load SBERT: %s", e)
            self.model = None

        # 2. Dynamic Search Integration
        try:
            if searcher is not None:
                self.searcher = searcher
            else:
                mod = _dynamic_import("academic_search")
                search_cls = getattr(mod, "AcademicSearch")
                self.searcher = search_cls.get_shared_instance() if hasattr(search_cls, "get_shared_instance") else search_cls()
        except Exception as e:
            logger.warning("AcademicSearch unavailable: %s", e)
            self.searcher = None

        # 3. Text Processor Integration
        try:
            tp_mod = _dynamic_import("text_processing")
            self.tp = getattr(tp_mod, "get_text_processor")()
        except Exception:
            self.tp = None

        # 4. Async Event Loop Management for Search Integration
        try:
            # Attempt to get the existing loop (needed for Streamlit environments)
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new loop if one doesn't exist in the current thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.match_cache = {}
        self.min_chunk_words = 10
        self.optimal_chunk_words = 50
        self.max_optimal_chunk_words = 140
        self.target_chunk_count = 36
        self.max_ranked_chunks_to_search = 30
        self.min_short_chunk_words = 15
        self.max_candidate_pool = 18
        self.search_results_per_query = 8
        self.max_queries_per_chunk = 2
        self.max_sources_per_query = 2
        self.chunk_retrieval_timeout_sec = 45.0
        self.backend_query_timeout_sec = 6.0
        self.citation_penalty = 0.12
        self.short_chunk_penalty = 0.10
        self.technical_vocab_boost = 0.05
        self.technical_density_threshold = 0.18
        self.min_act_threshold = 0.40
        self.max_act_threshold = 0.85
        self.domain_low_confidence = 0.26
        self.domain_flatten_strength = 0.45
        self.database_weight_ratio = 0.20
        self.database_priority = {
            "computer_science": {"arxiv": 1.0, "semantic_scholar": 0.9, "openalex": 0.6, "crossref": 0.5},
            "medical": {"crossref": 1.0, "openalex": 0.9, "semantic_scholar": 0.8, "arxiv": 0.4},
            "physics": {"arxiv": 1.0, "semantic_scholar": 0.9, "openalex": 0.6, "crossref": 0.5},
            "legal": {"crossref": 0.9, "openalex": 0.8, "semantic_scholar": 0.6, "arxiv": 0.2},
            "business": {"crossref": 0.9, "openalex": 0.8, "semantic_scholar": 0.7, "arxiv": 0.3},
            "general": {"semantic_scholar": 0.85, "openalex": 0.8, "crossref": 0.8, "arxiv": 0.65},
        }
        self.candidate_cache = {}
        self.rerank_cache = {}
        self.chunk_priority_cache = {}
        self.query_routing_cache = {}
        self.chunk_search_workers = 4
        self.chunk_search_executor = ThreadPoolExecutor(max_workers=self.chunk_search_workers)
        self.routing_search_executor = ThreadPoolExecutor(max_workers=4)
        self.fallback_search_executor = ThreadPoolExecutor(max_workers=2)

    def check_plagiarism(self, text: str, threshold: Optional[float] = None) -> List[Dict]:
        """
        Main API: Detects both lexical overlap and 'Idea Theft' via adaptive thresholding.
        """
        if not text or not text.strip() or not self.model:
            return []

        raw_chunks = self._process_text_into_chunks(text)
        chunks = raw_chunks
        if not chunks:
            return []
        chunks = self._prepare_unique_ranked_chunks(chunks)
        logger.info("Plagiarism chunk prep: raw=%s unique_ranked=%s", len(raw_chunks), len(chunks))

        chunk_analyses = []
        best_scores = []
        precomputed_candidates = self._retrieve_candidates_for_chunks(chunks)

        for chunk in chunks:
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            if chunk_hash in self.match_cache:
                analysis = self.match_cache[chunk_hash]
            else:
                candidates = precomputed_candidates.get(chunk, [])
                scored_candidates = self._align_manifold(chunk, candidates)
                analysis = {
                    "chunk": chunk,
                    "scored_candidates": scored_candidates,
                    "best_score": scored_candidates[0]["score"] if scored_candidates else 0.0,
                    "has_citation": self._has_citation_pattern(chunk),
                    "word_count": len(chunk.split()),
                    "technical_density": self._technical_density(chunk),
                    "domain": candidates[0].get("matched_domain", "general") if candidates else "general",
                    "domain_confidence": candidates[0].get("domain_confidence", 0.0) if candidates else 0.0,
                }
                self.match_cache[chunk_hash] = analysis

            chunk_analyses.append(analysis)
            if analysis["best_score"] > 0:
                best_scores.append(analysis["best_score"])

        base_threshold = self._calculate_dynamic_threshold(best_scores)
        all_matches = []

        for analysis in chunk_analyses:
            chunk_threshold = self._apply_context_layers(
                base_threshold,
                has_citation=analysis["has_citation"],
                word_count=analysis["word_count"],
                technical_density=analysis["technical_density"],
            )
            chunk_matches = self._finalize_chunk_matches(
                analysis["chunk"],
                analysis["scored_candidates"],
                chunk_threshold,
                base_threshold,
                analysis["has_citation"],
                analysis["technical_density"],
            )
            all_matches.extend(chunk_matches)

        return self._deduplicate(all_matches)

    def _retrieve_candidates_for_chunks(self, chunks: List[str]) -> Dict[str, List[Dict]]:
        if not chunks:
            return {}

        results: Dict[str, List[Dict]] = {}
        pending_chunks = []
        for chunk in chunks:
            cached = self.candidate_cache.get(chunk)
            if cached is not None:
                results[chunk] = cached
            else:
                pending_chunks.append(chunk)

        if not pending_chunks:
            return results

        logger.info("Starting plagiarism candidate retrieval for %s chunks", len(pending_chunks))
        completed = 0
        future_map = {self.chunk_search_executor.submit(self._retrieve_candidates, chunk): chunk for chunk in pending_chunks}
        pending_futures = set(future_map)
        deadline = time.monotonic() + self.chunk_retrieval_timeout_sec
        while pending_futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, pending_futures = wait(
                pending_futures,
                timeout=min(2.0, remaining),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                chunk = future_map[future]
                try:
                    results[chunk] = future.result() or []
                except Exception as exc:
                    logger.error("Candidate retrieval failed for chunk: %s", exc)
                    results[chunk] = []
                completed += 1
                if completed == len(pending_chunks) or completed % 5 == 0:
                    logger.info("Plagiarism candidate retrieval progress: %s/%s chunks", completed, len(pending_chunks))

        if pending_futures:
            logger.warning(
                "Plagiarism candidate retrieval timed out after %.1fs; returning partial results for %s/%s chunks",
                self.chunk_retrieval_timeout_sec,
                completed,
                len(pending_chunks),
            )
            for future in pending_futures:
                results[future_map[future]] = []
        return results

    def _align_manifold(self, chunk: str, candidates: list) -> List[Dict]:
        """
        Uses Siamese SBERT plus lexical/cross-encoder reranking to score candidates.
        """
        if not candidates:
            return []

        chunk_emb = self.model.encode(chunk, convert_to_tensor=True)
        source_texts = [c.get("abstract") or c.get("title") or "" for c in candidates]
        source_embs = self.model.encode(source_texts, convert_to_tensor=True)
        cosine_scores = util.cos_sim(chunk_emb, source_embs)[0]
        cross_scores = self._batch_cross_encoder_scores(chunk, source_texts)

        scored_candidates = []
        for i, score in enumerate(cosine_scores):
            semantic_score = float(score)
            candidate_text = source_texts[i]
            lexical_score = self._lexical_overlap(chunk, candidate_text)
            cross_score = cross_scores[i]
            final_score = self._ensemble_score(semantic_score, lexical_score, cross_score)
            candidate = dict(candidates[i])
            scored_candidates.append(
                {
                    "candidate": candidate,
                    "score": round(final_score, 3),
                    "semantic_score": round(semantic_score, 3),
                    "lexical_score": round(lexical_score, 3),
                    "cross_score": round(cross_score, 3),
                }
            )

        return sorted(scored_candidates, key=lambda item: item["score"], reverse=True)

    def _retrieve_candidates(self, chunk: str) -> List[Dict]:
        """
        Improve recall by searching the same chunk with multiple query forms and domain-weighted routing.
        """
        if not self.searcher:
            return []
        cached = self.candidate_cache.get(chunk)
        if cached is not None:
            return cached

        domain_info = self._classify_chunk_domain(chunk)
        db_weights = self._build_dynamic_db_weights(domain_info["domain"], domain_info["confidence"])
        queries = self._build_search_queries(chunk)
        merged = []
        for query in queries:
            merged.extend(
                self._search_with_routing(
                    query,
                    db_weights,
                    matched_domain=domain_info["domain"],
                    domain_confidence=domain_info["confidence"],
                )
            )
            if len(self._deduplicate_candidates(merged)) >= self.max_candidate_pool:
                break

        deduped = self._deduplicate_candidates(merged)
        if not deduped:
            return []

        reranked = self._rerank_candidates(chunk, deduped, db_weights)
        final_candidates = reranked[: self.max_candidate_pool]
        self.candidate_cache[chunk] = final_candidates
        return final_candidates

    def _build_search_queries(self, chunk: str) -> List[str]:
        normalized = " ".join(chunk.split())
        queries = [normalized[:300]]

        lead_words = normalized.split()[:18]
        if lead_words:
            queries.append(" ".join(lead_words))

        keywords = self._extract_keywords(normalized, limit=10)
        if keywords:
            queries.append(" ".join(keywords))

        compact = []
        seen = set()
        signatures = []
        for query in queries:
            cleaned = " ".join(query.split()).strip()
            if len(cleaned) < 20:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            signature = self._query_signature(cleaned)
            if any(self._signature_overlap(signature, prev_signature) >= 0.85 for prev_signature in signatures):
                continue
            seen.add(key)
            signatures.append(signature)
            compact.append(cleaned)
        return compact[: self.max_queries_per_chunk]

    def _query_signature(self, query: str) -> set:
        normalized = self._normalize_chunk_for_dedup(query)
        signature = self._build_chunk_signature(normalized)
        if signature:
            return signature
        return set(normalized.split())

    def _extract_keywords(self, text: str, limit: int = 10) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
        weighted = []
        seen = set()
        for token in tokens:
            if token in STOP_WORDS or token in seen:
                continue
            seen.add(token)
            weight = len(token)
            if token.endswith(("tion", "ment", "ology", "graphy", "model", "system", "network")):
                weight += 4
            weighted.append((weight, token))
        weighted.sort(reverse=True)
        return [token for _, token in weighted[:limit]]

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        unique = []
        seen = set()
        for candidate in candidates:
            key = (
                (candidate.get("doi") or "").lower(),
                hashlib.md5((candidate.get("title") or "").strip().lower().encode()).hexdigest()[:12],
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _rerank_candidates(self, chunk: str, candidates: List[Dict], db_weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        if not candidates or not self.searcher:
            return candidates
        normalized_chunk = self._normalize_chunk_for_dedup(chunk)[:500]
        candidate_signature = ",".join(
            self._normalize_chunk_for_dedup((c.get("doi") or c.get("title") or "")[:120])
            for c in candidates[:25]
        )
        cache_key = hashlib.md5(
            f"{normalized_chunk}::{candidate_signature}".encode()
        ).hexdigest()
        cached = self.rerank_cache.get(cache_key)
        if cached is not None:
            return [dict(candidate) for candidate in cached]

        pair_texts = [(chunk, candidate.get("abstract") or candidate.get("title") or "") for candidate in candidates]
        semantic_hints = self.searcher.batch_calculate_similarity(pair_texts, method="cosine")
        cross_scores = self.searcher.batch_calculate_similarity(pair_texts, method="cross_encoder")
        ranked = []
        for candidate, semantic_hint, cross_score in zip(candidates, semantic_hints, cross_scores):
            candidate_text = candidate.get("abstract") or candidate.get("title") or ""
            lexical_score = self._lexical_overlap(chunk, candidate_text)
            rank_score = self._ensemble_score(semantic_hint, lexical_score, cross_score)
            database_weight = self._resolve_database_weight(candidate.get("source", "unknown"), db_weights)
            rank_score = (1 - self.database_weight_ratio) * rank_score + (self.database_weight_ratio * database_weight)
            candidate_copy = dict(candidate)
            candidate_copy["database_weight"] = round(database_weight, 3)
            ranked.append((rank_score, candidate_copy))
        ranked.sort(key=lambda item: item[0], reverse=True)
        ordered = [candidate for _, candidate in ranked]
        self.rerank_cache[cache_key] = [dict(candidate) for candidate in ordered]
        return ordered

    def _classify_chunk_domain(self, chunk: str) -> Dict[str, float]:
        tokens = [tok for tok in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", chunk.lower()) if tok not in STOP_WORDS]
        if not tokens:
            return {"domain": "general", "confidence": 0.0}

        scores = {}
        token_count = max(len(tokens), 1)
        for domain, vocabulary in DOMAIN_VOCABULARIES.items():
            overlap = sum(1 for token in tokens if token in vocabulary)
            scores[domain] = overlap / token_count

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]
        sorted_scores = sorted(scores.values(), reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = max(0.0, best_score - second_score) + (0.5 * best_score)
        confidence = self._clip(confidence, 0.0, 1.0)
        if best_score <= 0:
            return {"domain": "general", "confidence": 0.0}
        return {"domain": best_domain, "confidence": confidence}

    def _build_dynamic_db_weights(self, domain: str, confidence: float) -> Dict[str, float]:
        base = dict(self.database_priority.get(domain, self.database_priority["general"]))
        if confidence < self.domain_low_confidence:
            flattened = {}
            neutral = 0.75
            for source, weight in base.items():
                flattened[source] = weight * (1 - self.domain_flatten_strength) + neutral * self.domain_flatten_strength
            return flattened
        multiplier = 0.6 + (0.4 * confidence)
        return {source: round(weight * multiplier, 3) for source, weight in base.items()}

    def _search_with_routing(
        self,
        query: str,
        db_weights: Dict[str, float],
        matched_domain: str,
        domain_confidence: float,
    ) -> List[Dict]:
        normalized_query = self._normalize_chunk_for_dedup(query)
        cache_key = hashlib.md5(
            f"{normalized_query}::{matched_domain}::{round(float(domain_confidence), 2)}".encode()
        ).hexdigest()
        cached = self.query_routing_cache.get(cache_key)
        if cached is not None:
            return [dict(result) for result in cached]

        ordered_sources = sorted(db_weights.items(), key=lambda item: item[1], reverse=True)
        ordered_sources = ordered_sources[: self.max_sources_per_query]
        merged = []
        future_map = {
            self.routing_search_executor.submit(self._search_single_backend, source_name, query): (source_name, weight)
            for source_name, weight in ordered_sources
        }
        pending_futures = set(future_map)
        deadline = time.monotonic() + self.backend_query_timeout_sec
        while pending_futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, pending_futures = wait(
                pending_futures,
                timeout=min(1.5, remaining),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                source_name, weight = future_map[future]
                try:
                    results = future.result() or []
                except Exception as e:
                    logger.error("Backend search failed for %s: %s", source_name, e)
                    results = []
                if not results:
                    continue
                for result in results:
                    result["database_weight"] = round(weight, 3)
                    result["matched_domain"] = matched_domain
                    result["domain_confidence"] = round(domain_confidence, 3)
                merged.extend(results)

            if len(self._deduplicate_candidates(merged)) >= self.max_candidate_pool:
                break

        if merged:
            deduped = self._deduplicate_candidates(merged)
            self.query_routing_cache[cache_key] = [dict(result) for result in deduped]
            return deduped

        try:
            fallback = self._run_parallel_search_fallback(query)
        except Exception as e:
            logger.error("Fallback search failed for query '%s': %s", query[:80], e)
            fallback = []

        for result in fallback or []:
            source_name = result.get("source", "unknown")
            result["database_weight"] = round(self._resolve_database_weight(source_name, db_weights), 3)
            result["matched_domain"] = matched_domain
            result["domain_confidence"] = round(domain_confidence, 3)
        fallback = self._deduplicate_candidates(fallback or [])
        self.query_routing_cache[cache_key] = [dict(result) for result in fallback]
        return fallback

    def _run_parallel_search_fallback(self, query: str) -> List[Dict]:
        if not self.searcher:
            return []

        future = self.fallback_search_executor.submit(
            lambda: asyncio.run(self.searcher.search_parallel(query, max_results=self.search_results_per_query))
        )
        try:
            return future.result(timeout=self.backend_query_timeout_sec + 1.0) or []
        except FutureTimeoutError:
            logger.warning("Fallback search timed out for query '%s'", query[:80])
            return []

    def _search_single_backend(self, source_name: str, query: str) -> List[Dict]:
        backend_map = {
            "semantic_scholar": "_search_semantic_scholar",
            "crossref": "_search_crossref",
            "arxiv": "_search_arxiv",
            "openalex": "_search_openalex",
        }
        method_name = backend_map.get(source_name)
        if not method_name:
            return []
        method = getattr(self.searcher, method_name, None)
        if not callable(method):
            return []
        try:
            return method(query, self.search_results_per_query) or []
        except Exception as e:
            logger.error("Backend search failed for %s: %s", source_name, e)
            return []

    def _resolve_database_weight(self, source_name: str, db_weights: Optional[Dict[str, float]]) -> float:
        if not db_weights:
            return 1.0
        return float(db_weights.get(source_name, 0.7))

    def _calculate_dynamic_threshold(self, best_scores: List[float]) -> float:
        if not best_scores:
            return self._clip(
                (self.min_act_threshold + self.max_act_threshold) / 2,
                self.min_act_threshold,
                self.max_act_threshold,
            )
        mu = float(np.mean(best_scores))
        sigma = float(np.std(best_scores))
        act_threshold = mu + (1.5 * sigma)
        return self._clip(act_threshold, self.min_act_threshold, self.max_act_threshold)

    def _apply_context_layers(
        self,
        base_threshold: float,
        has_citation: bool,
        word_count: int,
        technical_density: float,
    ) -> float:
        threshold = float(base_threshold)
        if has_citation:
            threshold += self.citation_penalty
        if word_count < self.min_short_chunk_words:
            threshold += self.short_chunk_penalty
        if technical_density >= self.technical_density_threshold:
            threshold += self.technical_vocab_boost
        return self._clip(threshold, self.min_act_threshold, self.max_act_threshold)

    def _finalize_chunk_matches(
        self,
        chunk: str,
        scored_candidates: List[Dict],
        threshold: float,
        base_threshold: float,
        has_citation: bool,
        technical_density: float,
    ) -> List[Dict]:
        if not scored_candidates:
            return []

        top_candidates = scored_candidates[:3]
        matches = []
        for scored in top_candidates:
            score_val = float(scored["score"])
            if score_val < threshold:
                continue

            semantic_score = float(scored["semantic_score"])
            lexical_score = float(scored["lexical_score"])
            cross_score = float(scored["cross_score"])
            if semantic_score >= 0.94 or lexical_score >= 0.82:
                match_type = "Verbatim"
            elif cross_score >= 0.82 or semantic_score >= 0.80:
                match_type = "Semantic/Idea Theft"
            else:
                match_type = "Contextual Similarity"

            candidate = scored["candidate"]
            matches.append(
                {
                    "text_chunk": chunk,
                    "score": round(score_val, 3),
                    "match_type": match_type,
                    "source": candidate.get("title", "Unknown Source"),
                    "source_backend": candidate.get("source", "unknown"),
                    "url": candidate.get("url", ""),
                    "details": {
                        "manifold_distance": round(1 - semantic_score, 4),
                        "source_year": candidate.get("year"),
                        "semantic_score": round(semantic_score, 3),
                        "lexical_score": round(lexical_score, 3),
                        "cross_score": round(cross_score, 3),
                        "base_threshold": round(base_threshold, 3),
                        "applied_threshold": round(threshold, 3),
                        "citation_adjusted": has_citation,
                        "technical_density": round(float(technical_density), 3),
                        "matched_domain": candidate.get("matched_domain", "general"),
                        "domain_confidence": round(float(candidate.get("domain_confidence", 0.0)), 3),
                        "database_weight": round(float(candidate.get("database_weight", 1.0)), 3),
                    },
                }
            )
        return matches

    def _has_citation_pattern(self, text: str) -> bool:
        patterns = [
            r"\([A-Z][A-Za-z]+,\s*\d{4}\)",
            r"\[[0-9,\-\s]+\]",
            r"\b[A-Z][A-Za-z]+\s+et\s+al\.\s*\(\d{4}\)",
            r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
        ]
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _technical_density(self, text: str) -> float:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        if not tokens:
            return 0.0
        technical_terms = 0
        for token in tokens:
            if token in STOP_WORDS:
                continue
            if len(token) >= 9 or token.endswith(
                ("tion", "sion", "ment", "graph", "metric", "model", "system", "network", "vector", "semantic")
            ):
                technical_terms += 1
        return technical_terms / max(len(tokens), 1)

    def _lexical_overlap(self, text1: str, text2: str) -> float:
        words1 = {w for w in re.findall(r"\w+", text1.lower()) if w not in STOP_WORDS}
        words2 = {w for w in re.findall(r"\w+", text2.lower()) if w not in STOP_WORDS}
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def _cross_encoder_score(self, text1: str, text2: str) -> float:
        if not self.searcher:
            return 0.0
        return self._safe_similarity(text1, text2, method="cross_encoder")

    def _batch_cross_encoder_scores(self, text1: str, texts2: List[str]) -> List[float]:
        if not self.searcher or not texts2:
            return [0.0 for _ in texts2]
        pairs = [(text1, text2) for text2 in texts2]
        return self.searcher.batch_calculate_similarity(pairs, method="cross_encoder")

    def _safe_similarity(self, text1: str, text2: str, method: str) -> float:
        try:
            return float(self.searcher.calculate_similarity(text1, text2, method=method))
        except Exception:
            return 0.0

    @staticmethod
    def _ensemble_score(semantic_score: float, lexical_score: float, cross_score: float) -> float:
        if cross_score > 0:
            return (0.55 * semantic_score) + (0.20 * lexical_score) + (0.25 * cross_score)
        return (0.75 * semantic_score) + (0.25 * lexical_score)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _process_text_into_chunks(self, text: str) -> List[str]:
        """Splits text into semantically viable chunks for manifold analysis."""
        sentences = []
        if self.tp:
            try:
                sentences = self.tp.split_sentences(text)
            except Exception:
                pass

        if not sentences:
            sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.split()) > 3]

        total_words = sum(len(sentence.split()) for sentence in sentences)
        adaptive_chunk_words = self._get_adaptive_chunk_word_target(total_words)
        chunks = []
        curr, curr_len = [], 0
        for s in sentences:
            words = s.split()
            if curr_len + len(words) > adaptive_chunk_words and curr:
                chunks.append(" ".join(curr))
                curr, curr_len = [s], len(words)
            else:
                curr.append(s)
                curr_len += len(words)
        if curr: chunks.append(" ".join(curr))
        
        return [c for c in chunks if len(c.split()) >= self.min_chunk_words]

    def _get_adaptive_chunk_word_target(self, total_words: int) -> int:
        if total_words <= 0:
            return self.optimal_chunk_words
        estimated = max(self.optimal_chunk_words, int(total_words / max(self.target_chunk_count, 1)))
        return min(self.max_optimal_chunk_words, estimated)

    def _prepare_unique_ranked_chunks(self, chunks: List[str]) -> List[str]:
        """Deduplicate repeated chunk content and process the strongest unique chunks first."""
        normalized_records = []
        seen_exact = set()
        for idx, chunk in enumerate(chunks):
            clean_chunk = " ".join(chunk.split()).strip()
            if not clean_chunk:
                continue
            exact_key = clean_chunk.lower()
            if exact_key in seen_exact:
                continue
            seen_exact.add(exact_key)
            normalized_records.append(
                {
                    "original_index": idx,
                    "chunk": clean_chunk,
                    "normalized": self._normalize_chunk_for_dedup(clean_chunk),
                }
            )

        unique_records = []
        seen_signatures = []
        for record in normalized_records:
            signature = self._build_chunk_signature(record["normalized"])
            if any(self._signature_overlap(signature, prev_signature) >= 0.88 for prev_signature in seen_signatures):
                continue
            seen_signatures.append(signature)
            unique_records.append(record)

        ranked_records = sorted(
            unique_records,
            key=lambda record: (
                -self._score_chunk_priority(record["chunk"]),
                record["original_index"],
            ),
        )

        if len(ranked_records) <= self.max_ranked_chunks_to_search:
            return [record["chunk"] for record in ranked_records]

        selected_records = ranked_records[: self.max_ranked_chunks_to_search]
        selected_keys = {record["chunk"].lower() for record in selected_records}
        citation_backfill = [
            record
            for record in ranked_records[self.max_ranked_chunks_to_search :]
            if self._has_citation_pattern(record["chunk"]) and record["chunk"].lower() not in selected_keys
        ]
        selected_records.extend(citation_backfill[:4])
        logger.info(
            "Plagiarism ranked chunk cap applied: using %s of %s chunks",
            len(selected_records),
            len(ranked_records),
        )
        return [record["chunk"] for record in selected_records]

    @staticmethod
    def _normalize_chunk_for_dedup(chunk: str) -> str:
        chunk = chunk.lower()
        chunk = re.sub(r"\s+", " ", chunk)
        chunk = re.sub(r"[^\w\s]", " ", chunk)
        return " ".join(chunk.split()).strip()

    def _build_chunk_signature(self, normalized_chunk: str) -> set:
        tokens = [token for token in normalized_chunk.split() if token not in STOP_WORDS]
        if len(tokens) <= 12:
            return set(tokens)
        signature = set(tokens[:8] + tokens[-8:])
        keywords = self._extract_keywords(" ".join(tokens), limit=8)
        signature.update(keywords)
        return signature

    @staticmethod
    def _signature_overlap(sig_a: set, sig_b: set) -> float:
        if not sig_a or not sig_b:
            return 0.0
        union = sig_a | sig_b
        if not union:
            return 0.0
        return len(sig_a & sig_b) / len(union)

    def _score_chunk_priority(self, chunk: str) -> float:
        cached = self.chunk_priority_cache.get(chunk)
        if cached is not None:
            return cached

        words = chunk.split()
        word_count = len(words)
        lexical_tokens = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", chunk.lower()) if w not in STOP_WORDS]
        lexical_diversity = len(set(lexical_tokens)) / max(len(lexical_tokens), 1)
        citation_bonus = 0.24 if self._has_citation_pattern(chunk) else 0.0
        technical_density = self._technical_density(chunk)
        technical_bonus = technical_density * 0.55
        keyword_bonus = min(len(self._extract_keywords(chunk, limit=12)) / 12.0, 1.0) * 0.20
        length_bonus = min(word_count / 90.0, 1.0) * 0.16
        diversity_bonus = lexical_diversity * 0.12
        numeric_evidence_bonus = min(len(re.findall(r"\b\d+(?:\.\d+)?%?\b", chunk)) / 6.0, 1.0) * 0.08
        proper_noun_bonus = min(len(re.findall(r"\b[A-Z][a-z]{2,}\b", chunk)) / 10.0, 1.0) * 0.06
        sentence_count = max(1, len([s for s in re.split(r"[.!?]+", chunk) if s.strip()]))
        sentence_balance = min(word_count / sentence_count, 28) / 28.0
        coherence_bonus = sentence_balance * 0.08
        priority_score = (
            citation_bonus
            + technical_bonus
            + keyword_bonus
            + length_bonus
            + diversity_bonus
            + numeric_evidence_bonus
            + proper_noun_bonus
            + coherence_bonus
        )
        if technical_density >= self.technical_density_threshold:
            priority_score += 0.04
        self.chunk_priority_cache[chunk] = priority_score
        return priority_score

    def _deduplicate(self, matches: List[Dict]) -> List[Dict]:
        """Removes overlapping or redundant flags."""
        unique, seen = [], set()
        for m in sorted(matches, key=lambda x: x["score"], reverse=True):
            key = hashlib.md5(
                f"{m.get('source_backend', '')}:{m['source']}:{m['text_chunk'][:30]}".encode()
            ).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(m)
        return unique
