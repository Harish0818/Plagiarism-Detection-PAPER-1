# citation_analyzer.py
import re
import json
import logging
import time
import asyncio
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedCitationAnalyzer:
    """
    Upgraded Citation Analyzer using Natural Language Inference (NLI).
    Verifies if a cited source actually entails the claim made in the text.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "nli_model": "cross-encoder/nli-deberta-v3-small",
            "context_window": 300,  # Increased window for better claim context
            "entailment_threshold": 0.6
        }
        
        # Initialize NLI Logic
        try:
            self.nli_verifier = pipeline(
                "text-classification", 
                model=self.config["nli_model"]
            )
            logger.info(f"NLI Verification Engine loaded: {self.config['nli_model']}")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            self.nli_verifier = None

        self.patterns = self._load_citation_patterns()

    def _load_citation_patterns(self) -> List[Dict]:
        """Expanded patterns to catch a wider variety of citation styles."""
        return [
            # Style: Smith (2020)
            {"name": "author_year", "pattern": r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((\d{4})\)", "groups": ["author", "year"]},
            # Style: [1] or [1, 2, 3]
            {"name": "bracketed", "pattern": r"\[(\d+(?:\s*,\s*\d+)*)\]", "groups": ["number"]},
            # Style: (Smith, 2020) or (Smith et al., 2020)
            {"name": "parenthetical", "pattern": r"\(([A-Z][a-z]+(?:\s+et\s+al\.)?),\s*(\d{4})\)", "groups": ["author", "year"]},
            # Style: Smith et al. (2020)
            {"name": "etal_active", "pattern": r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+et\s+al\.\s*\((\d{4})\)", "groups": ["author", "year"]},
            # Style: DOI strings
            {"name": "doi", "pattern": r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", "groups": ["doi"]}
        ]

    def analyze_citations(self, text: str, academic_search=None, document_id: Optional[str] = None) -> Tuple[Dict[str, Any], List[Tuple]]:
        """
        Extracts citations and verifies claim integrity.
        Handles async search calls using an event loop to fix the 'Zero' results issue.
        """
        start_time = time.time()
        citations = self.extract_citations(text)
        
        if not citations:
            logger.warning("No citations found in the provided text.")
            return self._create_empty_results(), []

        results = {
            "valid": [], 
            "invalid": [], 
            "irrelevant": [], 
            "fraudulent": [], 
            "statistics": {"total": len(citations), "processed": 0}
        }
        edges = []

        # Setup event loop for async search calls
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for c in citations:
            claim_context = c["context"]
            citation_query = c["text"]
            
            source_text = ""
            if academic_search:
                try:
                    # Execute async search and wait for results
                    search_results = loop.run_until_complete(
                        academic_search.search_parallel(citation_query, max_results=1)
                    )
                    if search_results and len(search_results) > 0:
                        # FALLBACK: Use Abstract if available, otherwise use Title for NLI
                        source_text = search_results[0].get("abstract") or search_results[0].get("title") or ""
                except Exception as e:
                    logger.error(f"Async search failed in Citation Analyzer: {e}")
            
            if not source_text:
                results["invalid"].append({"citation": citation_query, "status": "Source Not Found"})
                continue

            verification = self._verify_semantic_support(claim_context, source_text)
            
            citation_data = {
                "citation": citation_query,
                "context": claim_context,
                "status": verification["status"],
                "confidence": verification["confidence"],
                "source_used": "Abstract" if "abstract" in locals() and source_text == search_results[0].get("abstract") else "Title"
            }

            if verification["status"] == "Entailment":
                results["valid"].append(citation_data)
                edges.append((document_id or "Current Doc", citation_query))
            elif verification["status"] == "Contradiction":
                results["fraudulent"].append(citation_data)
                edges.append((document_id or "Current Doc", citation_query))
            else:
                results["irrelevant"].append(citation_data)

            results["statistics"]["processed"] += 1

        results["statistics"]["processing_time"] = time.time() - start_time
        return results, edges

    def _verify_semantic_support(self, claim: str, source_text: str) -> Dict[str, Any]:
        """Uses Natural Language Inference to verify claim support."""
        if not self.nli_verifier:
            return {"status": "Unverified", "confidence": 0.0}
        try:
            # The NLI model compares the 'premise' (source) against the 'hypothesis' (claim)
            nli_output = self.nli_verifier([{'text': source_text, 'text_pair': claim}])[0]
            
            label = nli_output['label'].lower()
            if "entailment" in label:
                status = "Entailment"
            elif "contradiction" in label:
                status = "Contradiction"
            else:
                status = "Neutral"
            return {"status": status, "confidence": round(nli_output['score'], 3)}
        except Exception as e:
            logger.error(f"NLI Verification Error: {e}")
            return {"status": "Error", "confidence": 0.0}

    def extract_citations(self, text: str) -> List[Dict]:
        """Identifies citation strings and captures their surrounding context."""
        found = []
        seen = set()
        for patt in self.patterns:
            for m in re.finditer(patt["pattern"], text):
                txt = m.group(0)
                # Avoid duplicate matches for the same text at the same position
                if (txt, m.start()) in seen: continue
                seen.add((txt, m.start()))
                
                found.append({
                    "text": txt,
                    "start": m.start(),
                    "end": m.end(),
                    "context": self._get_context(text, m.start(), m.end())
                })
        return sorted(found, key=lambda x: x["start"])

    def _get_context(self, text: str, start: int, end: int) -> str:
        """Retrieves text surrounding the citation to act as the 'claim'."""
        window = self.config["context_window"]
        s = max(0, start - window)
        e = min(len(text), end + window)
        return text[s:e].strip()

    def _create_empty_results(self) -> Dict:
        return {"valid": [], "invalid": [], "irrelevant": [], "fraudulent": [], "statistics": {"total": 0}}

# Singleton helper
_analyzer = None
def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = AdvancedCitationAnalyzer()
    return _analyzer