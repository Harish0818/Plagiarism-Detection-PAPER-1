import logging
import math
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, some AI detection features will be limited")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class NextGenAIDetector:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing NextGenAIDetector on %s", self.device)

        self.tokenizer = None
        self.model = None
        self.classifier = None

        if NLTK_AVAILABLE:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass

        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=False,
                ).to(self.device)
                self.model.eval()
            except Exception as exc:
                logger.error("Failed to load causal LM %s: %s", model_id, exc)
                self.tokenizer = None
                self.model = None

            try:
                self.classifier = self._load_classifier_pipeline()
            except Exception as exc:
                logger.warning("AI classifier unavailable: %s", exc)
                self.classifier = None

        self.config = {
            "sentence_threshold": 0.62,
            "chunk_size": 250,
            "min_words_for_flag": 8,
            "max_length": 512,
            "ensemble_weights": {
                "perplexity": 0.25,
                "entropy": 0.35,
                "classifier": 0.40,
            },
        }

    def _load_classifier_pipeline(self):
        classifier_model_id = "desklib/ai-text-detector-v1.01"
        classifier_device = 0 if self.device == "cuda" else -1
        try:
            return pipeline(
                "text-classification",
                model=classifier_model_id,
                device=classifier_device,
            )
        except Exception as exc:
            if "state dictionary" not in str(exc).lower() or not TRANSFORMERS_AVAILABLE:
                raise

            logger.warning("Classifier cache appears corrupted, forcing clean re-download.")
            tokenizer = AutoTokenizer.from_pretrained(
                classifier_model_id,
                force_download=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                classifier_model_id,
                force_download=True,
            )
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=classifier_device,
            )

    def _filter_technical_content(self, text: str) -> str:
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"St\.? ?Joseph's Institute of Technology", "", text, flags=re.I)
        text = re.sub(r"(Abstract|Introduction|Conclusion|References)", "", text, flags=re.I)
        return text.strip()

    def _calculate_stylometrics(self, text: str) -> Dict[str, float]:
        sentences = self._split_into_sentences(text)
        words = text.split()
        sentence_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
        burstiness = float(np.std(sentence_lengths)) if sentence_lengths else 0.0
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / len(words) if words else 0.0
        return {
            "burstiness": round(burstiness, 2),
            "lexical_diversity": round(ttr, 3),
            "avg_sentence_length": round(np.mean(sentence_lengths), 2) if sentence_lengths else 0,
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        clean_text = " ".join(text.split()).strip()
        if not clean_text:
            return []
        if NLTK_AVAILABLE:
            try:
                return [s.strip() for s in sent_tokenize(clean_text) if s.strip()]
            except Exception:
                pass
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]

    def _score_with_lm(self, text: str) -> Dict[str, float]:
        if not text or not self.model or not self.tokenizer:
            return {"log_likelihood": 0.0, "perplexity": 1.0, "entropy": 0.0}

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config["max_length"],
                padding=False,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])

            log_likelihood = -float(outputs.loss.item())
            perplexity = float(math.exp(-log_likelihood))

            logits = outputs.logits[:, :-1, :]
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean().item()
            return {
                "log_likelihood": log_likelihood,
                "perplexity": perplexity,
                "entropy": float(entropy),
            }
        except Exception as exc:
            logger.error("LM scoring failed: %s", exc)
            return {"log_likelihood": 0.0, "perplexity": 1.0, "entropy": 0.0}

    def get_log_likelihood(self, text: str) -> float:
        return self._score_with_lm(text).get("log_likelihood", 0.0)

    def _classifier_score(self, text: str) -> float:
        if not text or not self.classifier:
            return 0.0
        try:
            output = self.classifier(text[:1500], truncation=True)[0]
            label = str(output.get("label", "")).lower()
            score = float(output.get("score", 0.0))
            if "human" in label:
                return 1.0 - score
            if "ai" in label or "generated" in label:
                return score
            return score
        except Exception as exc:
            logger.error("Classifier scoring failed: %s", exc)
            return 0.0

    @staticmethod
    def _minmax_scale(values: List[float], invert: bool = False) -> List[float]:
        if not values:
            return []
        arr = np.array(values, dtype=float)
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if abs(vmax - vmin) < 1e-9:
            scaled = np.full_like(arr, 0.5, dtype=float)
        else:
            scaled = (arr - vmin) / (vmax - vmin)
        if invert:
            scaled = 1.0 - scaled
        return [float(x) for x in scaled]

    def _merge_flagged_sentences(self, sentences: List[Dict[str, Any]]) -> List[str]:
        segments: List[str] = []
        current: List[str] = []
        for sentence in sentences:
            if sentence.get("is_ai"):
                current.append(sentence["text"])
                continue
            if current:
                segments.append(" ".join(current))
                current = []
        if current:
            segments.append(" ".join(current))
        return segments

    def detect_ai_curvature(self, text: str) -> Dict[str, Any]:
        clean_text = self._filter_technical_content(text)
        if len(clean_text.split()) < 5:
            return {"is_ai": False, "confidence": 0.0, "curvature_score": 0.0, "stylometrics": {}}

        sentence_scores = self._analyze_sentences(clean_text)
        if not sentence_scores:
            return {"is_ai": False, "confidence": 0.0, "curvature_score": 0.0, "stylometrics": {}}

        mean_score = float(np.mean([s["ensemble_score"] for s in sentence_scores]))
        burstiness = float(np.std([s["perplexity"] for s in sentence_scores])) if sentence_scores else 0.0
        return {
            "is_ai": mean_score >= self.config["sentence_threshold"],
            "confidence": round(mean_score, 4),
            "curvature_score": round(mean_score, 4),
            "burstiness": round(burstiness, 4),
            "sentence_scores": sentence_scores,
            "stylometrics": self._calculate_stylometrics(clean_text),
        }

    def _analyze_sentences(self, text: str) -> List[Dict[str, Any]]:
        sentences = self._split_into_sentences(text)
        filtered = [sentence for sentence in sentences if len(sentence.split()) >= 3]
        if not filtered:
            return []

        raw_rows = []
        for sentence in filtered:
            lm_scores = self._score_with_lm(sentence)
            raw_rows.append(
                {
                    "text": sentence,
                    "log_likelihood": lm_scores["log_likelihood"],
                    "perplexity": lm_scores["perplexity"],
                    "entropy": lm_scores["entropy"],
                    "classifier_score": self._classifier_score(sentence),
                }
            )

        perplexity_scores = self._minmax_scale([row["perplexity"] for row in raw_rows], invert=True)
        entropy_scores = self._minmax_scale([row["entropy"] for row in raw_rows], invert=True)
        classifier_scores = [float(np.clip(row["classifier_score"], 0.0, 1.0)) for row in raw_rows]

        sentence_scores: List[Dict[str, Any]] = []
        for idx, row in enumerate(raw_rows):
            ensemble_score = (
                self.config["ensemble_weights"]["perplexity"] * perplexity_scores[idx]
                + self.config["ensemble_weights"]["entropy"] * entropy_scores[idx]
                + self.config["ensemble_weights"]["classifier"] * classifier_scores[idx]
            )
            sentence_scores.append(
                {
                    "text": row["text"],
                    "log_likelihood": round(row["log_likelihood"], 4),
                    "perplexity": round(row["perplexity"], 4),
                    "entropy": round(row["entropy"], 4),
                    "perplexity_score": round(perplexity_scores[idx], 4),
                    "entropy_score": round(entropy_scores[idx], 4),
                    "classifier_score": round(classifier_scores[idx], 4),
                    "ensemble_score": round(float(ensemble_score), 4),
                    "is_ai": bool(ensemble_score >= self.config["sentence_threshold"]),
                }
            )
        return sentence_scores

    def analyze_document_integrity(self, text: str) -> Dict[str, Any]:
        clean_text = self._filter_technical_content(text)
        total_word_count = len(clean_text.split())
        sentence_scores = self._analyze_sentences(clean_text)

        if not sentence_scores:
            return {
                "ai_percentage": 0.0,
                "mean_risk": 0.0,
                "ai_segments": [],
                "stylometrics": self._calculate_stylometrics(clean_text),
                "sentence_scores": [],
                "burstiness": 0.0,
                "avg_perplexity": 0.0,
                "avg_entropy": 0.0,
                "classifier_mean_score": 0.0,
                "ensemble_threshold": self.config["sentence_threshold"],
            }

        perplexities = [row["perplexity"] for row in sentence_scores]
        entropies = [row["entropy"] for row in sentence_scores]
        classifier_scores = [row["classifier_score"] for row in sentence_scores]
        ensemble_scores = [row["ensemble_score"] for row in sentence_scores]

        ai_word_count = sum(
            len(row["text"].split())
            for row in sentence_scores
            if row["is_ai"] and len(row["text"].split()) >= self.config["min_words_for_flag"]
        )
        ai_percentage = (ai_word_count / total_word_count) if total_word_count > 0 else 0.0
        mean_risk = float(np.mean(ensemble_scores)) if ensemble_scores else 0.0
        burstiness = float(np.std(perplexities)) if perplexities else 0.0
        ai_segments = self._merge_flagged_sentences(sentence_scores)

        return {
            "ai_percentage": round(ai_percentage, 4),
            "mean_risk": round(mean_risk, 4),
            "ai_segments": ai_segments,
            "stylometrics": self._calculate_stylometrics(clean_text),
            "sentence_scores": sentence_scores,
            "burstiness": round(burstiness, 4),
            "avg_perplexity": round(float(np.mean(perplexities)) if perplexities else 0.0, 4),
            "avg_entropy": round(float(np.mean(entropies)) if entropies else 0.0, 4),
            "classifier_mean_score": round(float(np.mean(classifier_scores)) if classifier_scores else 0.0, 4),
            "ensemble_threshold": self.config["sentence_threshold"],
        }

    def _split_into_chunks(self, text: str) -> List[str]:
        sentences = self._split_into_sentences(text)
        chunks: List[str] = []
        current: List[str] = []
        count = 0
        for sentence in sentences:
            w_count = len(sentence.split())
            if count + w_count > self.config["chunk_size"] and current:
                chunks.append(" ".join(current))
                current, count = [], 0
            current.append(sentence)
            count += w_count
        if current:
            chunks.append(" ".join(current))
        return chunks


_detector_singleton: Optional[NextGenAIDetector] = None


def get_detector() -> NextGenAIDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = NextGenAIDetector()
    return _detector_singleton


def detect_ai_content(text: str, threshold: float = 0.45) -> Dict[str, Any]:
    detector = get_detector()
    return detector.analyze_document_integrity(text)
