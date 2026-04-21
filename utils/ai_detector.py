import logging
import math
import os
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Config, DebertaV2Model, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, some AI detection features will be limited")

try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DebertaSingleLogitClassifier(nn.Module):
    def __init__(self, config: DebertaV2Config) -> None:
        super().__init__()
        self.model = DebertaV2Model(config)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        cls_token = self.dropout(cls_token)
        return self.classifier(cls_token).squeeze(-1)


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
                lm_source = self._resolve_model_source(model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(lm_source, local_files_only=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    lm_source,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
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
            "sentence_threshold": 0.58,
            "chunk_size": 250,
            "min_words_for_flag": 5,
            "max_length": 512,
            "lm_micro_batch_size": 4 if self.device == "cuda" else 16,
            "classifier_micro_batch_size": 8 if self.device == "cuda" else 32,
            "ensemble_weights": {
                "perplexity": 0.30,
                "entropy": 0.25,
                "classifier": 0.45,
            },
            "perplexity_center": 12.0,
            "perplexity_scale": 4.0,
            "entropy_center": 5.0,
            "entropy_scale": 0.9,
            "classifier_floor": 0.12,
            "document_ai_threshold": 0.22,
            "document_mean_risk_threshold": 0.57,
        }

    def _load_classifier_pipeline(self):
        classifier_model_id = "desklib/ai-text-detector-v1.01"
        source = self._resolve_model_source(classifier_model_id)
        classifier_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)

        if SAFETENSORS_AVAILABLE:
            model_path = os.path.join(source, "model.safetensors")
            config = DebertaV2Config.from_pretrained(source, local_files_only=True)
            classifier_model = DebertaSingleLogitClassifier(config)
            state_dict = load_safetensors_file(model_path)
            classifier_model.load_state_dict(state_dict, strict=True)
            classifier_model.to(self.device)
            classifier_model.eval()
            return {"tokenizer": classifier_tokenizer, "model": classifier_model, "mode": "single_logit"}

        fallback_model = AutoModelForSequenceClassification.from_pretrained(
            source,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
        fallback_model.to(self.device)
        fallback_model.eval()
        return {"tokenizer": classifier_tokenizer, "model": fallback_model, "mode": "transformers"}

    @staticmethod
    def _resolve_model_source(model_id: str) -> str:
        normalized = model_id.replace("/", "--")
        hub_root = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
            "hub",
        )
        snapshot_root = os.path.join(hub_root, f"models--{normalized}", "snapshots")
        if os.path.isdir(snapshot_root):
            snapshots = sorted(os.listdir(snapshot_root))
            if snapshots:
                return os.path.join(snapshot_root, snapshots[-1])
        return model_id

    def _filter_technical_content(self, text: str) -> str:
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"St\.? ?Joseph's Institute of Technology", "", text, flags=re.I)
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

    @staticmethod
    def _is_excluded_nonprose(text: str) -> bool:
        clean_text = " ".join(str(text).split()).strip()
        if not clean_text:
            return True

        lower = clean_text.lower()
        words = clean_text.split()

        if re.search(r"https?://|www\.|doi\.org/|10\.\d{4,9}/", lower):
            return True
        if lower.startswith(("references", "reference", "bibliography", "available:", "doi:", "source:")):
            return True
        if re.match(r"^\s*\[\d+\]", clean_text):
            return True
        if re.search(r"\b(pp?\.)\b", lower) and re.search(r"\b(19|20)\d{2}\b", lower):
            return True
        if re.search(r"\b(ieee|springer|elsevier|conference|journal|proceedings)\b", lower) and re.search(r"\b(19|20)\d{2}\b", lower):
            return True
        if clean_text.count('"') >= 2 and re.search(r"\b(19|20)\d{2}\b", lower):
            return True
        if lower.startswith(("formula", "equation", "eq.", "where:")):
            return True
        if re.match(r"^[A-Z0-9\s:._\-()]{4,}$", clean_text) and len(words) <= 10:
            return True
        if any(token in clean_text for token in ("=", "≈", "∑", "∫", "P(", "w(", "{", "}")):
            alpha_chars = sum(ch.isalpha() for ch in clean_text)
            symbol_chars = sum(not ch.isalnum() and not ch.isspace() for ch in clean_text)
            if symbol_chars >= max(4, alpha_chars // 4):
                return True
        if len(re.findall(r"[=+\-/*^_{}()[\]]", clean_text)) >= 4:
            return True
        alpha_tokens = re.findall(r"[A-Za-z]+", clean_text)
        if len(alpha_tokens) <= 3 and len(clean_text) < 40:
            return True
        return False

    def _sanitize_ai_segment(self, segment: str) -> str:
        clean_segment = " ".join(str(segment).split()).strip()
        if not clean_segment:
            return ""

        try:
            pieces = self._split_into_sentences(clean_segment)
        except Exception:
            pieces = [clean_segment]

        kept: List[str] = []
        for piece in pieces:
            normalized_piece = " ".join(piece.split()).strip()
            if not normalized_piece:
                continue
            if self._is_excluded_nonprose(normalized_piece):
                continue
            kept.append(normalized_piece)

        sanitized = " ".join(kept).strip()
        if not sanitized:
            return ""
        if self._is_excluded_nonprose(sanitized):
            return ""
        if len(sanitized.split()) < self.config["min_words_for_flag"]:
            return ""
        return sanitized

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
            with torch.inference_mode():
                outputs = self.model(**inputs, labels=inputs["input_ids"])

            log_likelihood = -float(outputs.loss.item())
            if not math.isfinite(log_likelihood):
                raise ValueError("Non-finite log likelihood from causal LM")
            perplexity = float(math.exp(-log_likelihood))
            if not math.isfinite(perplexity):
                perplexity = 1e6

            logits = outputs.logits[:, :-1, :]
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean().item()
            if not math.isfinite(entropy):
                raise ValueError("Non-finite entropy from causal LM")
            return {
                "log_likelihood": log_likelihood,
                "perplexity": perplexity,
                "entropy": float(entropy),
            }
        except Exception as exc:
            logger.error("LM scoring failed: %s", exc)
            return {"log_likelihood": 0.0, "perplexity": 1.0, "entropy": 0.0}

    def _score_with_lm_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        if not texts:
            return []
        if not self.model or not self.tokenizer:
            return [{"log_likelihood": 0.0, "perplexity": 1.0, "entropy": 0.0} for _ in texts]
        batch_size = max(1, int(self.config.get("lm_micro_batch_size", 4)))
        results: List[Dict[str, float]] = []
        index = 0
        while index < len(texts):
            current_batch_size = min(batch_size, len(texts) - index)
            batch_texts = texts[index:index + current_batch_size]
            try:
                results.extend(self._score_with_lm_batch_once(batch_texts))
                index += current_batch_size
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    logger.error("Batch LM scoring failed: %s", exc)
                    results.extend([self._score_with_lm(text) for text in batch_texts])
                    index += current_batch_size
                    continue
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if current_batch_size == 1:
                    logger.error("Batch LM scoring failed at minimum batch size: %s", exc)
                    results.append(self._score_with_lm(batch_texts[0]))
                    index += 1
                    continue
                batch_size = max(1, current_batch_size // 2)
                logger.warning("Reducing LM micro-batch size to %s after CUDA OOM", batch_size)
            except Exception as exc:
                logger.error("Batch LM scoring failed: %s", exc)
                results.extend([self._score_with_lm(text) for text in batch_texts])
                index += current_batch_size
        return results

    def _score_with_lm_batch_once(self, texts: List[str]) -> List[Dict[str, float]]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["max_length"],
            padding=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100

        with torch.inference_mode():
            outputs = self.model(**encoded, labels=labels)

        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        token_log_probs = gathered * valid_mask
        token_counts = valid_mask.sum(dim=1).clamp_min(1)
        mean_log_probs = token_log_probs.sum(dim=1) / token_counts

        probs = torch.softmax(logits, dim=-1)
        entropy_tensor = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        entropy_tensor = (entropy_tensor * valid_mask).sum(dim=1) / token_counts

        batch_scores = []
        for mean_log_prob, entropy in zip(mean_log_probs, entropy_tensor):
            log_likelihood = float(mean_log_prob.item())
            if not math.isfinite(log_likelihood):
                log_likelihood = 0.0
            perplexity = float(math.exp(-log_likelihood)) if math.isfinite(log_likelihood) else 1e6
            if not math.isfinite(perplexity):
                perplexity = 1e6
            entropy_value = float(entropy.item())
            if not math.isfinite(entropy_value):
                entropy_value = 0.0
            batch_scores.append(
                {
                    "log_likelihood": log_likelihood,
                    "perplexity": perplexity,
                    "entropy": entropy_value,
                }
            )
        return batch_scores

    def get_log_likelihood(self, text: str) -> float:
        return self._score_with_lm(text).get("log_likelihood", 0.0)

    def _classifier_score(self, text: str) -> float:
        if not text or not self.classifier:
            return 0.0
        try:
            tokenizer = self.classifier["tokenizer"]
            classifier_model = self.classifier["model"]
            encoded = tokenizer(
                text[:1500],
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
            with torch.inference_mode():
                logits = classifier_model(**encoded)
            if hasattr(logits, "logits"):
                logits_tensor = logits.logits
                if logits_tensor.ndim == 2 and logits_tensor.shape[-1] > 1:
                    score = torch.softmax(logits_tensor, dim=-1)[0, -1].item()
                else:
                    score = torch.sigmoid(logits_tensor.view(-1)[0]).item()
            else:
                score = torch.sigmoid(logits.view(-1)[0]).item()
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.error("Classifier scoring failed: %s", exc)
            return 0.0

    def _classifier_score_batch(self, texts: List[str]) -> List[float]:
        if not texts:
            return []
        if not self.classifier:
            return [0.0 for _ in texts]
        batch_size = max(1, int(self.config.get("classifier_micro_batch_size", 8)))
        scores: List[float] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            try:
                tokenizer = self.classifier["tokenizer"]
                classifier_model = self.classifier["model"]
                encoded = tokenizer(
                    [text[:1500] for text in batch_texts],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
                with torch.inference_mode():
                    logits = classifier_model(**encoded)
                if hasattr(logits, "logits"):
                    logits_tensor = logits.logits
                    if logits_tensor.ndim == 2 and logits_tensor.shape[-1] > 1:
                        probs = torch.softmax(logits_tensor, dim=-1)[:, -1]
                    else:
                        probs = torch.sigmoid(logits_tensor.view(-1))
                else:
                    probs = torch.sigmoid(logits.view(-1))
                scores.extend(float(prob.item()) for prob in probs)
            except RuntimeError as exc:
                if self.device == "cuda" and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    logger.warning("Classifier micro-batch OOM, falling back to single-item scoring for current batch")
                else:
                    logger.error("Batch classifier scoring failed: %s", exc)
                scores.extend([self._classifier_score(text) for text in batch_texts])
            except Exception as exc:
                logger.error("Batch classifier scoring failed: %s", exc)
                scores.extend([self._classifier_score(text) for text in batch_texts])
        return scores

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def _calibrate_low_signal(
        self,
        values: List[float],
        center: float,
        scale: float,
        *,
        low_is_ai: bool = True,
    ) -> List[float]:
        if not values:
            return []
        calibrated: List[float] = []
        denom = max(float(scale), 1e-6)
        for value in values:
            delta = (center - float(value)) / denom if low_is_ai else (float(value) - center) / denom
            calibrated.append(float(np.clip(self._sigmoid(delta), 0.0, 1.0)))
        return calibrated

    @staticmethod
    def _robust_rank(values: List[float], invert: bool = False) -> List[float]:
        if not values:
            return []
        arr = np.array(values, dtype=float)
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr), dtype=float)
        denom = max(len(arr) - 1, 1)
        scores = ranks / denom
        if invert:
            scores = 1.0 - scores
        return [float(x) for x in scores]

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _blend_neighbor_scores(self, sentence_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not sentence_scores:
            return []
        blended: List[Dict[str, Any]] = []
        for idx, row in enumerate(sentence_scores):
            prev_score = sentence_scores[idx - 1]["base_ensemble_score"] if idx > 0 else row["base_ensemble_score"]
            next_score = (
                sentence_scores[idx + 1]["base_ensemble_score"]
                if idx + 1 < len(sentence_scores)
                else row["base_ensemble_score"]
            )
            smoothed = (0.60 * row["base_ensemble_score"]) + (0.20 * prev_score) + (0.20 * next_score)
            classifier_score = row["classifier_score"]
            if classifier_score >= 0.85 and smoothed >= 0.50:
                smoothed += 0.06
            if classifier_score <= self.config["classifier_floor"] and smoothed <= 0.50:
                smoothed -= 0.08
            if self._is_excluded_nonprose(row["text"]):
                smoothed = min(smoothed, 0.12)
            enriched = dict(row)
            enriched["ensemble_score"] = round(self._clip01(smoothed), 4)
            enriched["is_ai"] = bool(enriched["ensemble_score"] >= self.config["sentence_threshold"])
            blended.append(enriched)
        return blended

    def _merge_flagged_sentences(self, sentences: List[Dict[str, Any]]) -> List[str]:
        segments: List[str] = []
        current: List[str] = []
        for sentence in sentences:
            if sentence.get("is_ai") and not self._is_excluded_nonprose(sentence["text"]):
                current.append(sentence["text"])
                continue
            if current:
                sanitized = self._sanitize_ai_segment(" ".join(current))
                if sanitized:
                    segments.append(sanitized)
                current = []
        if current:
            sanitized = self._sanitize_ai_segment(" ".join(current))
            if sanitized:
                segments.append(sanitized)
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

        lm_scores_batch = self._score_with_lm_batch(filtered)
        classifier_scores_batch = self._classifier_score_batch(filtered)
        raw_rows = []
        for sentence, lm_scores, classifier_score in zip(filtered, lm_scores_batch, classifier_scores_batch):
            raw_rows.append(
                {
                    "text": sentence,
                    "log_likelihood": lm_scores["log_likelihood"],
                    "perplexity": lm_scores["perplexity"],
                    "entropy": lm_scores["entropy"],
                    "classifier_score": classifier_score,
                }
            )

        perplexity_values = [row["perplexity"] for row in raw_rows]
        entropy_values = [row["entropy"] for row in raw_rows]
        perplexity_abs_scores = self._calibrate_low_signal(
            perplexity_values,
            self.config["perplexity_center"],
            self.config["perplexity_scale"],
            low_is_ai=True,
        )
        entropy_abs_scores = self._calibrate_low_signal(
            entropy_values,
            self.config["entropy_center"],
            self.config["entropy_scale"],
            low_is_ai=True,
        )
        perplexity_rank_scores = self._robust_rank(perplexity_values, invert=True)
        entropy_rank_scores = self._robust_rank(entropy_values, invert=True)
        perplexity_scores = [
            self._clip01((0.75 * abs_score) + (0.25 * rank_score))
            for abs_score, rank_score in zip(perplexity_abs_scores, perplexity_rank_scores)
        ]
        entropy_scores = [
            self._clip01((0.75 * abs_score) + (0.25 * rank_score))
            for abs_score, rank_score in zip(entropy_abs_scores, entropy_rank_scores)
        ]
        classifier_scores = [float(np.clip(row["classifier_score"], 0.0, 1.0)) for row in raw_rows]

        sentence_scores: List[Dict[str, Any]] = []
        for idx, row in enumerate(raw_rows):
            base_ensemble_score = (
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
                    "base_ensemble_score": round(float(base_ensemble_score), 4),
                }
            )
        return self._blend_neighbor_scores(sentence_scores)

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

        ai_weighted_word_count = 0.0
        ai_hard_word_count = 0
        for row in sentence_scores:
            word_count = len(row["text"].split())
            if word_count < self.config["min_words_for_flag"]:
                continue
            if self._is_excluded_nonprose(row["text"]):
                continue
            sentence_weight = max(0.0, (row["ensemble_score"] - 0.35) / 0.65)
            ai_weighted_word_count += word_count * sentence_weight
            if row["is_ai"]:
                ai_hard_word_count += word_count

        ai_segments = self._merge_flagged_sentences(sentence_scores)
        ai_segment_word_count = sum(len(segment.split()) for segment in ai_segments)
        ai_percentage_weighted = (ai_weighted_word_count / total_word_count) if total_word_count > 0 else 0.0
        ai_percentage_hard = (ai_hard_word_count / total_word_count) if total_word_count > 0 else 0.0
        ai_percentage_segments = (ai_segment_word_count / total_word_count) if total_word_count > 0 else 0.0
        ai_percentage = ai_percentage_segments
        mean_risk = float(np.mean(ensemble_scores)) if ensemble_scores else 0.0
        burstiness = float(np.std(perplexities)) if perplexities else 0.0
        is_ai_document = bool(
            ai_percentage >= self.config["document_ai_threshold"]
            or ai_percentage_weighted >= self.config["document_ai_threshold"]
            or mean_risk >= self.config["document_mean_risk_threshold"]
        )

        return {
            "ai_percentage": round(ai_percentage, 4),
            "ai_percentage_weighted": round(ai_percentage_weighted, 4),
            "ai_percentage_hard": round(ai_percentage_hard, 4),
            "ai_percentage_segments": round(ai_percentage_segments, 4),
            "mean_risk": round(mean_risk, 4),
            "ai_segments": ai_segments,
            "stylometrics": self._calculate_stylometrics(clean_text),
            "sentence_scores": sentence_scores,
            "burstiness": round(burstiness, 4),
            "avg_perplexity": round(float(np.mean(perplexities)) if perplexities else 0.0, 4),
            "avg_entropy": round(float(np.mean(entropies)) if entropies else 0.0, 4),
            "classifier_mean_score": round(float(np.mean(classifier_scores)) if classifier_scores else 0.0, 4),
            "ensemble_threshold": self.config["sentence_threshold"],
            "document_ai_threshold": self.config["document_ai_threshold"],
            "document_mean_risk_threshold": self.config["document_mean_risk_threshold"],
            "is_ai_document": is_ai_document,
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
