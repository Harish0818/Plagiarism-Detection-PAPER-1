# metrics.py
import time
import numpy as np
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

@dataclass
class MetricRecord:
    timestamp: datetime
    metric_type: str
    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedMetricsTracker:
    def __init__(self):
        self.metrics: List[MetricRecord] = []
        self.documents: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}
        self.execution_stats: Dict[str, List[float]] = {}
        self.ai_confidence_stats: Dict[str, List[float]] = {}
        self.similarity_stats: Dict[str, List[float]] = {}
        self.plagiarism_metrics = {"tp":0,"fp":0,"tn":0,"fn":0}

    def start_session(self, session_id: Optional[str] = None):
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        self.sessions[session_id] = {"start_time": datetime.now(), "documents": [], "warnings": []}
        self.record_metric("system", "session_start", 0.0, {"session_id": session_id})
        return session_id

    def end_session(self, session_id: Optional[str] = None):
        if not session_id:
            return
        s = self.sessions.get(session_id)
        if not s:
            return
        s["end_time"] = datetime.now()
        delta = (s["end_time"] - s["start_time"]).total_seconds()
        self.record_metric("system", "session_end", delta, {"session_id": session_id})

    def record_execution_time(self, operation: str, time_sec: float, details: Optional[Dict] = None):
        self.record_metric("execution_time", f"execution_{operation}", time_sec, details or {})
        self.execution_stats.setdefault(operation, []).append(time_sec)

    def record_document_metrics(self, document_id: str, filename: str, text_length: int, metadata: Optional[Dict] = None):
        metadata = metadata or {}
        word_count = metadata.get("word_count") or metadata.get("stats", {}).get("word_count") or int(text_length // 5)
        self.documents[document_id] = {"filename": filename, "text_length": text_length, "word_count": word_count, "metadata": metadata}
        self.record_metric("document", "document_size", text_length, {"filename": filename, "document_id": document_id})
        self.record_metric("document", "word_count", float(word_count), {"document_id": document_id})

    def record_plagiarism_metrics(self, document_id: str, results: List[Dict], threshold: float, ground_truth: Optional[List[int]] = None):
        scores = [r.get("score", 0.0) for r in results] if results else []
        if scores:
            avg = float(np.mean(scores))
            pct = float(len([s for s in scores if s >= threshold]) / len(scores))
            self.record_metric("plagiarism", "plagiarism_avg_score", avg, {"document_id": document_id})
            self.record_metric("plagiarism", "plagiarism_percentage", pct, {"document_id": document_id})
            self.similarity_stats.setdefault("plagiarism_similarity", []).extend(scores)

    def record_ai_detection_metrics(self, document_id: str, results: List[Dict], threshold: float, ground_truth: Optional[List[int]] = None):
        scores = [r.get("score", 0.0) for r in results] if results else []
        labels = [r.get("label", "unknown") for r in results] if results else []
        if scores:
            avg = float(np.mean(scores))
            pct = float(len([s for s in scores if s >= threshold]) / len(scores))
            self.record_metric("ai_detection", "ai_detection_avg_score", avg, {"document_id": document_id})
            self.record_metric("ai_detection", "ai_detection_percentage", pct, {"document_id": document_id})
            self.ai_confidence_stats.setdefault("ai_confidence", []).extend(scores)

    def record_citation_metrics(self, document_id: str, invalid_count: int, irrelevant_count: int, total_citations: int):
        self.record_metric("citation", "invalid_citations_count", float(invalid_count), {"document_id": document_id})
        self.record_metric("citation", "irrelevant_citations_count", float(irrelevant_count), {"document_id": document_id})

    def record_metric(self, metric_type: str, name: str, value: float, metadata: Optional[Dict] = None):
        self.metrics.append(MetricRecord(timestamp=datetime.now(), metric_type=metric_type, name=name, value=float(value), metadata=metadata or {}))

    def export_metrics(self) -> str:
        out = {"metrics":[{"timestamp":m.timestamp.isoformat(),"metric_type":m.metric_type,"name":m.name,"value":m.value,"metadata":m.metadata} for m in self.metrics],"documents":self.documents}
        return json.dumps(out, indent=2)