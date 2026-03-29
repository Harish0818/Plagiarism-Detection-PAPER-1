# text_processing.py
import io
import re
import json
import hashlib
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import logging
import gc

logger = logging.getLogger(__name__)

# Logic Fix: Use pymupdf as primary name to avoid conflicts
try:
    import pymupdf as fitz 
    FITZ_AVAILABLE = True
except Exception:
    try:
        import fitz
        FITZ_AVAILABLE = True
    except Exception:
        FITZ_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
    nltk.download("punkt", quiet=True)
except Exception:
    NLTK_AVAILABLE = False

# Vision/OCR logic
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
    logger.warning("pytesseract or PIL not available; Image OCR disabled")

class AdvancedTextProcessor:
    """Robust text extraction with strict validation to prevent silent failures."""

    def __init__(self):
        self.doc_cache: Dict[str, Dict] = {}
        self.cache_size_limit = 200

    def _read_file_bytes(self, file) -> bytes:
        if isinstance(file, (str, Path)):
            with open(file, "rb") as f:
                return f.read()
        if hasattr(file, "read"):
            try:
                file.seek(0)
            except Exception:
                pass
            return file.read()
        raise ValueError("Unsupported file handle/type")

    def _detect_file_type(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        mapping = {
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
        }
        return mapping.get(ext, "text/plain")

    def extract_text_from_images(self, fbytes: bytes) -> str:
        image_text = ""
        if not OCR_AVAILABLE or not FITZ_AVAILABLE:
            return ""
        try:
            doc = fitz.open(stream=fbytes, filetype="pdf")
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                for img in image_list:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base_image["image"]))
                    extracted = pytesseract.image_to_string(image)
                    if len(extracted.strip()) > 5:
                        image_text += f"\n[Vision-OCR Content from Page {page_index+1}]:\n{extracted}\n"
            doc.close()
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
        return image_text

    def extract_text(self, file, file_type: Optional[str] = None, extract_metadata: bool = True) -> Dict:
        result = {"text": "", "metadata": {}, "success": False, "error": ""}
        try:
            filename = getattr(file, "name", "unknown")
            fbytes = self._read_file_bytes(file)
            fingerprint = hashlib.sha256(fbytes).hexdigest()

            if file_type is None:
                file_type = self._detect_file_type(filename)

            text = ""
            metadata = {"filename": filename}

            # PDF Extraction logic
            if file_type == "application/pdf":
                if not FITZ_AVAILABLE:
                    result["error"] = "CRITICAL: PyMuPDF (pymupdf) not installed. Run 'pip install pymupdf'."
                    return result
                doc = fitz.open(stream=fbytes, filetype="pdf")
                pages = [p.get_text().strip() for p in doc if p.get_text().strip()]
                text = "\n\n".join(pages)
                metadata["pages"] = doc.page_count
                
                if OCR_AVAILABLE:
                    ocr_content = self.extract_text_from_images(fbytes)
                    if ocr_content:
                        text += "\n\n--- IMAGE DATA ---\n" + ocr_content
                doc.close()

            # DOCX Extraction logic
            elif file_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
                if not DOCX_AVAILABLE:
                    result["error"] = "CRITICAL: python-docx not installed. Run 'pip install python-docx'."
                    return result
                doc = Document(io.BytesIO(fbytes))
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

            else:
                text = fbytes.decode("utf-8", errors="replace")

            text = self._clean_text(text)
            
            # Final validation to prevent empty strings being marked as success
            if not text or len(text.strip()) < 10:
                result["error"] = "No readable text found. Check if the file is a scanned image."
                return result

            result.update({
                "text": text,
                "metadata": metadata,
                "success": True
            })
            return result

        except Exception as e:
            result["error"] = f"Extraction failed: {str(e)}"
            return result

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

# RESTORED SINGLETON HELPER FOR APP.PY
_processor_instance = None
def get_text_processor() -> AdvancedTextProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AdvancedTextProcessor()
    return _processor_instance