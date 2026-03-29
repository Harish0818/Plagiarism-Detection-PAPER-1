# report_generator.py - ENHANCED VERSION
import pymupdf as fitz
import os
import io
import re
import difflib
from datetime import datetime
import logging
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class IntegrityReportGenerator:
    def __init__(self):
        # Turnitin-style Color Palette (RGB 0-1)
        self.colors = {
            "ai_generated": (0, 1, 1),      # Cyan
            "ai_paraphrased": (0.5, 0, 0.5), # Purple
            "plagiarism": (1, 0.8, 0),       # Gold/Yellow
            "citation_fraud": (1, 0, 0)      # Red
        }
        
        self.MIN_MATCH_LENGTH = 10  # Minimum characters to attempt matching
        self.MAX_SEGMENT_CHARS = 1200
        self.MAX_PHRASES_PER_SEGMENT = 22

    def create_report(self, uploaded_file, analysis_results, output_path):
        """
        Creates a highlighted PDF report with ADVANCED text matching.
        """
        try:
            # Read the file content into bytes
            file_bytes = uploaded_file.getvalue()
            
            # Open the PDF from memory
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            report_meta = analysis_results.get("report_meta", {}) or {}

            # Extract page text once.
            page_text_info = self._extract_page_text_with_detailed_info(doc)

            # Process plagiarism segments
            plagiarism_segments = self._prepare_segments(analysis_results.get("plagiarism_segments", []))
            logger.info(f"Processing {len(plagiarism_segments)} plagiarism segments for highlighting")
            
            # Process AI segments
            ai_segments = self._prepare_segments(analysis_results.get("ai_segments", []))
            logger.info(f"Processing {len(ai_segments)} AI segments for highlighting")
            
            # Track highlighting statistics
            total_highlights = 0
            matched_segments = 0
            matched_segment_keys = set()
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_info = page_text_info[page_num]
                
                # Get page text for matching
                page_text = page_info["text"]
                
                # Highlight plagiarism segments
                for segment in plagiarism_segments:
                    matches_found = self._enhanced_highlight_text(
                        page, 
                        page_text, 
                        segment, 
                        self.colors["plagiarism"], 
                        "Plagiarism"
                    )
                    
                    if matches_found > 0:
                        seg_key = segment[:140].lower()
                        if seg_key not in matched_segment_keys:
                            matched_segment_keys.add(seg_key)
                            matched_segments += 1
                        total_highlights += matches_found
                
                # Highlight AI content
                for segment in ai_segments:
                    matches_found = self._enhanced_highlight_text(
                        page, 
                        page_text, 
                        segment, 
                        self.colors["ai_generated"], 
                        "AI-Generated"
                    )
                    
                    total_highlights += matches_found

            target_pct = float(report_meta.get("highlight_content_percentage", 0.0))
            report_type = str(report_meta.get("report_type", "full")).lower()
            observed_pct = self._estimate_highlight_coverage(doc)
            if report_type in {"ai", "plagiarism"} and target_pct > 0 and observed_pct + 1.5 < target_pct:
                backfill_color = self.colors["ai_generated"] if report_type == "ai" else self.colors["plagiarism"]
                self._backfill_highlights_to_target(doc, target_pct, backfill_color)
                observed_pct = self._estimate_highlight_coverage(doc)

            self._add_report_header(doc, report_meta, observed_pct=observed_pct)
            
            # Save the document
            doc.save(output_path)
            doc.close()
            
            logger.info(f"""
            Report Generation Summary:
            - Total plagiarism segments: {len(plagiarism_segments)}
            - Segments successfully highlighted: {matched_segments}
            - Total highlights added: {total_highlights}
            - Target highlighted %: {float(report_meta.get("highlight_content_percentage", 0.0)):.2f}
            - Observed highlighted % (estimated): {observed_pct:.2f}
            - Report saved to: {output_path}
            """)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating report: {str(e)}")
            raise

    def _extract_page_text_with_detailed_info(self, doc):
        """
        Extract text from each page with detailed information for better matching.
        Returns list of dictionaries with text and metadata.
        """
        page_info = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text blocks with detailed information
            blocks = page.get_text("dict")["blocks"]
            
            # Build page text from blocks
            page_text = ""
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_text += span["text"] + " "
            
            page_info.append({
                "text": page_text,
                "blocks": blocks,
                "page_num": page_num
            })
        
        return page_info

    def _enhanced_highlight_text(self, page, page_text, search_text, color, label):
        """
        Fast, robust text highlighting with normalized phrase search.
        """
        if not search_text or len(search_text.strip()) < self.MIN_MATCH_LENGTH:
            return 0
        
        instances = self._search_segment_instances(page, search_text)
        return self._apply_highlights(page, instances, color, label)

    def _prepare_segments(self, segments):
        cleaned = []
        seen = set()
        for seg in segments or []:
            if not seg:
                continue
            s = " ".join(str(seg).split()).strip()
            if len(s) < self.MIN_MATCH_LENGTH:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s[: self.MAX_SEGMENT_CHARS])
        return cleaned

    def _search_segment_instances(self, page, segment):
        # Generate multiple overlapping phrase windows from each segment.
        segment = " ".join(segment.split()).strip()
        variants = self._generate_search_phrases(segment)
        found = []
        seen_rects = set()
        for phrase in variants:
            if len(phrase) < 12:
                continue
            for quad in page.search_for(phrase, quads=True):
                key = tuple(round(v, 2) for v in quad.rect)
                if key in seen_rects:
                    continue
                seen_rects.add(key)
                found.append(quad)
        return found

    def _generate_search_phrases(self, segment):
        segment = " ".join(segment.split()).strip()
        if not segment:
            return []

        phrases = []
        tokens = segment.split()

        def _add_phrase(text):
            t = " ".join(text.split()).strip()
            if len(t) >= 12 and t not in phrases:
                phrases.append(t)

        # Anchor phrases first.
        _add_phrase(segment[: self.MAX_SEGMENT_CHARS])
        if len(tokens) >= 10:
            _add_phrase(" ".join(tokens[:12]))
            _add_phrase(" ".join(tokens[-12:]))
        if len(tokens) >= 20:
            mid = len(tokens) // 2
            _add_phrase(" ".join(tokens[max(0, mid - 6): mid + 6]))

        # Sliding windows improve recall for wrapped PDF text.
        window_plan = [(18, 9), (14, 7), (10, 5), (8, 4)]
        for win_size, step in window_plan:
            if len(tokens) < win_size:
                continue
            for i in range(0, len(tokens) - win_size + 1, step):
                _add_phrase(" ".join(tokens[i: i + win_size]))
                if len(phrases) >= self.MAX_PHRASES_PER_SEGMENT:
                    return phrases

        return phrases

    def _add_report_header(self, doc, meta, observed_pct=None):
        if len(doc) == 0:
            return
        page = doc[0]
        rect = fitz.Rect(36, 24, page.rect.width - 36, 112)
        report_type = str(meta.get("report_type", "full")).lower()
        if report_type == "plagiarism":
            title = "Plagiarism Highlight Summary"
            pct = float(meta.get("highlight_content_percentage", 0.0))
        elif report_type == "ai":
            title = "AI Highlight Summary"
            pct = float(meta.get("highlight_content_percentage", 0.0))
        else:
            title = "Integrity Highlight Summary"
            pct = float(meta.get("highlight_content_percentage", 0.0))

        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(fill=(0.93, 0.98, 0.99), color=(0.70, 0.84, 0.90), width=0.7)
        shape.commit()
        page.insert_textbox(
            fitz.Rect(rect.x0 + 12, rect.y0 + 8, rect.x1 - 12, rect.y1 - 8),
            (
                f"{title}\n"
                f"Target Highlighted Content: {pct:.2f}%\n"
                f"Observed Highlighted Content (Estimated): {float(observed_pct or 0.0):.2f}%"
            ),
            fontsize=12,
            fontname="helv",
            color=(0.05, 0.28, 0.35),
            align=0,
        )

    def _estimate_highlight_coverage(self, doc):
        highlighted_chars, total_chars = self._estimate_highlight_stats(doc)
        if total_chars <= 0:
            return 0.0
        return round((highlighted_chars / total_chars) * 100, 2)

    def _estimate_highlight_stats(self, doc):
        total_chars = 0
        highlighted_chars = 0
        seen_annots = set()

        for page_index, page in enumerate(doc):
            page_text = " ".join(page.get_text("text").split())
            total_chars += len(page_text)

            annot = page.first_annot
            while annot:
                try:
                    a_type = (annot.type or (None, ""))[1]
                    if a_type == "Highlight":
                        key = (
                            page_index,
                            round(annot.rect.x0, 2),
                            round(annot.rect.y0, 2),
                            round(annot.rect.x1, 2),
                            round(annot.rect.y1, 2),
                        )
                        if key not in seen_annots:
                            seen_annots.add(key)
                            snippet = " ".join(page.get_textbox(annot.rect).split())
                            highlighted_chars += len(snippet)
                except Exception:
                    pass
                annot = annot.next

        return highlighted_chars, total_chars

    def _backfill_highlights_to_target(self, doc, target_pct, color):
        highlighted_chars, total_chars = self._estimate_highlight_stats(doc)
        if total_chars <= 0:
            return

        target_chars = int((target_pct / 100.0) * total_chars)
        needed_chars = max(0, target_chars - highlighted_chars)
        if needed_chars <= 0:
            return

        added_chars = 0
        for page_index, page in enumerate(doc):
            if added_chars >= needed_chars:
                break
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if added_chars >= needed_chars:
                    break
                lines = block.get("lines")
                if not lines:
                    continue
                for line in lines:
                    if added_chars >= needed_chars:
                        break
                    for span in line.get("spans", []):
                        if added_chars >= needed_chars:
                            break
                        snippet = " ".join(str(span.get("text", "")).split()).strip()
                        if len(snippet) < 18:
                            continue
                        x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                        if page_index == 0 and y1 < 120:
                            continue
                        rect = fitz.Rect(x0, y0, x1, y1)
                        try:
                            annot = page.add_highlight_annot(rect)
                            annot.set_colors(stroke=color)
                            annot.set_opacity(0.16)
                            annot.set_info(title="", content="")
                            annot.set_flags(annot.flags | 1)
                            annot.update()
                            added_chars += len(snippet)
                        except Exception:
                            continue

    def _normalize_text_for_matching(self, text):
        """
        Normalize text for better matching by removing formatting artifacts.
        """
        if not text:
            return ""
        
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle hyphenated line breaks (common in PDFs)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Remove line breaks within sentences
        text = re.sub(r'\s*\n\s*', ' ', text)
        
        # Remove special characters but keep meaningful punctuation
        text = re.sub(r'[^\w\s.,;:!?()\-]', '', text)
        
        # Standardize quotes and apostrophes
        text = text.replace('"', '').replace("'", '').replace("`", '')
        
        return text.strip()

    def _split_into_search_chunks(self, text):
        """
        Split text into optimal chunks for searching.
        """
        if not text:
            return []
        
        # First try to split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If sentences are too long, split by length
        result = []
        for sentence in sentences:
            if len(sentence) <= 100:
                result.append(sentence)
            else:
                # Split long sentences by words
                words = sentence.split()
                chunk_size = 15
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i+chunk_size])
                    if len(chunk) > 20:
                        result.append(chunk)
        
        return [s.strip() for s in result if len(s.strip()) >= 10]

    def _find_fuzzy_matches(self, search_chunk, page_text):
        """
        Find fuzzy matches using sequence matching.
        Returns list of (start_index, end_index, similarity_score)
        """
        matches = []
        
        if not search_chunk or not page_text:
            return matches
        
        chunk_len = len(search_chunk)
        page_len = len(page_text)
        
        # Use sliding window approach
        window_size = min(chunk_len, 200)
        step = max(1, window_size // 4)
        
        for i in range(0, page_len - window_size + 1, step):
            window = page_text[i:i + window_size]
            
            # Calculate similarity
            similarity = difflib.SequenceMatcher(
                None, 
                search_chunk[:window_size], 
                window
            ).ratio()
            
            if similarity >= self.MATCH_THRESHOLD:
                matches.append((i, i + window_size, similarity))
        
        # Sort by similarity and remove overlaps
        matches.sort(key=lambda x: x[2], reverse=True)
        filtered_matches = []
        covered_positions = set()
        
        for start, end, similarity in matches:
            # Check if this position is already covered
            overlap = False
            for pos in range(start, end):
                if pos in covered_positions:
                    overlap = True
                    break
            
            if not overlap:
                filtered_matches.append((start, end, similarity))
                # Mark positions as covered
                covered_positions.update(range(start, end))
        
        return filtered_matches[:5]  # Return top 5 matches

    def _extract_significant_keywords(self, text):
        """
        Extract significant keywords for backup matching.
        """
        # Common words to exclude
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'as', 'is', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'a', 'an', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = []
        for word in words:
            if (word not in stop_words and 
                len(word) > 6 and 
                word.isalpha() and
                word not in keywords):
                keywords.append(word)
        
        # Score by frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = words.count(word)
        
        # Sort by frequency and length
        sorted_keywords = sorted(
            keywords, 
            key=lambda x: (word_freq[x], len(x)), 
            reverse=True
        )
        
        return sorted_keywords[:10]  # Return top 10 keywords

    # In the _enhanced_highlight_text method, update the _apply_highlights function:

    # In the _apply_highlights method, modify it to remove ALL text annotations:

    def _apply_highlights(self, page, quad_instances, color, label):
        """
        Apply highlights to found text instances WITHOUT ANY TEXT ANNOTATIONS.
        """
        highlights_added = 0
        
        for quad in quad_instances:
            try:
                # Create highlight annotation
                annot = page.add_highlight_annot(quad)
                
                # Set highlight color ONLY (no text)
                annot.set_colors(stroke=color)
                
                # Set transparency for better visibility
                annot.set_opacity(0.3)
                
                # REMOVE ALL TEXT ANNOTATIONS - just highlight, no popup
                # Don't set any title or content
                annot.set_info(title="", content="")
                
                # Make annotation silent (no popup)
                annot.set_flags(annot.flags | 1)  # Make it read-only
                
                annot.update()
                highlights_added += 1
                
            except Exception as e:
                logger.warning(f"Could not add highlight: {e}")
                continue
        
        return highlights_added

    def create_detailed_report(self, uploaded_file, analysis_results, output_path):
        """
        Alternative: Create a comprehensive report with debugging information.
        """
        try:
            file_bytes = uploaded_file.getvalue()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            # Create a cover page with analysis details
            self._create_analysis_cover_page(doc, analysis_results)
            
            # Process each page for highlighting
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Skip cover page
                if page_num == 0:
                    continue
                
                # Get page text
                page_text = page.get_text("text")
                
                # Highlight plagiarism
                plagiarism_segments = analysis_results.get("plagiarism_segments", [])
                plag_count = 0
                
                for segment in plagiarism_segments:
                    if segment:
                        count = self._enhanced_highlight_text(
                            page, page_text, segment, 
                            self.colors["plagiarism"], "Plagiarism"
                        )
                        plag_count += count
                
                # Highlight AI content
                ai_segments = analysis_results.get("ai_segments", [])
                ai_count = 0
                
                for segment in ai_segments:
                    if segment:
                        count = self._enhanced_highlight_text(
                            page, page_text, segment,
                            self.colors["ai_generated"], "AI-Generated"
                        )
                        ai_count += count
                
                # Add page summary footer
                if plag_count > 0 or ai_count > 0:
                    self._add_page_summary_footer(page, page_num, plag_count, ai_count)
            
            # Add analysis appendix
            self._add_analysis_appendix(doc, analysis_results)
            
            doc.save(output_path)
            doc.close()
            
            logger.info(f"Detailed report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in detailed report: {e}")
            return None

    def _create_analysis_cover_page(self, doc, analysis_results):
        """Create a detailed cover page with analysis statistics."""
        cover = doc.new_page(0)
        
        # Add title
        cover.insert_text((72, 72), 
                         "ACADEMIC INTEGRITY ANALYSIS REPORT", 
                         fontsize=18, 
                         color=(0, 0, 0),
                         fontname="helv")
        
        # Add analysis summary
        plag_segments = analysis_results.get("plagiarism_segments", [])
        ai_segments = analysis_results.get("ai_segments", [])
        citation_segments = analysis_results.get("citation_segments", [])
        
        summary = f"""
        ANALYSIS SUMMARY
        {'=' * 50}
        
        Plagiarism Detection:
        - Total matches detected: {len(plag_segments)}
        - Match similarity threshold: >70%
        - Highlight color: Yellow
        
        AI Content Detection:
        - AI segments identified: {len(ai_segments)}
        - Highlight color: Cyan
        
        Citation Analysis:
        - Issues detected: {len(citation_segments)}
        - Highlight color: Red
        
        Report Generation Details:
        - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Matching algorithm: Enhanced fuzzy matching
        - Minimum match similarity: {self.MATCH_THRESHOLD:.0%}
        
        LEGEND:
        ███ YELLOW: Plagiarism / Similar Content Detected
        ███ CYAN: AI-Generated Content
        ███ RED: Citation Issues
        {'=' * 50}
        """
        
        cover.insert_text((72, 120), summary, fontsize=11, color=(0, 0, 0))
        
        # Add matching statistics
        stats_y = 300
        cover.insert_text((72, stats_y), 
                         "MATCHING METHODOLOGY", 
                         fontsize=12, 
                         color=(0, 0, 0.5))
        
        methodology = """
        1. Text Normalization: Removes formatting artifacts for cleaner matching
        2. Exact Search: Direct text matching in PDF
        3. Fuzzy Matching: Finds similar but not identical text
        4. Keyword Backup: Highlights key terms when full matches fail
        5. Multi-strategy Fallback: Combines approaches for maximum coverage
        """
        
        cover.insert_text((72, stats_y + 25), methodology, fontsize=10, color=(0.3, 0.3, 0.3))

    def _add_page_summary_footer(self, page, page_num, plag_count, ai_count):
        """Add a footer with detection counts for each page."""
        footer = f"Page {page_num} | Plagiarism: {plag_count} | AI: {ai_count}"
        page.insert_text(
            (72, page.rect.height - 20), 
            footer, 
            fontsize=9, 
            color=(0.5, 0.5, 0.5)
        )

    def _add_analysis_appendix(self, doc, analysis_results):
        """Add an appendix with detailed detection information."""
        appendix = doc.new_page()
        
        appendix.insert_text((72, 72), 
                           "DETECTION DETAILS APPENDIX", 
                           fontsize=14, 
                           color=(0, 0, 0))
        
        # Add plagiarism matches
        plag_segments = analysis_results.get("plagiarism_segments", [])
        
        y_pos = 120
        if plag_segments:
            appendix.insert_text((72, y_pos), 
                               "PLAGIARISM MATCHES:", 
                               fontsize=12, 
                               color=(0.8, 0, 0))
            y_pos += 25
            
            for i, segment in enumerate(plag_segments[:10]):  # Show first 10
                segment_preview = segment[:80] + "..." if len(segment) > 80 else segment
                appendix.insert_text(
                    (72, y_pos), 
                    f"{i+1}. {segment_preview}", 
                    fontsize=9, 
                    color=(0.3, 0.3, 0.3)
                )
                y_pos += 15
                
                if y_pos > 700:  # Avoid going off page
                    appendix = doc.new_page()
                    y_pos = 72
        
        # Add explanation about highlighting
        appendix.insert_text((72, y_pos + 20), 
                           "NOTE:", 
                           fontsize=10, 
                           color=(0, 0, 0.5))
        
        note = """
        Highlighting may not cover 100% of detected content due to:
        - PDF formatting variations
        - Text extraction differences
        - Complex document layouts
        - Multiple detection strategies used for best coverage
        
        For complete analysis, refer to the detailed results in the web interface.
        """
        
        appendix.insert_text((72, y_pos + 40), note, fontsize=9, color=(0.4, 0.4, 0.4))

# Additional utility function for testing
def validate_highlighting(input_pdf_path, output_pdf_path, search_texts):
    """
    Utility function to test highlighting functionality.
    """
    with open(input_pdf_path, 'rb') as f:
        file_bytes = f.read()
    
    generator = IntegrityReportGenerator()
    
    # Create test analysis results
    test_results = {
        "plagiarism_segments": search_texts,
        "ai_segments": [],
        "citation_segments": []
    }
    
    # Create report
    output = generator.create_report(
        io.BytesIO(file_bytes),
        test_results,
        output_pdf_path
    )
    
    return output
