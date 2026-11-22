# Comprehensive Academic Integrity Verification Suite

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> An end-to-end multi-layered forensic framework designed to combat modern scholarly threats—identifying **AI-generated text**, detecting semantic **"Idea Theft"** plagiarism, and verifying **citation integrity** using Natural Language Inference (NLI) and live academic graph cross-referencing.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Core Analysis Engines](#-core-analysis-engines)
- [System Performance & Evaluation](#-system-performance--evaluation)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [API Reference](#-api-reference)
- [Forensic PDF Reporting](#-forensic-pdf-reporting)
- [Research & Publications](#-research--publications)
- [Contributors](#-contributors)

---

## 🔬 Executive Overview

Traditional academic integrity tools rely heavily on surface-level keyword matching and static string indexing. These legacy methods fail when confronted with modern scholarly integrity threats:
1. **Sophisticated Paraphrasing ("Idea Theft")**: Complex concepts rewritten using LLMs to evade string-overlap detectors.
2. **Citation Hallucinations & Fraud**: Papers citing real or fabricated sources where the cited literature does **not** actually entail or support the student's claims.
3. **Unrefined AI Detection**: High rate of false positives on academic writing due to lack of stylometric profiling (burstiness and perplexity calibration).

The **Academic Integrity Verification Suite** addresses these critical blind spots by combining causal language model curvature analysis ($Qwen2.5$), sentence transformer manifold alignment ($MPNet$), Natural Language Inference ($DeBERTa-v3$), dynamic live search cross-referencing, and interactive citation graph visualizers.

---

## ✨ Key Features

- **Multi-Format Ingress**: Native parsing for standard academic formats (`.pdf`, `.docx`, `.txt`) with integrated Tesseract OCR for embedded figure text extraction.
- **3-Layered Forensic Engine**:
  - 🤖 **AI Content Detection**: Probability curvature analysis (Perplexity & Entropy via Causal LM) combined with stylometric profiling (Burstiness & Type-Token Ratio).
  - ⚖️ **Semantic Plagiarism & Idea Theft**: Siamese SBERT manifold alignment, Auto-Calibrated Thresholds (ACT) for technical jargon, and domain-weighted academic search.
  - 🔍 **Citation Entailment & Hallucination Check**: DeBERTa-v3 NLI checking that verifies if a source premise logically entails or contradicts the written context.
- **Dynamic Academic Search Integration**: Queries live scholarly APIs (**Semantic Scholar**, **CrossRef**, **arXiv**, and **OpenAlex**) with rate-limiting, jitter, FAISS indexing, and Redis caching.
- **Interactive Network Visualization**: 2D, 3D, and community graph visualizations (via NetworkX & Plotly) mapping citation structures and identifying "citation bubbles".
- **Turnitin-Style PDF Reporting**: Automatic generation of color-coded PDF forensic audits with precision highlights.

---

## 📐 System Architecture

The framework is structured as a high-performance web architecture operating across 7 dedicated layers:

```
<img width="2012" height="1099" alt="image" src="https://github.com/user-attachments/assets/833aa1ed-0c03-447c-b6d9-ccda5d77d90a" />
```

---

## ⚙️ Core Analysis Engines

### 1. 🤖 AI Content Identification
- **Probability Curvature**: Utilizes causal LMs ($Qwen2.5-0.5B$) to score text predictability via Perplexity ($PPL$) and Token Entropy ($H$).
- **Stylometric Fingerprinting**:
  - **Burstiness**: Standard deviation of sentence lengths $\sigma(L_{\text{sentence}})$, distinguishing human structural variability from uniform AI outputs.
  - **Lexical Diversity**: Measured via Type-Token Ratio ($\text{TTR} = \frac{\text{Unique Words}}{\text{Total Words}}$).

### 2. ⚖️ Semantic Plagiarism & Idea Theft
- **Manifold Alignment**: Uses `paraphrase-multilingual-mpnet-base-v2` to calculate vector cosine similarity:
  $$\text{Similarity}(q, d) = \frac{\vec{v}_q \cdot \vec{v}_d}{\|\vec{v}_q\| \|\vec{v}_d\|}$$
- **Auto-Calibrated Thresholds (ACT)**: Dynamically adjusts decision boundaries based on technical term density, citation presence, and chunk size to prevent false alarms on standardized academic terminology.

### 3. 🔍 Citation Integrity Analysis
- **NLI Cross-Referencing**: Employs `cross-encoder/nli-deberta-v3-small` to evaluate logical entailment between claim context (hypothesis) and reference abstracts (premise):
  - **Entailment**: Claim is mathematically supported by the cited paper.
  - **Contradiction**: Claim directly conflicts with the cited paper (*Citation Fraud / Hallucination*).
  - **Neutral**: Reference is contextually irrelevant to the claim made.

---

## 📊 System Performance & Evaluation

The system was evaluated against benchmark scholarly datasets and complex paraphrased academic texts:

### Metric Breakdown
| Evaluation Domain | Metric Score | Primary Methodology |
| :--- | :---: | :--- |
| **Semantic Plagiarism Detection** | **88.0%** | Siamese SBERT + Lexical Alignment |
| **AI Content Identification** | **85.0%** | Perplexity Curvature + Stylometrics |
| **Citation Integrity Verification** | **82.0%** | DeBERTa-v3 Contextual NLI |

### System-Wide Performance
```
Overall System Accuracy :  89.0%
System Precision        :  0.900  (Minimizes false accusations on human work)
System Recall           :  0.830  (Captures heavily paraphrased text)
Overall F1-Score        :  0.850  (Balanced forensic reliability)
```

---

## 🛠️ Tech Stack

- **Core Backend**: Python 3.9+, Flask
- **Deep Learning / NLP**: PyTorch, Hugging Face `transformers`, `sentence-transformers`
- **Search & Vector Indexing**: FAISS, Redis, Requests HTTP Pool
- **Visualization**: Plotly, Matplotlib, NetworkX, Seaborn
- **PDF & Document Engine**: PyMuPDF (`fitz`), `python-docx`, `pytesseract`
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Tailwind / Custom Component Styling

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Tesseract OCR engine (Optional, for scanned PDF OCR support)
- Redis server (Optional, for multi-level query caching)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/academic-integrity-suite.git
cd academic-integrity-suite
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.

---

## 📡 API Reference

### 1. Single Document Analysis
- **Endpoint**: `POST /analyze`
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (PDF, DOCX, TXT)
- **Response**: JSON containing raw text, AI metrics, plagiarism matches, citation entailment status, and evaluation tables.

### 2. Batch Analysis
- **Endpoint**: `POST /analyze-batch`
- **Content-Type**: `multipart/form-data`
- **Body**: `files` (Multiple documents)
- **Response**: Comparative document risk matrix and aggregate statistics.

### 3. Export PDF Report
- **Endpoint**: `POST /generate-report`
- **Content-Type**: `multipart/form-data`
- **Form Data**: `file`, `report_type` (`full`, `ai`, `plagiarism`, `citations`)
- **Response**: Downloadable highlighted PDF document.

---

## 📄 Forensic PDF Reporting

The suite includes an automated PDF generation engine powered by PyMuPDF:
- 🟨 **Yellow Highlights**: Verbatim and semantic plagiarism matches.
- 🟦 **Cyan Highlights**: Passages identified as AI-generated.
- 🟥 **Red Highlights**: Citation contradictions and unverified claims.
- 📋 **Header Summary Box**: Embeds target vs. observed highlight coverage metadata.

---

## 📜 Research & Publications

This framework was presented at the **International Conference on FutureTech: AI-Driven Innovations for Sustainable Development Goals (FutureTech - 2025)**:

- **Paper Title**: *Comprehensive Academic Integrity Verification: Plagiarism, AI-Content and Citation Analysis*
- **Publication Partners**: SSRN / Elsevier
- **Organized By**: Institute of Technical and Scientific Research (ITSR), Jaipur, India (Dec 22-23, 2025)

---

## 👥 Contributors

- **Hariharan K** (Reg No: 312422243050) – St. Joseph's Institute of Technology
- **Harish B** (Reg No: 312422243053) – St. Joseph's Institute of Technology
- **Guided By**: Dr. M.K. Kirubakaran, M.E., Ph.D.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
