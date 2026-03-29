
🔍 Advanced Plagiarism Detection System
A comprehensive, AI-powered tool for detecting plagiarism, AI-generated content, and analyzing citations in academic documents. Built with Python and Streamlit, this system helps researchers, educators, and students maintain academic integrity.

✨ Features
📄 Multi-Format Support: Analyze PDF, DOCX, and TXT documents

🔍 Plagiarism Detection: Identify copied content using semantic similarity matching with academic databases

🤖 AI Content Detection: Detect AI-generated text using ensemble models and pattern analysis

📚 Citation Analysis: Validate citations and check contextual relevance

🌐 Network Visualization: Interactive citation network graphs

📊 Comprehensive Reporting: Detailed metrics, performance statistics, and export capabilities

⚙️ Configurable Thresholds: Adjustable sensitivity for different detection scenarios


🚀 Quick Start
Prerequisites
Python 3.8 or higher

pip (Python package manager)


🛠️ Usage
Upload Documents: Click "Browse files" to upload PDF, DOCX, or TXT files

Adjust Settings: Use the sliders in the sidebar to set detection thresholds

Run Analysis: The system will automatically process all uploaded files

Review Results: Explore the findings through the interactive tabs:

Summary: Overview of detected issues across all documents

Detailed Analysis: Document-specific findings and evidence

Citation Network: Visual representation of citation relationships

Metrics: Performance statistics and execution times

Evaluation: System performance metrics and confusion matrice


📁 Project Structure:
plagiarism-detection-system/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── utils/                 # Core functionality modules
│   ├── text_processing.py # Document extraction and text processing
│   ├── plagiarism.py      # Plagiarism detection logic
│   ├── ai_detector.py     # AI content detection
│   ├── citation_analyzer.py # Citation validation and analysis
│   ├── academic_search.py # Academic database integration
│   ├── graph_visualizer.py # Network visualization
│   ├── metrics.py         # Performance tracking
│   └── evaluation.py      # Evaluation metrics calculation
├── tests/                 # Test utilities
│   └── test_data_generator.py # Test case generation
└── README.md             # This file


🔧 Configuration
The system provides several configurable parameters:

Plagiarism Similarity Threshold (0.5-1.0): Controls sensitivity for plagiarism detection

AI Detection Confidence Threshold (0.1-1.0): Adjusts sensitivity for AI-generated content detection

API Rate Limiting: Built-in delays to respect academic API limits


🌐 API Integrations
The system integrates with several academic APIs:

Semantic Scholar API: For academic paper search and similarity matching

CrossRef API: Fallback for citation validation and metadata retrieval


📊 Outputs
The system generates comprehensive reports including:

Document-level summary statistics

Detailed evidence of detected issues

Citation network visualizations

Performance metrics and execution times

Exportable CSV files for further analysis
