# AI Medical Literature Research Assistant

A professional research assistant for discovering, analyzing, comparing, and questioning scientific literature in the medical domain.

## Features

- PubMed literature discovery with real NCBI E-utilities integration
- PDF upload and extraction for research papers
- Automatic section detection for common paper structures
- Evidence-grounded semantic retrieval and chat-style questioning
- Saved paper library and note-taking
- Comparison of research studies
- Clear abstention when evidence is insufficient

## Tech stack

- Python
- Streamlit
- PubMed E-utilities
- PyMuPDF
- Sentence Transformers
- Hugging Face Transformers

## Installation

1. Create a virtual environment:
   python -m venv venv
2. Activate it:
   - Windows: venv\\Scripts\\activate
   - macOS/Linux: source venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Launch the app:
   streamlit run app.py

## Usage

1. Search for papers in PubMed.
2. Upload a PDF document.
3. Review detected sections and summaries.
4. Index the uploaded paper for semantic search.
5. Ask evidence-based scientific questions.
6. Save papers to the local research library and add notes.

## Architecture

The app is organized into a small modular research stack:

- `app.py` – UI and orchestration
- `src/paper_search.py` – PubMed access and result parsing
- `src/pdf_processor.py` – PDF content extraction and validation
- `src/paper_analyzer.py` – section detection and summary helpers
- `src/rag_engine.py` – embedding-based retrieval for research evidence
- `src/answer_generator.py` – evidence-grounded answer generation with abstention
- `src/research_library.py` – local saved-paper storage

## Testing

Run the test suite with:

pytest

## Privacy and limitations

- This project does not require a paid API or user account.
- Uploaded PDFs are processed locally in the app environment.
- The AI answers are evidence-grounded only; unsupported claims are not invented.
- This tool is intended for academic exploration and should not replace clinical advice.
