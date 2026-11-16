# Architecture Overview

## System purpose

This application helps researchers discover, read, and interrogate clinical and biomedical literature. The design keeps the workflow transparent: search results come from PubMed, PDF text comes from local extraction, and answers are traceable to retrieved passages.

## Core modules

- `app.py`: Streamlit interface and orchestration.
- `src/paper_search.py`: PubMed E-utilities access and result parsing.
- `src/pdf_processor.py`: local PDF extraction and validation.
- `src/paper_analyzer.py`: section detection, summaries, and metadata hints.
- `src/rag_engine.py`: chunking and semantic retrieval using Sentence Transformers.
- `src/answer_generator.py`: evidence-based answer generation with abstention.
- `src/research_library.py`: local saved papers and notes storage.

## Design principles

- Evidence first: the app distinguishes retrieved text from generated interpretation.
- Transparency: users can inspect evidence passages before trusting an answer.
- Local-first processing: no mandatory paid cloud dependencies.
- Reliability: unsupported claims are explicitly abstained from instead of invented.

## Data flow

1. User searches PubMed.
2. Relevant abstracts and metadata are returned.
3. User uploads a PDF.
4. PDF text is extracted locally.
5. The uploaded paper is chunked and embedded.
6. Semantic retrieval finds the most relevant evidence.
7. The answer generator responds only using the retrieved passages.
8. Users can save notes and study comparisons locally.
