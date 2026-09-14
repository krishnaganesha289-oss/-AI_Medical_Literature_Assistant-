\# 🧬 AI Medical Literature Research Assistant



An AI-powered research assistant designed to make exploring medical research papers faster and easier.



The application allows users to search medical literature through PubMed, upload research papers, analyze their structure, retrieve relevant evidence using semantic search, and ask questions about the uploaded paper.



\## ✨ Features



\- 🔎 Search medical literature using PubMed

\- 📄 Upload research papers in PDF format

\- 📑 Automatically detect major paper sections

\- 📝 Generate a structured paper overview

\- 🧠 Create semantic embeddings of paper content

\- 🔍 Retrieve relevant evidence using semantic search

\- 💬 Ask natural-language questions about a research paper

\- 📚 Display retrieved evidence with similarity scores

\- 🔒 Designed to work locally without requiring an OpenAI API key

\- 🌐 Simple interactive interface built with Streamlit



\## 🏗️ How It Works



```text

&#x20;                ┌─────────────────────┐

&#x20;                │      PubMed Search  │

&#x20;                └──────────┬──────────┘

&#x20;                           │

&#x20;                           ▼

&#x20;                ┌─────────────────────┐

&#x20;                │   Research Paper    │

&#x20;                │       PDF Upload    │

&#x20;                └──────────┬──────────┘

&#x20;                           │

&#x20;                           ▼

&#x20;                ┌─────────────────────┐

&#x20;                │  Text Extraction \&  │

&#x20;                │  Section Detection  │

&#x20;                └──────────┬──────────┘

&#x20;                           │

&#x20;                           ▼

&#x20;                ┌─────────────────────┐

&#x20;                │  Semantic Chunking  │

&#x20;                │   \& Embeddings      │

&#x20;                └──────────┬──────────┘

&#x20;                           │

&#x20;                           ▼

&#x20;                ┌─────────────────────┐

&#x20;                │ Semantic Retrieval  │

&#x20;                └──────────┬──────────┘

&#x20;                           │

&#x20;                   User Question

&#x20;                           │

&#x20;                           ▼

&#x20;                ┌─────────────────────┐

&#x20;                │ Evidence-Grounded   │

&#x20;                │      Answer         │

&#x20;                └─────────────────────┘

🛠️ Technologies Used

Python

Streamlit

PubMed

PyPDF

Sentence Transformers

all-MiniLM-L6-v2

NumPy

Semantic Search

Retrieval-Augmented Generation (RAG) concepts

AI\_Medical\_Literature\_Assistant/

│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

│

└── src/

&#x20;   ├── paper\_search.py

&#x20;   ├── pdf\_processor.py

&#x20;   ├── paper\_analyzer.py

&#x20;   ├── rag\_engine.py

&#x20;   └── answer\_generator.py



