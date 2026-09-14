import streamlit as st

from src.paper_search import search_pubmed
from src.pdf_processor import (
    extract_text_from_pdf,
    get_page_count,
)
from src.paper_analyzer import (
    extract_sections,
    create_paper_summary,
)
from src.rag_engine import RAGEngine
from src.answer_generator import AnswerGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Medical Literature Research Assistant",
    page_icon="🧬",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "paper_text" not in st.session_state:
    st.session_state.paper_text = None

if "paper_name" not in st.session_state:
    st.session_state.paper_name = None

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "answer_generator" not in st.session_state:
    st.session_state.answer_generator = None

if "indexed_paper_name" not in st.session_state:
    st.session_state.indexed_paper_name = None


# ============================================================
# HEADER
# ============================================================

st.title("🧬 AI Medical Literature Research Assistant")

st.write(
    "Search medical literature, upload research papers, "
    "analyze their structure, and ask questions using "
    "semantic retrieval and a local AI model."
)

st.divider()


# ============================================================
# SECTION 1 — PUBMED SEARCH
# ============================================================

st.header("🔎 Search Medical Literature")

query = st.text_input(
    "Enter your research topic",
    placeholder=(
        "Example: artificial intelligence in cancer detection"
    ),
)

max_results = st.slider(
    "Number of papers to retrieve",
    min_value=5,
    max_value=30,
    value=10,
)


if st.button(
    "🔎 Search PubMed",
    type="primary",
):

    if not query.strip():

        st.warning(
            "Please enter a research topic."
        )

    else:

        with st.spinner(
            "Searching PubMed..."
        ):

            try:

                papers = search_pubmed(
                    query,
                    max_results,
                )

                if not papers:

                    st.info(
                        "No papers were found. "
                        "Try a different search term."
                    )

                else:

                    st.success(
                        f"Found {len(papers)} papers."
                    )

                    for index, paper in enumerate(
                        papers,
                        start=1,
                    ):

                        st.subheader(
                            f"{index}. {paper['title']}"
                        )

                        authors = paper.get(
                            "authors",
                            []
                        )

                        if authors:

                            if len(authors) > 5:

                                author_text = (
                                    ", ".join(
                                        authors[:5]
                                    )
                                    + " et al."
                                )

                            else:

                                author_text = ", ".join(
                                    authors
                                )

                        else:

                            author_text = (
                                "Authors unavailable"
                            )

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write(
                                f"**Authors:** "
                                f"{author_text}"
                            )

                            st.write(
                                f"**Journal:** "
                                f"{paper.get('journal', 'Unknown')}"
                            )

                        with col2:

                            st.write(
                                f"**Publication:** "
                                f"{paper.get('publication_date', 'Unknown')}"
                            )

                            st.link_button(
                                "📄 View on PubMed",
                                paper["pubmed_url"],
                            )

                        st.divider()

            except Exception as e:

                st.error(
                    f"PubMed search failed: {e}"
                )


# ============================================================
# SECTION 2 — PDF UPLOAD
# ============================================================

st.header("📄 Research Paper Reader")

uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"],
)


if uploaded_file is not None:

    # Detect a different/new paper
    if (
        st.session_state.paper_name
        != uploaded_file.name
    ):

        st.session_state.paper_text = None
        st.session_state.paper_name = None
        st.session_state.rag_engine = None
        st.session_state.answer_generator = None
        st.session_state.indexed_paper_name = None

    try:

        page_count = get_page_count(
            uploaded_file
        )

        st.success(
            f"Uploaded: {uploaded_file.name} "
            f"({page_count} pages)"
        )

        uploaded_file.seek(0)

        with st.spinner(
            "Extracting text from the paper..."
        ):

            extracted_text = extract_text_from_pdf(
                uploaded_file
            )

        if not extracted_text.strip():

            st.warning(
                "No text could be extracted from this PDF. "
                "It may be a scanned/image-only document."
            )

        else:

            st.session_state.paper_text = (
                extracted_text
            )

            st.session_state.paper_name = (
                uploaded_file.name
            )

            st.success(
                f"Extracted approximately "
                f"{len(extracted_text):,} characters."
            )

    except Exception as e:

        st.error(
            f"Could not process the PDF: {e}"
        )


# ============================================================
# SECTION 3 — PAPER ANALYSIS
# ============================================================

if st.session_state.paper_text:

    st.divider()

    st.header("🧠 Paper Analysis")

    sections = extract_sections(
        st.session_state.paper_text
    )

    # --------------------------------------------------------
    # Detected sections
    # --------------------------------------------------------

    st.subheader("📑 Detected Sections")

    if "Full Text" in sections:

        st.warning(
            "Standard research-paper sections "
            "could not be detected."
        )

    else:

        st.success(
            f"Detected {len(sections)} sections."
        )

        st.write(
            ", ".join(sections.keys())
        )

        for section_name, section_text in sections.items():

            with st.expander(
                f"📌 {section_name}"
            ):

                st.write(section_text)

    # --------------------------------------------------------
    # Structured summary
    # --------------------------------------------------------

    st.subheader(
        "📝 Structured Paper Summary"
    )

    summary = create_paper_summary(
        sections
    )

    if summary:

        for section_name, content in summary.items():

            st.markdown(
                f"### {section_name}"
            )

            st.write(content)

    else:

        st.info(
            "A structured summary could not be created."
        )


# ============================================================
# SECTION 4 — RAG INDEXING
# ============================================================

if st.session_state.paper_text:

    st.divider()

    st.header(
        "🔍 Semantic Search"
    )

    st.write(
        "Create a semantic index of the uploaded paper "
        "before asking questions."
    )

    if st.session_state.indexed_paper_name:

        st.success(
            f"Currently indexed: "
            f"{st.session_state.indexed_paper_name}"
        )

    if st.button(
        "🧠 Index Paper for Semantic Search"
    ):

        with st.spinner(
            "Loading embedding model and indexing paper..."
        ):

            try:

                # Get section-aware content
                sections = extract_sections(
                    st.session_state.paper_text
                )

                rag = RAGEngine()

                chunk_count = rag.index_document(
                    st.session_state.paper_text,
                    sections=sections,
                )

                st.session_state.rag_engine = rag

                st.session_state.indexed_paper_name = (
                    st.session_state.paper_name
                )

                st.success(
                    f"Paper indexed successfully. "
                    f"Created {chunk_count} searchable chunks."
                )

            except Exception as e:

                st.error(
                    f"Could not index the paper: {e}"
                )


# ============================================================
# SECTION 5 — QUESTION ANSWERING
# ============================================================

if (
    st.session_state.rag_engine is not None
    and st.session_state.paper_text is not None
):

    st.divider()

    st.header(
        "🤖 Ask Questions About the Paper"
    )

    st.write(
        f"Current paper: "
        f"**{st.session_state.paper_name}**"
    )

    question = st.text_input(
        "Ask a question about this paper",
        placeholder=(
            "Example: What dataset was used in this study?"
        ),
    )

    top_k = st.slider(
        "Number of evidence passages to retrieve",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button(
        "🔍 Ask the Paper",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ==================================================
            # RETRIEVE EVIDENCE
            # ==================================================

            with st.spinner(
                "Searching the paper..."
            ):

                try:

                    results = (
                        st.session_state.rag_engine.search(
                            question,
                            top_k=top_k,
                        )
                    )

                except Exception as e:

                    st.error(
                        f"Search failed: {e}"
                    )

                    results = []

            if not results:

                st.info(
                    "No relevant passages were found "
                    "in the uploaded paper."
                )

            else:

                # ==================================================
                # RETRIEVED EVIDENCE
                # ==================================================

                st.subheader(
                    "📚 Retrieved Evidence"
                )

                for index, result in enumerate(
                    results,
                    start=1,
                ):

                    section_name = result.get(
                        "section",
                        "Unknown section",
                    )

                    score = result.get(
                        "score",
                        0.0,
                    )

                    with st.expander(
                        f"Evidence {index} — "
                        f"{section_name} — "
                        f"similarity {score:.3f}"
                    ):

                        st.write(
                            result["chunk"]
                        )

                # ==================================================
                # LOCAL AI ANSWER
                # ==================================================

                st.subheader(
                    "🤖 AI Answer"
                )

                try:

                    if (
                        st.session_state.answer_generator
                        is None
                    ):

                        with st.spinner(
                            "Loading local AI model..."
                        ):

                            st.session_state.answer_generator = (
                                AnswerGenerator()
                            )

                    with st.spinner(
                        "Generating answer from the retrieved evidence..."
                    ):

                        answer = (
                            st.session_state
                            .answer_generator
                            .generate_answer(
                                question,
                                results,
                            )
                        )

                    if answer:

                        st.write(answer)

                    else:

                        st.warning(
                            "The AI model returned an empty answer."
                        )

                except Exception as e:

                    st.error(
                        f"Could not generate the AI answer: {e}"
                    )

                    st.info(
                        "The retrieved evidence above is still "
                        "available."
                    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ Project Status")

    st.write("### Components")

    st.write("✅ PubMed Literature Search")

    st.write("✅ PDF Text Extraction")

    st.write("✅ Paper Section Detection")

    st.write("✅ Section-Aware Chunking")

    st.write("✅ Sentence Embeddings")

    st.write("✅ Semantic Retrieval")

    st.write("✅ Local AI Answer Generation")

    st.divider()

    st.write("### 🔒 Privacy")

    st.caption(
        "The uploaded paper and retrieval pipeline "
        "are processed locally by this application. "
        "No OpenAI API key is required."
    )

    st.divider()

    st.caption(
        "This tool is intended for research and "
        "literature exploration. It is not a substitute "
        "for professional medical advice."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Medical Literature Research Assistant "
    "— PubMed search • PDF analysis • semantic retrieval "
    "• local AI question answering"
)