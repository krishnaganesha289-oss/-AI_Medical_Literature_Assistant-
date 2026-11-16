import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.answer_generator import AnswerGenerator
from src.cross_paper_analysis import CrossPaperEvidenceDetector
from src.paper_analyzer import create_paper_summary, extract_research_metadata, extract_sections
from src.paper_search import search_pubmed
from src.pdf_processor import extract_pdf_document, get_page_count
from src.rag_engine import RAGEngine
from src.research_library import build_saved_note, load_library, save_paper_entry


st.set_page_config(
    page_title="AI Medical Literature Research Assistant",
    page_icon="🧬",
    layout="wide",
)


PAGE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(180deg, #f4f8ff 0%, #eef4f6 100%);
        color: #132238;
    }
    .stApp {
        background: #f7fbff;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(17,60,105,0.99), rgba(21,109,137,0.90));
        color: white;
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 18px 40px rgba(15, 48, 82, 0.18);
    }
    .glass-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(20,40,90,0.08);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 10px 24px rgba(15, 48, 82, 0.06);
    }
    .metric-box {
        background: #ffffff;
        border: 1px solid rgba(18,45,75,0.08);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 6px 18px rgba(15, 48, 82, 0.05);
    }
    .tiny-note {
        color: #4a627a;
        font-size: 0.8rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background: #dff7f0;
        color: #0b6b46;
        margin-right: 0.5rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f9ff 0%, #eef6f8 100%);
    }
    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)


def reset_paper_state():
    st.session_state.paper_text = None
    st.session_state.paper_name = None
    st.session_state.paper_metadata = None
    st.session_state.rag_engine = None
    st.session_state.answer_generator = None
    st.session_state.indexed_paper_name = None
    st.session_state.extraction_summary = None


if "paper_text" not in st.session_state:
    st.session_state.paper_text = None
if "paper_name" not in st.session_state:
    st.session_state.paper_name = None
if "paper_metadata" not in st.session_state:
    st.session_state.paper_metadata = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "answer_generator" not in st.session_state:
    st.session_state.answer_generator = None
if "indexed_paper_name" not in st.session_state:
    st.session_state.indexed_paper_name = None
if "extraction_summary" not in st.session_state:
    st.session_state.extraction_summary = None
if "library" not in st.session_state:
    st.session_state.library = load_library()
if "research_notes" not in st.session_state:
    st.session_state.research_notes = []


st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div style="max-width:60%;">
                <div class="status-badge">Evidence-first research</div>
                <h1 style="margin:0.2rem 0 0.6rem 0; font-size:2.3rem;">AI Medical Literature Research Assistant</h1>
                <p style="margin:0; color: rgba(255,255,255,0.82); font-size:1.02rem; line-height:1.6;">
                    Discover medical papers, analyze PDF evidence, compare findings, and ask evidence-grounded questions about research documents.
                </p>
            </div>
            <div style="max-width:34%; min-width:260px;">
                <img src="https://images.unsplash.com/photo-1532187643603-ba119ca4109e?auto=format&fit=crop&w=1000&q=80" style="width:100%; border-radius:20px; object-fit:cover; max-height:220px;" />
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

main_cols = st.columns([1.2, 0.8])
with main_cols[0]:
    st.subheader("Research workflow")
    st.write("1. Search the literature.  2. Upload a paper.  3. Index and investigate evidence.  4. Save notes and compare findings.")
with main_cols[1]:
    st.subheader("Privacy")
    st.caption("Uploaded PDFs are processed locally. No paid API is required.")

st.divider()


st.header("🔎 Literature discovery")
search_col, filter_col = st.columns([2, 1])
with search_col:
    query = st.text_input("Search PubMed by topic, condition, author, or keyword", value="diabetes mellitus and machine learning", placeholder="Example: oncology biomarkers and AI")
with filter_col:
    sort_mode = st.selectbox("Sort result order", ["relevance", "date"])
search_page = st.number_input("Page number", min_value=1, value=1, step=1)
max_results = st.slider("Results per page", min_value=5, max_value=20, value=10, step=1)

if st.button("Search PubMed", type="primary"):
    if not query.strip():
        st.warning("Enter a search query before looking up literature.")
    else:
        with st.spinner("Querying PubMed..."):
            try:
                papers = search_pubmed(query=query, max_results=max_results, page=search_page, sort=sort_mode)
                if not papers:
                    st.info("No papers matched the current query. Try a broader keyword set or different search terms.")
                else:
                    st.success(f"Found {len(papers)} papers.")
                    for paper in papers:
                        title = paper.get("title", "Untitled paper")
                        authors = paper.get("authors", [])
                        abstract = paper.get("abstract", "")
                        authors_text = ", ".join(authors[:5]) if authors else "Authors unavailable"
                        if len(authors) > 5:
                            authors_text += " et al."
                        paper_card = st.container()
                        with paper_card:
                            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                            st.markdown(f"### {title}")
                            st.write(f"**Authors:** {authors_text}")
                            st.write(f"**Journal:** {paper.get('journal', 'Unknown journal')}")
                            st.write(f"**Publication date:** {paper.get('publication_date', 'Unknown')}")
                            if abstract:
                                st.write(f"**Abstract preview:** {abstract[:500]}{'...' if len(abstract) > 500 else ''}")
                            st.markdown(
                                f"<div class='tiny-note'>PMID: {paper.get('pmid', 'Unknown')} · <a href='{paper.get('pubmed_url', '#')}' target='_blank'>Open in PubMed</a></div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"PubMed request failed: {exc}")

st.divider()


st.header("📄 PDF analysis")
uploaded_file = st.file_uploader("Upload a research PDF", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.paper_name != uploaded_file.name:
        reset_paper_state()
        st.session_state.paper_name = uploaded_file.name

    try:
        page_count = get_page_count(uploaded_file)
        st.success(f"Uploaded: {uploaded_file.name} ({page_count} pages)")
        extracted = extract_pdf_document(uploaded_file)
        st.session_state.extraction_summary = extracted
        st.session_state.paper_text = extracted.get("text", "")
        st.session_state.paper_metadata = {
            "page_count": page_count,
            "scanned_pages": extracted.get("scanned_pages", 0),
            "is_scanned": extracted.get("is_scanned", False),
        }

        if not st.session_state.paper_text.strip():
            st.warning("No extractable text was found in the PDF. This may be a scanned or image-only document, and OCR is not available in this local workflow.")
        else:
            st.info(f"Extracted approximately {len(st.session_state.paper_text):,} characters from the document.")
            if extracted.get("is_scanned"):
                st.warning("The document appears to be image-based or scanned, and text extraction may be incomplete.")
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Failed to process the PDF: {exc}")

if st.session_state.paper_text:
    sections = extract_sections(st.session_state.paper_text)
    summary = create_paper_summary(sections)
    metadata = extract_research_metadata(sections)

    st.subheader("Detected paper structure")
    if "Full Text" in sections:
        st.warning("Standard section headings were not detected. The full text will be treated as a single document block.")
    else:
        st.success(f"Detected {len(sections)} major sections.")
        st.write(", ".join(sections.keys()))

    for section_name, section_text in sections.items():
        with st.expander(f"{section_name}"):
            st.write(section_text[:5000] if len(section_text) > 5000 else section_text)

    st.subheader("Research summary")
    if summary:
        for section_name, content in summary.items():
            st.markdown(f"### {section_name}")
            st.write(content)
    else:
        st.info("The paper did not contain enough structure for a detailed summary.")

    if metadata:
        st.subheader("Methodological cues")
        for key, value in metadata.items():
            st.markdown(f"**{key.replace('_', ' ').title()}:**")
            st.write(value[:1200] if len(value) > 1200 else value)

st.divider()


if st.session_state.paper_text:
    st.header("🧠 Evidence retrieval")
    st.write("Create a semantic index and retrieve the most relevant evidence passages from the uploaded paper.")
    if st.session_state.indexed_paper_name:
        st.success(f"Currently indexed: {st.session_state.indexed_paper_name}")

    if st.button("Index paper for retrieval"):
        with st.spinner("Loading the embedding model and indexing the paper..."):
            try:
                rag = RAGEngine()
                parsed_sections = extract_sections(st.session_state.paper_text)
                chunk_count = rag.index_document(st.session_state.paper_text, sections=parsed_sections)
                st.session_state.rag_engine = rag
                st.session_state.indexed_paper_name = st.session_state.paper_name
                st.success(f"Paper indexed successfully. {chunk_count} searchable chunks were created.")
            except Exception as exc:
                st.error(f"Could not index the paper: {exc}")

    if st.session_state.rag_engine is not None:
        question = st.text_input("Ask a question about the paper", value="What is the study objective?", key="paper_question")
        top_k = st.slider("Evidence passages to retrieve", min_value=1, max_value=5, value=3)
        if st.button("Retrieve evidence and answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question before retrieving evidence.")
            else:
                with st.spinner("Finding relevant passages..."):
                    try:
                        evidence = st.session_state.rag_engine.search(question, top_k=top_k)
                    except Exception as exc:
                        st.error(f"Retrieval failed: {exc}")
                        evidence = []

                if not evidence:
                    st.info("No passages in the paper matched the question closely enough to support an answer.")
                else:
                    st.subheader("Relevant evidence")
                    for index, result in enumerate(evidence, start=1):
                        section_name = result.get("section", "Unknown section")
                        score = result.get("score", 0.0)
                        with st.expander(f"Evidence {index} · {section_name} · similarity {score:.3f}"):
                            st.write(result.get("chunk", ""))

                    st.subheader("Evidence-based answer")
                    if st.session_state.answer_generator is None:
                        with st.spinner("Loading the local answer model..."):
                            st.session_state.answer_generator = AnswerGenerator()

                    with st.spinner("Generating evidence-grounded answer..."):
                        answer = st.session_state.answer_generator.generate_answer(question, evidence)

                    st.write(answer)
                    st.caption("Interpretation is anchored to the retrieved passages above; unsupported claims are not fabricated.")

st.divider()


st.header("🗂️ Saved research library")
col_save, col_note = st.columns([1, 1])
with col_save:
    if st.button("Save current paper to library"):
        if not st.session_state.paper_text:
            st.warning("Upload and process a paper before saving it.")
        else:
            sections_for_save = extract_sections(st.session_state.paper_text)
            paper_entry = {
                "title": st.session_state.paper_name or "Untitled paper",
                "type": "paper",
                "pmid": "",
                "journal": "Uploaded PDF",
                "publication_date": "Uploaded locally",
                "abstract": (st.session_state.paper_text[:500] if st.session_state.paper_text else ""),
                "text": st.session_state.paper_text,
                "sections": sections_for_save,
                "page_texts": st.session_state.extraction_summary.get("page_texts", []) if st.session_state.extraction_summary else [],
                "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "source": "local_upload",
            }
            save_paper_entry(paper_entry)
            st.session_state.library = load_library()
            st.success("The paper was saved to the local research library.")

with col_note:
    note_title = st.text_input("Note title", value="Research note")
    note_text = st.text_area("Write a note", height=130)
    if st.button("Save note"):
        if not note_text.strip():
            st.warning("Add content before saving a note.")
        else:
            entry = build_saved_note(note_title or "Research note", note_text, source="research-notes")
            st.session_state.research_notes.append(entry)
            st.success("The note was saved locally for this session.")

saved_library = st.session_state.library
if saved_library:
    st.subheader("Saved papers")
    for item in saved_library[:8]:
        with st.expander(item.get("title", "Saved paper")):
            st.write(f"**Type:** {item.get('type', 'paper')}")
            st.write(f"**Source:** {item.get('source', 'unknown')}")
            st.write(f"**Saved:** {item.get('saved_at', 'unknown')}")
            if item.get("abstract"):
                st.write(item["abstract"][:600])
else:
    st.info("No files are saved yet.")

if st.session_state.research_notes:
    st.subheader("Saved notes")
    for note in st.session_state.research_notes:
        with st.expander(note.get("title", "Research note")):
            st.write(note.get("content", ""))

st.divider()

st.header("🧪 Cross-paper evidence & contradiction detector")
comparison_intro = st.caption("Compare 2-5 research papers, identify comparable findings, and classify evidence as supporting, contradicting, mixed, insufficient, or not comparable.")
comparison_uploaded = st.file_uploader("Upload 2-5 PDFs for cross-paper comparison", type=["pdf"], accept_multiple_files=True)
comparison_saved = st.multiselect("Or select saved papers from the local library", [item.get("title", "Untitled paper") for item in saved_library])

if st.button("Analyze cross-paper evidence"):
    selected_papers = []
    for uploaded in comparison_uploaded or []:
        try:
            extracted = extract_pdf_document(uploaded)
            selected_papers.append(
                {
                    "title": uploaded.name,
                    "source": "uploaded_pdf",
                    "text": extracted.get("text", ""),
                    "page_texts": extracted.get("page_texts", []),
                }
            )
        except Exception as exc:
            st.warning(f"{uploaded.name} could not be processed: {exc}")

    for title in comparison_saved:
        matching = next((item for item in saved_library if item.get("title") == title), None)
        if matching and matching.get("text"):
            selected_papers.append({
                "title": title,
                "source": matching.get("source", "saved_library"),
                "text": matching.get("text", ""),
                "page_texts": matching.get("page_texts", []),
            })

    selected_papers = selected_papers[:5]
    if len(selected_papers) < 2:
        st.warning("Select at least two research papers to run a cross-paper evidence comparison.")
    else:
        detector = CrossPaperEvidenceDetector()
        analysis = detector.analyze(selected_papers)
        st.subheader("Comparison summary")
        st.write(analysis.get("summary", "No summary available."))

        if not analysis.get("comparisons"):
            st.info("No comparable claims were extracted. The selected papers do not provide enough directly comparable evidence to classify.")
        else:
            for index, comparison in enumerate(analysis["comparisons"], start=1):
                badge_color = "#dff7f0" if comparison["classification"] == "Supporting" else "#fff2d8" if comparison["classification"] in {"Mixed", "Contradicting"} else "#eef2ff"
                with st.container():
                    st.markdown(f"<div class='glass-card' style='background:{badge_color};'><h4>{index}. {comparison['theme']}</h4><p><strong>Classification:</strong> {comparison['classification']}</p><p>{comparison['summary']}</p></div>", unsafe_allow_html=True)
                    for evidence in comparison["evidence"]:
                        with st.expander(f"{evidence['paper_title']} · {evidence['section']} · {evidence['page_reference']}"):
                            st.write(f"**Evidence:** {evidence['evidence']}")
                            st.write(f"**Direction:** {evidence['direction']}")
                            st.caption(f"Source paper: {evidence['paper_title']} · Section: {evidence['section']}")
                    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        st.download_button(
            label="Download comparison report",
            data=analysis.get("download_text", "No report available."),
            file_name="cross_paper_evidence_report.txt",
            mime="text/plain",
        )

st.divider()

st.header("📊 Compare papers")
compare_candidates = [item.get("title", "Untitled") for item in saved_library]
if len(compare_candidates) >= 2:
    left_choice, right_choice = st.selectbox("Study A", compare_candidates), st.selectbox("Study B", compare_candidates)
    if st.button("Compare selected papers"):
        left_paper = next((item for item in saved_library if item.get("title") == left_choice), None)
        right_paper = next((item for item in saved_library if item.get("title") == right_choice), None)
        if left_paper and right_paper:
            st.subheader(f"Comparison: {left_choice} vs {right_choice}")
            comparison_cols = st.columns(2)
            for paper, col in [(left_paper, comparison_cols[0]), (right_paper, comparison_cols[1])]:
                with col:
                    st.markdown(f"### {paper.get('title', 'Untitled')}")
                    st.write(f"**Source:** {paper.get('source', 'unknown')}")
                    st.write(f"**Journal:** {paper.get('journal', 'Unknown journal')}")
                    st.write(f"**Abstract:** {paper.get('abstract', 'No summary available')[:800]}")
else:
    st.info("Save at least two papers to enable a side-by-side comparison view.")


with st.sidebar:
    st.header("ℹ️ Project status")
    st.write("✅ PubMed search")
    st.write("✅ PDF extraction and validation")
    st.write("✅ Section-aware analysis")
    st.write("✅ Semantic retrieval")
    st.write("✅ Evidence-grounded answer generation")
    st.write("✅ Local research library")
    st.divider()
    st.write("### Privacy")
    st.caption("Local processing only. No mandatory login, paid API, or external upload required.")
    st.caption("This workflow is for educational and research exploration and is not medical advice.")
    if st.button("Clear current paper"):
        reset_paper_state()
        st.rerun()
