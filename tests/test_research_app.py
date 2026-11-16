import io
import sys
from pathlib import Path

import numpy as np
import pytest
import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_paper_analysis import CrossPaperEvidenceDetector
from src.paper_search import search_pubmed
from src.paper_analyzer import create_paper_summary, extract_sections
from src.pdf_processor import extract_text_from_pdf, get_page_count


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_search_pubmed_returns_paper_metadata(monkeypatch):
    def fake_get(url, params, timeout):
        if "esearch.fcgi" in url:
            return DummyResponse({"esearchresult": {"idlist": ["123", "456"]}})
        return DummyResponse({
            "result": {
                "123": {
                    "title": "Study on cancer detection",
                    "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}],
                    "fulljournalname": "Journal of Clinical Research",
                    "pubdate": "2024",
                    "abstract": "Cancer detection improved with early screening.",
                },
                "456": {
                    "title": "Another study",
                    "authors": [{"name": "Carol Lee"}],
                    "fulljournalname": "Medical Review",
                    "pubdate": "2023",
                    "abstract": "This paper reports a different result.",
                },
            }
        })

    monkeypatch.setattr("requests.get", fake_get)
    papers = search_pubmed("cancer detection", max_results=2)

    assert len(papers) == 2
    assert papers[0]["title"] == "Study on cancer detection"
    assert papers[0]["authors"] == ["Alice Smith", "Bob Jones"]


def test_pdf_extraction_common_document():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Abstract\nThis study examined diagnostic accuracy.\n\nMethods\nFifty patients were enrolled.")
    pdf_bytes = doc.tobytes()
    doc.close()

    text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    assert "Abstract" in text
    assert "Methods" in text
    assert get_page_count(io.BytesIO(pdf_bytes)) == 1


def test_section_detection_handles_standard_headings():
    document = """
    Abstract
    This study evaluates an AI model.

    Introduction
    Background information is provided.

    Methods
    Patients were followed for 18 months.

    Results
    The model achieved strong performance.

    Conclusion
    Additional validation is needed.
    """

    sections = extract_sections(document)
    assert "Abstract" in sections
    assert "Introduction" in sections
    assert "Methods" in sections
    assert "Conclusion" in sections
    assert create_paper_summary(sections)


def test_rag_engine_uses_embedding_similarity(monkeypatch):
    import src.rag_engine as rag_module

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return np.array([1.0, 0.0], dtype=float)
            return np.array([
                [1.0, 0.0],
                [0.7, 0.7],
                [0.0, 1.0],
            ], dtype=float)[: len(texts)]

    monkeypatch.setattr(rag_module, "SentenceTransformer", FakeSentenceTransformer)
    engine = rag_module.RAGEngine()
    engine.index_document(
        "Abstract\nSample abstract\n\nMethods\nA supervised study.",
        sections={
            "Abstract": "Sample abstract with findings.",
            "Methods": "A supervised study was conducted.",
        },
    )

    results = engine.search("What was studied?", top_k=2)
    assert len(results) == 2
    assert results[0]["section"] in {"Abstract", "Methods"}


def test_answer_generator_abstains_when_evidence_missing(monkeypatch):
    import src.answer_generator as answer_module

    class DummyTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            return DummyTokenizer()

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [1]}

        def decode(self, outputs, skip_special_tokens=True):
            return "The paper does not provide enough information."

    class DummyModel:
        @staticmethod
        def from_pretrained(model_name):
            return DummyModel()

        def generate(self, **kwargs):
            return [[1]]

    monkeypatch.setattr(answer_module, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(answer_module, "AutoModelForSeq2SeqLM", DummyModel)

    generator = answer_module.AnswerGenerator()
    answer = generator.generate_answer("What was the sample size?", [])
    assert "does not provide enough information" in answer.lower()


def test_cross_paper_detector_supporting_evidence():
    detector = CrossPaperEvidenceDetector()
    papers = [
        {
            "title": "Paper A",
            "source": "synthetic",
            "text": "Abstract\nDiagnostic accuracy improved significantly after model deployment.\n\nResults\nThe model increased sensitivity and specificity, improving diagnostic accuracy across the cohort.",
        },
        {
            "title": "Paper B",
            "source": "synthetic",
            "text": "Abstract\nDiagnostic accuracy improved in this dataset compared with baseline.\n\nResults\nSensitivity and specificity improved, confirming better diagnostic accuracy.",
        },
    ]

    analysis = detector.analyze(papers)
    assert analysis["comparisons"]
    assert any(item["classification"] == "Supporting" for item in analysis["comparisons"])


def test_cross_paper_detector_reduces_false_contradictions():
    detector = CrossPaperEvidenceDetector()
    papers = [
        {
            "title": "Paper A",
            "source": "synthetic",
            "text": "Abstract\nThe model improved diagnostic accuracy.\n\nResults\nSensitivity was higher in the final cohort.",
        },
        {
            "title": "Paper B",
            "source": "synthetic",
            "text": "Abstract\nParticipants experienced lower mortality.\n\nResults\nMortality was reduced in the intervention arm.",
        },
    ]

    analysis = detector.analyze(papers)
    assert all(item["classification"] != "Contradicting" for item in analysis.get("comparisons", []))


def test_cross_paper_detector_abstains_when_evidence_is_insufficient():
    detector = CrossPaperEvidenceDetector()
    papers = [
        {
            "title": "Paper A",
            "source": "synthetic",
            "text": "Abstract\nThe study reports an exploratory analysis.\n\nDiscussion\nThe findings may be relevant but are not directly comparable.",
        },
        {
            "title": "Paper B",
            "source": "synthetic",
            "text": "Abstract\nThe study explores a different patient group.\n\nDiscussion\nThe clinical context differs substantially from the prior study.",
        },
    ]

    analysis = detector.analyze(papers)
    assert analysis["comparisons"] == [] or all(item["classification"] in {"Insufficient evidence", "Not comparable"} for item in analysis["comparisons"])
