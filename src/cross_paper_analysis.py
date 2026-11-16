from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.paper_analyzer import extract_sections


POSITIVE_WORDS = {
    "improved",
    "increase",
    "increased",
    "higher",
    "better",
    "effective",
    "reduction",
    "reduced",
    "associated with improved",
    "supported",
    "benefit",
    "significant",
    "superior",
    "favorable",
    "lower risk",
    "less likely",
    "improves",
    "improve",
}
NEGATIVE_WORDS = {
    "worse",
    "worsened",
    "decreased",
    "lower",
    "less effective",
    "not effective",
    "inadequate",
    "insufficient",
    "no improvement",
    "no significant difference",
    "no benefit",
    "weak",
    "adverse",
    "increased risk",
    "higher risk",
    "did not improve",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "with",
    "was",
    "were",
    "is",
    "are",
    "on",
    "by",
    "from",
    "as",
    "at",
    "this",
    "that",
    "these",
    "those",
    "it",
    "be",
    "been",
    "their",
    "our",
    "study",
    "paper",
    "results",
    "method",
    "methods",
}


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned


def extract_page_reference(section_name: str, evidence_text: str, paper_pages: List[str]) -> str:
    """Return a page number only when a section text can be confidently matched to a source page."""
    if not paper_pages:
        return "Page reference unavailable"
    section_lookup = (section_name or "").lower()
    normalized_evidence = normalize_text(evidence_text).lower()
    for index, page_text in enumerate(paper_pages, start=1):
        page_lower = page_text.lower()
        if section_lookup in page_lower and normalized_evidence in page_lower:
            return f"Page {index}"
        if normalized_evidence in page_lower:
            return f"Page {index}"
    return "Page reference unavailable"


def infer_direction(text: str) -> str:
    normalized = normalize_text(text).lower()
    if any(word in normalized for word in POSITIVE_WORDS):
        return "positive"
    if any(word in normalized for word in NEGATIVE_WORDS):
        return "negative"
    return "neutral"


def infer_theme(text: str) -> str:
    normalized = normalize_text(text).lower()
    if any(word in normalized for word in ["accuracy", "sensitivity", "specificity", "precision", "diagnostic"]):
        return "diagnostic_performance"
    if any(word in normalized for word in ["mortality", "survival", "death", "outcome", "events"]):
        return "clinical_outcome"
    if any(word in normalized for word in ["risk", "adverse", "toxicity", "complication"]):
        return "risk_or_safety"
    if any(word in normalized for word in ["dataset", "population", "sample", "cohort", "patients"]):
        return "cohort_characteristics"
    tokens = []
    for token in re.findall(r"[a-zA-Z]+", normalized):
        if token not in STOPWORDS and len(token) > 4:
            tokens.append(token)
    return "_".join(tokens[:3]) if tokens else "general_findings"


def normalize_for_compare(text: str) -> str:
    normalized = normalize_text(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = [token for token in normalized.split() if token not in STOPWORDS and len(token) > 2]
    return " ".join(tokens)


class CrossPaperEvidenceDetector:
    """Compare findings across multiple papers and classify evidence strength."""

    def build_paper_profile(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        text = paper.get("text", "")
        sections = extract_sections(text)
        paper_profile: Dict[str, Any] = {
            "title": paper.get("title", "Untitled paper"),
            "source": paper.get("source", "local_upload"),
            "text": text,
            "sections": sections,
            "objective": self._extract_objective(sections),
            "methodology": self._extract_methodology(sections),
            "dataset": self._extract_dataset(sections),
            "sample_size": self._extract_sample_size(sections),
            "findings": self._extract_findings(sections),
            "limitations": self._extract_limitations(sections),
            "page_texts": paper.get("page_texts", []),
        }
        return paper_profile

    def _extract_objective(self, sections: Dict[str, str]) -> str:
        for section_name in ["Abstract", "Introduction"]:
            content = sections.get(section_name)
            if content:
                return content[:800]
        return "Objective not clearly extracted from the available sections."

    def _extract_methodology(self, sections: Dict[str, str]) -> str:
        for section_name in ["Methods", "Materials and Methods"]:
            content = sections.get(section_name)
            if content:
                return content[:1200]
        return "Methodology not clearly identified."

    def _extract_dataset(self, sections: Dict[str, str]) -> str:
        text = "\n".join(sections.values())
        dataset_patterns = [
            r"dataset[s]?\s*[:\-]?\s*([^\n.]{0,200})",
            r"cohort[s]?\s*[:\-]?\s*([^\n.]{0,200})",
            r"registry\s*[:\-]?\s*([^\n.]{0,200})",
            r"population\s*[:\-]?\s*([^\n.]{0,200})",
        ]
        for pattern in dataset_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Dataset information was not clearly stated in the extracted text."

    def _extract_sample_size(self, sections: Dict[str, str]) -> str:
        text = "\n".join(sections.values())
        patterns = [
            r"(\d+(?:,\d+)?)\s+(patients|participants|subjects|samples|cases)",
            r"sample size\s*[:\-]?\s*(\d+(?:,\d+)?)",
            r"n\s*=\s*(\d+(?:,\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1)
                return f"{value} participants or cases (extracted from paper text)"
        return "Sample size was not clearly stated in the available text."

    def _extract_findings(self, sections: Dict[str, str]) -> List[str]:
        text_blocks = []
        for section_name in ["Results", "Conclusion", "Discussion"]:
            block = sections.get(section_name)
            if block:
                text_blocks.append(block)
        if not text_blocks:
            return []
        combined = "\n".join(text_blocks)
        sentences = split_sentences(combined)
        findings = []
        for sentence in sentences:
            lowered = sentence.lower()
            if len(sentence) < 25:
                continue
            if any(keyword in lowered for keyword in ["improved", "increased", "decreased", "higher", "lower", "effective", "sensitivity", "accuracy", "risk", "mortality", "association", "benefit", "no significant"]):
                findings.append(sentence)
        return findings[:8]

    def _extract_limitations(self, sections: Dict[str, str]) -> str:
        text = "\n".join(sections.values())
        for section_name in ["Limitations", "Discussion"]:
            content = sections.get(section_name)
            if content:
                return content[:1200]
        if "limitation" in text.lower() or "limitations" in text.lower():
            return text[:1200]
        return "Limitations were not explicitly identified in the extracted text."

    def _extract_claims(self, paper_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        sections = paper_profile.get("sections", {})
        relevant_sections = ["Results", "Conclusion", "Discussion", "Abstract"]
        for section_name in relevant_sections:
            content = sections.get(section_name)
            if not content:
                continue
            for sentence in split_sentences(content):
                if len(sentence) < 30:
                    continue
                direction = infer_direction(sentence)
                theme = infer_theme(sentence)
                claims.append(
                    {
                        "paper_title": paper_profile.get("title"),
                        "source": paper_profile.get("source"),
                        "section": section_name,
                        "text": sentence,
                        "direction": direction,
                        "theme": theme,
                        "page_reference": extract_page_reference(section_name, sentence, paper_profile.get("page_texts", [])),
                    }
                )
        return claims

    def analyze(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(papers) < 2:
            return {
                "comparisons": [],
                "summary": "At least two papers are required to compare findings.",
                "download_text": "No comparison report generated because fewer than two papers were selected.",
            }

        profiles = [self.build_paper_profile(paper) for paper in papers]
        claim_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for profile in profiles:
            for claim in self._extract_claims(profile):
                claim_buckets[claim["theme"]].append(claim)

        comparisons: List[Dict[str, Any]] = []
        for theme, claims in claim_buckets.items():
            comparable_claims = []
            for claim in claims:
                normalized = normalize_for_compare(claim["text"])
                if normalized:
                    comparable_claims.append((claim["text"], claim))
            if not comparable_claims or len(comparable_claims) < 2:
                continue

            directions = [claim["direction"] for _, claim in comparable_claims]
            unique_directions = set(directions)
            unique_papers = {claim["paper_title"] for _, claim in comparable_claims}

            if len(unique_papers) < 2:
                classification = "Insufficient evidence"
            elif len(unique_directions) == 1 and "neutral" not in unique_directions:
                classification = "Supporting"
            elif "positive" in unique_directions and "negative" in unique_directions:
                classification = "Contradicting"
            elif "neutral" in unique_directions and len(unique_directions) > 1:
                classification = "Mixed"
            elif all(direction == "neutral" for direction in directions):
                classification = "Insufficient evidence"
            else:
                classification = "Not comparable"

            if classification == "Not comparable" and len(comparable_claims) < 2:
                classification = "Insufficient evidence"

            evidence_items = [
                {
                    "paper_title": claim["paper_title"],
                    "source": claim["source"],
                    "section": claim["section"],
                    "page_reference": claim.get("page_reference", "Page reference unavailable"),
                    "evidence": claim["text"],
                    "direction": claim["direction"],
                }
                for _, claim in comparable_claims
            ]

            comparisons.append(
                {
                    "theme": theme,
                    "classification": classification,
                    "evidence": evidence_items,
                    "paper_count": len({item["paper_title"] for item in evidence_items}),
                    "summary": self._build_theme_summary(theme, evidence_items),
                }
            )

        if not comparisons:
            return {
                "comparisons": [],
                "summary": "No comparable claims were extracted. The selected papers do not provide enough directly comparable evidence for a trustworthy cross-paper comparison.",
                "download_text": "No comparable findings detected. No confident cross-paper evidence classification was produced.",
            }

        report_text = self.generate_report_text(comparisons)
        return {
            "comparisons": comparisons,
            "summary": self._summarize_findings(comparisons),
            "download_text": report_text,
        }

    def _build_theme_summary(self, theme: str, evidence_items: List[Dict[str, Any]]) -> str:
        directions = [item["direction"] for item in evidence_items]
        if any(direction == "positive" for direction in directions) and any(direction == "negative" for direction in directions):
            return f"The selected papers show mixed directionality for the {theme} theme."
        if all(direction == "positive" for direction in directions):
            return f"The selected papers consistently support the {theme} direction."
        if all(direction == "negative" for direction in directions):
            return f"The selected papers consistently describe a negative {theme} signal."
        return f"The available evidence for the {theme} theme is not sufficient to draw a confident conclusion."

    def _summarize_findings(self, comparisons: List[Dict[str, Any]]) -> str:
        total = len(comparisons)
        classification_counts = defaultdict(int)
        for item in comparisons:
            classification_counts[item["classification"]] += 1
        return (
            f"Compared {total} evidence themes across the selected papers. "
            + "; ".join(f"{key}: {value}" for key, value in sorted(classification_counts.items()))
        )

    def generate_report_text(self, comparisons: List[Dict[str, Any]]) -> str:
        lines = [
            "CROSS-PAPER EVIDENCE REPORT",
            "==========================",
            "",
            "This report is generated from extracted paper text only. Evidence is classified from the available passages and should be reviewed by a researcher before drawing conclusions.",
            "",
        ]
        for index, item in enumerate(comparisons, start=1):
            lines.append(f"{index}. Theme: {item['theme']}")
            lines.append(f"   Classification: {item['classification']}")
            lines.append(f"   Summary: {item['summary']}")
            for evidence in item["evidence"]:
                lines.append(f"   - Source: {evidence['paper_title']}")
                lines.append(f"     Section: {evidence['section']}")
                lines.append(f"     Page: {evidence['page_reference']}")
                lines.append(f"     Evidence: {evidence['evidence']}")
                lines.append(f"     Direction: {evidence['direction']}")
            lines.append("")
        return "\n".join(lines)
