import re
from typing import Dict, List


SECTION_ALIASES = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction", "background"],
    "Methods": ["methods", "methodology", "materials and methods", "materials & methods", "experimental methods"],
    "Results": ["results", "findings"],
    "Discussion": ["discussion"],
    "Limitations": ["limitations", "study limitations"],
    "Conclusion": ["conclusion", "conclusions", "summary and conclusion"],
    "References": ["references", "bibliography"],
}


def normalize_text(text: str) -> str:
    """Normalize extracted PDF text for section detection."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_section_positions(text: str) -> List[tuple[int, str, int]]:
    """Locate likely section headings in the paper."""
    positions: List[tuple[int, str, int]] = []
    text = normalize_text(text)

    for section_name, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?im)^\s*{re.escape(alias)}\s*$"
            match = re.search(pattern, text)
            if match:
                positions.append((match.start(), section_name, match.end()))
                break

    positions.sort(key=lambda item: item[0])
    return positions


def extract_sections(text: str) -> Dict[str, str]:
    """Split a paper into detected sections while preserving unstructured fallback."""
    normalized = normalize_text(text)
    if not normalized:
        return {"Full Text": ""}

    positions = find_section_positions(normalized)
    if not positions:
        return {"Full Text": normalized}

    sections: Dict[str, str] = {}
    for index, (start_position, section_name, end_position) in enumerate(positions):
        next_start = positions[index + 1][0] if index + 1 < len(positions) else len(normalized)
        section_text = normalized[end_position:next_start].strip()
        if section_text:
            sections[section_name] = section_text
    return sections


def create_paper_summary(sections: Dict[str, str]) -> Dict[str, str]:
    """Summarize the most relevant sections of a paper."""
    summary: Dict[str, str] = {}
    for section_name in ["Abstract", "Introduction", "Methods", "Results", "Limitations", "Conclusion"]:
        content = sections.get(section_name)
        if content:
            summary[section_name] = content[:1500]
    return summary


def extract_research_metadata(sections: Dict[str, str]) -> Dict[str, str]:
    """Extract a few methodologically relevant fields from a paper."""
    metadata: Dict[str, str] = {}
    methods_text = sections.get("Methods", "")
    results_text = sections.get("Results", "")
    abstract_text = sections.get("Abstract", "")

    if methods_text:
        metadata["methodology"] = methods_text[:1200]
    if results_text:
        metadata["results_summary"] = results_text[:1200]
    if abstract_text:
        metadata["objective"] = abstract_text[:800]

    return metadata