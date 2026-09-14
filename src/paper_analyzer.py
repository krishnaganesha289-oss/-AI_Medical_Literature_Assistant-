import re


SECTION_ALIASES = {
    "Abstract": [
        "abstract"
    ],
    "Introduction": [
        "introduction",
        "background"
    ],
    "Methods": [
        "methods",
        "methodology",
        "materials and methods",
        "materials & methods",
        "experimental methods"
    ],
    "Results": [
        "results",
        "findings"
    ],
    "Discussion": [
        "discussion"
    ],
    "Limitations": [
        "limitations",
        "study limitations"
    ],
    "Conclusion": [
        "conclusion",
        "conclusions",
        "summary and conclusion"
    ],
    "References": [
        "references",
        "bibliography"
    ],
}


def normalize_text(text):
    """Clean extracted PDF text."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def find_section_positions(text):
    """Find positions of common research-paper sections."""

    positions = []

    for section_name, aliases in SECTION_ALIASES.items():

        for alias in aliases:

            pattern = rf"(?im)^\s*{re.escape(alias)}\s*$"

            match = re.search(pattern, text)

            if match:
                positions.append(
                    (
                        match.start(),
                        section_name,
                        match.end()
                    )
                )
                break

    positions.sort(
        key=lambda item: item[0]
    )

    return positions


def extract_sections(text):
    """Split a research paper into detected sections."""

    text = normalize_text(text)

    positions = find_section_positions(text)

    sections = {}

    if not positions:
        sections["Full Text"] = text
        return sections

    for index, current in enumerate(positions):

        start_position = current[2]

        if index + 1 < len(positions):
            end_position = positions[index + 1][0]
        else:
            end_position = len(text)

        section_name = current[1]

        section_text = text[
            start_position:end_position
        ].strip()

        if section_text:
            sections[section_name] = section_text

    return sections


def create_paper_summary(sections):
    """Create a basic structured summary."""

    summary = {}

    summary_sections = [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Limitations",
        "Conclusion"
    ]

    for section_name in summary_sections:

        content = sections.get(section_name)

        if content:
            summary[section_name] = content[:1500]

    return summary