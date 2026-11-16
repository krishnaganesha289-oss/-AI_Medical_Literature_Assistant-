from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_LIBRARY_PATH = Path(__file__).resolve().parents[1] / "data" / "research_library.json"


def _ensure_library_file(path: Path | str = DEFAULT_LIBRARY_PATH) -> Path:
    library_path = Path(path)
    library_path.parent.mkdir(parents=True, exist_ok=True)
    if not library_path.exists():
        library_path.write_text(json.dumps({"papers": []}, indent=2), encoding="utf-8")
    return library_path


def load_library(path: Path | str = DEFAULT_LIBRARY_PATH) -> List[Dict[str, Any]]:
    library_path = _ensure_library_file(path)
    try:
        with library_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    papers = payload.get("papers", []) if isinstance(payload, dict) else []
    return papers if isinstance(papers, list) else []


def save_library(papers: List[Dict[str, Any]], path: Path | str = DEFAULT_LIBRARY_PATH) -> None:
    library_path = _ensure_library_file(path)
    with library_path.open("w", encoding="utf-8") as handle:
        json.dump({"papers": papers}, handle, indent=2, ensure_ascii=False)


def save_paper_entry(paper: Dict[str, Any], path: Path | str = DEFAULT_LIBRARY_PATH) -> Dict[str, Any]:
    library = load_library(path)
    key = paper.get("pmid") or paper.get("title", "")
    if key:
        for index, existing in enumerate(library):
            existing_key = existing.get("pmid") or existing.get("title", "")
            if existing_key == key:
                library[index] = paper
                save_library(library, path)
                return paper
    library.append(paper)
    save_library(library, path)
    return paper


def build_saved_note(title: str, content: str, source: str = "User note") -> Dict[str, Any]:
    return {
        "title": title,
        "type": "note",
        "source": source,
        "content": content,
        "saved_at": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
