import time
from typing import Any, Dict, List

import requests


PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def search_pubmed(query: str, max_results: int = 10, page: int = 1, sort: str = "relevance") -> List[Dict[str, Any]]:
    """Search PubMed and return structured paper metadata."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("A search query is required.")

    max_results = max(1, min(int(max_results), 50))
    page = max(1, int(page))
    sort = sort if sort in {"relevance", "date"} else "relevance"

    search_params = {
        "db": "pubmed",
        "term": cleaned_query,
        "retmode": "json",
        "retmax": max_results,
        "retstart": (page - 1) * max_results,
        "sort": sort,
    }

    try:
        response = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"PubMed search request failed: {exc}") from exc

    esearch_result = data.get("esearchresult", {}) if isinstance(data, dict) else {}
    paper_ids = esearch_result.get("idlist", [])
    if not paper_ids:
        return []

    return get_paper_details(paper_ids)


def get_paper_details(paper_ids: List[str]) -> List[Dict[str, Any]]:
    """Collect summary metadata for PubMed article IDs."""
    if not paper_ids:
        return []

    summary_params = {
        "db": "pubmed",
        "id": ",".join(str(paper_id) for paper_id in paper_ids),
        "retmode": "json",
    }

    try:
        response = requests.get(PUBMED_SUMMARY_URL, params=summary_params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"PubMed summary request failed: {exc}") from exc

    results = data.get("result", {}) if isinstance(data, dict) else {}
    papers: List[Dict[str, Any]] = []

    for paper_id in paper_ids:
        paper = results.get(str(paper_id), {})
        if not paper:
            continue

        authors = get_authors(paper)
        abstract = extract_abstract(paper)

        papers.append(
            {
                "pmid": str(paper_id),
                "title": paper.get("title", "No title available"),
                "authors": authors,
                "journal": paper.get("fulljournalname", "Unknown journal"),
                "publication_date": paper.get("pubdate", "Unknown"),
                "abstract": abstract,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
            }
        )

    return papers


def get_authors(paper: Dict[str, Any]) -> List[str]:
    """Extract author names while tolerating schema variations."""
    authors = paper.get("authors", [])
    names: List[str] = []
    for author in authors:
        if isinstance(author, dict):
            name = author.get("name")
            if name:
                names.append(name)
        elif isinstance(author, str):
            names.append(author)
    return names


def extract_abstract(paper: Dict[str, Any]) -> str:
    """Return the abstract text if available, otherwise an empty string."""
    abstract = paper.get("abstract")
    if isinstance(abstract, str):
        return abstract.strip()
    if isinstance(abstract, list):
        return " ".join(str(part) for part in abstract if part).strip()
    return ""
