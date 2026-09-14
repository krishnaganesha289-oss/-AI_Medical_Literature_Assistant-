import requests


PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)

PUBMED_SUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
)


def search_pubmed(query, max_results=10):
    """Search PubMed and return basic paper information."""

    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
    }

    response = requests.get(
        PUBMED_SEARCH_URL,
        params=search_params,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    paper_ids = data.get(
        "esearchresult",
        {}
    ).get(
        "idlist",
        []
    )

    if not paper_ids:
        return []

    return get_paper_details(paper_ids)


def get_paper_details(paper_ids):
    """Get details for PubMed paper IDs."""

    summary_params = {
        "db": "pubmed",
        "id": ",".join(paper_ids),
        "retmode": "json",
    }

    response = requests.get(
        PUBMED_SUMMARY_URL,
        params=summary_params,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    papers = []

    for paper_id in paper_ids:

        paper = data.get(
            "result",
            {}
        ).get(
            paper_id,
            {}
        )

        if not paper:
            continue

        papers.append({
            "pmid": paper_id,
            "title": paper.get(
                "title",
                "No title available"
            ),
            "authors": get_authors(paper),
            "journal": paper.get(
                "fulljournalname",
                "Unknown journal"
            ),
            "publication_date": paper.get(
                "pubdate",
                "Unknown"
            ),
            "pubmed_url": (
                f"https://pubmed.ncbi.nlm.nih.gov/"
                f"{paper_id}/"
            ),
        })

    return papers


def get_authors(paper):
    """Extract author names."""

    authors = paper.get(
        "authors",
        []
    )

    names = []

    for author in authors:

        name = author.get("name")

        if name:
            names.append(name)

    return names