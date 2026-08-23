import pandas as pd

from npj_embeddings_review.bibliography import (
    attach_abstracts_to_dataframe,
    merge_bibliographic_dataframes,
    normalize_doi,
    parse_pubmed_tagged_text,
)


def test_normalize_doi_removes_prefix_and_punctuation():
    assert normalize_doi("https://doi.org/10.1000/ABC.123.") == "10.1000/abc.123"
    assert normalize_doi("") is None


def test_parse_and_attach_pubmed_abstract_by_doi_or_pmid():
    tagged = """PMID- 123
LID - 10.1000/one [doi]
AB  - First line
      second line.

PMID- 456
AB  - PMID-only abstract.
"""
    records = parse_pubmed_tagged_text(tagged)
    assert records[0]["AB"] == "First line second line."
    table = pd.DataFrame(
        {"DOI": ["doi:10.1000/ONE", None], "PMID": [999, 456], "Abstract": [None, None]}
    )
    output = attach_abstracts_to_dataframe(table, tagged)
    assert output["Abstract"].tolist() == ["First line second line.", "PMID-only abstract."]


def test_merge_prefers_primary_doi_and_retains_missing_doi():
    primary = pd.DataFrame({"DOI": ["10.1000/a", None], "Title": ["A", "No DOI primary"]})
    pubmed = pd.DataFrame(
        {"DOI": ["https://doi.org/10.1000/A", "10.1000/b", None], "Title": ["duplicate", "B", "No DOI PubMed"]}
    )
    merged, stats = merge_bibliographic_dataframes(primary, pubmed)
    assert merged["Title"].tolist() == ["A", "No DOI primary", "B", "No DOI PubMed"]
    assert stats["pubmed_duplicates_removed"] == 1
