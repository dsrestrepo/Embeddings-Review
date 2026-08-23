"""Functions for preprocessing and merging bibliographic database exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

PUBMED_FIELD_RE = re.compile(r"^([A-Z][A-Z0-9]{1,3})\s*-\s?(.*)$")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def normalize_doi(value: Any) -> str | None:
    """Return a canonical DOI or ``None`` when no DOI can be identified.

    URL and ``doi:`` prefixes are removed, matching is case-insensitive, and
    common trailing punctuation from bibliographic exports is stripped.
    """

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.I)
    match = DOI_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;:)]}").lower()


def parse_pubmed_tagged_text(text: str) -> list[dict[str, str]]:
    """Parse a PubMed MEDLINE/tagged-text export into record dictionaries.

    Repeated fields are joined with a single space. Continuation lines are
    appended to the most recent field, which is important for multi-line
    abstracts.
    """

    records: list[dict[str, str]] = []
    current: dict[str, list[str]] = {}
    current_field: str | None = None

    def finish_record() -> None:
        nonlocal current, current_field
        if current:
            records.append({key: " ".join(parts).strip() for key, parts in current.items()})
        current = {}
        current_field = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = PUBMED_FIELD_RE.match(line)
        if match:
            field, value = match.groups()
            if field == "PMID" and current:
                finish_record()
            current.setdefault(field, []).append(value.strip())
            current_field = field
        elif line.strip() and current_field:
            current[current_field].append(line.strip())
        elif not line.strip() and current:
            finish_record()
    finish_record()
    return records


def pubmed_abstract_lookup(records: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    """Build DOI- and PMID-indexed abstract lookups from parsed records."""

    by_doi: dict[str, str] = {}
    by_pmid: dict[str, str] = {}
    for record in records:
        abstract = record.get("AB", "").strip()
        if not abstract:
            continue
        pmid = record.get("PMID", "").strip()
        doi = normalize_doi(record.get("LID")) or normalize_doi(record.get("AID"))
        if pmid:
            by_pmid[pmid] = abstract
        if doi:
            by_doi[doi] = abstract
    return by_doi, by_pmid


def attach_abstracts_to_dataframe(
    pubmed: pd.DataFrame,
    tagged_text: str,
    *,
    doi_column: str = "DOI",
    pmid_column: str = "PMID",
    abstract_column: str = "Abstract",
) -> pd.DataFrame:
    """Attach abstracts to a PubMed table, matching DOI first and PMID second."""

    missing = [column for column in (doi_column, pmid_column) if column not in pubmed.columns]
    if missing:
        raise ValueError(f"PubMed table is missing required columns: {missing}")
    by_doi, by_pmid = pubmed_abstract_lookup(parse_pubmed_tagged_text(tagged_text))
    output = pubmed.copy()

    def lookup(row: pd.Series) -> str | None:
        doi = normalize_doi(row.get(doi_column))
        pmid_value = row.get(pmid_column)
        pmid = None if pd.isna(pmid_value) else str(pmid_value).strip().removesuffix(".0")
        return (by_doi.get(doi) if doi else None) or (by_pmid.get(pmid) if pmid else None)

    found = output.apply(lookup, axis=1)
    if abstract_column in output.columns:
        existing = output[abstract_column].where(output[abstract_column].notna(), "").astype(str).str.strip()
        output[abstract_column] = existing.where(existing.ne(""), found)
    else:
        output[abstract_column] = found
    return output


def attach_pubmed_abstracts(
    pubmed_csv: str | Path,
    pubmed_text: str | Path,
    output_csv: str | Path,
    *,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read PubMed exports, attach abstracts, and write the updated CSV."""

    table = pd.read_csv(pubmed_csv)
    text = Path(pubmed_text).read_text(encoding=encoding)
    output = attach_abstracts_to_dataframe(table, text)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def merge_bibliographic_dataframes(
    scopus_wos: pd.DataFrame,
    pubmed: pd.DataFrame,
    *,
    doi_column: str = "DOI",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Merge bibliographic tables with DOI-based de-duplication.

    Scopus/Web of Science records take precedence when a DOI appears in both
    inputs. Records with missing DOIs are retained because their identity cannot
    be resolved safely from DOI alone.
    """

    for label, frame in (("Scopus/Web of Science", scopus_wos), ("PubMed", pubmed)):
        if doi_column not in frame.columns:
            raise ValueError(f"{label} table is missing required column {doi_column!r}")

    primary = scopus_wos.copy()
    secondary = pubmed.copy()
    primary_dois = primary[doi_column].map(normalize_doi)
    secondary_dois = secondary[doi_column].map(normalize_doi)
    duplicate_mask = secondary_dois.notna() & secondary_dois.isin(set(primary_dois.dropna()))
    unique_pubmed = secondary.loc[~duplicate_mask].copy()

    for column in primary.columns:
        if column not in unique_pubmed.columns:
            unique_pubmed[column] = pd.NA
    unique_pubmed = unique_pubmed.reindex(columns=primary.columns)
    merged = pd.concat([primary, unique_pubmed], ignore_index=True)
    stats = {
        "scopus_wos_records": len(primary),
        "pubmed_records": len(secondary),
        "pubmed_duplicates_removed": int(duplicate_mask.sum()),
        "pubmed_unique_added": len(unique_pubmed),
        "merged_records": len(merged),
    }
    return merged, stats


def merge_bibliographic_databases(
    scopus_wos_csv: str | Path,
    pubmed_csv: str | Path,
    output_csv: str | Path,
) -> dict[str, int]:
    """Read, merge, and save Scopus/Web of Science and PubMed CSV files."""

    scopus_wos = pd.read_csv(scopus_wos_csv)
    pubmed = pd.read_csv(pubmed_csv)
    merged, stats = merge_bibliographic_dataframes(scopus_wos, pubmed)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return stats

