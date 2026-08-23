# Embeddings Review: reproducible preprocessing and analysis

This repository provides a script-based version of the preprocessing and plotting workflow used for the review. It covers:

1. attaching abstracts from a PubMed text export to a PubMed CSV export;
2. merging PubMed records with a preprocessed Scopus/Web of Science table while removing DOI duplicates;
3. validating and normalizing features entered manually by reviewers; and
4. generating the frequency tables, summary statistics, cross-tabulation, and publication-ready plots used in the analysis. 


## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Expected inputs

### Bibliographic preprocessing

- A PubMed CSV containing at least `DOI` and `PMID`.
- A PubMed plain-text export containing tagged records such as `PMID-`, `LID -`, and `AB  -`.
- A preprocessed Scopus/Web of Science CSV containing a `DOI` column.

The merged output follows the Scopus/Web of Science column order. PubMed-only records receive empty values for fields unavailable in the PubMed export.

### Manual screening

Reviewers apply these eligibility criteria to titles and abstracts:

- publication year from 2021 through 2025 and English-language publication;
- embeddings or vector representations applied to clinical or patient-level data;
- clinical data such as EHRs, notes, laboratories/codes, medical imaging, waveforms, or multimodal combinations;
- an implemented or evaluated clinical downstream task, such as diagnosis, prognosis, risk prediction, survival analysis, retrieval, decision support, or patient stratification; and
- exclusion of non-clinical applications, reviews or conceptual papers without an implementation/evaluation, and records too vague to verify eligibility.

The included-record CSV is the input to annotation and plotting. Keep `Title`, `Abstract`, `DOI`, `Year`, and any source identifiers needed for auditability.

### Manual reviewer features

The general analysis expects these columns. List-valued cells may be entered as literal lists (for example `['EHR', 'clinical notes']`) or as semicolon-separated values.

| Column | Reviewer-entered content |
|---|---|
| `data_type` | Clinical data modalities, such as EHR, notes, imaging, labs, text, or waveforms |
| `is_multimodal` | Whether more than one data modality is combined |
| `downstream_task` | Clinical downstream tasks |
| `medical_field` | Clinical specialty or field |
| `training_model` | Model name or model family |
| `method_used` | Representation-learning or training method |
| `resulting_model` | Named model produced by the study, when applicable |
| `use_case` | Specific application, such as classification, regression, or retrieval |
| `datasets` | Named public datasets or a private/local data source |
| `metrics` | Named evaluation metrics |

An optional, more detailed taxonomy is also supported:

- `Data_Class_Main` and `Data_Class_imaging_Sub`
- `Clinical_Class` and `Clinical_Class_sp`
- `Task_classification_regression`
- `Task_representation_learning`
- `Task_localization`
- `Task_generative`
- `Task_others`, `Summary`, and `Details`

Allowed values for the detailed taxonomy are defined in `src/embeddings_review/schema.py`. Reviewers can use the generated annotation template to enter them consistently.

See [`docs/annotation_codebook.md`](docs/annotation_codebook.md) for definitions, allowed values, and conditional entry rules.

## Command-line workflow

### 1. Add PubMed abstracts

```bash
review attach-pubmed-abstracts \
  --pubmed-csv data/pubmed2.csv \
  --pubmed-text data/pubmed1.txt \
  --output data/pubmed.csv
```

### 2. Merge databases and remove DOI duplicates

```bash
review merge-databases \
  --scopus-wos data/Scopus_WOS_Preprocessed.csv \
  --pubmed data/pubmed.csv \
  --output data/Merged_Scopus_WoS_Pubmed_NoDuplicates.csv
```

Records without a DOI are retained. DOI duplicates are resolved in favor of the Scopus/Web of Science table, matching the original workflow.

### 3. Create a reviewer annotation template

```bash
review make-template \
  --input data/included_papers.csv \
  --output data/reviewer_annotations.csv \
  --schema both
```

Use `--schema general` or `--schema detailed` to request only one annotation set.

### 4. Validate and normalize completed annotations

```bash
review normalize-annotations \
  --input data/reviewer_annotations_completed.csv \
  --output data/reviewer_annotations_normalized.csv \
  --report results/normalization_report.json
```

Unknown values are preserved where appropriate and recorded in the report so reviewers can resolve vocabulary differences transparently.

### 5. Generate general analysis outputs

```bash
review analyze \
  --input data/reviewer_annotations_normalized.csv \
  --output-dir results/general
```

This creates count tables and plots for modalities, multimodality, model families, methods, downstream tasks, use cases, medical fields, datasets, metrics, papers by year, a field-by-modality table, and `summary.json`.

### 6. Generate detailed taxonomy plots

```bash
review plot-taxonomy \
  --input data/reviewer_annotations_completed.csv \
  --output-dir results/taxonomy
```

The command creates plots and count tables for the main data class, imaging subtype, clinical class, four task families, and classification/regression tasks by data class.

All commands accept `--help`. The Python functions can also be imported directly for use in other pipelines.
