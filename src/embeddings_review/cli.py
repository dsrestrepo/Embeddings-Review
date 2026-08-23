"""Command-line interface for review preprocessing and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .analysis import analyze_detailed_taxonomy, analyze_general_annotations
from .annotations import (
    clean_general_annotations,
    create_annotation_template,
    normalize_detailed_annotations,
    save_normalization_report,
    serialize_list_columns,
)
from .bibliography import attach_pubmed_abstracts, merge_bibliographic_databases
from .schema import DETAILED_LIST_COLUMNS, GENERAL_COLUMNS


def _parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_annotation_csv(path: str) -> pd.DataFrame:
    """Read a comma- or semicolon-delimited reviewer annotation file."""

    return pd.read_csv(path, sep=None, engine="python")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="review",
        description="Preprocess bibliographic exports and analyze manual reviewer annotations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    attach = commands.add_parser("attach-pubmed-abstracts", help="Attach abstracts from a PubMed text export.")
    attach.add_argument("--pubmed-csv", required=True)
    attach.add_argument("--pubmed-text", required=True)
    attach.add_argument("--output", required=True)

    merge = commands.add_parser("merge-databases", help="Merge PubMed with a Scopus/Web of Science CSV.")
    merge.add_argument("--scopus-wos", required=True)
    merge.add_argument("--pubmed", required=True)
    merge.add_argument("--output", required=True)

    template = commands.add_parser("make-template", help="Add empty manual annotation columns.")
    template.add_argument("--input", required=True)
    template.add_argument("--output", required=True)
    template.add_argument("--schema", choices=["general", "detailed", "both"], default="both")

    normalize = commands.add_parser("normalize-annotations", help="Validate and normalize reviewer features.")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--report", required=True)

    analyze = commands.add_parser("analyze", help="Create general frequency tables, plots, and summary statistics.")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output-dir", required=True)

    taxonomy = commands.add_parser("plot-taxonomy", help="Create detailed taxonomy tables and plots.")
    taxonomy.add_argument("--input", required=True)
    taxonomy.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected workflow command."""

    args = build_parser().parse_args(argv)
    if args.command == "attach-pubmed-abstracts":
        output = attach_pubmed_abstracts(args.pubmed_csv, args.pubmed_text, args.output)
        print(json.dumps({"records": len(output), "output": args.output}, indent=2))
    elif args.command == "merge-databases":
        stats = merge_bibliographic_databases(args.scopus_wos, args.pubmed, args.output)
        print(json.dumps({**stats, "output": args.output}, indent=2))
    elif args.command == "make-template":
        data = _read_annotation_csv(args.input)
        output = create_annotation_template(data, args.schema)
        _parent(args.output)
        output.to_csv(args.output, index=False)
        print(json.dumps({"records": len(output), "output": args.output}, indent=2))
    elif args.command == "normalize-annotations":
        data = _read_annotation_csv(args.input)
        general, general_report = clean_general_annotations(data)
        detailed, detailed_report = normalize_detailed_annotations(general)
        serialized = serialize_list_columns(detailed, [*GENERAL_COLUMNS, *DETAILED_LIST_COLUMNS])
        _parent(args.output)
        serialized.to_csv(args.output, index=False)
        report = {"general": general_report, "detailed": detailed_report}
        save_normalization_report(report, args.report)
        print(json.dumps({"records": len(serialized), "output": args.output, "report": args.report}, indent=2))
    elif args.command == "analyze":
        summary = analyze_general_annotations(_read_annotation_csv(args.input), args.output_dir)
        print(json.dumps(summary, indent=2))
    elif args.command == "plot-taxonomy":
        summary = analyze_detailed_taxonomy(_read_annotation_csv(args.input), args.output_dir)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
