"""Frequency tables, summary statistics, and plots for reviewer annotations."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .annotations import clean_general_annotations, normalize_detailed_annotations, normalize_tokens, parse_list_cell


def configure_plotting() -> None:
    """Apply a consistent, colorblind-friendly publication style."""

    matplotlib.use("Agg", force=True)
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def explode_list_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Return non-empty individual tokens from a list-valued column."""

    if column not in data.columns:
        return pd.Series(dtype="object", name=column)
    values = data[column].apply(parse_list_cell).apply(normalize_tokens).explode().dropna()
    values = values[values.astype(str).str.strip().ne("")]
    values.name = column
    return values


def top_counts(series: pd.Series, top_n: int | None = None, min_count: int = 1) -> pd.Series:
    """Count values and optionally limit the result by count or rank."""

    counts = series.value_counts(dropna=True)
    counts = counts[counts >= min_count]
    return counts.head(top_n) if top_n is not None else counts


def _wrap_labels(labels: list[Any], width: int) -> list[str]:
    return ["\n".join(textwrap.wrap(str(label), width=width, break_long_words=False)) for label in labels]


def save_bar_plot(
    counts: pd.Series,
    title: str,
    destination: str | Path,
    *,
    xlabel: str = "Count",
    horizontal: bool = True,
    annotate: bool = True,
    wrap_width: int = 24,
    figsize: tuple[float, float] = (8.5, 5.2),
    palette: str = "colorblind",
) -> Path | None:
    """Save an annotated bar plot from a count series."""

    counts = counts.dropna().sort_values(ascending=False)
    if counts.empty:
        return None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = _wrap_labels(counts.index.tolist(), wrap_width)
    values = counts.to_numpy(dtype=float)
    plot_data = pd.DataFrame({"label": labels, "value": values})
    colors = sns.color_palette(palette, n_colors=len(plot_data))
    figure, axis = plt.subplots(figsize=figsize)

    if horizontal:
        sns.barplot(data=plot_data, y="label", x="value", hue="label", palette=colors, legend=False, ax=axis)
        axis.invert_yaxis()
        axis.set(xlabel=xlabel, ylabel="")
        axis.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        if annotate:
            pad = max(values.max() * 0.008, 0.02)
            for bar in axis.patches:
                value = bar.get_width()
                label = str(int(value)) if float(value).is_integer() else f"{value:.2f}"
                axis.text(value + pad, bar.get_y() + bar.get_height() / 2, label, va="center")
    else:
        sns.barplot(data=plot_data, x="label", y="value", hue="label", palette=colors, legend=False, ax=axis)
        axis.set(ylabel=xlabel, xlabel="")
        axis.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        if annotate:
            pad = max(values.max() * 0.01, 0.02)
            for bar in axis.patches:
                value = bar.get_height()
                label = str(int(value)) if float(value).is_integer() else f"{value:.2f}"
                axis.text(bar.get_x() + bar.get_width() / 2, value + pad, label, ha="center")

    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_donut_plot(
    counts: pd.Series,
    title: str,
    destination: str | Path,
    *,
    min_label_percent: float = 1.0,
) -> Path | None:
    """Save a donut chart with outside percentage labels."""

    counts = counts.dropna().sort_values(ascending=False)
    if counts.empty or counts.sum() == 0:
        return None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = counts.to_numpy(dtype=float)
    percentages = values / values.sum() * 100
    colors = sns.color_palette("colorblind", n_colors=len(values))
    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    wedges, _ = axis.pie(values, startangle=90, colors=colors, wedgeprops={"width": 0.42, "edgecolor": "white"})
    for wedge, percentage, label in zip(wedges, percentages, counts.index):
        if percentage < min_label_percent:
            continue
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(angle), np.sin(angle)
        axis.annotate(
            f"{label}: {percentage:.1f}%",
            xy=(x, y),
            xytext=(1.25 * np.sign(x), 1.15 * y),
            ha="left" if x >= 0 else "right",
            va="center",
            arrowprops={"arrowstyle": "-", "connectionstyle": "angle3,angleA=0,angleB=90", "lw": 0.8},
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "lw": 0.6, "alpha": 0.9},
        )
    axis.set_title(title)
    axis.axis("equal")
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def field_by_modality_table(
    data: pd.DataFrame,
    field_counts: pd.Series,
    modality_counts: pd.Series,
    *,
    top_k_fields: int = 10,
    top_k_modalities: int = 10,
) -> pd.DataFrame:
    """Count papers for each top-field/top-modality combination."""

    fields = list(field_counts.head(top_k_fields).index)
    modalities = list(modality_counts.head(top_k_modalities).index)
    matrix = pd.DataFrame(0, index=fields, columns=modalities, dtype=int)
    for _, row in data.iterrows():
        row_fields = set(normalize_tokens(parse_list_cell(row.get("medical_field"))))
        row_modalities = set(normalize_tokens(parse_list_cell(row.get("data_type"))))
        for field in row_fields.intersection(fields):
            for modality in row_modalities.intersection(modalities):
                matrix.loc[field, modality] += 1
    return matrix


def analyze_general_annotations(data: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    """Normalize general annotations and create all general analysis outputs."""

    configure_plotting()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, normalization_report = clean_general_annotations(data)

    column_specs = {
        "data_type": ("modalities", "Top Data Modalities", "plot_modalities_top20.png", 20),
        "training_model": ("training_models", "Top Training Model Families", "plot_training_models_top20.png", 20),
        "method_used": ("methods", "Top Methods", "plot_methods_top30.png", 30),
        "downstream_task": ("downstream_tasks", "Top Downstream Tasks", "plot_downstream_tasks_top20.png", 20),
        "use_case": ("use_cases", "Top Use Cases", "plot_use_cases_top20.png", 20),
        "medical_field": ("medical_fields", "Top Medical Fields", "plot_medical_fields_top20.png", 20),
        "datasets": ("datasets", "Top Datasets", "plot_datasets_top20.png", 20),
        "metrics": ("metrics", "Top Evaluation Metrics", "plot_metrics_top20.png", 20),
    }
    all_counts: dict[str, pd.Series] = {}
    for source_column, (name, title, plot_name, plot_limit) in column_specs.items():
        counts = top_counts(explode_list_series(data, source_column))
        all_counts[name] = counts
        counts.to_csv(output_dir / f"table_{name}.csv", header=["count"])
        save_bar_plot(counts.head(plot_limit), title, output_dir / plot_name, wrap_width=60)

    multimodal = data["is_multimodal"].dropna().astype(bool)
    multimodal_counts = multimodal.value_counts().rename(index={True: "Yes", False: "No"})
    multimodal_counts.to_csv(output_dir / "table_multimodal.csv", header=["count"])
    save_donut_plot(multimodal_counts, "Is Multimodal", output_dir / "pie_multimodal.png")

    cross_tab = field_by_modality_table(data, all_counts["medical_fields"], all_counts["modalities"])
    cross_tab.to_csv(output_dir / "table_field_by_modality.csv")

    if "Year" in data.columns:
        years = pd.to_numeric(data["Year"], errors="coerce").dropna().astype(int)
        year_counts = years.value_counts().sort_index()
        year_counts.to_csv(output_dir / "table_papers_by_year.csv", header=["count"])
        save_bar_plot(year_counts, "Number of Papers Published by Year", output_dir / "plot_papers_by_year.png", horizontal=False)

    total = len(data)
    n_multimodal = int(multimodal.sum())
    summary = {
        "n_included": total,
        "n_multimodal": n_multimodal,
        "pct_multimodal": round(100 * n_multimodal / total, 2) if total else 0.0,
        **{f"top_{name}": counts.head(5).to_dict() for name, counts in all_counts.items()},
        "normalization_report": normalization_report,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _countplot_table_and_figure(
    data: pd.DataFrame,
    column: str,
    output_dir: Path,
    *,
    title: str,
    stem: str,
    palette: str,
) -> pd.Series:
    values = explode_list_series(data, column)
    counts = top_counts(values)
    counts.to_csv(output_dir / f"table_{stem}.csv", header=["count"])
    save_bar_plot(counts, title, output_dir / f"plot_{stem}.png", palette=palette, wrap_width=44)
    return counts


def analyze_detailed_taxonomy(data: pd.DataFrame, output_dir: str | Path) -> dict[str, dict[str, int]]:
    """Validate and plot the detailed reviewer taxonomy."""

    configure_plotting()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, validation_report = normalize_detailed_annotations(data)

    specifications = [
        ("Data_Class_Main", "Distribution of Data Class (Main)", "data_class_main", "viridis"),
        ("Data_Class_imaging_Sub", "Imaging Subtypes", "imaging_subtypes", "magma"),
        ("Clinical_Class", "Clinical Specialty Distribution", "clinical_class", "cubehelix"),
        ("Task_classification_regression", "Tasks: Classification / Regression", "classification_regression_tasks", "viridis"),
        ("Task_representation_learning", "Representation Learning Tasks", "representation_learning_tasks", "plasma"),
        ("Task_localization", "Localization Tasks", "localization_tasks", "crest"),
        ("Task_generative", "Generative Tasks", "generative_tasks", "inferno"),
    ]
    output_counts: dict[str, dict[str, int]] = {}
    for column, title, stem, palette in specifications:
        source = data[data["Data_Class_Main"] == "imaging"] if column == "Data_Class_imaging_Sub" else data
        counts = _countplot_table_and_figure(source, column, output_dir, title=title, stem=stem, palette=palette)
        output_counts[stem] = {str(key): int(value) for key, value in counts.items()}

    task_rows = data[["Data_Class_Main", "Task_classification_regression"]].copy()
    task_rows["Task_classification_regression"] = task_rows["Task_classification_regression"].apply(parse_list_cell)
    task_rows = task_rows.explode("Task_classification_regression").dropna()
    task_rows = task_rows[task_rows["Task_classification_regression"].ne("")]
    cross_tab = pd.crosstab(task_rows["Data_Class_Main"], task_rows["Task_classification_regression"])
    cross_tab.to_csv(output_dir / "table_tasks_by_data_class.csv")
    if not task_rows.empty:
        figure, axis = plt.subplots(figsize=(12, 6))
        sns.countplot(
            data=task_rows,
            x="Data_Class_Main",
            hue="Task_classification_regression",
            palette="tab20",
            ax=axis,
        )
        axis.set(title="Task Distribution by Data Class", ylabel="Count", xlabel="Data Class")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(output_dir / "plot_tasks_by_data_class.png", bbox_inches="tight")
        plt.close(figure)

    result = {"counts": output_counts, "validation_report": validation_report}
    (output_dir / "taxonomy_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
