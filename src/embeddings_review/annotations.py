"""Validate and normalize manual reviewer annotations."""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .schema import (
    CLINICAL_CLASS,
    DATA_CLASS_IMAGING_SUB,
    DATA_CLASS_MAIN,
    DETAILED_ALLOWED_VALUES,
    DETAILED_COLUMNS,
    DETAILED_LIST_COLUMNS,
    GENERAL_COLUMNS,
)

DATA_TYPE_MAP = {
    "ehr": "EHR",
    "ehrs": "EHR",
    "emr": "EHR",
    "structured ehr": "EHR",
    "longitudinal ehr": "EHR",
    "tabular ehr": "EHR",
    "patient visit": "EHR",
    "admission record": "EHR",
    "cxr": "CXR",
    "chest xray": "CXR",
    "chest x ray": "CXR",
    "chest radiograph": "CXR",
    "wsi": "WSI",
    "whole slide image": "WSI",
    "oct": "OCT",
    "fundus": "fundus",
    "ct": "CT",
    "mri": "MRI",
    "dce mri": "MRI",
    "pet": "PET",
    "fdg pet": "PET",
    "ultrasound": "ultrasound",
    "clinical note": "clinical notes",
    "free text": "clinical text",
    "text report": "clinical text",
    "radiology report": "radiology reports",
    "ecg": "ECG",
    "time series": "time-series",
    "multivariate time series": "time-series",
    "lab test": "labs",
    "laboratory test": "labs",
    "knowledge graph": "knowledge graph",
}

MODEL_MAP = {
    "bert": "BERT",
    "clinicalbert": "ClinicalBERT",
    "llama": "Llama",
    "vit": "ViT",
    "resnet": "ResNet",
    "resnet50": "ResNet",
    "moco": "MoCo",
    "simclr": "SimCLR",
    "byol": "BYOL",
    "dino": "DINO",
    "dinov2": "DINOv2",
    "cnn": "CNN",
    "transformer": "Transformer",
    "graph neural network": "GNN",
    "gcn": "GCN",
    "gat": "GAT",
    "vae": "VAE",
    "autoencoder": "Autoencoder",
    "masked auto encoder": "MAE",
    "mae": "MAE",
    "clip": "CLIP",
    "vision language": "Vision-Language Model",
    "vision language model": "Vision-Language Model",
    "vision language foundation model": "Vision-Language Model",
    "multimodal foundation model": "Vision-Language Model",
    "contrastive vision language model": "Vision-Language Model",
    "multimodal encoder": "Vision-Language Model",
    "large language model": "Large Language Model",
    "pretrained language model": "Pretrained Language Model",
    "pre trained language model": "Pretrained Language Model",
    "language model": "Language Model",
    "foundation model": "Foundation Model",
    "temporal foundation model": "Foundation Model",
    "vision foundation model": "Foundation Model",
    "imagenet pretrained": "ImageNet Pretrained Model",
    "convolutional neural network": "Convolutional Neural Network",
    "densenet201": "Convolutional Neural Network",
    "chexnet": "Convolutional Neural Network",
    "inceptionv3": "Convolutional Neural Network",
    "unet": "UNet Variant",
    "swin unetr": "UNet Variant",
    "nnformer": "UNet Variant",
    "deep learning": "Neural Network",
    "deep neural network": "Neural Network",
    "neural network": "Neural Network",
    "mlp": "Neural Network",
    "gru": "Recurrent Neural Network",
    "lstm": "Recurrent Neural Network",
    "bilstm": "Recurrent Neural Network",
    "random forest": "Classical Machine Learning",
    "logistic regression": "Classical Machine Learning",
    "xgboost": "Classical Machine Learning",
    "word2vec": "Word Embeddings",
    "glove": "Word Embeddings",
    "skip gram": "Word Embeddings",
    "word embeddings": "Word Embeddings",
    "gan": "Generative Adversarial Network",
    "siamese neural network": "Siamese Network",
    "siamese network": "Siamese Network",
}

METHOD_MAP = {
    "self supervised learning": "self-supervised learning",
    "supervised contrastive learning": "supervised contrastive learning",
    "contrastive learning": "contrastive learning",
    "masked autoencoding": "masked autoencoding",
    "masked image modelling": "masked autoencoding",
    "pretraining": "pre-training",
    "pre training": "pre-training",
    "fine tuning": "fine-tuning",
    "multimodal pretraining": "multimodal pre-training",
    "federated learning": "federated learning",
    "knowledge distillation": "knowledge distillation",
    "prototype learning": "prototype learning",
    "graph contrastive learning": "graph contrastive learning",
    "multiple instance learning": "multiple instance learning",
    "domain adaptation": "domain adaptation",
    "metric learning": "metric learning",
    "zero shot learning": "zero-shot learning",
    "few shot learning": "few-shot learning",
    "simclr": "SimCLR",
    "moco": "MoCo",
    "byol": "BYOL",
    "dino": "DINO",
    "dinov2": "DINOv2",
    "mae": "MAE",
}

METHOD_BLACKLIST = {
    "data augmentation", "grad cam", "grad cam++", "influence function",
    "tracin", "t sne", "tsne", "pca", "umap", "ablation",
    "explainability", "interpretability", "feature extraction", "attention",
    "self attention", "cross attention",
}

METRIC_MAP = {
    "auc": "AUC", "auroc": "AUC", "roc auc": "AUC", "mauc": "AUC",
    "auprc": "AUPRC", "pr auc": "AUPRC", "ap": "AP", "f1": "F1",
    "f1 score": "F1", "kappa": "Kappa",
    "quadratic weighted kappa": "Kappa (QWK)", "accuracy": "accuracy",
    "precision": "precision", "recall": "recall", "sensitivity": "recall",
    "specificity": "specificity", "npv": "NPV", "mae": "MAE",
    "mse": "MSE", "rmse": "RMSE", "c index": "C-index",
    "dice": "Dice", "dsc": "Dice", "95hd": "HD95", "hd95": "HD95",
    "spearman rho": "Spearman rho", "retrieval rate": "retrieval rate",
    "p value": "P-value",
}

DATASET_MAP = {
    "mimic iii": "MIMIC-III", "mimic iv": "MIMIC-IV",
    "mimic iv v3.0": "MIMIC-IV", "mimic cxr": "MIMIC-CXR",
    "eicu": "eICU-CRD", "eicu crd": "eICU-CRD", "open i": "OpenI-IU",
    "open i iu": "OpenI-IU", "chexpert": "CheXpert", "nih 14": "NIH-14",
    "uk biobank": "UK Biobank", "tcga": "TCGA", "adni": "ADNI",
    "opkdiat": "OPHDIAT", "ophdiat": "OPHDIAT",
}


def parse_list_cell(value: Any) -> list[Any]:
    """Parse a CSV cell containing a list or semicolon-separated values."""

    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
        return [parsed]
    except (SyntaxError, ValueError):
        separator = ";" if ";" in text else ","
        return [part.strip() for part in text.split(separator) if part.strip()]


def normalize_tokens(tokens: Iterable[Any]) -> list[str]:
    """Trim, de-duplicate, and normalize whitespace in annotation tokens."""

    output: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token is None:
            continue
        value = " ".join(str(token).replace("\u00ad", "").split()).strip()
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def clean_token(token: Any) -> str:
    """Return a lowercase comparison token with stable punctuation/spacing."""

    text = unicodedata.normalize("NFKC", str(token)).strip().lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^\w\s./+]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("x ray", "xray")


def _map_items(
    value: Any,
    table: dict[str, str],
    *,
    preserve_unknown: bool = True,
) -> tuple[list[str], list[str]]:
    output: list[str] = []
    unknown: list[str] = []
    for original in normalize_tokens(parse_list_cell(value)):
        key = clean_token(original)
        mapped = table.get(key)
        if mapped is None:
            mapped = next((replacement for phrase, replacement in table.items() if len(phrase) > 2 and phrase in key), None)
        if mapped:
            output.append(mapped)
        elif preserve_unknown:
            output.append(original)
            unknown.append(original)
        else:
            unknown.append(original)
    return normalize_tokens(output), normalize_tokens(unknown)


def normalize_boolean(value: Any) -> bool | None:
    """Normalize a scalar or one-element list to a boolean value."""

    values = parse_list_cell(value)
    if not values:
        values = [value]
    for item in values:
        if isinstance(item, bool):
            return item
        text = str(item).strip().lower()
        if text in {"true", "yes", "1", "y"}:
            return True
        if text in {"false", "no", "0", "n"}:
            return False
    return None


def clean_general_annotations(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Normalize general reviewer-entered feature columns.

    Unrecognized model and dataset names are preserved and logged so novel
    study-specific names are not lost. Method labels outside the review's
    controlled method vocabulary are logged and omitted from method plots.
    """

    output = data.copy()
    report: defaultdict[str, list[str]] = defaultdict(list)
    for column in GENERAL_COLUMNS:
        if column not in output.columns:
            output[column] = pd.Series([pd.NA] * len(output), dtype="object")

    specifications = {
        "data_type": (DATA_TYPE_MAP, True),
        "training_model": (MODEL_MAP, True),
        "metrics": (METRIC_MAP, True),
        "datasets": (DATASET_MAP, True),
    }
    for column, (table, preserve_unknown) in specifications.items():
        cleaned: list[list[str]] = []
        for value in output[column]:
            values, unknown = _map_items(value, table, preserve_unknown=preserve_unknown)
            cleaned.append(values)
            report[f"{column}_unmapped"].extend(unknown)
        output[column] = cleaned

    cleaned_methods: list[list[str]] = []
    for value in output["method_used"]:
        items = [item for item in normalize_tokens(parse_list_cell(value)) if clean_token(item) not in METHOD_BLACKLIST]
        values, unknown = _map_items(items, METHOD_MAP, preserve_unknown=False)
        cleaned_methods.append(values)
        report["method_used_unmapped"].extend(unknown)
    output["method_used"] = cleaned_methods

    for column in ["downstream_task", "medical_field", "resulting_model", "use_case"]:
        output[column] = output[column].apply(lambda value: normalize_tokens(parse_list_cell(value)))
    output["is_multimodal"] = output["is_multimodal"].apply(normalize_boolean)

    cleaned_report = {
        key: normalize_tokens(clean_token(value) for value in values if value)
        for key, values in report.items()
        if values
    }
    return output, cleaned_report


def _normalize_choice(value: Any, allowed: list[str], default: str) -> str:
    text = clean_token(value)
    aliases = {clean_token(option): option for option in allowed}
    if text == "opthalmic":
        text = "ophthalmic"
    return aliases.get(text, default)


def _normalize_multiple(value: Any, allowed: list[str], *, default_none: bool) -> tuple[list[str], list[str]]:
    aliases = {clean_token(option): option for option in allowed}
    values: list[str] = []
    invalid: list[str] = []
    for item in parse_list_cell(value):
        token = clean_token(item)
        if token == "opthalmic":
            token = "ophthalmic"
        if token in aliases:
            values.append(aliases[token])
        elif token:
            invalid.append(str(item))
    values = normalize_tokens(values)
    if not values and default_none and "none" in allowed:
        values = ["none"]
    return values, normalize_tokens(invalid)


def normalize_detailed_annotations(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Validate the detailed reviewer taxonomy and apply conditional rules."""

    output = data.copy()
    report: defaultdict[str, list[str]] = defaultdict(list)
    for column in DETAILED_COLUMNS:
        if column not in output.columns:
            output[column] = ""
        output[column] = output[column].astype("object")

    output["Data_Class_Main"] = output["Data_Class_Main"].apply(
        lambda value: _normalize_choice(value, DATA_CLASS_MAIN, "other/unknown")
    )
    output["Clinical_Class"] = output["Clinical_Class"].apply(
        lambda value: _normalize_choice(value, CLINICAL_CLASS, "others")
    )

    for index in output.index:
        is_imaging = output.at[index, "Data_Class_Main"] == "imaging"
        for column in DETAILED_LIST_COLUMNS:
            allowed = DETAILED_ALLOWED_VALUES[column]
            if column == "Data_Class_imaging_Sub" and not is_imaging:
                output.at[index, column] = []
                continue
            values, invalid = _normalize_multiple(
                output.at[index, column],
                allowed,
                default_none=column != "Data_Class_imaging_Sub",
            )
            if column == "Data_Class_imaging_Sub" and not values:
                values = ["other"]
            output.at[index, column] = values
            report[f"{column}_invalid"].extend(invalid)

        clinical = output.at[index, "Clinical_Class"]
        if clinical not in {"others", "general medicine/cross-domain"}:
            output.at[index, "Clinical_Class_sp"] = ""
        else:
            output.at[index, "Clinical_Class_sp"] = str(output.at[index, "Clinical_Class_sp"] or "").strip()

        has_other = any(
            "others" in output.at[index, column]
            for column in DETAILED_LIST_COLUMNS
            if column != "Data_Class_imaging_Sub"
        )
        output.at[index, "Task_others"] = (
            str(output.at[index, "Task_others"] or "").strip() if has_other else ""
        )
        for column in ("Summary", "Details"):
            value = output.at[index, column]
            output.at[index, column] = "" if pd.isna(value) else str(value).strip()

    return output, {key: normalize_tokens(values) for key, values in report.items() if values}


def create_annotation_template(data: pd.DataFrame, schema: str = "both") -> pd.DataFrame:
    """Add empty reviewer feature columns to an included-record table."""

    if schema not in {"general", "detailed", "both"}:
        raise ValueError("schema must be 'general', 'detailed', or 'both'")
    output = data.copy()
    columns: list[str] = []
    if schema in {"general", "both"}:
        columns.extend(GENERAL_COLUMNS)
    if schema in {"detailed", "both"}:
        columns.extend(DETAILED_COLUMNS)
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    return output


def save_normalization_report(report: dict[str, list[str]], path: str | Path) -> None:
    """Write a normalization/validation report as UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def serialize_list_columns(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Serialize list columns as semicolon-separated strings for CSV output."""

    output = data.copy()
    for column in columns:
        if column in output.columns:
            output[column] = output[column].apply(
                lambda value: "; ".join(map(str, value)) if isinstance(value, list) else value
            )
    return output
