from pathlib import Path

import pandas as pd

from embeddings_review.analysis import analyze_detailed_taxonomy, analyze_general_annotations


def example_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2022, 2023],
            "data_type": ["EHR; clinical notes", "chest x-ray"],
            "is_multimodal": ["yes", "no"],
            "downstream_task": ["risk prediction", "diagnosis"],
            "medical_field": ["cardiology", "radiology"],
            "training_model": ["BERT", "ResNet50"],
            "method_used": ["self supervised learning", "contrastive learning"],
            "resulting_model": ["Model A", "Model B"],
            "use_case": ["classification", "retrieval"],
            "datasets": ["MIMIC IV", "CheXpert"],
            "metrics": ["AUROC", "F1"],
            "Data_Class_Main": ["multimodal", "imaging"],
            "Data_Class_imaging_Sub": ["", "radiology"],
            "Clinical_Class": ["cardiology", "radiology"],
            "Task_classification_regression": ["binary classification", "multiclass classification"],
            "Task_representation_learning": ["patient-level representation", "retrieval"],
            "Task_localization": ["none", "none"],
            "Task_generative": ["none", "none"],
        }
    )


def test_general_analysis_writes_expected_outputs(tmp_path: Path):
    summary = analyze_general_annotations(example_annotations(), tmp_path)
    assert summary["n_included"] == 2
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "table_field_by_modality.csv").is_file()
    assert (tmp_path / "plot_modalities_top20.png").is_file()


def test_detailed_analysis_writes_expected_outputs(tmp_path: Path):
    summary = analyze_detailed_taxonomy(example_annotations(), tmp_path)
    assert "data_class_main" in summary["counts"]
    assert (tmp_path / "taxonomy_summary.json").is_file()
    assert (tmp_path / "plot_tasks_by_data_class.png").is_file()
