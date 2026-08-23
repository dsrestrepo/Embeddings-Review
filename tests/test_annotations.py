import pandas as pd

from npj_embeddings_review.annotations import (
    clean_general_annotations,
    normalize_detailed_annotations,
    parse_list_cell,
)


def test_parse_list_cell_supports_literal_and_semicolon_formats():
    assert parse_list_cell("['EHR', 'notes']") == ["EHR", "notes"]
    assert parse_list_cell("EHR; notes") == ["EHR", "notes"]
    assert parse_list_cell(None) == []


def test_clean_general_annotations_normalizes_common_values():
    data = pd.DataFrame(
        {
            "data_type": ["EHR; chest x-ray"],
            "is_multimodal": ["yes"],
            "training_model": ["ResNet50"],
            "method_used": ["self supervised learning; PCA"],
            "metrics": ["AUROC; F1 score"],
            "datasets": ["MIMIC IV; New Cohort"],
        }
    )
    output, report = clean_general_annotations(data)
    assert output.at[0, "data_type"] == ["EHR", "CXR"]
    assert bool(output.at[0, "is_multimodal"]) is True
    assert output.at[0, "training_model"] == ["ResNet"]
    assert output.at[0, "method_used"] == ["self-supervised learning"]
    assert output.at[0, "metrics"] == ["AUC", "F1"]
    assert "new cohort" in report["datasets_unmapped"]


def test_detailed_annotation_rules_clear_inapplicable_fields():
    data = pd.DataFrame(
        {
            "Data_Class_Main": ["text", "imaging"],
            "Data_Class_imaging_Sub": ["radiology", "opthalmic"],
            "Clinical_Class": ["cardiology", "others"],
            "Clinical_Class_sp": ["should clear", "dentistry"],
            "Task_classification_regression": ["none", "binary classification"],
            "Task_representation_learning": ["retrieval", ""],
            "Task_localization": ["", "segmentation"],
            "Task_generative": ["", "none"],
            "Task_others": ["should clear", ""],
        }
    )
    output, _ = normalize_detailed_annotations(data)
    assert output.at[0, "Data_Class_imaging_Sub"] == []
    assert output.at[0, "Clinical_Class_sp"] == ""
    assert output.at[1, "Data_Class_imaging_Sub"] == ["ophthalmic"]
    assert output.at[1, "Task_representation_learning"] == ["none"]
