"""Define reviewer annotation columns and controlled vocabularies."""

GENERAL_COLUMNS = [
    "data_type",
    "is_multimodal",
    "downstream_task",
    "medical_field",
    "training_model",
    "method_used",
    "resulting_model",
    "use_case",
    "datasets",
    "metrics",
]

DETAILED_COLUMNS = [
    "Data_Class_Main",
    "Data_Class_imaging_Sub",
    "Clinical_Class",
    "Clinical_Class_sp",
    "Task_classification_regression",
    "Task_representation_learning",
    "Task_localization",
    "Task_generative",
    "Task_others",
    "Summary",
    "Details",
]

DATA_CLASS_MAIN = [
    "imaging", "waveform", "text", "tabular", "multimodal",
    "genomics/omics", "video", "other/unknown",
]

DATA_CLASS_IMAGING_SUB = [
    "radiology", "pathology", "dermatology", "ophthalmic", "endoscopy",
    "surgical/operative", "multimodal imaging", "microscopy", "other",
]

CLINICAL_CLASS = [
    "cardiology", "pulmonology", "infectious disease", "neurology",
    "psychiatry", "endocrinology", "nephrology",
    "hepatology/gastroenterology", "hematology/oncology", "rheumatology",
    "dermatology", "ophthalmology", "obstetrics/gynecology", "pediatrics",
    "geriatrics", "emergency medicine", "critical care", "anesthesiology",
    "surgery", "primary care", "rehabilitation", "radiology",
    "general medicine/cross-domain", "others",
]

TASK_CLASSIFICATION_REGRESSION = [
    "binary classification", "multiclass classification",
    "multilabel classification", "classification (detail unspecified)",
    "regression (detail unspecified)", "ordinal regression/ranking",
    "survival analysis", "time-series forecasting", "causal inference",
    "trajectory modeling", "others", "none",
]

TASK_REPRESENTATION_LEARNING = [
    "clustering", "retrieval", "cross-modal alignment", "metric learning",
    "patient-level representation", "others", "none",
]

TASK_LOCALIZATION = [
    "object detection", "segmentation", "landmark detection", "others", "none",
]

TASK_GENERATIVE = [
    "data synthesis", "simulation", "imputation/reconstruction", "denoising",
    "report generation", "captioning", "image generation",
    "multimodal generation", "others", "none",
]

DETAILED_ALLOWED_VALUES = {
    "Data_Class_Main": DATA_CLASS_MAIN,
    "Data_Class_imaging_Sub": DATA_CLASS_IMAGING_SUB,
    "Clinical_Class": CLINICAL_CLASS,
    "Task_classification_regression": TASK_CLASSIFICATION_REGRESSION,
    "Task_representation_learning": TASK_REPRESENTATION_LEARNING,
    "Task_localization": TASK_LOCALIZATION,
    "Task_generative": TASK_GENERATIVE,
}

DETAILED_LIST_COLUMNS = [
    "Data_Class_imaging_Sub",
    "Task_classification_regression",
    "Task_representation_learning",
    "Task_localization",
    "Task_generative",
]
