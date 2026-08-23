# Manual reviewer annotation codebook

Complete annotations only for records that passed the manual title/abstract screening described in the main README. Enter only information supported by the title, abstract, or full text used in the review. Leave a field empty when the information is unavailable; do not guess.

List-valued fields may be entered as semicolon-separated values, such as `EHR; clinical notes`, or as a list. Use short, consistent labels. The normalization report identifies unfamiliar variants without silently discarding study-specific model or dataset names.

## General analysis features

### `data_type`

One or more types of clinical data used in the study. Examples include EHR, clinical notes, clinical text, laboratories, structured codes, CXR, CT, MRI, WSI, OCT, fundus imaging, ultrasound, ECG, and time-series data.

### `is_multimodal`

Enter `yes` when the analyzed representation combines two or more distinct data modalities; otherwise enter `no`.

### `downstream_task`

One or more evaluated clinical tasks, such as diagnosis, prognosis, risk prediction, survival analysis, patient stratification, retrieval, segmentation, or clinical decision support.

### `medical_field`

The clinical specialty or medical field of the application. Use a broad field when a paper spans several specialties.

### `training_model`

The model name or model family used to learn or produce representations. Prefer the named architecture when provided; otherwise use a family such as BERT, ResNet, transformer, CNN, GNN, recurrent neural network, autoencoder, or vision-language model.

### `method_used`

The central learning method. Examples include self-supervised learning, contrastive learning, supervised contrastive learning, masked autoencoding, pre-training, fine-tuning, federated learning, knowledge distillation, multiple-instance learning, domain adaptation, metric learning, zero-shot learning, and few-shot learning.

Analysis-only techniques such as PCA, t-SNE, UMAP, attribution maps, and ablations are not counted as representation-learning methods.

### `resulting_model`

The study's named resulting model, if one is introduced. Leave empty when the paper does not name a resulting model.

### `use_case`

The concrete application of the representation, such as classification, regression, retrieval, clustering, or image analysis. This may be more specific than `downstream_task`.

### `datasets`

One or more named datasets or data sources. Use the common public name when available, such as MIMIC-III, MIMIC-IV, MIMIC-CXR, eICU-CRD, CheXpert, NIH-14, UK Biobank, TCGA, or ADNI. For non-public data, use a consistent label such as `private dataset` plus a site name when needed to distinguish cohorts.

### `metrics`

Only named evaluation metrics, such as AUC/AUROC, AUPRC, average precision, F1, accuracy, precision, recall/sensitivity, specificity, NPV, MAE, MSE, RMSE, C-index, Dice, or HD95.

## Detailed taxonomy

### Main data class

`Data_Class_Main` is single-choice:

- `imaging`
- `waveform`
- `text`
- `tabular`
- `multimodal`
- `genomics/omics`
- `video`
- `other/unknown`

`Data_Class_imaging_Sub` is list-valued and is completed only when the main class is `imaging`:

- `radiology`
- `pathology`
- `dermatology`
- `ophthalmic`
- `endoscopy`
- `surgical/operative`
- `multimodal imaging`
- `microscopy`
- `other`

### Clinical class

`Clinical_Class` is single-choice:

- `cardiology`, `pulmonology`, `infectious disease`, `neurology`, `psychiatry`
- `endocrinology`, `nephrology`, `hepatology/gastroenterology`
- `hematology/oncology`, `rheumatology`, `dermatology`, `ophthalmology`
- `obstetrics/gynecology`, `pediatrics`, `geriatrics`
- `emergency medicine`, `critical care`, `anesthesiology`, `surgery`
- `primary care`, `rehabilitation`, `radiology`
- `general medicine/cross-domain`, `others`

Complete `Clinical_Class_sp` only when `Clinical_Class` is `others` or `general medicine/cross-domain`.

### Classification and regression tasks

`Task_classification_regression` is list-valued:

- `binary classification`
- `multiclass classification`
- `multilabel classification`
- `classification (detail unspecified)`
- `regression (detail unspecified)`
- `ordinal regression/ranking`
- `survival analysis`
- `time-series forecasting`
- `causal inference`
- `trajectory modeling`
- `others`
- `none`

### Representation-learning tasks

`Task_representation_learning` is list-valued:

- `clustering`
- `retrieval`
- `cross-modal alignment`
- `metric learning`
- `patient-level representation`
- `others`
- `none`

### Localization tasks

`Task_localization` is list-valued:

- `object detection`
- `segmentation`
- `landmark detection`
- `others`
- `none`

### Generative tasks

`Task_generative` is list-valued:

- `data synthesis`
- `simulation`
- `imputation/reconstruction`
- `denoising`
- `report generation`
- `captioning`
- `image generation`
- `multimodal generation`
- `others`
- `none`

Complete `Task_others` only when `others` is selected in at least one task family.

### Narrative fields

- `Summary`: a one- or two-sentence plain-language study summary.
- `Details`: brief model, pre-training, or loss details explicitly reported by the paper.

The normalization command enforces the conditional fields, converts spelling variants such as `opthalmic` to `ophthalmic`, substitutes `none` for empty task families, and writes invalid controlled-vocabulary values to the report for reviewer resolution.
