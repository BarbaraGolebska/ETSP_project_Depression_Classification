# Multimodal Fusion


## EARLY FUSION

This folder contains scripts for performing **early fusion** of audio and text features, training multimodal classifiers, and formatting evaluation results. The pipeline focuses on combining the best-performing unimodal models to improve patient-level prediction.

## Folder Structure and Script Description

### `concat_features.py`
Performs **early fusion** by concatenating selected feature files into a single multimodal feature set.

**Key functionality:**
- Acts as a factory for feature aggregation.
- Allows selection of specific files to include in the fusion.


---

### `early_fusion_LR_TM.py`
Trains multimodal classifiers using the **best-performing models from unimodal audio classification**.

**Key functionality:**
- Trains Logistic Regression (baseline) and LightGBM (tree-based) models on the fused feature set.
- Uses the fused data from `concat_features.py` to leverage complementary information from multiple modalities.

To run, use python -m multimodal_fusion_model.early_fusion_LR_TM command

---

### `describe_results.py`
Formats and organizes final evaluation metrics for multimodal models.

**Key functionality:**
- Collects metrics from multiple experiments.
- Identifies the best-performing models based on chosen criteria (e.g., AUC, F1-score, Youden index).
- Prepares results for reporting and comparison.

---

## Typical Workflow

1. **Perform early fusion**
   - Run `concat_features.py` to generate a combined feature set for the desired modalities.

2. **Train multimodal models**
   - Run `early_fusion_LR_TM.py` using the fused features.
   - Trains Logistic Regression and LightGBM models with optimized hyperparameters.

3. **Format and review results**
   - Run `describe_results.py` to summarize metrics and detect the best-performing models.


## LATE FUSION
