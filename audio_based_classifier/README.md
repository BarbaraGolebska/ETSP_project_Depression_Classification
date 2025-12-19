# Audio-Based Classifier

This folder contains all scripts required to build, train, and evaluate **audio-based machine learning classifiers** for patient-level prediction. The pipeline covers data acquisition, feature preprocessing, model training, hyperparameter optimization, and evaluation on a held-out test set.

## Folder Structure and Script Description

### `01_download_audio_data.py`
Responsible for downloading the raw audio data required for feature extraction and modeling.

**Usage:**
- Run this script first to ensure all audio data are available locally.
- Handles dataset retrieval and storage in the expected directory structure.

---

### `02_preprocessing.py`
Performs feature aggregation and preprocessing to obtain **one feature vector per patient**.

**Key functionality:**
- Aggregates audio features over time and produces a single row per patient.

**Important notes:**
- Call the `main()` function to run preprocessing.
- Specify the feature files to aggregate via the `ftypes` argument.
  - You may aggregate a **single feature file** or **multiple feature files jointly**.
- All features are processed through the generic aggregation pipeline **except HuBERT embeddings**.
  - HuBERT features must be processed separately using `preprocess_hubert()`.

---

### `baseline.py`
Implements the **baseline audio-based classifier** using Logistic Regression.

**Key functionality:**
- `train_models()`:
  - Trains a Logistic Regression model.
  - Evaluates multiple oversampling strategies to address class imbalance.
  - Uses Optuna for hyperparameter optimization.
- Runs systematic experiments to identify the best-performing configuration on the development set.

**Evaluation:**
- `evaluate_on_test_set()`:
  - Applies the selected models to the test set.
  - Computes performance metrics such as AUC, Youden index, precision, recall, F1-score, and confusion matrix.

---

### `classic_tree_models.py`
Trains and evaluates **classical tree-based ensemble models**. To provide a comparison between linear baseline models and more expressive non-linear tree-based methods.
Uses the same preprocessed, patient-level audio features for consistency.

**Models included:**
- Random Forest
- XGBoost


---

### `project_utils.py`
Contains shared utility functions used across the pipeline.

**Includes utilities for:**
- Loading feature files
- Splitting data into development and test sets
- Model evaluation and metric computation
- Common helper functions to avoid code duplication

---

## Typical Workflow

1. **Download audio data**: 
run 01_download_audio_data.py
   

2. **Preprocess and aggregate features**:

Run main() in 02_preprocessing.py

Specify feature types via ftypes

Process HuBERT features separately using preprocess_hubert()

3. **Train baseline model**:

Run baseline.py to train Logistic Regression models with oversampling and Optuna tuning

4. **Train tree-based models**:

Run classic_tree_models.py to train Random Forest and XGBoost classifiers


