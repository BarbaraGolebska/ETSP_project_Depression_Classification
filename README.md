# Automatic Detection of Psychological Conditions from Clinical Interviews

This project focuses on the automatic detection of depression using interview transcripts and audio data. 

## Table of Contents
- [Data](#data)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Audio Based Classifier](#audio_based_classifier)

## Data

We utilize the **E-DAIC dataset**, a component of the DAIC Corpus, which features:

- **Virtual Interviewer**: Interviews conducted by an animated virtual interviewer called Ellie
- **Multimodal Recordings**: Audio recordings with extracted acoustic features
- **Text Transcripts**: Complete utterance transcriptions with timestamps and speaker labels
- **Clinical Labels**: PHQ-8 scores for depression

  ## Evaluation

**Metrics**:
- **AUC-ROC**: Primary ranking metric
- **Youden's J statistic** (Sensitivity + Specificity - 1): Threshold selection

  ## Project Structure

```
ESTP_project/
├── README.md
├── audio_based_classifier/     # Audio processing and classification
├── data/                       # Dataset and preprocessing scripts
├── multimodal_fusion_model/    # Fusion strategies and models
└── text_based_classifier/      # Text processing and classification
```
# Audio-Based Classifier (audio_based_classifier/)

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


# Text-Based Classifier

This module contains all scripts for preprocessing interview transcripts and training text-only depression classifiers.

## Folder Structure

```text
text_based_classifier/
├── 01_download_text_data.py   # Download / prepare raw text data
├── 02_EDA.ipynb               # Exploratory data analysis (Jupyter notebook)
├── 03_preprocessing.py        # Cleaning, concatenating transcripts and labels
├── 04_baseline.py             # Simple baseline text classifiers
├── 05_embeddings-based.py     # Embedding-based models
├── 06_bilstm_attn.py          # BiLSTM + attention model (sequence-based)
├── punctuationmodel.py        # Auxiliary punctuation restoration model
├── results/                   # Saved models, thresholds, metrics, plots
├── optuna/                    # Saved optuna studies
└── README.md                  # This file
```

## Running the Pipeline

All commands below are run from the `text_based_classifier` folder:

```bash
cd text_based_classifier
```

1. **Download / prepare text data**

```bash
python 01_download_text_data.py
```

2. **Preprocess data (generates ../data/processed/text_combined.csv)**

```bash
python 03_preprocessing.py
```

3. **Train baseline models**

```bash
python 04_baseline.py
```

4. **Train embeddings-based models**

```bash
python 05_embeddings-based.py
```

5. **Train BiLSTM + attention model**

```bash
python 06_bilstm_attn.py
```

Trained models, thresholds, and evaluation metrics are written to the `results/` directory.








