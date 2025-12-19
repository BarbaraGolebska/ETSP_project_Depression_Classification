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