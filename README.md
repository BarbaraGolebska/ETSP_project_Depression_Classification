# Automatic Detection of Psychological Conditions from Clinical Interviews

This project focuses on the automatic detection of depression using interview transcripts and audio data. 

## Table of Contents
- [Data](#data)
- [Methodology](#methodology)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

## Data

We utilize the **E-DAIC dataset**, a component of the DAIC Corpus, which features:

- **Virtual Interviewer**: Interviews conducted by an animated virtual interviewer called Ellie
- **Multimodal Recordings**: Audio recordings with extracted acoustic features
- **Text Transcripts**: Complete utterance transcriptions with timestamps and speaker labels
- **Clinical Labels**: PHQ-8 scores for depression

## Methodology

### Text-Based Classifier

...

### Audio-Based Classifier

...

### Multimodal Fusion Model

...

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