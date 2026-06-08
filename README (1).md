# Toxic Comment Classifier — AI & Semantic Analysis

An AI-powered system that detects toxic and harmful online comments using TF-IDF semantic analysis and a TensorFlow deep learning model. Features a real-time Streamlit web app with toxicity scoring, keyword-level analysis, and prediction history.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
- [Streamlit App Features](#streamlit-app-features)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Model Performance](#model-performance)
- [Future Improvements](#future-improvements)

---

## Project Overview

Online platforms struggle with toxic, abusive, and harmful comments that damage communities and mental health. This project builds an **end-to-end ML pipeline** to automatically detect toxic comments using:

- **TF-IDF Vectorization** — converts text into numerical semantic features
- **Deep Learning (TensorFlow/Keras)** — binary classification (Toxic / Non-Toxic)
- **SMOTE** — handles class imbalance in real-world datasets
- **Streamlit** — interactive, real-time web interface

Users paste any comment into the app and instantly receive a toxicity prediction, probability score, keyword-level breakdown, and session history.

---

## Repository Structure

```
toxic-comment-classifier/
│
├── toxic_comment_classifier.ipynb    # Main notebook: EDA, training, Streamlit app
│
├── Toxic.py                          # Streamlit app v1 — binary classification + HuggingFace toggle
├── Toxic2.py                         # Streamlit app v2 — probability gauge chart
│
├── toxic_comment_model.h5            # Saved Keras model
├── tfidf_vectorizer.pkl              # Saved TF-IDF vectorizer
│
├── data/
│   ├── Toxic_classification.csv      # Dataset 1: Kaggle toxic comment classification
│   └── youtoxic_english_1000.csv     # Dataset 2: YouTube toxic comments
│
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Data Pipeline

Two datasets are merged into a unified binary format (`Text` | `Toxic 0/1`):

- `Toxic_classification.csv` — multi-label data (toxic, obscene, threat, insult, hate)
- `youtoxic_english_1000.csv` — YouTube comment toxicity data

### 2. Text Preprocessing

- Lowercasing and contraction expansion (`can't` → `cannot`, `i'm` → `i am`)
- Removing special characters and extra whitespace
- Duplicate removal and null value handling

### 3. Feature Extraction

TF-IDF Vectorizer with `max_features=5000` and English stop words removed.

### 4. Class Balancing

SMOTE (Synthetic Minority Oversampling Technique) is applied to correct the natural imbalance between toxic and non-toxic samples.

### 5. Model Architecture

```
Input (TF-IDF features)
    └── Dense(64, activation='relu')
    └── Dropout(0.3)
    └── Dense(32, activation='relu')
    └── Dropout(0.3)
    └── Dense(1, activation='sigmoid')   →   Toxic / Non-Toxic
```

Optimizer: `Adam` | Loss: `binary_crossentropy`

### 6. Semantic Toxic Word Analysis

The app independently scans comments for toxic keywords across 5 categories:

| Category | Examples |
|---|---|
| Profanity | Common swear words |
| Hate Speech | Slurs and discriminatory terms |
| Insults | idiot, stupid, moron |
| Threats | kill, hurt, attack |
| Harassment | stalker, creep, harassment |

---

## Streamlit App Features

| Feature | Description |
|---|---|
| Text Input | Accepts comments up to 200 words |
| Model Selection | Custom Model or HuggingFace `toxic-bert` |
| Toxicity Gauge | Visual probability gauge chart (Plotly) |
| Toxic Word Breakdown | Category-wise bar chart of flagged words |
| Prediction History | Running log with total, toxic, and non-toxic counts |
| Clear History | Reset session state |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/toxic-comment-classifier.git
cd toxic-comment-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
numpy==1.23.5
pandas
scikit-learn==1.2.2
imbalanced-learn==0.10.1
tensorflow==2.12.0
streamlit
plotly
wordcloud
matplotlib
seaborn
transformers
```

### 3. Run the App

```bash
# Version 1 — binary result + HuggingFace model toggle
streamlit run Toxic.py

# Version 2 — probability gauge chart
streamlit run Toxic2.py
```

### Running on Google Colab

```bash
!curl ipv4.icanhazip.com
!streamlit run Toxic.py &>./logs.txt & npx localtunnel --port 8501
```

---

## Datasets

| Dataset | Source | Size |
|---|---|---|
| Toxic Comment Classification | [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | ~160,000 rows |
| YouToxic English | [Kaggle](https://www.kaggle.com/datasets/nikulinjurij/youtoxic-english) | ~1,000 rows |

---

## Model Performance

Evaluated using:

- Precision, Recall, F1-Score via `classification_report`
- Confusion Matrix
- Train / Test Split: 80% / 20%

---

## Future Improvements

- [ ] LSTM or BERT-based model for deeper contextual understanding
- [ ] Multi-label classification (toxic, threat, insult separately)
- [ ] REST API deployment with FastAPI
- [ ] Browser extension integration
- [ ] Multilingual toxic comment detection
