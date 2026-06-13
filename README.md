<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/SVM-One--vs--Rest-success" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?logo=react" />
  <img src="https://img.shields.io/badge/Vite-Build-646CFF?logo=vite" />
</p>

<h1 align="center">
News Domain Classification using One-vs-Rest SVM
</h1>

<p align="center">
Course Project — CS419 Information Retrieval
</p>

<p align="center">
An end-to-end news article classification system built using TF-IDF feature extraction,
Chi-Square feature selection, and One-vs-Rest Support Vector Machines.
The project includes a FastAPI backend for model serving and a React-based frontend for interactive prediction.
</p>

---

# Table of Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Model Training and Optimization](#4-model-training-and-optimization)
5. [System Architecture](#5-system-architecture)
6. [Project Structure](#6-project-structure)
7. [Installation](#7-installation)
8. [Running the Application](#8-running-the-application)
9. [Prediction Workflow](#9-prediction-workflow)
10. [Technologies Used](#10-technologies-used)
11. [References](#11-references)

---

# 1. Overview

## Problem Statement

Automatic document classification is a fundamental task in Information Retrieval and Text Mining. News organizations continuously publish articles across various domains, making manual categorization expensive and difficult to scale.

This project addresses the problem of multiclass news classification by automatically assigning a news article to its corresponding domain based on textual content.

## Objectives

- Build a complete text classification pipeline.
- Explore TF-IDF document representation.
- Apply Chi-Square feature selection to reduce dimensionality.
- Implement multiclass classification using the One-vs-Rest SVM strategy.
- Optimize model performance through Grid Search.
- Deploy the trained model through a web-based application.

## Main Contributions

- End-to-end machine learning pipeline for news classification.
- Feature engineering using TF-IDF and Chi-Square statistics.
- Multiclass classification with One-vs-Rest Support Vector Machines.
- Hyperparameter tuning using Grid Search.
- FastAPI inference service.
- Interactive React frontend for real-time predictions.

---

# 2. Dataset

## BBC News Dataset

The project uses the BBC News dataset, a widely used benchmark corpus for document classification research.

The dataset contains articles belonging to five categories:

| Category | Description |
|-----------|-------------|
| Business | Economics, finance, and corporate news |
| Entertainment | Movies, television, music, and media |
| Politics | Government, elections, and public policy |
| Sport | Sporting events and athletes |
| Tech | Technology, software, and innovation |

## Data Preparation

Before training, the dataset is transformed through several preprocessing stages:

- Text normalization
- Lowercasing
- Removal of punctuation
- Tokenization
- TF-IDF vectorization

The resulting representation is a sparse high-dimensional feature matrix suitable for SVM-based learning.

---

# 3. Methodology

## Overall Pipeline

```text
Raw News Article
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Chi-Square Feature Selection
        │
        ▼
One-vs-Rest SVM
        │
        ▼
Predicted Category
```

## TF-IDF Representation

Term Frequency–Inverse Document Frequency (TF-IDF) converts textual documents into numerical vectors by weighting terms according to both their frequency within a document and their rarity across the corpus.

Benefits:

- Highlights discriminative terms.
- Suppresses common words.
- Produces efficient sparse representations.
- Widely adopted in Information Retrieval systems.

## Chi-Square Feature Selection

Chi-Square statistics are used to identify features that exhibit strong dependence on class labels.

Feature selection provides:

- Reduced dimensionality
- Faster training
- Lower memory usage
- Improved generalization
- Better interpretability

## One-vs-Rest Support Vector Machine

Support Vector Machines are inherently binary classifiers.

To support multiclass classification, the One-vs-Rest (OvR) strategy is employed:

- Train one classifier for each category.
- Each classifier learns to distinguish its target class from all remaining classes.
- During inference, the classifier with the highest decision score determines the final prediction.

For a dataset containing K classes:

```text
K Binary SVM Classifiers
        ↓
Decision Scores
        ↓
Highest Score Wins
```

---

# 4. Model Training and Optimization

## Hyperparameter Tuning

Model parameters are optimized using Grid Search.

The search process evaluates multiple parameter combinations and selects the configuration yielding the best validation performance.

### Tuned Parameters

- Kernel type
- Regularization parameter (C)
- Gamma (where applicable)

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

Additional analyses may include:

- Confusion Matrix
- Per-class performance
- Feature importance based on Chi-Square statistics

---

# 5. System Architecture

The application follows a client-server architecture.

```text
┌─────────────────────┐
│    React Frontend   │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│   FastAPI Backend   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     OvR SVM Model   │
│   TF-IDF Vectorizer │
│   Chi² Selector     │
└─────────────────────┘
```

## Backend Responsibilities

- Load trained artifacts
- Preprocess incoming text
- Execute inference
- Return prediction results

## Frontend Responsibilities

- User interaction
- News article submission
- Display prediction results
- Communicate with backend APIs

---

# 6. Project Structure

```text
CS419-News-Classifier-Using-OvR-SVM
│
├── Backend
│   ├── main.py
│   ├── model_utils.py
│   ├── requirements.txt
│   ├── svm_model.joblib
│   ├── vectorizer.joblib
│   ├── selector.joblib
│   ├── grid_search.joblib
│   ├── text_classification.ipynb
│   ├── bbc-news-data.csv
│   └── chi2_result.csv
│
├── Frontend
│   ├── public
│   ├── src
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── index.html
│
├── .gitignore
└── README.md
```

### Important Files

| File | Purpose |
|--------|---------|
| `text_classification.ipynb` | Model development and experimentation |
| `vectorizer.joblib` | Saved TF-IDF vectorizer |
| `selector.joblib` | Saved Chi-Square feature selector |
| `svm_model.joblib` | Trained OvR-SVM classifier |
| `grid_search.joblib` | Best model configuration from Grid Search |
| `main.py` | FastAPI application entry point |

---

# 7. Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/CS419-News-Classifier-Using-OvR-SVM.git

cd CS419-News-Classifier-Using-OvR-SVM
```

---

## Backend Setup

```bash
cd Backend

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Frontend Setup

```bash
cd Frontend

npm install
```

---

# 8. Running the Application

## Start Backend

```bash
cd Backend

uvicorn main:app --reload
```

Backend endpoint:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

---

## Start Frontend

```bash
cd Frontend

npm run dev
```

Frontend endpoint:

```text
http://localhost:5173
```

---

# 9. Prediction Workflow

1. User submits a news article through the web interface.
2. Frontend sends a request to the FastAPI backend.
3. Backend preprocesses the text.
4. TF-IDF vectorization is applied.
5. Chi-Square feature selection is performed.
6. The OvR-SVM model generates decision scores.
7. The category with the highest score is returned.
8. Frontend displays the prediction result.

---

# 10. Technologies Used

## Machine Learning

- Scikit-learn
- NumPy
- Pandas
- Joblib

## Backend

- FastAPI
- Uvicorn

## Frontend

- React
- Vite
- JavaScript

## Development

- Python 3.10+
- Node.js
- Git

---

# 11. References

1. Joachims, T. (1998). *Text Categorization with Support Vector Machines: Learning with Many Relevant Features*.

2. Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*.

3. BBC News Dataset.

4. Scikit-learn Documentation.

5. Vapnik, V. N. *The Nature of Statistical Learning Theory*.