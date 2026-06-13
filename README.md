# <p align="center">

# &#x20; <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python" />

# &#x20; <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn" />

# &#x20; <img src="https://img.shields.io/badge/SVM-One--vs--Rest-success" />

# &#x20; <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" />

# &#x20; <img src="https://img.shields.io/badge/React-Frontend-61DAFB?logo=react" />

# &#x20; <img src="https://img.shields.io/badge/Vite-Build-646CFF?logo=vite" />

# </p>

# 

# <h1 align="center">

# News Domain Classification using One-vs-Rest SVM

# </h1>

# 

# <p align="center">

# Course Project — CS419 Information Retrieval

# </p>

# 

# <p align="center">

# An end-to-end news article classification system built using TF-IDF feature extraction,

# Chi-Square feature selection, and One-vs-Rest Support Vector Machines.

# The project includes both a machine learning backend and a modern React-based web interface.

# </p>

# Giới thiệu

# \## 1. Introduction

# 

# \### Problem Statement

# 

# News portals publish thousands of articles every day across multiple domains such as business, politics, sports, technology, and entertainment. Automatically categorizing incoming articles is an important task in Information Retrieval and Text Mining systems.

# 

# This project focuses on multiclass news classification using traditional machine learning techniques. Given the title and content of a news article, the system predicts its corresponding category.

<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python" />

&#x20; <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn" />

&#x20; <img src="https://img.shields.io/badge/SVM-One--vs--Rest-success" />

&#x20; <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" />

&#x20; <img src="https://img.shields.io/badge/React-Frontend-61DAFB?logo=react" />

&#x20; <img src="https://img.shields.io/badge/Vite-Build-646CFF?logo=vite" />

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

The project includes both a machine learning backend and a modern React-based web interface.

</p>



\## 1. Introduction



\### Problem Statement



News portals publish thousands of articles every day across multiple domains such as business, politics, sports, technology, and entertainment. Automatically categorizing incoming articles is an important task in Information Retrieval and Text Mining systems.



This project focuses on multiclass news classification using traditional machine learning techniques. Given the title and content of a news article, the system predicts its corresponding category.



\### Objectives



\- Build a complete text classification pipeline.

\- Investigate the effectiveness of TF-IDF representations.

\- Evaluate feature selection using Chi-Square statistics.

\- Train a multiclass classifier using One-vs-Rest Support Vector Machines.

\- Deploy the model as a web application for real-time prediction.



\## 2. Dataset



The project uses the BBC News dataset, a benchmark corpus frequently used for document classification research.



\### Categories



The dataset contains articles from five news domains:



| Label | Category |

|---------|---------|

| 0 | Business |

| 1 | Entertainment |

| 2 | Politics |

| 3 | Sport |

| 4 | Tech |



\### Data Preparation



Several preprocessing steps were applied before training:



\- Lowercasing

\- Punctuation removal

\- Stopword filtering

\- Text normalization

\- TF-IDF vectorization



The final feature representation is a sparse TF-IDF matrix suitable for linear SVM training.



\## 3. Methodology



\### Overall Pipeline



Raw News Article

↓

Text Preprocessing

↓

TF-IDF Vectorization

↓

Chi-Square Feature Selection

↓

One-vs-Rest SVM

↓

Predicted Category



\### TF-IDF Representation



Term Frequency–Inverse Document Frequency (TF-IDF) is used to transform textual documents into numerical vectors. The representation emphasizes discriminative words while reducing the impact of common terms.



\### Chi-Square Feature Selection



To reduce dimensionality and improve generalization, Chi-Square statistics are employed to select the most informative features for classification.



Benefits:



\- Reduced training time

\- Lower memory consumption

\- Improved interpretability

\- Reduced noise



\### One-vs-Rest Support Vector Machine



Because SVM is inherently a binary classifier, multiclass classification is handled using the One-vs-Rest (OvR) strategy.



For K classes:



\- Train K independent binary classifiers.

\- Each classifier distinguishes one class from all remaining classes.

\- The classifier with the highest decision score determines the final prediction.



\## 4. Model Optimization



Hyperparameters were optimized using Grid Search.



\### Tuned Parameters



\- C

\- Kernel

\- Gamma (where applicable)



Grid Search systematically evaluates candidate parameter combinations and selects the configuration yielding the best validation performance.

\## 5. System Architecture



The application follows a client-server architecture.



┌───────────────┐

│ React Frontend│

└───────┬───────┘

&#x20;       │ HTTP

&#x20;       ▼

┌──────────────────┐

│ FastAPI Backend  │

└───────┬──────────┘

&#x20;       │

&#x20;       ▼

┌──────────────────┐

│ OvR SVM Model    │

│ TF-IDF Vectorizer│

│ Chi² Selector    │

└──────────────────┘



\### Backend Responsibilities



\- Load trained artifacts

\- Process user input

\- Execute inference

\- Return prediction results



\### Frontend Responsibilities



\- User interaction

\- Article submission

\- Visualization of prediction results

\- Communication with backend API



\## 6. Project Structure



```text

CS419-News-Classifier-Using-OvR-SVM

│

├── Backend/

│   ├── main.py

│   ├── model\_utils.py

│   ├── requirements.txt

│   ├── svm\_model.joblib

│   ├── vectorizer.joblib

│   ├── selector.joblib

│   ├── grid\_search.joblib

│   ├── text\_classification.ipynb

│   ├── bbc-news-data.csv

│   └── chi2\_result.csv

│

├── Frontend/

│   ├── src/

│   ├── public/

│   ├── package.json

│   └── vite.config.js

│

└── README.md

```

\## 7. Installation



\### Clone Repository



```bash

git clone https://github.com/your-username/CS419-News-Classifier-Using-OvR-SVM.git



cd CS419-News-Classifier-Using-OvR-SVM

```



\---



\## 8. Backend Setup



```bash

cd Backend



python -m venv venv



venv\\Scripts\\activate



pip install -r requirements.txt

```



Start FastAPI server:



```bash

uvicorn main:app --reload

```



Backend will be available at:



```text

http://localhost:8000

```



\---



\## 9. Frontend Setup



```bash

cd Frontend



npm install



npm run dev

```



Frontend will be available at:



```text

http://localhost:5173

```

\## 10. Example Prediction



\### Input



Apple unveils its latest AI-powered chip designed for next-generation devices.



\### Output



Category: Tech



Confidence: 96.4%



\## 11. Experimental Results



Evaluation metrics:



\- Accuracy

\- Precision

\- Recall

\- F1-score



The OvR-SVM model achieved strong performance on the BBC News benchmark, demonstrating that traditional machine learning approaches remain highly competitive for medium-scale text classification tasks.



\## 12. Technologies Used



\### Machine Learning



\- Scikit-learn

\- NumPy

\- Pandas

\- Joblib



\### Backend



\- FastAPI

\- Uvicorn



\### Frontend



\- React

\- Vite

\- JavaScript



\### Development



\- Python 3.10+

\- Node.js

\- Git



\## 13. References



1\. Joachims, T. (1998). Text Categorization with Support Vector Machines.

2\. Manning, C., Raghavan, P., Schütze, H. Introduction to Information Retrieval.

3\. BBC News Dataset.

4\. Scikit-Learn Documentation.

