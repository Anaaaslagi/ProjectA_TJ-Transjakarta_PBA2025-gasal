# Transjakarta Sentiment Analysis — NLP Project

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-purple)
![IndoBERT](https://img.shields.io/badge/Model-IndoBERT-orange)
![Sklearn](https://img.shields.io/badge/Sklearn-Logistic%20Regression-green?logo=scikitlearn&logoColor=white)


> **An end-to-end NLP pipeline for sentiment analysis of Transjakarta (Jakarta BRT) public reviews — from web scraping to IndoBERT fine-tuning, comparing traditional ML vs. transformer-based approaches.**

---

## Project Overview

**Transjakarta** is Jakarta's Bus Rapid Transit (BRT) system — one of the largest BRT networks in Southeast Asia. This project analyzes **public sentiment** from user reviews and feedback to understand what passengers like and dislike about the service.

The analysis covers the **full NLP pipeline**:

```
Scraping → Preprocessing → Labeling → EDA → POS/NER → Modeling → Fine-tuning → Evaluation
```

### Objectives

- Classify public reviews about Transjakarta into **Positive / Negative / Neutral** sentiment
- Compare performance of **TF-IDF + Logistic Regression** vs **IndoBERT** (Indonesian BERT)
- Apply **data augmentation** to handle class imbalance in neutral sentiment
- Perform **linguistic analysis** using POS Tagging and Named Entity Recognition (NER)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **BeautifulSoup / Selenium** | Web scraping reviews |
| **Pandas, NumPy** | Data manipulation |
| **NLTK, Sastrawi** | Indonesian text preprocessing |
| **Scikit-learn** | TF-IDF vectorization + Logistic Regression |
| **HuggingFace Transformers** | IndoBERT fine-tuning |
| **Matplotlib, Seaborn, WordCloud** | Visualization & EDA |
| **Jupyter Notebook** | Analysis & documentation |

---

## Project Structure

```
ProjectA_TJ-Transjakarta_PBA2025-gasal/
│
├── Data/                           # Raw and processed datasets
│
└── Notebooks/
    ├── 1_Scrape.ipynb                               # Web scraping reviews
    ├── 2_preprocessing.ipynb                        # Text cleaning & preprocessing
    ├── 2b_preprocessing_bert.ipynb                  # BERT-specific preprocessing
    ├── 3_Sentiment Labeling.ipynb                   # Sentiment label assignment
    ├── 4_EDA.ipynb                                  # Exploratory Data Analysis
    ├── 5_POS.ipynb                                  # Part-of-Speech tagging
    ├── 6_NER.ipynb                                  # Named Entity Recognition
    ├── 7_TFIDF_dan_Evaluasi_LR_dan_IndoBERT.ipynb  # Model training & evaluation
    ├── 7b_finetuning_bert.ipynb                     # IndoBERT fine-tuning
    ├── 7c_error_analysis.ipynb                      # Error analysis
    ├── 8_Augmentasi Data Sentimen Netral.ipynb      # Neutral class augmentation
    ├── 9_Preprocessing Ulang Hasil Augmentasi.ipynb # Re-preprocessing
    └── 10_Evaluasi_Ulang_Model.ipynb                # Final model re-evaluation
```

---

## Pipeline Walkthrough

### 1. Data Collection (`1_Scrape.ipynb`)
- Scraped user reviews about Transjakarta from public sources
- Collected review text, ratings, and metadata

### 2. Text Preprocessing (`2_preprocessing.ipynb`, `2b_preprocessing_bert.ipynb`)
- Lowercasing, punctuation removal, URL & emoji removal
- Indonesian stopword removal using **Sastrawi**
- Stemming for traditional ML; WordPiece tokenization for BERT
- Separate pipelines for TF-IDF and BERT models

### 3. Sentiment Labeling (`3_Sentiment Labeling.ipynb`)
- Labeled reviews into **3 classes**: Positive | Neutral | Negative
- Used lexicon-based approach for initial labeling
- Manual verification for edge cases

### 4. Exploratory Data Analysis (`4_EDA.ipynb`)
- Class distribution analysis
- Word frequency analysis & word clouds
- Review length distribution
- Most common positive vs. negative words

### 5. Linguistic Analysis
- **POS Tagging** (`5_POS.ipynb`): Identified nouns, verbs, adjectives in reviews
- **NER** (`6_NER.ipynb`): Extracted entities like station names, route names, places

### 6. Model Training & Evaluation (`7_*.ipynb`)

#### Approach A — TF-IDF + Logistic Regression
- Feature extraction: TF-IDF vectorizer
- Classifier: Logistic Regression
- Baseline model for comparison

#### Approach B — IndoBERT Fine-tuning
- Pre-trained model: `indobenchmark/indobert-base-p1`
- Fine-tuned on Transjakarta review dataset
- Evaluated with accuracy, precision, recall, F1-score

### 7. Handling Class Imbalance (`8_Augmentasi.ipynb`)
- Neutral class had significantly fewer samples
- Applied **data augmentation** techniques (synonym replacement, back-translation)
- Re-preprocessed augmented data

### 8. Final Evaluation (`10_Evaluasi_Ulang_Model.ipynb`)
- Compared model performance before and after augmentation
- Generated final confusion matrix

---

## Model Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| TF-IDF + Logistic Regression | ~baseline | Fast, interpretable |
| IndoBERT (before augmentation) | Higher | Better context understanding |
| IndoBERT (after augmentation) | **Best** | Improved neutral class recall |


---

## Key Findings

- **Negative sentiment** is the most common category — passengers frequently complain about:
  - Overcrowding during peak hours
  - Long waiting times
  - AC & cleanliness issues
- **Positive sentiment** centers around:
  - Affordable fare prices
  - Wide route coverage
  - Dedicated busway lanes reducing travel time
- **IndoBERT** significantly outperforms TF-IDF on understanding context and sarcasm in Indonesian text
- **Data augmentation** for neutral class improved overall macro F1-score

---

## How to Run

### Prerequisites
```bash
pip install transformers pandas numpy scikit-learn matplotlib seaborn wordcloud sastrawi
```

### Run the Pipeline (in order)
```
1. Notebooks/1_Scrape.ipynb
2. Notebooks/2_preprocessing.ipynb
3. Notebooks/3_Sentiment Labeling.ipynb
4. Notebooks/4_EDA.ipynb
5. Notebooks/7_TFIDF_dan_Evaluasi_...ipynb
6. Notebooks/7b_finetuning_bert.ipynb
7. Notebooks/10_Evaluasi_Ulang_Model.ipynb
```

> Fine-tuning IndoBERT requires a GPU. Google Colab (free tier) is recommended if no local GPU is available.

---

## References

- [IndoBERT — IndoNLU Benchmark](https://huggingface.co/indobenchmark/indobert-base-p1)
- [Transjakarta Official](https://www.transjakarta.co.id/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Sastrawi — Indonesian Stemmer](https://github.com/har07/PySastrawi)

