# Spam vs Non-Spam Classification System

##  Project Overview
This project implements a **Spam vs Non-Spam text classification system** using classical **Machine Learning** techniques.  
The model is trained on labeled text data and deployed as a **web application using Streamlit**, allowing users to classify new messages in real time.

The focus of this project is:
- Proper **data preprocessing**
- **Algorithm comparison**
- **Hyperparameter tuning**
- Selection of the **best-performing model**
- Simple and clean **deployment**

---

## Problem Statement
Spam messages cause inconvenience, security risks, and productivity loss.  
The objective of this project is to **automatically classify a given message as Spam or Non-Spam** using machine learning techniques.

This is a **binary classification problem**.

---

##  Dataset
- Dataset used: **SMS Spam Collection Dataset**
- Each data sample contains:
  - `label` → spam / ham
  - `text` → message content

### Why this dataset?
- Publicly available and well-structured
- Commonly used for benchmarking spam classifiers
- Suitable for beginners and academic projects

---

## ⚙️ Data Preprocessing
Steps applied:
1. Converted text to lowercase
2. Removed punctuation
3. Converted labels:
   - `spam → 1`
   - `ham → 0`

 Heavy text cleaning (stemming/lemmatization) was intentionally avoided, as **TF-IDF already captures word importance effectively**.

---

## Feature Extraction
Text data cannot be directly used by ML models, so it was converted into numerical form using:

### **TF-IDF Vectorization**
- Removes the impact of common words
- Highlights important and rare words
- Performs well with linear models

**Parameters used:**
- Stop words removal
- Unigrams + Bigrams (`ngram_range = (1,2)`)
- Maximum features = 5000

---

## Algorithms Implemented
Multiple algorithms were trained and evaluated to ensure fair comparison:

### 1. Naive Bayes
- Fast baseline model
- Works well with text data
- Assumes feature independence (limitation)

### 2. Logistic Regression
- Strong linear classifier
- Handles high-dimensional sparse data well
- Supports probability estimation
- Tuned using hyperparameter optimization


---

## Evaluation Metrics
Since the dataset is **imbalanced**, accuracy alone is misleading.

The following metrics were used:
- **Precision** → Avoid false spam classification
- **Recall** → Capture as much spam as possible
- **F1-score** → Balance between precision and recall

**F1-score for spam class was the primary metric for model selection.**

---

## Hyperparameter Tuning
Logistic Regression was tuned using **GridSearchCV** with:
- Regularization strength (`C`)
- Class balancing (`class_weight`)
- 5-fold cross-validation
- Scoring metric: **F1-score**

This significantly improved spam detection performance.

---

## Final Model Selection
After evaluation and tuning:

**Tuned Logistic Regression with TF-IDF** was selected as the final model.

### Reasons:
- High F1-score for spam class
- Good generalization on unseen data
- Stable and interpretable
- Industry-relevant approach for text classification

---

## Web Application (Streamlit)
The trained model was deployed using **Streamlit**.

### Features:
- User-friendly interface
- Real-time message classification
- Clear spam / non-spam output

### Workflow:
1. User enters text
2. Text is transformed using TF-IDF
3. Model predicts spam or non-spam
4. Result is displayed on the web interface

---

## 📁 Project Structure
SpamClassifier/
│
├── app.py
├── models/
│ ├── spam_model.pkl
│ └── tfidf_vectorizer.pkl
├── data/
|    |__ spam.csv
├── model_training.ipynb
└── README.md



---

## How to Run the Project
1. Install dependencies:
```bash
pip install pandas scikit-learn streamlit

streamlit run app.py


