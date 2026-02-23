<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/3f15d122-4a48-4045-bcb9-d69d9ebbdb82" />


![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blue?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Data%20Visualization-4C72B0?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue?style=for-the-badge)

---

## 🏢 Business Understanding

The rapid spread of misinformation across digital platforms has created a significant societal challenge. Fake news influences public opinion, political stability, and trust in media institutions.

This project builds an automated system capable of classifying news articles as **Fake** or **Real** using Natural Language Processing and Machine Learning techniques. The objective is to provide a scalable and reliable solution for detecting misleading information.

---

## 📖 Project Overview

This project develops and compares multiple machine learning and deep learning models for fake news detection. The workflow includes:

- Exploratory Data Analysis (EDA)
- Text preprocessing and feature engineering
- Training traditional ML models
- Training deep learning models (LSTM, BiLSTM)
- Fine-tuning DistilBERT transformer
- Model evaluation and comparison
- Local deployment using FastAPI and React

The dataset primarily contains long, multi-paragraph articles, which significantly influences model behavior.

---

## ❓ Problem Statement

Manual verification of online news is inefficient and prone to error. There is a need for an automated system that can:

- Accurately distinguish between fake and real news articles  
- Handle long-form content effectively  
- Provide prediction confidence for reliability  

The main challenge is building a model that generalizes well while maintaining high accuracy and F1-score.

---

## 🎯 Objectives

- Train and compare:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - LSTM
  - BiLSTM
  - DistilBERT
- Evaluate using Accuracy and F1-score
- Optimize performance for multi-paragraph articles
- Deploy the best-performing model locally
- Document model strengths and limitations

---

## 📊 Success Metrics

Evaluation metrics used:

- Accuracy
- F1-Score
- Confidence scores
- Stability on long-form articles

Target benchmark:

- Accuracy ≥ 95%
- F1-score ≥ 95%

---

## 📚 Data Understanding

The dataset consists of labeled news articles categorized as **Fake** or **Real**.

Key findings from EDA:

- Articles are typically multi-paragraph and long-form
- Balanced class distribution
- Significant variation in article length
- Long articles provide richer contextual information

Important insight:
Transformer models perform better when sufficient contextual information is available.

---

## 🧹 Data Cleaning and Preprocessing

Steps performed:

- Removal of special characters and extra whitespace
- Lowercasing (for classical models)
- Tokenization
- Padding and truncation for neural networks
- Train-test split
- Tensor conversion for deep learning models

For DistilBERT:
- Defined maximum token length
- Truncated long articles to fit transformer constraints

---

## 📊 ExPloratory Data Analysis (EDA)

<img width="644" height="508" alt="image" src="https://github.com/user-attachments/assets/81db2514-df0a-4839-892e-5e83ae3ba418" />

<img width="681" height="580" alt="image" src="https://github.com/user-attachments/assets/2f995fad-b08c-4727-99e7-e40b6009b196" />

<img width="676" height="503" alt="image" src="https://github.com/user-attachments/assets/2a1d425c-9220-4bcc-8130-6da9b8f50c4e" />

<img width="624" height="334" alt="image" src="https://github.com/user-attachments/assets/4db25ffe-6d05-4688-baae-85d2ba3aba05" />

### Key Observations

- Imbalanced Classes: "Fake News" (0) significantly outweighs "Real News" (1).
- Political Focus: Both categories are dominated by US politics, specifically terms like "Donald Trump," "Hillary Clinton," and "White House."
-  Neutral Language: The most frequent words across both are functional terms like "said," "one," and "state," suggesting similar reporting styles on the surface.

---

## 🤖 Modeling and Evaluation

| Model        | Accuracy (%) | F1-Score (%) |
|--------------|-------------|--------------|
| Naive Bayes  | 85          | 84           |
| SVM          | 96          | 96           |
| LSTM         | 91          | 90           |
| BiLSTM       | 96          | 95           |
| DistilBERT   | 98          | 98           |

### Key Observations

- DistilBERT achieved the highest performance.
- SVM performed exceptionally well among traditional models.
- LSTM models required sufficient context for optimal results.
- Multi-paragraph articles significantly improved transformer performance.
- Very short inputs may reduce reliability.

---

## 🚀 Deployment

### Backend

- Built using FastAPI
- Hosts trained models
- Exposes a prediction endpoint
- Returns:
  - Predicted label (Fake/Real)
  - Confidence score

### Frontend

- Built using React
- Accepts user article input
- Sends request to backend
- Displays prediction and confidence

### Architecture Flow

User Input → Frontend → Backend API → Model Inference → Prediction → Frontend Display

### Important Note

The system performs best with full-length, multi-paragraph articles, as models were trained primarily on long-form content.

---

## ✅ Conclusions

- DistilBERT is the best-performing model (98% Accuracy, 98% F1).
- SVM is strong for shorter articles.
- Article length significantly affects transformer performance.
- Combining classical ML and deep learning improves robustness.

---

## 💡 Recommendations

- Use DistilBERT as the primary production model.
- Implement model selection logic based on input length.
- Expand dataset with more short-form content.
- Consider cloud deployment for scalability.
- Add monitoring and caching for performance optimization.

## How to test

``` # Clone repository
git clone <your-repo-url>

# Navigate to backend
cd Fake_News_Backend

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn app:app --reload

# Open new terminal and navigate to frontend
cd Fake_News_Frontend

# Install frontend dependencies
npm install

# Start frontend
npm start
```
## ⭐ Expected Results 

<img width="658" height="374" alt="image" src="https://github.com/user-attachments/assets/cfdd66f2-b4dc-491d-9333-c65cd168a27e" />

