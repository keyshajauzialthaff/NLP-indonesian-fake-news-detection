# NLP-indonesian-fake-news-detection
Indonesian Fake News Detection using NLP, TF-IDF, and Logistic Regression for Hoax and Valid news classification.

# Indonesian Fake News Detection Using TF-IDF and Logistic Regression

## Project Overview

This project aims to develop a machine learning model to classify Indonesian news articles into two categories: **Hoax** and **Valid**.

The project applies Natural Language Processing (NLP) techniques to process textual data and uses TF-IDF (Term Frequency–Inverse Document Frequency) for feature extraction. A Logistic Regression classifier with class weighting is then implemented to predict news categories.

This project serves as an introductory application of NLP and machine learning for Indonesian text classification.

## Objectives

- Process and clean Indonesian-language news text.
- Transform textual data into numerical features using TF-IDF.
- Develop a binary text classification model using Logistic Regression.
- Evaluate model performance using classification metrics.
- Explore the application of NLP techniques in misinformation detection.

## Dataset

The dataset consists of **600 Indonesian news articles** with two classification labels:

| Label | Number of Articles |
|---|---:|
| Valid | 372 |
| Hoax | 228 |
| **Total** | **600** |

The dataset contains two main columns:
- `berita`: News article text.
- `tagging`: News classification label (Valid or Hoax).

## Methodology

The project follows these steps:

### 1. Text Preprocessing

Text preprocessing is performed to prepare raw news articles for machine learning, including:
- Case folding (converting text to lowercase).
- Removing numbers and special characters.
- Tokenization.
- Stopword removal using Indonesian stopwords.
- Stemming using Sastrawi.

### 2. Feature Extraction

TF-IDF is used to transform text into numerical representations.

The vectorizer configuration:
- Maximum features: 5,000
- N-gram range: (1, 2), including unigrams and bigrams.

### 3. Model Development

Logistic Regression is used as the classification algorithm.

To address the imbalance between Hoax and Valid labels, the model applies:

`class_weight='balanced'`

The dataset is divided into:
- Training data: 80% (480 articles)
- Testing data: 20% (120 articles)

Stratified splitting is applied to maintain class proportions.

### 4. Model Evaluation

Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

## Model Performance

The balanced Logistic Regression model achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 74.17% |
| Macro Precision | 0.73 |
| Macro Recall | 0.74 |
| Macro F1-score | 0.73 |
| Weighted F1-score | 0.74 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Hoax | 0.65 | 0.72 | 0.68 |
| Valid | 0.81 | 0.76 | 0.78 |

The model demonstrates a more balanced classification performance between the two categories after applying class weighting.

## Libraries

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Sastrawi
- Matplotlib
- Seaborn
- Joblib

## Project Structure

```text
fake-news-detection/
│
├── NLP_Hoax_Detection.ipynb
├── 600 news with valid hoax label.csv
├── logistic_regression_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
└── README.md
```

## Limitations

This project has several limitations:

- The dataset contains only 600 news articles, which may limit model generalization.
- Model performance depends on the quality and reliability of dataset labels.
- The model identifies patterns learned from labeled data and does not independently verify factual claims.
- Further evaluation using larger and more diverse datasets is needed.

## Future Improvements

Potential improvements include:
- Expanding the dataset with more diverse Indonesian news sources.
- Exploring advanced NLP methods such as LSTM and transformer-based models.
- Improving model performance through hyperparameter tuning.
- Evaluating model robustness on unseen news articles.

## Conclusion

This project demonstrates the implementation of an NLP-based text classification pipeline for Indonesian news using TF-IDF and Logistic Regression.

With an accuracy of 74.17%, the model provides a baseline approach to distinguishing Hoax and Valid news articles. The project also highlights the importance of text preprocessing, feature extraction, class imbalance handling, and model evaluation in NLP applications.

---

**Project Focus:** Natural Language Processing (NLP) | Text Classification | Machine Learning

**Author:** Keysha Jauzi Althaff

**Purpose:** Machine Learning and NLP Portfolio Project
