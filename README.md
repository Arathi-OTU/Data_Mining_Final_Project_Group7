# 📱 Amazon Cell Phones & Accessories Review Sentiment Analysis

## 📌 Project Overview

This project performs **Sentiment Analysis** on the **Amazon Cell Phones and Accessories Reviews** dataset using Natural Language Processing (NLP) and Machine Learning techniques. The objective is to classify customer reviews as **Positive** or **Negative** based on the review text.

The project includes data exploration, text preprocessing, feature engineering using TF-IDF, and sentiment classification using multiple machine learning algorithms.

---

## 🎯 Objectives

- Analyze customer review data.
- Perform exploratory data analysis (EDA).
- Clean and preprocess textual data.
- Convert text into numerical features using TF-IDF.
- Train machine learning models for sentiment classification.
- Evaluate model performance.
- Predict sentiment for unseen customer reviews.

---

## 📂 Dataset

**Dataset:** Amazon Cell Phones and Accessories Reviews

The dataset contains customer reviews with features such as:

- Review Text
- Rating (1–5)
- Reviewer Information
- Product Information
- Review Summary
- Review Time

The notebook uses the JSON dataset:

```
Cell_Phones_and_Accessories_5.json
```

---

## 🛠 Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- WordCloud

---

## 📊 Exploratory Data Analysis (EDA)

The project performs several exploratory analyses including:

- Review length distribution
- Average number of words per review
- Distribution of customer ratings
- Word Cloud visualization
- Class balance analysis

---

## 🧹 Text Preprocessing

The following NLP preprocessing techniques are applied:

- Convert text to lowercase
- Remove special characters
- Remove punctuation
- Remove stopwords
- Remove high-frequency and low-frequency words
- Tokenization
- Stemming
- Lemmatization

These preprocessing steps improve the quality of textual features before model training.

---

## 🔄 Sentiment Labeling

Customer ratings are converted into sentiment classes.

Example:

| Rating | Sentiment |
|---------|-----------|
| 4–5 | Positive |
| 1–2 | Negative |

*(Neutral ratings are handled according to the preprocessing strategy used in the notebook.)*

---

## ⚙️ Feature Engineering

Text is transformed into numerical vectors using:

- **TF-IDF (Term Frequency–Inverse Document Frequency)**

This representation captures the importance of words across all reviews.

---

## 🤖 Machine Learning Models

The following models are implemented:

### Logistic Regression

- TF-IDF Features
- Sentiment Prediction
- Feature Importance Analysis
- Top Positive Words
- Top Negative Words

### Multinomial Naive Bayes

- TF-IDF Features
- Sentiment Classification
- Performance Evaluation

---

## 📈 Model Evaluation

The models are evaluated using standard classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🔍 Prediction on New Reviews

The notebook demonstrates how to classify unseen customer reviews into:

- Positive
- Negative

using the trained machine learning model.

---

## 📁 Project Structure

```
Data-Mining-Final-Project/
│
├── Data_Mining_Final_Project.ipynb
├── Cell_Phones_and_Accessories_5.json
├── README.md
└── requirements.txt (optional)
```

---

## 🚀 How to Run

1. Clone this repository.

```bash
git clone https://github.com/yourusername/Data-Mining-Final-Project.git
```

2. Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
```

3. Download the dataset.

4. Update the dataset path inside the notebook if necessary.

5. Run the notebook sequentially.

---

## 📌 Project Workflow

```
Load Dataset
      ↓
Exploratory Data Analysis
      ↓
Text Preprocessing
      ↓
Feature Engineering (TF-IDF)
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Sentiment Prediction
```

---

## 📚 Key Learning Outcomes

- Natural Language Processing (NLP)
- Text Cleaning Techniques
- Feature Engineering using TF-IDF
- Sentiment Analysis
- Machine Learning Classification
- Data Visualization
- Model Evaluation

---

## 📄 License

This project is intended for educational and academic purposes.
