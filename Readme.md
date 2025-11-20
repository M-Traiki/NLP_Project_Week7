# Fake News Classification — NLP Project (Week 7)

## 1. Project Overview
This project focuses on classifying news headlines as **real** or **fake** using traditional machine learning models and text preprocessing techniques. Multiple vectorization strategies and classifiers were evaluated to determine the most effective setup.

The primary goal is to:
- Build a robust fake–real text classifier
- Compare multiple ML algorithms
- Evaluate the impact of preprocessing choices
- Identify the best-performing model

The project is implemented in the notebook:
`NLP_Project_Week7.ipynb`

---

## 2. Preprocessing Steps
Text cleaning and normalization proved critical for model performance. The following preprocessing steps were applied:

### **2.1 Basic Cleaning**
- Lowercasing
- Removing/preserving punctuation depending on experiment
- Unicode normalization
- Removal of political entity names  
  *(e.g., “trump”, “obama”, “putin”, “senate”, “congress”, etc.)*
- Removal of headlines with fewer than 4 words
- Standardization of different quote styles  
  (`“ ” „ ’ ‘` → `"` or `'`)

### **2.2 Tokenization**
- Custom tokenizer for word-level vectorizers
- Character-level TF-IDF tokenizer (`analyzer="char"`) for best model

### **2.3 Stopword Removal**
- Experiments were run with and without stopwords
- Removing stopwords typically reduced performance for character n-grams

---

## 3. Results

### **3.1 Model Performance Summary**
Multiple models were evaluated using TF-IDF (word), Bag-of-Words, and TF-IDF (character-level):

| Model                         | Accuracy | F1 Score |
|-------------------------------|----------|----------|
| **Linear SVM + TF-IDF Char**  | **0.972** | **0.972** |
| Logistic Regression + TF-IDF  | 0.946 | 0.945 |
| Multinomial NB + TF-IDF       | 0.942 | 0.939 |
| Random Forest + TF-IDF        | 0.917 | 0.917 |
| XGBoost + TF-IDF              | 0.912 | 0.913 |

The clear winner is the **Linear SVM with character-level TF-IDF**, achieving the strongest accuracy and F1 score.

### **3.2 Preprocessing Variations**
Additional comparisons included:
- Removing punctuation  
- Removing stopwords  
- Lemmatization  
- Minimal preprocessing  

The best performance consistently came from:
- **No lemmatization**  
- **No stemming**  
- **Minimal cleaning**  
- **Character-level TF-IDF (2–5 or 3–6 n-grams)**
---

## 4. Future Work

Several improvements can further enhance the classifier:

### **4.1 BERT Fine-Tuning**
Instead of using an off-the-shelf Transformer model, fine-tune:
- `bert-base-uncased`
- `roberta-base`
- `distilbert-base-uncased`

on your dataset to achieve higher performance.

### **4.2 Expand the Dataset**
Real vs. fake news classification requires:
- A larger and more diverse dataset
- Balanced political, scientific, economic news

### **4.3 Ensemble Models**
Combine:
- Linear SVM (char-level)
- Transformer classifier

to build a hybrid system.

### **4.4 Error Analysis**
Investigate:
- Which headlines are frequently misclassified
- Whether certain linguistic patterns cause errors

### **4.5 Deploy the Model**
Export your SVM model + vectorizer using:
- `joblib`
- Flask/FastAPI web service  
- Streamlit UI

---

## 5. Notebook
Full implementation is available in:

`NLP_Project_Week7.ipynb`

---

