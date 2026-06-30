# machine-learning-classification-projects
Built end-to-end classification models for credit risk and marketing analytics using Logistic Regression, Decision Tree, WOE, and IV techniques.
# 🏦 Credit Risk Analysis & Fraud Detection

## 📌 Project Overview
This project implements an end-to-end **Credit Risk & Fraud Detection system** using an industry-standard machine learning pipeline.

The goal is to classify transactions as **Fraud (Bad)** or **Non-Fraud (Good)** while also performing key risk analytics such as **Roll Rate Analysis** and **Vintage Analysis**.

---

## ⚙️ Key Features

### 🔹 Feature Engineering
- Time-based bucketing  
- Weight of Evidence (WOE) Encoding  
- Information Value (IV) Analysis  
- Feature selection based on IV  

### 🔹 Machine Learning Models
- Logistic Regression  
- Decision Tree Classifier  
- K-Fold Cross Validation  

### 🔹 Model Evaluation
- Accuracy, Precision, Recall, F1 Score  
- Confusion Matrix  
- ROC Curve & AUC  

### 🔹 Risk Analytics
- Roll Rate Analysis  
- Vintage (Cohort) Analysis  

---

## 📊 Project Outputs

| File                      | Description                          |
|---------------------------|--------------------------------------|
| iv_values.csv             | Feature importance using IV           |
| transformed_dataset.csv   | Dataset with WOE & IV features        |
| final_model_metrics.csv   | Model performance summary             |
| model_predictions.csv     | Actual vs predicted values            |
| roll_rate.csv             | Time-based default trends             |
| vintage_analysis.csv      | Cohort-wise risk analysis             |

---

## 🧠 Key Insights
- WOE transformation improves model interpretability  
- Logistic Regression performs well on structured data  
- Decision Tree captures non-linear relationships  
- Time-based cohorts reveal changing fraud patterns  

---

## 🚀 Tools & Technologies
- Python (Pandas, NumPy)  
- Scikit-learn  
- Matplotlib  
- Pycharm

---

## 📌 Conclusion
This project demonstrates a complete **Credit Risk Modeling pipeline**, combining **Machine Learning** with **Business Risk Analytics**.

It reflects real-world practices used in banking, NBFCs, and fintech companies.
