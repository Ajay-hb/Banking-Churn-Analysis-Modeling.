# 🏦 Customer Churn Prediction in Banking

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Model](https://img.shields.io/badge/Model-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 📌 Overview

Customer churn is a critical problem in the banking industry, leading to significant revenue loss and reduced customer lifetime value.

This project builds a **machine learning pipeline** to predict whether a customer is likely to leave the bank, enabling proactive and data-driven retention strategies.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA) to understand churn behavior  
- Identify key factors influencing customer churn  
- Build and evaluate machine learning models  
- Develop a reliable churn prediction system  
- Translate insights into business decisions  

---

## 📊 Dataset Description

The dataset contains **10,000 customer records** with the following features:

### 🔹 Customer Information
- Credit Score  
- Geography  
- Gender  
- Age  

### 🔹 Banking Details
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  

### 🔹 Activity & Financial
- Is Active Member  
- Estimated Salary  

### 🎯 Target Variable
- `Exited` → 1 (Churn), 0 (Retained)

---

## 🔍 Exploratory Data Analysis

### 📌 Key Findings

- Inactive customers show significantly higher churn rates  
- Older customers are more likely to churn  
- Customers with fewer products tend to leave  
- Balance alone is not a strong indicator of churn  

---

## ⚙️ Model Workflow

### 🔄 End-to-End Pipeline

```
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Prediction & Insights
```

---

### 🔹 Step 1: Data Preprocessing
- Removed irrelevant columns (`CustomerId`, `Surname`)  
- Checked and handled missing values  
- Converted categorical variables using One-Hot Encoding  

---

### 🔹 Step 2: Feature Engineering
- Created model-ready dataset  
- Ensured no data leakage  
- Structured features for ML algorithms  

---

### 🔹 Step 3: Train-Test Split
- 80% Training Data  
- 20% Testing Data  

---

### 🔹 Step 4: Model Building

Models evaluated:
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Final Selected Model:** Random Forest Classifier  

---

### 🔹 Step 5: Model Evaluation

Evaluation metrics used:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  

---

## 📈 Model Performance

| Metric        | Score |
|--------------|------|
| Accuracy      | ~0.84 |
| Precision     | High |
| Recall        | Balanced |

The model demonstrates strong ability to distinguish between churned and retained customers.

---

## 🧠 Key Insights

- Customer engagement is the strongest predictor of churn  
- Increasing product adoption improves retention  
- Predictive modeling enables proactive decision-making  
- Data-driven strategies outperform reactive approaches  

---

## 💼 Business Impact

This solution helps banks to:

- Identify high-risk customers early  
- Reduce churn-related revenue loss  
- Improve customer retention strategies  
- Optimize marketing efforts  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📂 Project Structure

```
Banking-Churn-Analysis/
│── Banking_Churn_Analysis.ipynb
│── README.md
```

---

## 🚀 How to Run the Project

```bash
# Clone repository
git clone https://github.com/your-username/banking-churn-analysis.git

# Navigate to folder
cd banking-churn-analysis

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
```

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV)  
- Handle class imbalance (SMOTE)  
- Try advanced models (XGBoost, LightGBM)  
- Deploy using Streamlit for real-time predictions  

---

## 👨‍💻 Author

**Ajay Ponnuru**  
Aspiring Data Scientist | Machine Learning Enthusiast  

---

## 📌 Conclusion

This project demonstrates how machine learning can be applied to solve a real-world business problem by transforming raw customer data into actionable insights.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
