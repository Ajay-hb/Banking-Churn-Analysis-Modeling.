# 🏦 Customer Churn Prediction in Banking  
### 🚀 Turning Data into Retention Strategy

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Model](https://img.shields.io/badge/Model-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 📖 Business Problem

Banks lose millions due to **customer churn**.

But the real challenge is not knowing:
> ❓ *Which customer will leave next?*

Without prediction:
- Retention strategies are reactive  
- Marketing budgets are wasted  
- Customer lifetime value drops  

---

## 💡 Solution

This project builds a **machine learning system** that:

✔ Predicts churn in advance  
✔ Identifies high-risk customers  
✔ Enables data-driven retention strategies  

---

## 📊 Dataset Overview

- 📦 10,000+ customer records  
- 🧾 Features: Demographics, Financial behavior, Engagement  
- 🎯 Target: `Exited (Churn)`  

---

## 🔍 Key Insights from Data

### 📌 Churn Distribution
![Churn Distribution](images/churn_distribution.png)

👉 Majority stay, but churn segment drives business loss  

---

### 📌 Age Impact
![Age vs Churn](images/age_vs_churn.png)

👉 Older customers show higher churn tendency  

---

### 📌 Engagement Matters
![Active vs Churn](images/active_vs_churn.png)

👉 Inactive users = **high churn risk**  

---

### 📌 Feature Relationships
![Correlation Heatmap](images/correlation_heatmap.png)

👉 Strong signals:
- Age  
- Balance  
- Activity Status  

---

## ⚙️ ML Pipeline

### 1️⃣ Data Processing
- Removed irrelevant features (CustomerId, Surname)  
- Encoded categorical variables  
- Feature scaling applied  

### 2️⃣ Model Development
Models tested:
- Logistic Regression  
- Decision Tree  
- **Random Forest (Best)** 🌟  

---

## 🧠 Final Model Performance

### 📊 Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### 📈 Metrics

- Accuracy: **~84%**  
- Precision: High  
- Recall: Strong  
- F1 Score: Balanced  

---

## 💼 Business Impact

This model enables:

🎯 Targeted retention campaigns  
💰 Reduction in customer loss  
📊 Better marketing ROI  
🤖 Automation of churn prediction  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn  

---

## 📂 Project Structure

    📁 Banking-Churn-Analysis
    │── 📄 Banking_Churn_Analysis.ipynb
    │── 📄 README.md
    │── 📁 images/
    │ ├── churn_distribution.png
    │ ├── age_vs_churn.png
    │ ├── active_vs_churn.png
    │ ├── correlation_heatmap.png
    │ ├── confusion_matrix.png



## 🚀 How to Run


git clone https://github.com/your-username/banking-churn-analysis.git
cd banking-churn-analysis
pip install -r requirements.txt
jupyter notebook

## 🔮 Future Improvements
Hyperparameter tuning (GridSearchCV)
Class imbalance handling (SMOTE)
XGBoost / LightGBM
Deploy using Streamlit

👨‍💻 Author

Ajay Ponnuru
Aspiring Data Scientist | ML Engineer
