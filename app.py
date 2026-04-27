import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Churn Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ------------------ LOAD ------------------
model = joblib.load("model.pkl")
df = pd.read_csv("Churn_Modelling.csv")

# ------------------ SIDEBAR ------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Analytics Dashboard"])

# ------------------ PREDICTION PAGE ------------------
if page == "Prediction":

    st.title("🏦 Customer Churn Prediction")
    st.markdown("Predict churn using ML model")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 600)
        age = st.slider("Age", 18, 92, 35)
        tenure = st.slider("Tenure", 0, 10, 3)

    with col2:
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
        salary = st.number_input("Salary", 0.0, 200000.0, 50000.0)

    with col3:
        num_products = st.selectbox("Products", [1,2,3,4])
        has_card = st.selectbox("Credit Card", ["Yes","No"])
        is_active = st.selectbox("Active Member", ["Yes","No"])

    geography = st.selectbox("Geography", ["France","Germany","Spain"])
    gender = st.selectbox("Gender", ["Male","Female"])

    input_data = pd.DataFrame({
        'CreditScore':[credit_score],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_products],
        'HasCrCard':[1 if has_card=="Yes" else 0],
        'IsActiveMember':[1 if is_active=="Yes" else 0],
        'EstimatedSalary':[salary],
        'Geography_Germany':[1 if geography=="Germany" else 0],
        'Geography_Spain':[1 if geography=="Spain" else 0],
        'Gender_Male':[1 if gender=="Male" else 0]
    })

    if st.button("🚀 Predict"):

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(f"⚠️ High Churn Risk ({prob:.2%})")
        else:
            st.success(f"✅ Likely to Stay ({1-prob:.2%})")

        st.progress(int(prob * 100))

# ------------------ ANALYTICS DASHBOARD ------------------
else:

    st.title("📊 Analytics Dashboard")

    st.subheader("Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Exited", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Age vs Churn")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Exited", y="Age", data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Active Members vs Churn")
    fig3, ax3 = plt.subplots()
    sns.countplot(x="IsActiveMember", hue="Exited", data=df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Correlation Heatmap")
    fig4, ax4 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), ax=ax4)
    st.pyplot(fig4)

    # Feature importance
    st.subheader("Top Feature Importance")

    X = df.drop(['RowNumber','CustomerId','Surname','Exited'], axis=1)
    X = pd.get_dummies(X, drop_first=True)

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:10]

    fig5, ax5 = plt.subplots()
    feat_imp.plot(kind='barh', ax=ax5)
    st.pyplot(fig5)
