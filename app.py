import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Intelligence System", layout="wide")

# ---------------- LOAD ----------------
model = joblib.load("model.pkl")
df = pd.read_csv("Churn_Modelling.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 Churn Intelligence System")
page = st.sidebar.radio("Navigation", ["Prediction", "Analytics", "Data Explorer"])

# ===================== PREDICTION =====================
if page == "Prediction":

    st.title("🏦 Customer Churn Prediction")

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

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"⚠️ High Churn Risk ({prob:.2%})")
        else:
            st.success(f"✅ Likely to Stay ({1-prob:.2%})")

        st.progress(int(prob * 100))

        # Download prediction
        result_df = input_data.copy()
        result_df["Prediction"] = pred
        result_df["Probability"] = prob

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction",
            data=csv,
            file_name="prediction.csv",
            mime="text/csv"
        )

# ===================== ANALYTICS =====================
elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    st.subheader("Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Exited", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Age vs Churn")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Exited", y="Age", data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Feature Importance")

    X = df.drop(['RowNumber','CustomerId','Surname','Exited'], axis=1)
    X = pd.get_dummies(X, drop_first=True)

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:10]

    fig3, ax3 = plt.subplots()
    feat_imp.plot(kind='barh', ax=ax3)
    st.pyplot(fig3)

# ===================== DATA EXPLORER =====================
else:

    st.title("🔍 Data Explorer")

    st.subheader("Filter Data")

    # Filters
    age_filter = st.slider("Age Range", int(df.Age.min()), int(df.Age.max()), (20,50))
    balance_filter = st.slider("Balance Range", int(df.Balance.min()), int(df.Balance.max()), (0,150000))

    filtered_df = df[
        (df["Age"] >= age_filter[0]) & (df["Age"] <= age_filter[1]) &
        (df["Balance"] >= balance_filter[0]) & (df["Balance"] <= balance_filter[1])
    ]

    st.write("Filtered Data Shape:", filtered_df.shape)

    st.dataframe(filtered_df.head(20))

    # Download filtered data
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Filtered Data",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )
