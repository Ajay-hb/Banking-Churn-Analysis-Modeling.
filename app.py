import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Prediction System", layout="wide")

# ---------------- LOAD ----------------
model = joblib.load("model.pkl")
df = pd.read_csv("Churn_Modelling.csv")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center;'>🏦 Customer Churn Prediction System</h1>
<p style='text-align: center;'>📊 Predict • Analyze • Improve Retention</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Analytics", "Data Explorer"])

# ===================== PREDICTION =====================
if page == "Prediction":

    st.subheader("🔮 Predict Customer Churn")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 600)
        age = st.slider("Age", 18, 92, 35)
        tenure = st.slider("Tenure", 0, 10, 3)

    with col2:
        balance = st.slider("Balance (₹)", 0, 2500000, 50000)
        salary = st.slider("Estimated Salary (₹)", 0, 300000, 50000)

    with col3:
        num_products = st.selectbox("Products", [1,2,3,4])
        has_card = st.selectbox("Credit Card", ["Yes","No"])
        is_active = st.selectbox("Active Member", ["Yes","No"])

    geography = st.selectbox("Geography", ["France","Germany","Spain"])
    gender = st.selectbox("Gender", ["Male","Female"])

    # Input dataframe
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

    if st.button("🚀 Predict Churn"):

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("📈 Prediction Result")

        col4, col5 = st.columns(2)

        with col4:
            if pred == 1:
                st.error("⚠️ High Churn Risk")
            else:
                st.success("✅ Customer Likely to Stay")

        with col5:
            st.metric("Churn Probability", f"{prob:.2%}")

        st.progress(int(prob * 100))

        # ---------------- IMPROVEMENT SUGGESTIONS ----------------
        st.markdown("---")
        st.subheader("📊 Key Areas to Improve (Retention Strategy)")

        improvements = []

        if is_active == "No":
            improvements.append("Increase customer engagement (offers, notifications, app usage)")

        if num_products <= 1:
            improvements.append("Encourage use of additional banking products")

        if balance < 50000:
            improvements.append("Promote savings or investment plans to increase balance")

        if credit_score < 500:
            improvements.append("Offer credit improvement solutions")

        if age > 50:
            improvements.append("Provide personalized support and relationship management")

        if geography == "Germany":
            improvements.append("Apply targeted retention strategies for this region")

        if improvements:
            for imp in improvements:
                st.write("•", imp)
        else:
            st.success("Customer profile is strong. Maintain current engagement strategy.")

        # ---------------- FEATURE IMPORTANCE ----------------
        st.markdown("---")
        st.subheader("📊 Top Influencing Features")

        X = df.drop(['RowNumber','CustomerId','Surname','Exited'], axis=1)
        X = pd.get_dummies(X, drop_first=True)

        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        for f in feat_imp.head(5).index:
            st.write("•", f)

        # ---------------- DOWNLOAD ----------------
        result_df = input_data.copy()
        result_df["Prediction"] = pred
        result_df["Probability"] = prob

        st.download_button(
            "📥 Download Prediction",
            result_df.to_csv(index=False),
            file_name="prediction.csv"
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

    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]

    fig3, ax3 = plt.subplots()
    feat_imp.plot(kind='barh', ax=ax3)
    st.pyplot(fig3)

# ===================== DATA EXPLORER =====================
else:

    st.title("🔍 Data Explorer")

    age_range = st.slider("Age Range", int(df.Age.min()), int(df.Age.max()), (20,50))
    balance_range = st.slider("Balance Range", int(df.Balance.min()), int(df.Balance.max()), (0,150000))

    filtered_df = df[
        (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) &
        (df["Balance"] >= balance_range[0]) & (df["Balance"] <= balance_range[1])
    ]

    st.write("Filtered Data:", filtered_df.shape)
    st.dataframe(filtered_df.head(20))

    st.download_button(
        "📥 Download Filtered Data",
        filtered_df.to_csv(index=False),
        file_name="filtered_data.csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")
