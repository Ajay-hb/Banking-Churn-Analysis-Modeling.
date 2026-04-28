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

# ---------------- BANNER ----------------
st.image("images/banner.png", use_container_width=True)

st.markdown("""
<h1 style='text-align: center;'>🏦 Customer Churn Intelligence System</h1>
<p style='text-align: center;'>🚀 Predict • Analyze • Explore • Retain</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 Navigation")
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
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
        salary = st.number_input("Salary", 0.0, 200000.0, 50000.0)

    with col3:
        num_products = st.selectbox("Products", [1,2,3,4])
        has_card = st.selectbox("Credit Card", ["Yes","No"])
        is_active = st.selectbox("Active Member", ["Yes","No"])

    geography = st.selectbox("Geography", ["France","Germany","Spain"])
    gender = st.selectbox("Gender", ["Male","Female"])

    # Prepare input
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

        # ---------------- AI INSIGHT ----------------
        st.markdown("### 🤖 AI Insight")

        st.info(f"""
        This customer has a churn probability of **{prob:.2%}**.

        Key influencing factors may include:
        - Engagement level
        - Product usage
        - Financial activity
        """)

        # ---------------- RETENTION ADVISOR ----------------
        st.markdown("---")
        st.subheader("🧠 AI Retention Advisor")

        recommendations = []

        if prob > 0.7:
            st.error("🔴 High Risk Customer")

            if is_active == "No":
                recommendations.append("🚨 Re-engage with personalized offers")

            if num_products <= 1:
                recommendations.append("📦 Recommend additional banking products")

            if balance < 50000:
                recommendations.append("💰 Provide financial incentives")

            if age > 50:
                recommendations.append("👤 Offer dedicated support")

        elif prob > 0.4:
            st.warning("🟡 Medium Risk Customer")
            recommendations.append("📧 Send targeted campaigns")
            recommendations.append("🎁 Offer loyalty rewards")

        else:
            st.success("🟢 Low Risk Customer")
            recommendations.append("👍 Maintain engagement")
            recommendations.append("📊 Monitor behavior")

        for rec in recommendations:
            st.write("-", rec)

        # ---------------- DOWNLOAD ----------------
        result_df = input_data.copy()
        result_df["Prediction"] = pred
        result_df["Probability"] = prob

        st.download_button(
            "📥 Download Prediction",
            result_df.to_csv(index=False),
            file_name="prediction.csv"
        )

        # ---------------- AI CHAT ----------------
        st.markdown("---")
        st.subheader("💬 AI Assistant")

        query = st.text_input("Ask something about this customer")

        def ai_response(q):
            if "why" in q.lower():
                return "Churn risk is influenced by engagement, product usage, and financial activity."
            elif "reduce" in q.lower():
                return "Increase engagement, offer products, and provide incentives."
            elif "probability" in q.lower():
                return f"Churn probability is {prob:.2%}"
            else:
                return "Try asking: Why churn? How to reduce?"

        if query:
            st.info(ai_response(query))

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
st.markdown("""
---
<p style='text-align: center;'>Built with ❤️ using Machine Learning & Streamlit</p>
""", unsafe_allow_html=True)
