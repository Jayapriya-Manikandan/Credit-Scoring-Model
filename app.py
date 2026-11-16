
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)

# ==========================================
# 2. LOAD SAVED MODEL FILES
# ==========================================

@st.cache_resource
def load_models():
    model = joblib.load("best_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    comparison = pd.read_csv("model_comparison.csv", index_col=0)

    return model, feature_columns, comparison


try:
    model, feature_columns, comparison = load_models()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# ==========================================
# 3. CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
}

.main-title {
    font-size: 38px;
    font-weight: bold;
    color: #58a6ff;
}

.subtitle {
    color: #aab4c3;
    font-size: 17px;
}

.metric-card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30363d;
    text-align: center;
}

.result-card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #30363d;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("💳 Credit Scoring")
st.sidebar.caption("Machine Learning Application")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Credit Risk Prediction",
        "Model Performance",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Credit Default Prediction using Machine Learning")


# ==========================================
# 5. HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<p class="main-title">Credit Scoring Model</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">An intelligent machine learning system '
        'for predicting credit card payment default risk.</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records", "30,000")

    with col2:
        st.metric("ML Algorithms", "3")

    with col3:
        st.metric("Best Model", "Decision Tree")

    st.markdown("### Project Overview")

    st.write(
        "This application uses machine learning algorithms to analyze "
        "customer financial information and predict whether a customer "
        "may default on their credit card payment next month."
    )

    st.markdown("### Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info("📊 Financial Data Analysis")
        st.info("🤖 Machine Learning Prediction")

    with col2:
        st.info("📈 Model Performance Comparison")
        st.info("⚡ Instant Risk Assessment")

    st.markdown("---")

    st.caption(
        "Educational project developed as part of the CodeAlpha "
        "Machine Learning Internship."
    )


# ==========================================
# 6. CREDIT RISK PREDICTION PAGE
# ==========================================

elif page == "Credit Risk Prediction":

    st.title("🔍 Credit Risk Prediction")

    st.write(
        "Enter the customer's financial information "
        "to predict the possibility of credit card payment default."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        st.subheader("Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            limit_bal = st.number_input(
                "Credit Limit",
                min_value=10000,
                max_value=1000000,
                value=200000,
                step=10000
            )

        with col2:
            sex = st.selectbox(
                "Gender",
                [1, 2],
                format_func=lambda x: (
                    "Male" if x == 1 else "Female"
                )
            )

        with col3:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )

        col1, col2 = st.columns(2)

        with col1:
            education = st.selectbox(
                "Education",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Graduate School",
                    2: "University",
                    3: "High School",
                    4: "Others"
                }[x]
            )

        with col2:
            marriage = st.selectbox(
                "Marital Status",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Married",
                    2: "Single",
                    3: "Others"
                }[x]
            )

        st.markdown("---")
        st.subheader("Repayment History")

        st.caption(
            "Payment status: -2 = No consumption, -1 = Paid duly, "
            "0 = Revolving credit, 1 or above = Payment delay."
        )

        payment_status = {}

        col1, col2, col3 = st.columns(3)

        pay_columns = [
            ("PAY_0", "September"),
            ("PAY_2", "August"),
            ("PAY_3", "July"),
            ("PAY_4", "June"),
            ("PAY_5", "May"),
            ("PAY_6", "April")
        ]

        for i, (column, month) in enumerate(pay_columns):
            with [col1, col2, col3][i % 3]:
                payment_status[column] = st.selectbox(
                    f"{month} Payment Status",
                    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                    index=2,
                    key=column
                )

        st.markdown("---")
        st.subheader("Bill Amounts")

        bill_amounts = {}

        bill_columns = [
            ("BILL_AMT1", "September"),
            ("BILL_AMT2", "August"),
            ("BILL_AMT3", "July"),
            ("BILL_AMT4", "June"),
            ("BILL_AMT5", "May"),
            ("BILL_AMT6", "April")
        ]

        for i, (column, month) in enumerate(bill_columns):
            with [col1, col2, col3][i % 3]:
                bill_amounts[column] = st.number_input(
                    f"{month} Bill Amount",
                    min_value=-1000000,
                    max_value=1000000,
                    value=5000,
                    step=1000,
                    key=column
                )

        st.markdown("---")
        st.subheader("Previous Payment Amounts")

        payment_amounts = {}

        payment_columns = [
            ("PAY_AMT1", "September"),
            ("PAY_AMT2", "August"),
            ("PAY_AMT3", "July"),
            ("PAY_AMT4", "June"),
            ("PAY_AMT5", "May"),
            ("PAY_AMT6", "April")
        ]

        for i, (column, month) in enumerate(payment_columns):
            with [col1, col2, col3][i % 3]:
                payment_amounts[column] = st.number_input(
                    f"{month} Payment Amount",
                    min_value=0,
                    max_value=1000000,
                    value=2000,
                    step=1000,
                    key=column
                )

        submitted = st.form_submit_button(
            "🔍 Predict Credit Risk",
            use_container_width=True
        )

    if submitted:

        input_data = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **payment_status,
            **bill_amounts,
            **payment_amounts
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠️ Higher Default Risk")
                st.write(
                    "The model predicts that the customer may "
                    "default on the next month's payment."
                )
            else:
                st.success("✅ Lower Default Risk")
                st.write(
                    "The model predicts that the customer may "
                    "not default on the next month's payment."
                )

        with col2:
            st.metric(
                "Predicted Default Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

        st.caption(
            "This is a machine learning estimate, not a guaranteed outcome."
        )


# ==========================================
# 7. MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Comparison of the three machine learning algorithms "
        "trained for credit default prediction."
    )

    st.markdown("---")

    st.subheader("Model Comparison Table")

    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")

    metric = st.selectbox(
        "Select Evaluation Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ]
    )

    chart_data = comparison.reset_index()
    chart_data = chart_data.rename(
        columns={"index": "Model"}
    )

    fig = px.bar(
        chart_data,
        x="Model",
        y=metric,
        color="Model",
        title=f"{metric} Comparison",
        text_auto=".3f",
        template="plotly_dark"
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    best_model_name = comparison["F1-Score"].idxmax()
    best_score = comparison["F1-Score"].max()

    st.success(
        f"Best Model: {best_model_name} "
        f"(F1-Score: {best_score:.4f})"
    )


# ==========================================
# 8. ABOUT PAGE
# ==========================================

elif page == "About":

    st.title("ℹ️ About This Project")

    st.markdown("---")

    st.subheader("Credit Scoring Model")

    st.write(
        "This project predicts whether a credit card customer "
        "may default on their payment in the following month "
        "using historical financial and repayment information."
    )

    st.subheader("Machine Learning Algorithms")

    st.write("1. Logistic Regression")
    st.write("2. Decision Tree")
    st.write("3. Random Forest")

    st.subheader("Dataset")

    st.write(
        "Default of Credit Card Clients Dataset "
        "(UCI Machine Learning Repository)."
    )

    st.subheader("Technology Stack")

    st.write(
        "Python, Pandas, Scikit-learn, Streamlit, "
        "Plotly and Joblib."
    )

    st.markdown("---")

    st.info(
        "Internship Project – CodeAlpha Machine Learning Internship."
    )

    st.warning(
        "Disclaimer: This application is developed for educational "
        "and demonstration purposes only. It should not be used as "
        "the sole basis for real financial or credit approval decisions."
    )


# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Credit Scoring Model | Machine Learning Project | CodeAlpha"
)
