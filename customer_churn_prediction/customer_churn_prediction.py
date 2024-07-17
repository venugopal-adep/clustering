import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="Customer Churn Prediction", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .sub-header {
        font-size: 24px !important;
        font-weight: bold;
        color: #4682B4;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .text-content {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .highlight {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #1E90FF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    data = pd.read_csv(data_url)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
    return data

data = load_data()

# Prepare data for modeling
def prepare_data(data):
    categorical_columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                           "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                           "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
    encoded_data = pd.get_dummies(data, columns=categorical_columns)
    X = encoded_data.drop(["customerID", "Churn"], axis=1)
    y = encoded_data["Churn"].map({"Yes": 1, "No": 0})
    return X, y

X, y = prepare_data(data)

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    return clf, X_test, y_test, y_pred, scaler

clf, X_test, y_test, y_pred, scaler = train_model(X, y)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Main app
def main():
    st.markdown("<h1 class='main-header'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'><strong>Developed by : Venugopal Adep</strong></p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🧮 Explore", "📊 Model Performance", "🔮 Predict", "📈 Feature Importance"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
        st.write(data.head())
        st.write(f"Dataset shape: {data.shape}")

        # Add interactive data exploration
        st.markdown("<h3 class='sub-header'>Data Exploration</h3>", unsafe_allow_html=True)
        feature = st.selectbox("Select a feature to visualize", data.columns)
        if data[feature].dtype == "object":
            fig = px.pie(data, names=feature, title=f"Distribution of {feature}")
        else:
            fig = px.histogram(data, x=feature, title=f"Distribution of {feature}")
        st.plotly_chart(fig)

    with tab2:
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.markdown("<p class='text-content'><strong>Classification Report:</strong></p>", unsafe_allow_html=True)
        st.text(report)

        st.markdown("<h3 class='sub-header'>Confusion Matrix</h3>", unsafe_allow_html=True)
        cm_df = pd.DataFrame(cm, index=["Not Churned", "Churned"], columns=["Not Churned", "Churned"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig)

        # Add ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(scaler.transform(X_test))[:, 1])
        roc_auc = auc(fpr, tpr)
        fig = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                      title=f'Receiver Operating Characteristic (ROC) Curve (AUC = {roc_auc:.2f})')
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig)

    with tab3:
        st.markdown("<h2 class='sub-header'>Predict Churn for a New Customer</h2>", unsafe_allow_html=True)
        st.markdown("<p class='text-content'>Enter customer details:</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

        with col3:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            tenure = st.number_input("Tenure (months)", min_value=0, value=0, step=1)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0, step=0.01)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=0.0, step=0.01)

        if st.button("Predict"):
            new_customer = {
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            new_customer_df = pd.DataFrame([new_customer])
            new_customer_encoded = pd.get_dummies(new_customer_df)
            new_customer_encoded = new_customer_encoded.reindex(columns=X.columns, fill_value=0)
            new_customer_scaled = scaler.transform(new_customer_encoded)

            prediction = clf.predict(new_customer_scaled)[0]
            churn_probability = clf.predict_proba(new_customer_scaled)[0][1]

            if prediction == 1:
                st.error(f"The customer is predicted to churn with a probability of {churn_probability:.2f}")
            else:
                st.success(f"The customer is predicted to stay with a probability of {1-churn_probability:.2f}")

    with tab4:
        st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': clf.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
        fig = px.bar(feature_importance, x='importance', y='feature', orientation='h', 
                     title='Top 10 Most Important Features')
        st.plotly_chart(fig)

        # Add partial dependence plots
        st.markdown("<h3 class='sub-header'>Partial Dependence Plots</h3>", unsafe_allow_html=True)
        from sklearn.inspection import PartialDependenceDisplay
        numeric_features = X.select_dtypes(include=[np.number]).columns
        feature_to_plot = st.selectbox("Select a feature for Partial Dependence Plot", numeric_features)
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(clf, X, [feature_to_plot], ax=ax)
        st.pyplot(fig)

if __name__ == '__main__':
    main()
