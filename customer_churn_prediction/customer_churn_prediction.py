import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data = pd.read_csv(data_url)

# Preprocess the data
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# Perform one-hot encoding for categorical features
categorical_columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                       "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                       "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
encoded_data = pd.get_dummies(data, columns=categorical_columns)

# Split the data into features and target
X = encoded_data.drop(["customerID", "Churn"], axis=1)
y = encoded_data["Churn"].map({"Yes": 1, "No": 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit application
def main():
    st.write("## Customer Churn Prediction")
    st.write("**Developed by : Venugopal Adep**")

    # Display dataset information
    st.subheader("Dataset Information")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")

    # Display model performance metrics
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(report)

    # Visualize the confusion matrix using Plotly
    cm_df = pd.DataFrame(cm, index=["Not Churned", "Churned"], columns=["Not Churned", "Churned"])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig)

    # Sidebar - Predict churn for a new customer
    st.sidebar.subheader("Predict Churn for a New Customer")
    st.sidebar.write("Enter customer details:")

    # Create input fields for customer details in the sidebar
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure = st.sidebar.number_input("Tenure (months)", min_value=0, value=0, step=1)
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=0.0, step=0.01)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=0.0, step=0.01)

    if st.sidebar.button("Predict"):
        # Create a dictionary with the input values
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

        # Convert the dictionary to a DataFrame
        new_customer_df = pd.DataFrame([new_customer])

        # Perform one-hot encoding for categorical features
        new_customer_encoded = pd.get_dummies(new_customer_df, columns=categorical_columns)

        # Align the columns with the training data columns
        missing_cols = set(X.columns) - set(new_customer_encoded.columns)
        for c in missing_cols:
            new_customer_encoded[c] = 0
        new_customer_encoded = new_customer_encoded[X.columns]

        # Make the prediction
        prediction = clf.predict(new_customer_encoded)[0]

        # Display the prediction result
        if prediction == 1:
            st.sidebar.error("The customer is predicted to churn.")
        else:
            st.sidebar.success("The customer is predicted to stay.")

if __name__ == '__main__':
    main()