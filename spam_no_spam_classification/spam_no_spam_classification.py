import streamlit as st
import pandas as pd
import numpy as npimport streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import requests
import zipfile
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="Email Spam Prediction", page_icon="📧")

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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        file_name = zip_ref.namelist()[0]
        with zip_ref.open(file_name) as file:
            data = pd.read_csv(file, sep="\t", names=["label", "message"])
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    return data

# Prepare data for modeling
@st.cache_data
def prepare_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_features, X_test_features, vectorizer

# Train model
@st.cache_resource
def train_model(X_train_features, y_train):
    clf = MultinomialNB()
    clf.fit(X_train_features, y_train)
    return clf

# Plot wordcloud
def plot_wordcloud(data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

# Main app
def main():
    st.markdown("<h1 class='main-header'>📧 Email Spam Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'><strong>Developed by : Venugopal Adep</strong></p>", unsafe_allow_html=True)

    # Load data
    data = load_data()

    # Prepare data
    X_train, X_test, y_train, y_test, X_train_features, X_test_features, vectorizer = prepare_data(data)

    # Train model
    clf = train_model(X_train_features, y_train)

    # Calculate metrics
    y_pred = clf.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "🧮 Explore Data"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Predict Spam Emails</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            email_input = st.text_area("Enter an email message")
        
        with col2:
            sample_emails = X_test.sample(n=5, random_state=42).tolist()
            sample_emails_with_labels = [(email, label) for email, label in zip(sample_emails, y_test[X_test.isin(sample_emails)])]
            selected_email = st.selectbox("Or select a sample email", [""] + [f"{email} ({'Spam' if label == 1 else 'Not Spam'})" for email, label in sample_emails_with_labels])
        
        if st.button("Predict"):
            if email_input:
                input_email = [email_input]
            elif selected_email:
                input_email = [selected_email.split(" (")[0]]
            else:
                st.warning("Please enter an email message or select a sample email.")
                return
            
            input_features = vectorizer.transform(input_email)
            prediction = clf.predict(input_features)[0]
            
            if prediction == 1:
                st.error("This email is predicted to be spam.")
            else:
                st.success("This email is predicted to be not spam.")

    with tab2:
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.markdown("<p class='text-content'><strong>Classification Report:</strong></p>", unsafe_allow_html=True)
        st.text(report)

        st.markdown("<h3 class='sub-header'>Confusion Matrix</h3>", unsafe_allow_html=True)
        cm_df = pd.DataFrame(cm, index=["Not Spam", "Spam"], columns=["Not Spam", "Spam"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig)

    with tab3:
        st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
        st.write(data.head(10))
        st.write(f"Dataset shape: {data.shape}")

        # Add interactive data exploration
        st.markdown("<h3 class='sub-header'>Data Exploration</h3>", unsafe_allow_html=True)
        
        # Message length distribution
        data['message_length'] = data['message'].apply(len)
        fig = px.histogram(data, x='message_length', color='label', 
                           labels={'message_length': 'Message Length', 'label': 'Email Type'},
                           title='Distribution of Message Lengths')
        st.plotly_chart(fig)

        # Word cloud
        st.markdown("<h3 class='sub-header'>Word Clouds</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            plot_wordcloud(data[data['label'] == 0]['message'], 'Non-Spam Emails')
        with col2:
            plot_wordcloud(data[data['label'] == 1]['message'], 'Spam Emails')

if __name__ == '__main__':
    main()
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import requests
import zipfile
import io

# Download the dataset from the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)

# Extract the data from the ZIP archive
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    file_name = zip_ref.namelist()[0]  # Assumes the ZIP archive contains only one file
    with zip_ref.open(file_name) as file:
        data = pd.read_csv(file, sep="\t", names=["label", "message"])

# Preprocess the data
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text to numerical features
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_features, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_features)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Streamlit application
def main():
    st.write("## Email Spam Prediction")
    st.write("**Developed by : Venugopal Adep**")
    
    # User input for email message
    email_input = st.text_area("Enter an email message")
    
    # Sample email selection
    sample_emails = X_test.sample(n=5, random_state=42).tolist()
    sample_emails_with_labels = [(email, label) for email, label in zip(sample_emails, y_test[X_test.isin(sample_emails)])]
    selected_email = st.selectbox("Or select a sample email", [""] + [f"{email} ({'Spam' if label == 1 else 'Not Spam'})" for email, label in sample_emails_with_labels])
    
    if st.button("Predict"):
        if email_input:
            input_email = [email_input]
        elif selected_email:
            input_email = [selected_email.split(" (")[0]]  # Extract the email message from the selected option
        else:
            st.warning("Please enter an email message or select a sample email.")
            return
        
        # Preprocess the input email
        input_features = vectorizer.transform(input_email)
        
        # Predict using the trained model
        prediction = clf.predict(input_features)[0]
        
        # Display the prediction result
        if prediction == 1:
            st.error("This email is predicted to be spam.")
        else:
            st.success("This email is predicted to be not spam.")
    
    # Display model performance metrics
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    
    # Visualize the confusion matrix using Plotly
    cm_df = pd.DataFrame(cm, index=["Not Spam", "Spam"], columns=["Not Spam", "Spam"])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig)

    # Display the first 10 rows of the dataset
    st.subheader("Dataset Preview")
    st.write(data.head(10))

if __name__ == '__main__':
    main()
