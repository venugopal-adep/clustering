import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    data = pd.read_excel(url)
    
    # Data preprocessing
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['InvoiceYear'] = data['InvoiceDate'].dt.year
    data['InvoiceMonth'] = data['InvoiceDate'].dt.month
    
    # Remove rows with missing CustomerID
    data = data.dropna(subset=['CustomerID'])
    
    # Aggregate data by customer
    customer_data = data.groupby('CustomerID').agg({
        'InvoiceNo': 'count',
        'TotalPrice': 'sum',
        'InvoiceYear': 'max',
        'InvoiceMonth': 'max',
        'InvoiceDate': 'max'
    }).reset_index()
    
    customer_data.columns = ['CustomerID', 'Frequency', 'MonetaryValue', 'LastPurchaseYear', 'LastPurchaseMonth', 'LastPurchaseDate']
    
    # Calculate Recency
    last_date = data['InvoiceDate'].max()
    customer_data['Recency'] = (last_date - customer_data['LastPurchaseDate']).dt.days
    
    # Remove 'LastPurchaseDate' as it's no longer needed
    customer_data = customer_data.drop('LastPurchaseDate', axis=1)
    
    # Impute missing values
    customer_data = customer_data.fillna(customer_data.mean())
    
    return customer_data

def main():
    st.set_page_config(page_title="Customer Segmentation Demo", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #ff6e40;
        color: white !important;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
        color: white !important;
    }
    .quiz-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🛍️ Customer Segmentation Analysis")
    
    tabs = st.tabs(["📚 Learn", "🧪 Experiment", "📊 Visualization", "🧠 Quiz"])
    
    with tabs[0]:
        learn_section()
    
    with tabs[1]:
        experiment_section()
    
    with tabs[2]:
        visualization_section()
    
    with tabs[3]:
        quiz_section()

def learn_section():
    st.header("Understanding Customer Segmentation")
    
    st.write("""
    Customer segmentation is the process of dividing customers into groups based on common characteristics 
    so companies can market to each group effectively and appropriately.

    Key aspects of customer segmentation include:
    """)

    aspects = {
        "Data Collection": "Gathering relevant information about customers",
        "Feature Selection": "Choosing the most informative customer attributes",
        "Clustering Algorithms": "Using machine learning to group similar customers",
        "Interpretation": "Understanding the characteristics of each segment",
        "Strategy Development": "Creating targeted marketing strategies for each segment",
        "Evaluation": "Assessing the effectiveness of segmentation and adjusting as needed"
    }

    for aspect, description in aspects.items():
        st.subheader(f"{aspect}")
        st.write(description)

    st.write("""
    Customer segmentation is important because:
    - It allows for more personalized marketing strategies
    - It helps in identifying high-value customer groups
    - It can improve customer satisfaction and loyalty
    - It enables more efficient allocation of marketing resources
    """)

def experiment_section():
    st.header("🧪 Experiment with Customer Segmentation")

    data = load_data()

    st.subheader("Data Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")

    # Prepare the data
    X = data[['Recency', 'Frequency', 'MonetaryValue']]
    
    if X.shape[0] == 0:
        st.error("No data available for clustering. Please check the dataset.")
        return

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    n_clusters = st.slider("Select number of clusters", 2, 10, 4)

    if st.button("Perform Clustering"):
        with st.spinner("Clustering in progress... This may take a moment."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Add cluster labels to the original dataframe
            data['Cluster'] = cluster_labels

            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Visualization of clusters
            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=cluster_labels,
                             labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                             title='Customer Segments')
            st.plotly_chart(fig)

            # Cluster profiles
            st.subheader("Cluster Profiles")
            cluster_profiles = data.groupby('Cluster').mean()
            st.write(cluster_profiles)

            # Radar chart for cluster comparison
            categories = ['Recency', 'Frequency', 'MonetaryValue']
            fig = go.Figure()

            for i in range(n_clusters):
                values = cluster_profiles.iloc[i][categories].values.tolist()
                values += values[:1]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=f'Cluster {i}'
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True)
                ),
                showlegend=True,
                title="Cluster Comparison"
            )

            st.plotly_chart(fig)

def visualization_section():
    st.header("📊 Visualizing Customer Data")

    data = load_data()

    # Recency distribution
    st.subheader("Recency Distribution")
    fig = px.histogram(data, x="Recency", nbins=50, title="Distribution of Customer Recency")
    st.plotly_chart(fig)

    # Frequency vs. Monetary Value
    st.subheader("Frequency vs. Monetary Value")
    fig = px.scatter(data, x="Frequency", y="MonetaryValue", 
                     color="Recency", hover_data=['CustomerID'],
                     title="Frequency vs. Monetary Value")
    st.plotly_chart(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = data[['Recency', 'Frequency', 'MonetaryValue']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    title="Correlation Heatmap of Customer Features")
    st.plotly_chart(fig)

    # Monthly purchase trends
    st.subheader("Monthly Purchase Trends")
    monthly_data = data.groupby(['LastPurchaseYear', 'LastPurchaseMonth'])['MonetaryValue'].sum().reset_index()
    monthly_data['Date'] = pd.to_datetime(monthly_data['LastPurchaseYear'].astype(str) + '-' + monthly_data['LastPurchaseMonth'].astype(str))
    fig = px.line(monthly_data, x='Date', y='MonetaryValue', title='Monthly Purchase Trends')
    st.plotly_chart(fig)

def quiz_section():
    st.header("🧠 Test Your Knowledge")
    
    questions = [
        {
            "question": "What is the main goal of customer segmentation?",
            "options": [
                "To increase the number of customers",
                "To divide customers into groups with similar characteristics",
                "To predict customer churn",
                "To set prices for products"
            ],
            "correct": "To divide customers into groups with similar characteristics",
            "explanation": "The main goal of customer segmentation is to divide customers into groups based on common characteristics. This allows businesses to tailor their marketing strategies and offerings to each group more effectively."
        },
        {
            "question": "Which of the following is NOT typically a feature used in RFM analysis for customer segmentation?",
            "options": [
                "Recency",
                "Frequency",
                "Monetary Value",
                "Customer Age"
            ],
            "correct": "Customer Age",
            "explanation": "While Recency, Frequency, and Monetary Value are the three main components of RFM analysis, Customer Age is not typically part of this specific segmentation method. RFM focuses on customer behavior rather than demographic information."
        },
        {
            "question": "What does the 'Recency' metric represent in customer segmentation?",
            "options": [
                "How recently a customer made a purchase",
                "How frequently a customer makes purchases",
                "How much money a customer spends",
                "How long a customer has been with the company"
            ],
            "correct": "How recently a customer made a purchase",
            "explanation": "In RFM analysis, 'Recency' refers to how recently a customer made their last purchase. It's usually measured in days since the last transaction and is an important indicator of customer engagement."
        },
        {
            "question": "Why is it important to scale features before applying clustering algorithms?",
            "options": [
                "To make the data more colorful",
                "To ensure all features contribute equally to the clustering",
                "To reduce the number of features",
                "To increase the number of clusters"
            ],
            "correct": "To ensure all features contribute equally to the clustering",
            "explanation": "Scaling features (e.g., using StandardScaler) is important before applying clustering algorithms to ensure that all features contribute equally to the clustering process. Without scaling, features with larger magnitudes would have a disproportionate impact on the clustering results."
        }
    ]
    
    for i, q in enumerate(questions, 1):
        st.subheader(f"Question {i}")
        with st.container():
            st.write(q["question"])
            answer = st.radio("Select your answer:", q["options"], key=f"q{i}")
            if st.button("Check Answer", key=f"check{i}"):
                if answer == q["correct"]:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Incorrect. The correct answer is: {q['correct']}")
                st.info(f"Explanation: {q['explanation']}")
            st.write("---")

if __name__ == "__main__":
    main()
