import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="Customer Segmentation Explorer", page_icon="👥")

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

# Helper functions
@st.cache_data
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/venugopal-adep/clustering/main/clustering_usecase_income_spend/customer_data.csv")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[["Annual Income", "Spending Score"]])
    return data, scaled_data

def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

# Main app
def main():
    st.markdown("<h1 class='main-header'>👥 Customer Segmentation using Clustering</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Developed by: Venugopal Adep</p>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>This application allows you to segment customers based on their annual income and spending score.</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding Customer Segmentation</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>What is Customer Segmentation?</h3>
        <p class='text-content'>
        Customer segmentation is the process of dividing customers into groups based on common characteristics. In this case, we're using:

        1. Annual Income: How much a customer earns per year
        2. Spending Score: A score (1-100) assigned to a customer based on their spending behavior

        By grouping similar customers together, businesses can:
        - Tailor marketing strategies
        - Improve customer service
        - Develop targeted products or services
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore Customer Segments</h2>", unsafe_allow_html=True)
        
        # Load the customer data
        data, scaled_data = load_data()

        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display the raw data
            if st.checkbox("Show raw data"):
                st.write(data)

            # Get user input for the number of clusters
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
            
            if st.button('Perform Clustering'):
                # Perform clustering
                labels = perform_clustering(scaled_data, n_clusters)
                
                # Add the cluster labels to the data
                data["Cluster"] = labels
                st.session_state.data = data
                st.session_state.n_clusters = n_clusters
        
        with col2:
            if 'data' in st.session_state:
                # Display the clustering results
                st.subheader("Clustering Results")
                fig = px.scatter(st.session_state.data, x="Annual Income", y="Spending Score", color="Cluster", 
                                 hover_data=["CustomerID"], color_continuous_scale=px.colors.qualitative.Bold)
                st.plotly_chart(fig, use_container_width=True)

                # Display additional insights
                st.subheader("Cluster Insights")
                for i in range(st.session_state.n_clusters):
                    cluster_data = st.session_state.data[st.session_state.data["Cluster"] == i]
                    st.markdown(f"""
                    <div class='highlight'>
                    <h4>Cluster {i+1}:</h4>
                    <ul>
                    <li>Number of customers: {len(cluster_data)}</li>
                    <li>Average annual income: ${cluster_data['Annual Income'].mean():.2f}</li>
                    <li>Average spending score: {cluster_data['Spending Score'].mean():.2f}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What are the two main features used for customer segmentation in this application?",
                "options": [
                    "Age and Gender",
                    "Annual Income and Spending Score",
                    "Location and Purchase History",
                    "Customer ID and Cluster Number"
                ],
                "correct": 1,
                "explanation": "This application uses Annual Income and Spending Score to segment customers. These two features provide insights into both the earning potential and spending behavior of customers."
            },
            {
                "question": "What is the purpose of customer segmentation?",
                "options": [
                    "To rank customers from best to worst",
                    "To assign unique IDs to each customer",
                    "To group similar customers together for targeted strategies",
                    "To predict future purchases"
                ],
                "correct": 2,
                "explanation": "Customer segmentation aims to group similar customers together. This allows businesses to develop targeted strategies, improve customer service, and tailor their offerings to specific customer groups."
            },
            {
                "question": "What algorithm is used for clustering in this application?",
                "options": [
                    "Linear Regression",
                    "Random Forest",
                    "K-Means",
                    "Neural Network"
                ],
                "correct": 2,
                "explanation": "This application uses the K-Means algorithm for clustering. K-Means is a popular algorithm for customer segmentation as it can group data points into a specified number of clusters based on their similarity."
            }
        ]

        for i, q in enumerate(questions):
            st.markdown(f"<p class='text-content'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
            user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
            
            if st.button("Check Answer", key=f"check{i}"):
                if q['options'].index(user_answer) == q['correct']:
                    st.success("Correct! 🎉")
                else:
                    st.error("Incorrect. Try again! 🤔")
                st.info(q['explanation'])
            st.markdown("---")

if __name__ == "__main__":
    main()
