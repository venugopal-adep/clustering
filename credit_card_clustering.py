#Dataset : https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration and custom CSS
st.set_page_config(page_title="Credit Card Customer Clustering", layout="wide")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 24px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #e6f3ff;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #3498db;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Interactive Credit Card Customer Clustering Demo")

    # Sidebar
    with st.sidebar:
        st.header("Clustering Settings")
        algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "Agglomerative Clustering", "DBSCAN"])
        n_clusters = st.slider("Number of Clusters", 2, 10, 3) if algorithm != "DBSCAN" else None
        eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5) if algorithm == "DBSCAN" else None
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5) if algorithm == "DBSCAN" else None

    # Load and process data
    X, feature_names, credit_card_data = load_and_process_data()

    # Perform clustering
    labels = cluster_data(X, algorithm, n_clusters, eps, min_samples)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualization", "📘 Interpretation", "🧠 Algorithms", "🎓 Quiz", "📋 Dataset"])

    with tab1:
        st.header("3D Visualization of Clustering Results")
        fig = create_3d_scatter(X, labels, feature_names, f"{algorithm} Clustering")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("How to Interpret the Visualization")
        interpretation_guide()

    with tab3:
        st.header("Understanding the Clustering Algorithms")
        explain_algorithms()

    with tab4:
        st.header("Test Your Understanding")
        quiz()

    with tab5:
        st.header("Dataset Overview")
        dataset_overview(credit_card_data)

def load_and_process_data():
    # Load the credit card dataset
    credit_card_data = pd.read_csv('credit_card_dataset1.csv')
    
    # Select features for clustering
    features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
    X = credit_card_data[features]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for visualization
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled)
    
    return X_3d, features, credit_card_data

def cluster_data(X, algorithm, n_clusters=None, eps=None, min_samples=None):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:  # DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def create_3d_scatter(X, labels, feature_names, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.8),
        text=[f"Cluster: {l}" for l in labels],
        hoverinfo='text'
    )])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", zaxis_title="PCA Component 3"),
        height=700,
        margin=dict(r=0, b=0, l=0, t=40)
    )
    return fig

def interpretation_guide():
    st.markdown("""
    <div style="background-color: #e6f3ff; padding: 20px; border-radius: 10px;">
    <h3 style="color: #2c3e50;">Key Points:</h3>
    <ul>
        <li>Each point represents a credit card customer from the dataset.</li>
        <li>Colors indicate the cluster assigned by the chosen algorithm.</li>
        <li>Proximity of points suggests similarity in financial behavior.</li>
    </ul>

    <h3 style="color: #2c3e50;">Example:</h3>
    <p>If you see a tight group of blue points in one area, this could represent a cluster of customers with similar spending and payment patterns.</p>

    <h3 style="color: #2c3e50;">What to Look For:</h3>
    <ol>
        <li><strong>Well-separated clusters:</strong> Distinct groups of colors might indicate that the algorithm has effectively separated different customer segments based on their financial behavior.</li>
        <li><strong>Mixed clusters:</strong> Areas where colors are intermingled could suggest overlapping financial patterns between different customer groups.</li>
        <li><strong>Outliers:</strong> Isolated points might represent customers with unusual spending or payment behaviors.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def explain_algorithms():
    st.subheader("K-Means")
    st.markdown("""
    <div style="background-color: #e8f8f5; padding: 20px; border-radius: 10px;">
    <p>K-Means tries to find a specified number of cluster centers and assign each customer to the nearest center.</p>
    <p><strong>Analogy:</strong> Imagine organizing credit card customers into groups based on their financial behavior. K-Means is like deciding on a number of customer segments (clusters) and then putting each customer in the segment they're most similar to.</p>
    <p><strong>Example:</strong> If we set K=3, K-Means might create clusters for "high spenders", "average users", and "low activity" customers based on their credit card usage patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Agglomerative Clustering")
    st.markdown("""
    <div style="background-color: #fef9e7; padding: 20px; border-radius: 10px;">
    <p>This algorithm starts with each customer as their own cluster and progressively merges the closest clusters.</p>
    <p><strong>Analogy:</strong> Think of grouping customers like building a family tree. We start with individuals, then group similar pairs, then extend to larger groups with similar characteristics.</p>
    <p><strong>Example:</strong> It might start by grouping very similar customers (like two with almost identical spending patterns), then gradually combine these small groups into larger customer segments.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("DBSCAN")
    st.markdown("""
    <div style="background-color: #f4ecf7; padding: 20px; border-radius: 10px;">
    <p>DBSCAN groups together customers that are closely packed in the feature space, marking customers in low-density regions as outliers.</p>
    <p><strong>Analogy:</strong> Imagine looking at a map of customer behaviors. DBSCAN would identify dense groups of customers with similar patterns and label isolated customers as outliers.</p>
    <p><strong>Example:</strong> In our credit card dataset, DBSCAN might identify dense clusters of customers with similar spending and payment behaviors while labeling customers with unique financial patterns as outliers.</p>
    </div>
    """, unsafe_allow_html=True)

def quiz():
    questions = [
        {
            "question": "What does each point in the 3D visualization represent?",
            "options": ["A cluster center", "A credit card customer", "A transaction", "A feature"],
            "correct": 1,
            "explanation": "Each point in the 3D visualization represents a single credit card customer from the dataset. The position of the point in 3D space is determined by the customer's financial behavior, with similar customers appearing closer together."
        },
        {
            "question": "In K-Means clustering, what does 'K' represent?",
            "options": ["Number of iterations", "Number of clusters", "Number of customers", "Number of features"],
            "correct": 1,
            "explanation": "In K-Means clustering, 'K' represents the number of clusters. It's the number of groups you want the algorithm to divide your data into. For example, if K=3, the algorithm will try to organize the credit card customers into 3 distinct groups based on their financial behavior."
        },
        {
            "question": "Which clustering algorithm is best for detecting outliers?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN"],
            "correct": 2,
            "explanation": "DBSCAN is particularly good at detecting outliers. Unlike K-Means and Agglomerative Clustering, which assign every point to a cluster, DBSCAN can label points in low-density regions as outliers. This makes it useful for identifying unusual customers with unique financial patterns in our credit card dataset."
        }
    ]

    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        user_answer = st.radio(q["question"], q["options"])
        if st.button(f"Check Answer {i+1}"):
            if q["options"].index(user_answer) == q["correct"]:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is: {q['options'][q['correct']]}")
            st.markdown(f"<div style='background-color: #e8f8f5; padding: 15px; border-radius: 10px;'><strong>Explanation:</strong> {q['explanation']}</div>", unsafe_allow_html=True)

def dataset_overview(credit_card_data):
    st.subheader("Sample Data")
    st.dataframe(credit_card_data.head())

    st.subheader("Dataset Statistics")
    total_customers = len(credit_card_data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", total_customers)
        st.metric("Number of Features", len(credit_card_data.columns) - 1)  # Excluding 'CUST_ID' column

    with col2:
        st.metric("Average Credit Limit", f"${credit_card_data['CREDIT_LIMIT'].mean():.2f}")
        st.metric("Average Balance", f"${credit_card_data['BALANCE'].mean():.2f}")

    st.subheader("Feature Distributions")
    numeric_features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
    fig = px.box(credit_card_data, y=numeric_features, title="Distribution of Key Financial Indicators")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()