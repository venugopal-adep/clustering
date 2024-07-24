#Dataset : https://www.kaggle.com/datasets/singhnproud77/eastwestairlines-heirarchical-clustering
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration and custom CSS
st.set_page_config(page_title="EastWest Airlines Clustering Analysis", layout="wide")
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
    st.title("Interactive EastWest Airlines Clustering Analysis")

    # Sidebar
    with st.sidebar:
        st.header("Clustering Settings")
        algorithm = st.selectbox("Select Clustering Algorithm", 
                                 ["K-Means", "Agglomerative Clustering", "DBSCAN", "BIRCH", "Gaussian Mixture"])
        n_clusters = st.slider("Number of Clusters", 2, 10, 3) if algorithm != "DBSCAN" else None
        eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5) if algorithm == "DBSCAN" else None
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5) if algorithm == "DBSCAN" else None

    # Load and process data
    X, feature_names, airline_data = load_and_process_data()

    # Perform clustering
    labels = cluster_data(X, algorithm, n_clusters, eps, min_samples)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Visualization", "📘 Interpretation", "🧠 Algorithms", "🔍 Cluster Insights", "🎓 Quiz", "📋 Dataset"])

    with tab1:
        st.header("3D Visualization of Clustering Results")
        fig = create_3d_scatter(X, labels, airline_data['ID'], feature_names, f"{algorithm} Clustering")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("How to Interpret the Visualization")
        interpretation_guide()

    with tab3:
        st.header("Understanding the Clustering Algorithms")
        explain_algorithms()

    with tab4:
        st.header("Cluster Insights")
        cluster_insights(airline_data, labels, feature_names)

    with tab5:
        st.header("Test Your Understanding")
        quiz()

    with tab6:
        st.header("Dataset Overview")
        dataset_overview(airline_data)

def load_and_process_data():
    # Load the EastWest Airlines dataset
    airline_data = pd.read_csv('EastWestAirlines.csv')
    
    # Select features for clustering
    features = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll']
    X = airline_data[features]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for visualization
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled)
    
    return X_3d, features, airline_data

def cluster_data(X, algorithm, n_clusters=None, eps=None, min_samples=None):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == "BIRCH":
        model = Birch(n_clusters=n_clusters)
    else:  # Gaussian Mixture
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    
    labels = model.fit_predict(X)
    
    # Calculate silhouette score if applicable
    if algorithm != "DBSCAN":
        score = silhouette_score(X, labels)
        st.sidebar.metric("Silhouette Score", f"{score:.3f}")
    
    return labels

def create_3d_scatter(X, labels, customer_ids, feature_names, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.8),
        text=[f"Customer ID: {id}<br>Cluster: {label}" for id, label in zip(customer_ids, labels)],
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
        <li>Each point represents a customer from the EastWest Airlines dataset.</li>
        <li>Colors indicate the cluster assigned by the chosen algorithm.</li>
        <li>Proximity of points suggests similarity in customer behavior and attributes.</li>
    </ul>

    <h3 style="color: #2c3e50;">Example:</h3>
    <p>If you see a tight group of blue points in one area, this could represent a cluster of customers with similar flying patterns and reward program usage.</p>

    <h3 style="color: #2c3e50;">What to Look For:</h3>
    <ol>
        <li><strong>Well-separated clusters:</strong> Distinct groups of colors might indicate that the algorithm has effectively separated different types of customers based on their attributes.</li>
        <li><strong>Mixed clusters:</strong> Areas where colors are intermingled could suggest overlapping characteristics between different customer groups.</li>
        <li><strong>Outliers:</strong> Isolated points might represent customers with unique flying patterns or reward program usage.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def explain_algorithms():
    st.subheader("K-Means")
    st.markdown("""
    <div style="background-color: #e8f8f5; padding: 20px; border-radius: 10px;">
    <p>K-Means tries to find a specified number of cluster centers and assign each customer to the nearest center.</p>
    <p><strong>Pros:</strong> Simple, fast, and works well on globular clusters.</p>
    <p><strong>Cons:</strong> Sensitive to initial centroids, assumes spherical clusters, and requires specifying the number of clusters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Agglomerative Clustering")
    st.markdown("""
    <div style="background-color: #fef9e7; padding: 20px; border-radius: 10px;">
    <p>This algorithm starts with each customer as its own cluster and progressively merges the closest clusters.</p>
    <p><strong>Pros:</strong> Can uncover hierarchical structure in data, doesn't assume cluster shape.</p>
    <p><strong>Cons:</strong> Computationally intensive for large datasets, can be sensitive to noise.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("DBSCAN")
    st.markdown("""
    <div style="background-color: #f4ecf7; padding: 20px; border-radius: 10px;">
    <p>DBSCAN groups together customers that are closely packed in the feature space, marking customers in low-density regions as outliers.</p>
    <p><strong>Pros:</strong> Can find arbitrarily shaped clusters, robust to outliers, doesn't require specifying number of clusters.</p>
    <p><strong>Cons:</strong> Sensitive to parameters, struggles with varying density clusters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("BIRCH")
    st.markdown("""
    <div style="background-color: #eaeded; padding: 20px; border-radius: 10px;">
    <p>BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) builds a tree structure to incrementally cluster the data.</p>
    <p><strong>Pros:</strong> Efficient for large datasets, handles outliers well.</p>
    <p><strong>Cons:</strong> May not work well with non-spherical clusters, sensitive to data order.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Gaussian Mixture")
    st.markdown("""
    <div style="background-color: #ebf5fb; padding: 20px; border-radius: 10px;">
    <p>Gaussian Mixture Models assume the data is generated from a mixture of a finite number of Gaussian distributions with unknown parameters.</p>
    <p><strong>Pros:</strong> Flexible, can model complex cluster shapes, provides probabilistic cluster assignments.</p>
    <p><strong>Cons:</strong> Sensitive to initialization, can overfit with too many components.</p>
    </div>
    """, unsafe_allow_html=True)

def cluster_insights(airline_data, labels, feature_names):
    df = airline_data.copy()
    df['Cluster'] = labels
    
    for cluster in sorted(df['Cluster'].unique()):
        st.subheader(f"Cluster {cluster}")
        cluster_data = df[df['Cluster'] == cluster]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Customers", len(cluster_data))
            st.write("Top 5 Customer IDs:")
            st.write(", ".join(map(str, cluster_data['ID'].head().tolist())))
        
        with col2:
            st.write("Cluster Characteristics:")
            for feature in feature_names:
                mean_value = cluster_data[feature].mean()
                overall_mean = df[feature].mean()
                difference = ((mean_value - overall_mean) / overall_mean) * 100
                emoji = "🔼" if difference > 0 else "🔽"
                st.write(f"{feature}: {emoji} {abs(difference):.1f}% {'above' if difference > 0 else 'below'} average")
    
    st.subheader("Conclusions")
    st.write("""
    Based on the cluster analysis, we can draw the following conclusions:
    1. There appear to be distinct groups of customers with similar flying patterns and reward program usage.
    2. Some clusters show higher balance and bonus miles, possibly indicating frequent flyers or high-value customers.
    3. Other clusters exhibit lower flight miles and transactions, which might represent occasional travelers.
    4. The clustering reveals different customer segments that could be targeted with tailored marketing strategies.
    5. Further investigation into outlier customers could provide insights into unique travel patterns or potential VIP customers.
    """)

def quiz():
    questions = [
        {
            "question": "Which clustering algorithm is particularly good at identifying outliers?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN", "BIRCH"],
            "correct": 2,
            "explanation": "DBSCAN is particularly good at identifying outliers because it can label points in low-density regions as noise or outliers, unlike other algorithms that assign every point to a cluster."
        },
        {
            "question": "What does the silhouette score measure?",
            "options": ["The number of clusters", "The quality of clustering", "The size of the largest cluster", "The number of outliers"],
            "correct": 1,
            "explanation": "The silhouette score measures the quality of clustering. It quantifies how similar an object is to its own cluster compared to other clusters. Higher values indicate better-defined clusters."
        },
        {
            "question": "Which algorithm builds a tree structure to incrementally cluster the data?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN", "BIRCH"],
            "correct": 3,
            "explanation": "BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) builds a tree structure to incrementally cluster the data, making it efficient for large datasets."
        },
        {
            "question": "What is a potential drawback of the K-Means algorithm?",
            "options": ["It's too slow", "It requires specifying the number of clusters beforehand", "It can only handle small datasets", "It always produces the same result"],
            "correct": 1,
            "explanation": "A potential drawback of K-Means is that it requires specifying the number of clusters beforehand. This can be challenging if you don't know the optimal number of clusters for your data."
        },
        {
            "question": "Which algorithm assumes the data is generated from a mixture of Gaussian distributions?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture"],
            "correct": 3,
            "explanation": "Gaussian Mixture Models assume the data is generated from a mixture of a finite number of Gaussian distributions with unknown parameters."
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

def dataset_overview(airline_data):
    st.subheader("Sample Data")
    st.dataframe(airline_data.head())

    st.subheader("Dataset Statistics")
    total_customers = len(airline_data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", total_customers)
        st.metric("Average Balance", f"{airline_data['Balance'].mean():.2f}")

    with col2:
        st.metric("Average Bonus Miles", f"{airline_data['Bonus_miles'].mean():.2f}")
        st.metric("Average Days Since Enroll", f"{airline_data['Days_since_enroll'].mean():.2f}")

    st.subheader("Feature Distributions")
    numeric_features = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll']
    fig = px.box(airline_data, y=numeric_features, title="Distribution of Customer Attributes")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = airline_data[numeric_features].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap of Features")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Award Distribution")
    award_counts = airline_data['Award'].value_counts()
    fig = px.pie(values=award_counts.values, names=award_counts.index, title="Distribution of Award")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()