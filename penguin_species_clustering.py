#Dataset : https://www.kaggle.com/datasets/youssefaboelwafa/clustering-penguins-species
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration and custom CSS
st.set_page_config(page_title="Penguin Species Clustering", layout="wide")
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
    st.title("Interactive Penguin Species Clustering Demo")

    # Sidebar
    with st.sidebar:
        st.header("Clustering Settings")
        algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "Agglomerative Clustering", "DBSCAN"])
        n_clusters = st.slider("Number of Clusters", 2, 10, 3) if algorithm != "DBSCAN" else None
        eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5) if algorithm == "DBSCAN" else None
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5) if algorithm == "DBSCAN" else None

    # Load and process data
    X, true_labels, feature_names, penguins = load_and_process_data()

    # Perform clustering
    labels = cluster_data(X, algorithm, n_clusters, eps, min_samples)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualization", "📘 Interpretation", "🧠 Algorithms", "🎓 Quiz", "📋 Dataset"])

    with tab1:
        st.header("3D Visualization of Clustering Results")
        fig = create_3d_scatter(X, labels, true_labels, feature_names, f"{algorithm} Clustering")
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
        dataset_overview(penguins)

def load_and_process_data():
    # Load the penguins dataset
    penguins = pd.read_csv('penguins.csv')
    
    # Select numerical features for clustering
    features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = penguins[features]
    
    # Handle missing values
    X = X.dropna()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for visualization
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled)
    
    # Get true labels (sex)
    true_labels = penguins.loc[X.index, 'sex']
    
    return X_3d, true_labels, features, penguins

def cluster_data(X, algorithm, n_clusters=None, eps=None, min_samples=None):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:  # DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def create_3d_scatter(X, labels, true_labels, feature_names, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.8),
        text=[f"True Sex: {l}" for l in true_labels],
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
        <li>Each point represents a penguin from the dataset.</li>
        <li>Colors indicate the cluster assigned by the chosen algorithm.</li>
        <li>Hover over points to see the true sex of each penguin.</li>
        <li>Proximity of points suggests similarity in measurements.</li>
    </ul>

    <h3 style="color: #2c3e50;">Example:</h3>
    <p>If you see a tight group of blue points in one area, this could represent a cluster of penguins with similar physical characteristics, possibly belonging to the same sex or species.</p>

    <h3 style="color: #2c3e50;">What to Look For:</h3>
    <ol>
        <li><strong>Well-separated clusters:</strong> Distinct groups of colors might indicate that the algorithm has effectively separated different penguin groups based on their measurements.</li>
        <li><strong>Mixed clusters:</strong> Areas where colors are intermingled could suggest overlapping characteristics between different penguin groups.</li>
        <li><strong>Outliers:</strong> Isolated points might represent penguins with unusual measurements for their sex or species.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def explain_algorithms():
    st.subheader("K-Means")
    st.markdown("""
    <div style="background-color: #e8f8f5; padding: 20px; border-radius: 10px;">
    <p>K-Means tries to find a specified number of cluster centers and assign each penguin to the nearest center.</p>
    <p><strong>Analogy:</strong> Imagine organizing penguins into groups based on their size. K-Means is like deciding on a number of size categories (clusters) and then putting each penguin in the category it's most similar to.</p>
    <p><strong>Example:</strong> If we set K=3, K-Means might create clusters for "small", "medium", and "large" penguins based on their measurements.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Agglomerative Clustering")
    st.markdown("""
    <div style="background-color: #fef9e7; padding: 20px; border-radius: 10px;">
    <p>This algorithm starts with each penguin as its own cluster and progressively merges the closest clusters.</p>
    <p><strong>Analogy:</strong> Think of penguins huddling together for warmth. They start individually, then form small groups, which gradually combine into larger groups.</p>
    <p><strong>Example:</strong> It might start by grouping very similar penguins (like two with almost identical measurements), then gradually combine these small groups into larger categories.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("DBSCAN")
    st.markdown("""
    <div style="background-color: #f4ecf7; padding: 20px; border-radius: 10px;">
    <p>DBSCAN groups together penguins that are closely packed, marking penguins in low-density regions as outliers.</p>
    <p><strong>Analogy:</strong> Imagine looking at a colony of penguins. DBSCAN would identify dense groups of penguins standing close together and label lone penguins as outliers.</p>
    <p><strong>Example:</strong> In our penguin dataset, DBSCAN might identify dense clusters of penguins with similar measurements while labeling penguins with unique characteristics as outliers.</p>
    </div>
    """, unsafe_allow_html=True)

def quiz():
    questions = [
        {
            "question": "What does each point in the 3D visualization represent?",
            "options": ["A cluster center", "A penguin", "A measurement", "A species"],
            "correct": 1,
            "explanation": "Each point in the 3D visualization represents a single penguin from the dataset. The position of the point in 3D space is determined by the penguin's measurements, with similar penguins appearing closer together."
        },
        {
            "question": "In K-Means clustering, what does 'K' represent?",
            "options": ["Number of iterations", "Number of clusters", "Number of penguins", "Number of features"],
            "correct": 1,
            "explanation": "In K-Means clustering, 'K' represents the number of clusters. It's the number of groups you want the algorithm to divide your data into. For example, if K=3, the algorithm will try to organize the penguins into 3 distinct groups based on their measurements."
        },
        {
            "question": "Which clustering algorithm is best for detecting outliers?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN"],
            "correct": 2,
            "explanation": "DBSCAN is particularly good at detecting outliers. Unlike K-Means and Agglomerative Clustering, which assign every point to a cluster, DBSCAN can label points in low-density regions as outliers. This makes it useful for identifying unusual penguins with measurements that don't fit well with the main groups."
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

def dataset_overview(penguins):
    st.subheader("Sample Data")
    st.dataframe(penguins.head())

    st.subheader("Dataset Statistics")
    total_penguins = len(penguins)
    sex_counts = penguins['sex'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Penguins", total_penguins)
        st.metric("Number of Features", len(penguins.columns) - 1)  # Excluding 'sex' column

    with col2:
        st.metric("Most Common Sex", sex_counts.index[0])
        st.metric("Number of NA values", penguins.isna().sum().sum())

    st.subheader("Sex Distribution")
    fig = px.pie(values=sex_counts.values, names=sex_counts.index, title="Distribution of Penguin Sex")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Distributions")
    numeric_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
    fig = px.box(penguins, y=numeric_features, points="all", title="Distribution of Penguin Measurements")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()