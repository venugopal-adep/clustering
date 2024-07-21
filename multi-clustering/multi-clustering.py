import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris, make_s_curve
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, OPTICS, MeanShift, AffinityPropagation, Birch, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go

# Set page config
st.set_page_config(layout="wide", page_title="Advanced Clustering Explorer", page_icon="🔬")

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
def generate_data(dataset):
    if dataset == 'Blobs':
        X, y = make_blobs(n_samples=200, centers=3, n_features=3, random_state=42)
    elif dataset == 'Circles':
        X, y = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
        X = np.hstack((X, np.zeros((X.shape[0], 1))))
    elif dataset == 'Moons':
        X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
        X = np.hstack((X, np.zeros((X.shape[0], 1))))
    elif dataset == 'Iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
    elif dataset == 'S-Curve':
        X, y = make_s_curve(n_samples=200, random_state=42)
    return X, y

def perform_clustering(X, method, n_clusters):
    models = {
        'K-Means': KMeans(n_clusters=n_clusters, random_state=42),
        'Mini-Batch K-Means': MiniBatchKMeans(n_clusters=n_clusters, random_state=42),
        'Spectral Clustering': SpectralClustering(n_clusters=n_clusters, random_state=42),
        'Gaussian Mixture': GaussianMixture(n_components=n_clusters, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Agglomerative Clustering': AgglomerativeClustering(n_clusters=n_clusters),
        'OPTICS': OPTICS(min_samples=5),
        'Mean Shift': MeanShift(),
        'Affinity Propagation': AffinityPropagation(random_state=42),
        'Birch': Birch(n_clusters=n_clusters),
        't-SNE': TSNE(n_components=3, random_state=42)
    }
    
    if method == 't-SNE':
        return models[method].fit_transform(X)
    
    model = models[method]
    model.fit(X)
    return model.labels_ if hasattr(model, 'labels_') else model.predict(X)

def plot_3d(X, labels):
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=labels,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ), height=600)
    return fig

# Main app
def main():
    st.markdown("<h1 class='main-header'>🔬 Advanced Clustering Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Developed by: Venugopal Adep</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <p class='text-content'>
        Clustering is an unsupervised learning technique that groups similar data points together. This explorer demonstrates various clustering algorithms on different datasets:

        1. K-Means: Partitions data into K clusters, each represented by its centroid.
        2. Mini-Batch K-Means: A variant of K-Means that uses mini-batches to reduce computation time.
        3. Spectral Clustering: Uses eigenvalues of the similarity matrix to reduce dimensionality before clustering.
        4. Gaussian Mixture: Assumes data is generated from a mixture of Gaussian distributions.
        5. DBSCAN: Density-based clustering that can find arbitrarily shaped clusters.
        6. And more...

        Each algorithm has its strengths and is suited for different types of data and clustering tasks.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dataset = st.selectbox('Select a dataset', ['Blobs', 'Circles', 'Moons', 'Iris', 'S-Curve'])
            method = st.selectbox('Select a clustering method', ['K-Means', 'Mini-Batch K-Means', 'Spectral Clustering', 'Gaussian Mixture', 'DBSCAN', 'Agglomerative Clustering', 'OPTICS', 'Mean Shift', 'Affinity Propagation', 'Birch', 't-SNE'])
            n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)
            
            if st.button('Run Clustering'):
                X, y = generate_data(dataset)
                if method == 't-SNE':
                    X_transformed = perform_clustering(X, method, n_clusters)
                    st.session_state.X = X_transformed
                    st.session_state.labels = y
                else:
                    labels = perform_clustering(X, method, n_clusters)
                    st.session_state.X = X
                    st.session_state.labels = labels
        
        with col2:
            if 'X' in st.session_state and 'labels' in st.session_state:
                fig = plot_3d(st.session_state.X, st.session_state.labels)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "Which clustering algorithm is known for its ability to find arbitrarily shaped clusters?",
                "options": ["K-Means", "DBSCAN", "Gaussian Mixture", "Birch"],
                "correct": 1,
                "explanation": "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is known for its ability to find arbitrarily shaped clusters. It groups together points that are closely packed together, making it effective for clusters of various shapes."
            },
            {
                "question": "What's the main difference between K-Means and Mini-Batch K-Means?",
                "options": [
                    "K-Means uses Euclidean distance, Mini-Batch K-Means uses Manhattan distance",
                    "K-Means processes all data points, Mini-Batch K-Means uses subsets of data",
                    "K-Means is more accurate, Mini-Batch K-Means is faster",
                    "K-Means works with continuous data, Mini-Batch K-Means with categorical data"
                ],
                "correct": 1,
                "explanation": "The main difference is that K-Means processes all data points in each iteration, while Mini-Batch K-Means uses small random batches of data in each iteration. This makes Mini-Batch K-Means faster but potentially slightly less accurate than standard K-Means."
            },
            {
                "question": "What does t-SNE stand for?",
                "options": [
                    "t-Supervised Network Embedding",
                    "t-Stochastic Neighbor Embedding",
                    "time-Series Network Exploration",
                    "topological-Spatial Neighbor Estimation"
                ],
                "correct": 1,
                "explanation": "t-SNE stands for t-Distributed Stochastic Neighbor Embedding. It's a machine learning algorithm for visualization used to reduce high-dimensional data into two or three dimensions, making it easier to visualize."
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
