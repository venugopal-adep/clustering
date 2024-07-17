import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(layout="wide", page_title="K-Means Elbow Method Explorer", page_icon="📊")

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

# Helper functions (same as before)
def calculate_wcss(data, max_k):
    wcss = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

def calculate_silhouette_scores(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)
    return silhouette_scores

def plot_elbow_and_silhouette(wcss, silhouette_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(2, len(wcss) + 2)), y=wcss, mode='lines+markers', name='WCSS'))
    fig.add_trace(go.Scatter(x=list(range(2, len(silhouette_scores) + 2)), y=silhouette_scores, mode='lines+markers', name='Silhouette Score', yaxis='y2'))
    fig.update_layout(title='Elbow Method and Silhouette Score',
                      xaxis_title='Number of Clusters (k)',
                      yaxis=dict(title='WCSS'),
                      yaxis2=dict(title='Silhouette Score', overlaying='y', side='right'))
    return fig

def plot_clusters(data, kmeans):
    fig = go.Figure()
    for i in range(kmeans.n_clusters):
        cluster_data = data[kmeans.labels_ == i]
        fig.add_trace(go.Scatter(x=cluster_data[:, 0], y=cluster_data[:, 1], mode='markers', name=f'Cluster {i+1}'))
    fig.update_layout(title='Clustering Results',
                      xaxis_title='X',
                      yaxis_title='Y')
    return fig

def find_optimal_k_elbow(wcss):
    n = len(wcss)
    x1, y1 = 2, wcss[0]
    x2, y2 = n + 1, wcss[n - 1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        distances.append(numerator / denominator)
    return distances.index(max(distances)) + 2

def find_optimal_k_silhouette(silhouette_scores):
    return silhouette_scores.index(max(silhouette_scores)) + 2

# Main app
def main():
    st.markdown("<h1 class='main-header'>📊 K-Means Elbow Method Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Explore the Elbow Method and Silhouette Score for K-Means Clustering</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding K-Means and the Elbow Method</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>K-Means Clustering</h3>
        <p class='text-content'>
        K-Means is an unsupervised learning algorithm that partitions data into k clusters. It works by:
        
        1. Initializing k centroids randomly
        2. Assigning each data point to the nearest centroid
        3. Recalculating the centroids based on the assigned points
        4. Repeating steps 2-3 until convergence
        
        The challenge is determining the optimal number of clusters (k).
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Elbow Method and Silhouette Score</h3>
        <p class='text-content'>
        Two common methods for finding the optimal k are:
        
        1. Elbow Method: Plots the Within-Cluster Sum of Squares (WCSS) against k. The "elbow" of this curve suggests the optimal k.
        2. Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters. Higher scores indicate better-defined clusters.
        
        By considering both methods, we can make a more informed decision about the optimal number of clusters.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore K-Means Clustering</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_samples = st.slider("Number of samples", min_value=100, max_value=1000, value=200, step=100)
            n_features = st.slider("Number of features", min_value=2, max_value=5, value=2, step=1)
            max_k = st.slider("Maximum number of clusters", min_value=5, max_value=20, value=10, step=1)
            
            if st.button("Generate Data and Analyze"):
                n_centers = np.random.randint(3, 7)
                X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=np.random.randint(100))
                
                wcss = calculate_wcss(X, max_k)
                silhouette_scores = calculate_silhouette_scores(X, max_k)
                
                wcss_optimal_k = find_optimal_k_elbow(wcss)
                silhouette_optimal_k = find_optimal_k_silhouette(silhouette_scores)
                optimal_k = silhouette_optimal_k if silhouette_optimal_k == wcss_optimal_k else wcss_optimal_k
                
                kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans.fit(X)
                
                st.session_state.X = X
                st.session_state.wcss = wcss
                st.session_state.silhouette_scores = silhouette_scores
                st.session_state.wcss_optimal_k = wcss_optimal_k
                st.session_state.silhouette_optimal_k = silhouette_optimal_k
                st.session_state.optimal_k = optimal_k
                st.session_state.kmeans = kmeans
        
        with col2:
            if 'X' in st.session_state:
                fig1 = plot_elbow_and_silhouette(st.session_state.wcss, st.session_state.silhouette_scores)
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown(f"<p class='text-content'>Optimal k (Elbow Method): {st.session_state.wcss_optimal_k}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='text-content'>Optimal k (Silhouette Score): {st.session_state.silhouette_optimal_k}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='text-content'>Best k: {st.session_state.optimal_k}</p>", unsafe_allow_html=True)
                
                fig2 = plot_clusters(st.session_state.X, st.session_state.kmeans)
                st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What does the Elbow Method use to determine the optimal number of clusters?",
                "options": [
                    "Silhouette Score",
                    "Within-Cluster Sum of Squares (WCSS)",
                    "Between-Cluster Sum of Squares",
                    "Euclidean Distance"
                ],
                "correct": 1,
                "explanation": "The Elbow Method uses the Within-Cluster Sum of Squares (WCSS) to determine the optimal number of clusters. It plots WCSS against the number of clusters, and the 'elbow' point suggests the optimal k."
            },
            {
                "question": "What does a higher Silhouette Score indicate?",
                "options": [
                    "Poorly defined clusters",
                    "Better defined clusters",
                    "More clusters needed",
                    "Overfitting"
                ],
                "correct": 1,
                "explanation": "A higher Silhouette Score indicates better-defined clusters. It measures how similar an object is to its own cluster compared to other clusters, with values ranging from -1 to 1."
            },
            {
                "question": "Why might the optimal k from the Elbow Method and Silhouette Score be different?",
                "options": [
                    "One method is always wrong",
                    "They measure different aspects of clustering quality",
                    "It's impossible for them to be different",
                    "The data is not suitable for clustering"
                ],
                "correct": 1,
                "explanation": "The Elbow Method and Silhouette Score might suggest different optimal k values because they measure different aspects of clustering quality. The Elbow Method focuses on compactness within clusters, while the Silhouette Score considers both cohesion and separation between clusters."
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
