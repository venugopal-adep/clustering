import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(layout="wide", page_title="KMeans vs KMeans++ Explorer", page_icon="🔬")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .sub-header {
        font-size: 24px !important;
        font-weight: bold;
        color: #4682B4;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .text-content {
        font-size: 16px !important;
        line-height: 1.4;
    }
    .highlight {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #1E90FF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 14px;
        padding: 8px 16px;
        border-radius: 8px;
    }
    .developer-credit {
        font-size: 14px !important;
        font-style: italic;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def generate_data(n_samples=300, n_centers=4, cluster_std=0.7, random_state=42):
    np.random.seed(random_state)
    centers = np.random.rand(n_centers, 2) * 10
    data = []
    for i, center in enumerate(centers):
        cluster = np.random.randn(n_samples // n_centers, 2) * cluster_std + center
        data.append(cluster)
    data = np.vstack(data)
    return pd.DataFrame(data, columns=['X', 'Y'])

def perform_clustering(data, n_clusters, init_method='k-means++'):
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    return labels, centroids, inertia

def plot_clusters(data, kmeans_labels, kmeans_centroids, kmeanspp_labels, kmeanspp_centroids):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("KMeans Clustering", "KMeans++ Clustering"))
    
    # KMeans plot
    fig.add_trace(
        go.Scatter(x=data['X'], y=data['Y'], mode='markers', marker=dict(color=kmeans_labels, colorscale='Viridis'), showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=kmeans_centroids[:, 0], y=kmeans_centroids[:, 1], mode='markers', marker=dict(color='red', size=10, symbol='x'), name='KMeans Centroids'),
        row=1, col=1
    )
    
    # KMeans++ plot
    fig.add_trace(
        go.Scatter(x=data['X'], y=data['Y'], mode='markers', marker=dict(color=kmeanspp_labels, colorscale='Viridis'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=kmeanspp_centroids[:, 0], y=kmeanspp_centroids[:, 1], mode='markers', marker=dict(color='red', size=10, symbol='x'), name='KMeans++ Centroids'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, width=800, title_text="Clustering Comparison")
    return fig

# Main app
def main():
    st.markdown("<h1 class='main-header'>🔬 KMeans vs KMeans++ Clustering Demo</h1>", unsafe_allow_html=True)
    st.markdown("<p class='developer-credit'>Developed by: Venugopal Adep</p>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>This application demonstrates the difference between KMeans and KMeans++ clustering algorithms.</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding KMeans and KMeans++</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>KMeans Clustering</h3>
        <p class='text-content'>
        KMeans is a popular clustering algorithm that aims to partition n observations into k clusters. The algorithm works as follows:
        1. Randomly initialize k centroids
        2. Assign each data point to the nearest centroid
        3. Recalculate centroids as the mean of all points in the cluster
        4. Repeat steps 2-3 until convergence

        However, the random initialization can sometimes lead to suboptimal results.
        </p>
        </div>

        <div class='highlight'>
        <h3>KMeans++ Clustering</h3>
        <p class='text-content'>
        KMeans++ is an improvement over the standard KMeans algorithm. It addresses the initialization problem by using a smarter approach:
        1. Choose the first centroid randomly from the data points
        2. For each subsequent centroid, choose a data point with probability proportional to its squared distance from the nearest existing centroid
        3. Repeat step 2 until k centroids are chosen
        4. Proceed with standard KMeans algorithm

        This initialization method tends to choose centroids that are spread out, leading to better and more consistent results.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore KMeans vs KMeans++</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_samples = st.slider("Number of Samples", min_value=100, max_value=1000, value=300, step=100)
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
            cluster_std = st.slider("Cluster Std Dev", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
            
            if st.button('Generate & Cluster'):
                data = generate_data(n_samples, n_clusters, cluster_std)
                
                # Perform KMeans clustering
                kmeans_labels, kmeans_centroids, kmeans_inertia = perform_clustering(data, n_clusters, init_method='random')
                
                # Perform KMeans++ clustering
                kmeanspp_labels, kmeanspp_centroids, kmeanspp_inertia = perform_clustering(data, n_clusters, init_method='k-means++')
                
                st.session_state.data = data
                st.session_state.kmeans_labels = kmeans_labels
                st.session_state.kmeans_centroids = kmeans_centroids
                st.session_state.kmeanspp_labels = kmeanspp_labels
                st.session_state.kmeanspp_centroids = kmeanspp_centroids
                st.session_state.kmeans_inertia = kmeans_inertia
                st.session_state.kmeanspp_inertia = kmeanspp_inertia

            # Check if clustering has been performed before displaying inertia
            if 'kmeans_inertia' in st.session_state and 'kmeanspp_inertia' in st.session_state:
                st.markdown("""
                <div class='highlight'>
                <h4>Inertia Comparison</h4>
                <p class='text-content'>
                KMeans Inertia: {:.2f}<br>
                KMeans++ Inertia: {:.2f}<br><br>
                Lower inertia indicates better-defined clusters.
                </p>
                </div>
                """.format(st.session_state.kmeans_inertia, st.session_state.kmeanspp_inertia), unsafe_allow_html=True)
        
        with col2:
            if 'data' in st.session_state:
                # Display the clustering results
                fig = plot_clusters(st.session_state.data, st.session_state.kmeans_labels, st.session_state.kmeans_centroids, 
                                    st.session_state.kmeanspp_labels, st.session_state.kmeanspp_centroids)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What is the main difference in initialization between KMeans and KMeans++?",
                "options": [
                    "KMeans uses random initialization, while KMeans++ uses a smarter approach",
                    "KMeans++ uses random initialization, while KMeans uses a smarter approach",
                    "There is no difference in initialization between KMeans and KMeans++",
                    "KMeans uses only one centroid, while KMeans++ uses multiple centroids"
                ],
                "correct": 0,
                "explanation": "KMeans uses random initialization for its centroids, which can sometimes lead to suboptimal results. KMeans++, on the other hand, uses a smarter approach that tends to choose initial centroids that are spread out, leading to better and more consistent results."
            },
            {
                "question": "What does lower inertia indicate in clustering results?",
                "options": [
                    "Worse clustering performance",
                    "Better-defined clusters",
                    "More random initialization",
                    "Slower algorithm convergence"
                ],
                "correct": 1,
                "explanation": "Inertia is the sum of squared distances of samples to their closest cluster center. Lower inertia indicates that data points are closer to their respective cluster centers, which means better-defined clusters."
            },
            {
                "question": "Why is KMeans++ often preferred over standard KMeans?",
                "options": [
                    "It's faster than standard KMeans",
                    "It always finds the global optimum",
                    "It tends to produce more consistent and better results",
                    "It can handle non-linear data better"
                ],
                "correct": 2,
                "explanation": "KMeans++ is often preferred because it tends to produce more consistent and better results. Its smart initialization method helps avoid the poor clusterings that can sometimes result from the random initialization of standard KMeans."
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