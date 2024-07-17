import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import plotly.express as px

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
@st.cache_data
def load_data():
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def plot_clusters(data, labels):
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = labels
    
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', 
                        labels={'Cluster': 'Cluster'}, title='PCA 3D Cluster Plot',
                        color_continuous_scale=px.colors.qualitative.Bold)
    fig.update_traces(marker=dict(size=5))
    return fig

# Main app
def main():
    st.markdown("<h1 class='main-header'>🔬 Advanced Clustering Explorer: Spectral Clustering and Gaussian Mixture Models</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Explore advanced clustering algorithms on the Iris dataset</p>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'><strong>Developed by: Venugopal Adep</strong></p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding Advanced Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Spectral Clustering</h3>
        <p class='text-content'>
        Spectral Clustering is a technique that:
        
        1. Uses the spectrum (eigenvalues) of the similarity matrix of the data
        2. Can find non-convex clusters
        3. Is particularly effective when the dataset has a complex structure
        
        The number of clusters needs to be specified in advance.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <h3>Gaussian Mixture Models (GMM)</h3>
        <p class='text-content'>
        Gaussian Mixture Models are probabilistic models that assume:
        
        1. Data points are generated from a mixture of a finite number of Gaussian distributions
        2. Each cluster corresponds to one Gaussian distribution
        3. Can handle clusters of different sizes and shapes
        
        The number of components and covariance type are key parameters.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            data = load_data()
            features = st.multiselect('Select features to cluster', options=data.columns[:-1].tolist(), default=data.columns[:4].tolist())
            
            method = st.selectbox('Select clustering method', options=['Spectral Clustering', 'Gaussian Mixture Models'])
            if method == 'Spectral Clustering':
                n_clusters = st.slider('Select number of clusters', min_value=2, max_value=10, value=3, step=1)
            else:
                n_components = st.slider('Select number of components', min_value=2, max_value=10, value=3, step=1)
                covariance_type = st.selectbox('Select covariance type', ['full', 'tied', 'diag', 'spherical'])
            
            if st.button('Perform Clustering'):
                if method == 'Spectral Clustering':
                    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
                elif method == 'Gaussian Mixture Models':
                    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
                    
                labels = model.fit_predict(data[features])
                st.session_state.labels = labels
                st.session_state.features = features
        
        with col2:
            if 'labels' in st.session_state:
                fig = plot_clusters(data[st.session_state.features], st.session_state.labels)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What is a key characteristic of Spectral Clustering?",
                "options": [
                    "It always requires specifying the number of clusters",
                    "It uses the eigenvalues of the similarity matrix",
                    "It only works with categorical data",
                    "It's particularly slow for large datasets"
                ],
                "correct": 1,
                "explanation": "Spectral Clustering uses the spectrum (eigenvalues) of the similarity matrix of the data to perform dimensionality reduction before clustering."
            },
            {
                "question": "What assumption does Gaussian Mixture Models make about the data?",
                "options": [
                    "All clusters must be the same size",
                    "Data points are generated from a mixture of Gaussian distributions",
                    "The number of clusters must be odd",
                    "All features must be normally distributed"
                ],
                "correct": 1,
                "explanation": "Gaussian Mixture Models assume that data points are generated from a mixture of a finite number of Gaussian distributions, with each cluster corresponding to one Gaussian."
            },
            {
                "question": "What is an advantage of Gaussian Mixture Models?",
                "options": [
                    "They don't require specifying the number of clusters",
                    "They always converge to the global optimum",
                    "They can handle clusters of different sizes and shapes",
                    "They work well with very small datasets"
                ],
                "correct": 2,
                "explanation": "Gaussian Mixture Models can handle clusters of different sizes and shapes due to their flexibility in modeling the covariance structure of the data."
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
