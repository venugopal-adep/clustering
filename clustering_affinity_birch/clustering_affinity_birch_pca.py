import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import AffinityPropagation, Birch
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
    st.markdown("<h1 class='main-header'>🔬 Advanced Clustering Explorer: Affinity Propagation and Birch</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Explore advanced clustering algorithms on the Iris dataset</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding Advanced Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Affinity Propagation</h3>
        <p class='text-content'>
        Affinity Propagation is a clustering algorithm that:
        
        1. Does not require the number of clusters to be specified in advance
        2. Works by exchanging messages between data points to find exemplars (cluster centers)
        3. Can find non-spherical clusters and is particularly good with small to medium-sized datasets
        
        The 'damping' parameter controls the convergence of the algorithm.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <h3>Birch (Balanced Iterative Reducing and Clustering using Hierarchies)</h3>
        <p class='text-content'>
        Birch is a clustering algorithm designed for large datasets. It:
        
        1. Builds a tree-like data structure (CFT) to summarize the data
        2. Can handle large datasets efficiently
        3. Is particularly good when the number of clusters is unknown
        
        The 'threshold' parameter controls the size of sub-clusters.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            data = load_data()
            features = st.multiselect('Select features to cluster', options=data.columns[:-1].tolist(), default=data.columns[:4].tolist())
            
            method = st.selectbox('Select clustering method', options=['Affinity Propagation', 'Birch'])
            if method == 'Birch':
                threshold = st.slider('Select threshold for Birch', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            else:
                damping = st.slider('Select damping for Affinity Propagation', min_value=0.5, max_value=1.0, value=0.9, step=0.1)
            
            if st.button('Perform Clustering'):
                if method == 'Affinity Propagation':
                    model = AffinityPropagation(damping=damping)
                elif method == 'Birch':
                    model = Birch(threshold=threshold)
                    
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
                "question": "What is a key advantage of Affinity Propagation?",
                "options": [
                    "It's always faster than other algorithms",
                    "It doesn't require specifying the number of clusters in advance",
                    "It only works with categorical data",
                    "It can handle millions of data points efficiently"
                ],
                "correct": 1,
                "explanation": "Affinity Propagation doesn't require the number of clusters to be specified in advance. It determines the number of clusters based on the data and the parameter settings."
            },
            {
                "question": "What does the 'damping' parameter control in Affinity Propagation?",
                "options": [
                    "The number of clusters",
                    "The size of the dataset",
                    "The convergence of the algorithm",
                    "The feature selection process"
                ],
                "correct": 2,
                "explanation": "The 'damping' parameter in Affinity Propagation controls the convergence of the algorithm. Higher values result in more stable convergence but may require more iterations."
            },
            {
                "question": "What is Birch particularly good at?",
                "options": [
                    "Handling very small datasets",
                    "Working with high-dimensional data",
                    "Dealing with large datasets efficiently",
                    "Clustering time-series data"
                ],
                "correct": 2,
                "explanation": "Birch is particularly good at handling large datasets efficiently. It uses a tree-like data structure to summarize the data, allowing it to process large amounts of data quickly."
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
