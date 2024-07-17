import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage

# Set page config
st.set_page_config(layout="wide", page_title="Clustering Explorer", page_icon="🔍")

# Custom CSS (same as before)
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

def plot_clusters_3d(data, labels):
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = labels
    
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3',
                        color='Cluster', labels={'Cluster': 'Cluster'},
                        title='PCA 3D Cluster Plot',
                        color_continuous_scale=px.colors.qualitative.Bold)
    fig.update_traces(marker=dict(size=5))
    return fig

def plot_dendrogram(X):
    # Create linkage matrix
    Z = linkage(X, 'ward')
    
    # Create dendrogram
    fig = go.Figure()
    dn = dendrogram(Z)
    
    # Extract x and y coordinates from dendrogram
    x = dn['icoord']
    y = dn['dcoord']
    
    # Plot the dendrogram lines
    for xi, yi in zip(x, y):
        fig.add_trace(go.Scatter(x=xi, y=yi, mode='lines', line=dict(color='#636EFA'), showlegend=False))
    
    fig.update_layout(
        title='Agglomerative Clustering Dendrogram',
        xaxis_title='Sample Index',
        yaxis_title='Distance',
        width=800,
        height=600
    )
    return fig

def plot_dbscan_density(data, labels, eps):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels

    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                     title='DBSCAN Clustering',
                     color_continuous_scale=px.colors.qualitative.Bold)

    # Add circles to represent eps
    for _, row in df_pca.iterrows():
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=row['PC1'] - eps, y0=row['PC2'] - eps,
                      x1=row['PC1'] + eps, y1=row['PC2'] + eps,
                      line_color="LightSeaGreen", line_width=1, opacity=0.3)

    fig.update_layout(width=800, height=600)
    return fig

# Main app
def main():
    st.markdown("<h1 class='main-header'>🔍 Clustering Explorer: Agglomerative and DBSCAN</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Explore clustering algorithms on the Iris dataset</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Agglomerative Clustering</h3>
        <p class='text-content'>
        Agglomerative Clustering is a hierarchical clustering method that works by:
        
        1. Starting with each data point as a single cluster
        2. Merging the closest clusters at each step
        3. Continuing until the desired number of clusters is reached
        
        It's useful for discovering hierarchical relationships in the data.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <h3>DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</h3>
        <p class='text-content'>
        DBSCAN is a density-based clustering algorithm that:
        
        1. Groups together points that are closely packed together
        2. Marks points as outliers if they're in low-density regions
        
        It's particularly good at finding clusters of arbitrary shape and identifying noise in the data.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            data = load_data()
            features = st.multiselect('Select features to cluster', options=data.columns[:-1].tolist(), default=data.columns[:4].tolist())
            
            method = st.selectbox('Select clustering method', options=['Agglomerative Clustering', 'DBSCAN'])
            if method == 'Agglomerative Clustering':
                n_clusters = st.slider('Select number of clusters', min_value=2, max_value=10, value=3, step=1)
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif method == 'DBSCAN':
                eps = st.slider('Select epsilon (eps)', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
                min_samples = st.slider('Select minimum samples', min_value=1, max_value=10, value=5, step=1)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            
            if st.button('Perform Clustering'):
                labels = model.fit_predict(data[features])
                st.session_state.labels = labels
                st.session_state.features = features
                st.session_state.model = model
                st.session_state.method = method
                if method == 'DBSCAN':
                    st.session_state.eps = eps
        
        with col2:
            if 'labels' in st.session_state:
                fig1 = plot_clusters_3d(data[st.session_state.features], st.session_state.labels)
                st.plotly_chart(fig1, use_container_width=True)
                
                if st.session_state.method == 'Agglomerative Clustering':
                    fig2 = plot_dendrogram(data[st.session_state.features].values)
                    st.plotly_chart(fig2, use_container_width=True)
                elif st.session_state.method == 'DBSCAN':
                    fig2 = plot_dbscan_density(data[st.session_state.features], st.session_state.labels, st.session_state.eps)
                    st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What is the main difference between Agglomerative Clustering and DBSCAN?",
                "options": [
                    "Agglomerative Clustering is faster",
                    "DBSCAN requires specifying the number of clusters",
                    "Agglomerative Clustering builds a hierarchy, while DBSCAN is density-based",
                    "DBSCAN only works on 2D data"
                ],
                "correct": 2,
                "explanation": "Agglomerative Clustering builds a hierarchy of clusters, merging the closest clusters at each step. DBSCAN, on the other hand, is density-based and groups together points that are closely packed, without requiring a specified number of clusters."
            },
            {
                "question": "What is an advantage of DBSCAN over Agglomerative Clustering?",
                "options": [
                    "It's always faster",
                    "It can identify noise and outliers in the data",
                    "It always produces better clusters",
                    "It works on categorical data"
                ],
                "correct": 1,
                "explanation": "DBSCAN has the advantage of being able to identify noise and outliers in the data. It marks points in low-density regions as outliers, which can be useful in many real-world scenarios."
            },
            {
                "question": "Why do we use PCA before visualizing the clusters?",
                "options": [
                    "To make the clustering algorithm faster",
                    "To reduce the dimensionality for visualization",
                    "To improve clustering accuracy",
                    "To normalize the data"
                ],
                "correct": 1,
                "explanation": "We use PCA (Principal Component Analysis) to reduce the dimensionality of the data. This allows us to visualize high-dimensional data (like the 4D Iris dataset) in a 3D or 2D plot."
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
