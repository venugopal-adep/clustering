import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(layout="wide", page_title="PCA Explorer", page_icon="🔍")

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
def load_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = load_iris()
    elif dataset_name == 'Wine':
        data = load_wine()
    elif dataset_name == 'Breast Cancer':
        data = load_breast_cancer()
    elif dataset_name == 'Digits':
        data = load_digits()
    
    X = data.data
    y = data.target
    feature_names = data.feature_names if hasattr(data, 'feature_names') else [f'Feature {i+1}' for i in range(X.shape[1])]
    return X, y, feature_names

def perform_pca(X, n_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    return X_pca, explained_variance_ratio

def plot_2d_scatter(X_pca, y):
    df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Class': y})
    fig = px.scatter(df, x='PC1', y='PC2', color='Class', 
                     title='PCA 2D Scatter Plot',
                     color_continuous_scale='viridis',
                     hover_data=['Class'])
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(plot_bgcolor='rgba(240,240,240,0.8)', 
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend_title_text='Class')
    return fig

def plot_3d_scatter(X_pca, y):
    df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'PC3': X_pca[:, 2], 'Class': y})
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Class',
                        title='PCA 3D Scatter Plot',
                        color_continuous_scale='viridis',
                        hover_data=['Class'])
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(scene=dict(xaxis=dict(backgroundcolor='rgba(240,240,240,0.8)'),
                                 yaxis=dict(backgroundcolor='rgba(240,240,240,0.8)'),
                                 zaxis=dict(backgroundcolor='rgba(240,240,240,0.8)')),
                      legend_title_text='Class')
    return fig

def plot_explained_variance(explained_variance_ratio):
    df = pd.DataFrame({'PC': range(1, len(explained_variance_ratio) + 1),
                       'Explained Variance Ratio': explained_variance_ratio})
    fig = px.bar(df, x='PC', y='Explained Variance Ratio',
                 title='Explained Variance Ratio',
                 color='Explained Variance Ratio',
                 color_continuous_scale='viridis')
    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.8)
    fig.update_layout(plot_bgcolor='rgba(240,240,240,0.8)', 
                      paper_bgcolor='rgba(0,0,0,0)',
                      coloraxis_showscale=False)
    return fig

def plot_image_grid(X, y):
    if len(X[0]) == 64:  # Digits dataset
        img_size = (8, 8)
    else:
        return None
    
    unique_labels = np.unique(y)
    fig = go.Figure()
    for label in unique_labels:
        img = X[y == label][0].reshape(img_size)
        fig.add_trace(go.Heatmap(z=img, colorscale='Viridis', showscale=False))
    
    fig.update_layout(title='Sample Images from the Dataset',
                      grid=dict(rows=1, columns=len(unique_labels), pattern='independent'),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

# Main app (same as before)
def main():
    st.markdown("<h1 class='main-header'>🔍 PCA Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Developed by: Venugopal Adep</p>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Explore the wonders of Principal Component Analysis (PCA) with interactive visualizations!</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding Principal Component Analysis (PCA)</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>What is PCA?</h3>
        <p class='text-content'>
        PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while retaining the most important information. The process involves:

        1. Standardizing the original data to have zero mean and unit variance.
        2. Computing the principal components by finding the directions of maximum variance in the data.
        3. Projecting the data onto the selected principal components.
        4. Visualizing the resulting lower-dimensional data using scatter plots.
        5. Analyzing the explained variance ratio to understand the proportion of variance explained by each principal component.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore PCA with Different Datasets</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dataset_name = st.selectbox('Select a dataset', ['Iris', 'Wine', 'Breast Cancer', 'Digits'])
            n_components = st.slider('Number of principal components', min_value=2, max_value=3, value=2)
            
            if st.button('Perform PCA'):
                X, y, feature_names = load_dataset(dataset_name)
                X_pca, explained_variance_ratio = perform_pca(X, n_components)
                
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.feature_names = feature_names
                st.session_state.X_pca = X_pca
                st.session_state.explained_variance_ratio = explained_variance_ratio
        
        with col2:
            if 'X' in st.session_state:
                st.subheader('Original Data')
                if dataset_name == 'Digits':
                    fig = plot_image_grid(st.session_state.X, st.session_state.y)
                    if fig:
                        st.plotly_chart(fig)
                else:
                    st.write(pd.DataFrame(st.session_state.X, columns=st.session_state.feature_names))
                
                st.subheader('PCA Results')
                if n_components == 2:
                    fig = plot_2d_scatter(st.session_state.X_pca, st.session_state.y)
                else:
                    fig = plot_3d_scatter(st.session_state.X_pca, st.session_state.y)
                st.plotly_chart(fig)
                
                st.subheader('Explained Variance Ratio')
                fig = plot_explained_variance(st.session_state.explained_variance_ratio)
                st.plotly_chart(fig)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What is the main purpose of PCA?",
                "options": [
                    "To classify data",
                    "To reduce dimensionality while retaining important information",
                    "To increase the number of features",
                    "To normalize the data"
                ],
                "correct": 1,
                "explanation": "PCA is primarily used for dimensionality reduction, transforming high-dimensional data into a lower-dimensional space while keeping the most important information."
            },
            {
                "question": "What does the explained variance ratio represent?",
                "options": [
                    "The accuracy of the PCA model",
                    "The proportion of variance explained by each principal component",
                    "The number of features in the original dataset",
                    "The classification error rate"
                ],
                "correct": 1,
                "explanation": "The explained variance ratio shows the proportion of variance in the original data that is explained by each principal component."
            },
            {
                "question": "Why is the data standardized before applying PCA?",
                "options": [
                    "To make the data more visually appealing",
                    "To ensure all features contribute equally to the principal components",
                    "To increase the number of features",
                    "To classify the data"
                ],
                "correct": 1,
                "explanation": "Standardizing the data ensures that all features are on the same scale and contribute equally to the computation of principal components, preventing features with larger scales from dominating the analysis."
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
