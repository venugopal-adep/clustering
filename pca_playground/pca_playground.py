import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

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
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y, title='PCA 2D Scatter Plot', labels={'x': 'PC1', 'y': 'PC2'})
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)

def plot_3d_scatter(X_pca, y):
    fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=y, title='PCA 3D Scatter Plot', labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'})
    fig.update_layout(scene=dict(xaxis=dict(backgroundcolor='rgba(0,0,0,0)'), yaxis=dict(backgroundcolor='rgba(0,0,0,0)'), zaxis=dict(backgroundcolor='rgba(0,0,0,0)')))
    st.plotly_chart(fig)

def plot_explained_variance(explained_variance_ratio):
    fig = px.bar(x=range(1, len(explained_variance_ratio) + 1), y=explained_variance_ratio, title='Explained Variance Ratio', labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'})
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)

def plot_image_grid(X, y):
    if len(X[0]) == 64:  # Digits dataset
        img_size = (8, 8)
    else:
        return
    
    fig = go.Figure(data=[go.Image(z=X[y == label][0].reshape(img_size)) for label in np.unique(y)])
    fig.update_layout(title='Sample Images from the Dataset', grid=dict(rows=1, columns=len(np.unique(y))), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title='PCA Explorer', layout='wide')
    st.title('PCA Explorer 🔍')
    st.write('Explore the wonders of Principal Component Analysis (PCA) with interactive visualizations!')
    
    dataset_name = st.sidebar.selectbox('Select a dataset', ['Iris', 'Wine', 'Breast Cancer', 'Digits'])
    n_components = st.sidebar.slider('Number of principal components', min_value=2, max_value=3, value=2)
    
    X, y, feature_names = load_dataset(dataset_name)
    X_pca, explained_variance_ratio = perform_pca(X, n_components)
    
    st.subheader('Original Data')
    if dataset_name in ['Digits']:
        plot_image_grid(X, y)
    else:
        st.write(pd.DataFrame(X, columns=feature_names))
    
    st.subheader('PCA Results')
    if n_components == 2:
        plot_2d_scatter(X_pca, y)
    else:
        plot_3d_scatter(X_pca, y)
    
    st.subheader('Explained Variance Ratio')
    plot_explained_variance(explained_variance_ratio)
    
    st.sidebar.subheader('How it works')
    st.sidebar.write('PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while retaining the most important information.')
    st.sidebar.write('1. The original data is standardized to have zero mean and unit variance.')
    st.sidebar.write('2. The principal components are computed by finding the directions of maximum variance in the data.')
    st.sidebar.write('3. The data is projected onto the selected principal components.')
    st.sidebar.write('4. The resulting lower-dimensional data is visualized using scatter plots.')
    st.sidebar.write('5. The explained variance ratio shows the proportion of variance explained by each principal component.')

if __name__ == '__main__':
    main()