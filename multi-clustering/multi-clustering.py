import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris, make_s_curve
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, OPTICS, MeanShift, AffinityPropagation, Birch
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
import plotly.express as px
import plotly.graph_objects as go

def generate_data(dataset):
    if dataset == 'Blobs':
        X, y = make_blobs(n_samples=200, centers=3, n_features=3, random_state=42)
    elif dataset == 'Circles':
        X, y = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
        X = np.hstack((X, np.zeros((X.shape[0], 1)))) # Add third dimension for visualization
    elif dataset == 'Moons':
        X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
        X = np.hstack((X, np.zeros((X.shape[0], 1)))) # Add third dimension for visualization
    elif dataset == 'Iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
    elif dataset == 'S-Curve':
        X, y = make_s_curve(n_samples=200, random_state=42)
    return X, y

def perform_clustering(X, method, n_clusters):
    if method == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'K-Medoids':
        model = KMedoids(n_clusters=n_clusters, random_state=42)
    elif method == 'Spectral Clustering':
        model = SpectralClustering(n_clusters=n_clusters, random_state=42)
    elif method == 'Gaussian Mixture':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    elif method == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
    elif method == 'Agglomerative Clustering':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'OPTICS':
        model = OPTICS(min_samples=5)
    elif method == 'Mean Shift':
        model = MeanShift()
    elif method == 'Affinity Propagation':
        model = AffinityPropagation(random_state=42)
    elif method == 'Birch':
        model = Birch(n_clusters=n_clusters)
    elif method == 't-SNE':
        model = TSNE(n_components=3, random_state=42)
        X_transformed = model.fit_transform(X)
        return X_transformed
    
    model.fit(X)
    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
    return labels

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
    st.plotly_chart(fig)

def main():
    st.title('Clustering')
    st.caption('Venugopal Adep')

    dataset = st.sidebar.selectbox('Select a dataset', ['Blobs', 'Circles', 'Moons', 'Iris', 'S-Curve'])
    X, y = generate_data(dataset)

    method = st.sidebar.selectbox('Select a clustering method', ['K-Means', 'K-Medoids', 'Spectral Clustering', 'Gaussian Mixture', 'DBSCAN', 'Agglomerative Clustering', 'OPTICS', 'Mean Shift', 'Affinity Propagation', 'Birch', 't-SNE'])
    n_clusters = st.sidebar.slider('Number of clusters', min_value=2, max_value=10, value=3)

    if st.sidebar.button('Run Clustering'):
        if method == 't-SNE':
            X_transformed = perform_clustering(X, method, n_clusters)
            plot_3d(X_transformed, y)
        else:
            labels = perform_clustering(X, method, n_clusters)
            plot_3d(X, labels)

if __name__ == '__main__':
    main()
