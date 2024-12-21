import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
import pandas as pd
from typing import Tuple, List
import plotly.figure_factory as ff

class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4):
        """Initialize KMeans clustering algorithm"""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.iteration_history = []
        
    def fit(self, X: np.ndarray) -> 'KMeans':
        """Fit KMeans to the data and store iteration history"""
        np.random.seed(42)
        idx = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centroids = X[idx]
        self.iteration_history = [(X.copy(), self.centroids.copy(), None)]
        
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Store current state
            self.iteration_history.append((X.copy(), self.centroids.copy(), labels))
            
            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) 
                                    for k in range(self.n_clusters)])
            
            # Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
                
            self.centroids = new_centroids
        
        self.labels_ = labels
        return self
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each data point to nearest centroid"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for input data"""
        return self._assign_clusters(X)

def create_clustering_plot(X: np.ndarray, kmeans: KMeans, iteration: int = -1) -> go.Figure:
    """Create interactive plotting using Plotly"""
    data = X.copy()
    centroids = kmeans.iteration_history[iteration][1]
    labels = kmeans.iteration_history[iteration][2]
    
    if labels is None:
        labels = np.zeros(len(data))
    
    df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2'])
    df['Cluster'] = labels
    
    fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Cluster',
                    title=f'K-means Clustering (Iteration {iteration})',
                    color_continuous_scale='viridis')
    
    # Add centroids
    fig.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(symbol='x', size=15, color='red', line=dict(width=2)),
        name='Centroids'
    ))
    
    fig.update_layout(
        title_x=0.5,
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig

def calculate_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Calculate distances between points and centroids"""
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def main():
    st.set_page_config(page_title="K-means Explorer", layout="wide")
    
    st.title("🎯 Interactive K-means Clustering Explorer")
    st.markdown("""
    This application demonstrates the K-means clustering algorithm with detailed visualizations 
    and mathematical calculations.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Parameters")
        n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)
        n_samples = st.slider("Number of Samples", 50, 500, 300)
        cluster_std = st.slider("Cluster Standard Deviation", 0.1, 2.0, 0.6)
        show_calculations = st.checkbox("Show Detailed Calculations", True)
        
        if st.button("Generate New Data"):
            st.session_state.X, _ = make_blobs(
                n_samples=n_samples,
                centers=n_clusters,
                cluster_std=cluster_std,
                random_state=np.random.randint(1000)
            )
    
    # Initialize or load data
    if 'X' not in st.session_state:
        st.session_state.X, _ = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=42
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Visualization")
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(st.session_state.X)
        
        # Iteration slider
        iteration = st.slider("Iteration", 0, len(kmeans.iteration_history)-1, 0)
        
        # Plot
        fig = create_clustering_plot(st.session_state.X, kmeans, iteration)
        st.plotly_chart(fig)
    
    with col2:
        st.header("Algorithm Details")
        
        st.markdown(r"""
        ## Mathematical Foundation

        ### 1. Distance Calculation
        The Euclidean distance between point $x$ and centroid $c$ is defined as:

        $$
        d(x,c) = \sqrt{\sum_{i=1}^n (x_i-c_i)^2}
        $$


        where:
        * $x_i$ is the $i^{th}$ coordinate of point $x$
        * $c_i$ is the $i^{th}$ coordinate of centroid $c$
        * $n$ is the number of dimensions

        ### 2. Centroid Update
        For each cluster $j$, the new centroid position is calculated as:

        $$
        c_j = \frac{1}{|S_j|}\sum_{x \in S_j} x
        $$


        where:
        * $S_j$ is the set of points in cluster $j$
        * $|S_j|$ is the number of points in cluster $j$
        * $x$ represents each point in the cluster

        ### 3. Objective Function
        The algorithm minimizes the within-cluster sum of squares:

        $$
        J = \sum_{j=1}^k \sum_{x \in S_j} \|x - c_j\|^2
        $$


        where:
        * $k$ is the number of clusters
        * $S_j$ is the set of points in cluster $j$
        * $c_j$ is the centroid of cluster $j$
        * $\|x - c_j\|^2$ is the squared Euclidean distance
        """)
        
        if show_calculations and iteration > 0:
            st.markdown("### Numerical Calculations")
            
            # Show sample calculations for first few points
            X = st.session_state.X[:5]  # Take first 5 points
            centroids = kmeans.iteration_history[iteration][1]
            
            distances = calculate_distances(X, centroids)
            
            # Create distance table
            df_distances = pd.DataFrame(
                distances,
                columns=[f'Centroid {i+1}' for i in range(n_clusters)]
            )
            df_distances.index = [f'Point {i+1}' for i in range(len(X))]
            
            st.markdown("#### Distance Matrix (First 5 Points)")
            st.dataframe(df_distances.style.highlight_min(axis=1))
            
            # Show cluster assignments
            assignments = kmeans.predict(X)
            df_assignments = pd.DataFrame({
                'Point': [f'Point {i+1}' for i in range(len(X))],
                'Assigned Cluster': assignments[:5]
            })
            
            st.markdown("#### Cluster Assignments")
            st.dataframe(df_assignments)
            
            # Show centroid updates
            if iteration > 0:
                old_centroids = kmeans.iteration_history[iteration-1][1]
                new_centroids = kmeans.iteration_history[iteration][1]
                
                df_centroids = pd.DataFrame({
                    'Centroid': [f'Centroid {i+1}' for i in range(n_clusters)],
                    'Old Position': [f'({old_centroids[i,0]:.2f}, {old_centroids[i,1]:.2f})' 
                                   for i in range(n_clusters)],
                    'New Position': [f'({new_centroids[i,0]:.2f}, {new_centroids[i,1]:.2f})' 
                                   for i in range(n_clusters)]
                })
                
                st.markdown("#### Centroid Updates")
                st.dataframe(df_centroids)

if __name__ == "__main__":
    main()
