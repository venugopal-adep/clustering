import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set page title
st.set_page_config(page_title="K-Means Elbow Method Demo")

# Function to calculate WCSS (Within-Cluster Sum of Squares)
def calculate_wcss(data, max_k):
    wcss = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Function to calculate silhouette scores
def calculate_silhouette_scores(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)
    return silhouette_scores

# Function to visualize the elbow curve and silhouette score
def plot_elbow_and_silhouette(wcss, silhouette_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(2, len(wcss) + 2)), y=wcss, mode='lines+markers', name='WCSS'))
    fig.add_trace(go.Scatter(x=list(range(2, len(silhouette_scores) + 2)), y=silhouette_scores, mode='lines+markers', name='Silhouette Score', yaxis='y2'))
    fig.update_layout(title='Elbow Method and Silhouette Score',
                      xaxis_title='Number of Clusters (k)',
                      yaxis=dict(title='WCSS'),
                      yaxis2=dict(title='Silhouette Score', overlaying='y', side='right'))
    st.plotly_chart(fig)

# Function to visualize the clusters
def plot_clusters(data, kmeans):
    fig = go.Figure()
    for i in range(kmeans.n_clusters):
        cluster_data = data[kmeans.labels_ == i]
        fig.add_trace(go.Scatter(x=cluster_data[:, 0], y=cluster_data[:, 1], mode='markers', name=f'Cluster {i+1}'))
    fig.update_layout(title='Clustering Results',
                      xaxis_title='X',
                      yaxis_title='Y')
    st.plotly_chart(fig)

# Function to find the optimal number of clusters based on the elbow method
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

# Function to find the optimal number of clusters based on the silhouette score
def find_optimal_k_silhouette(silhouette_scores):
    return silhouette_scores.index(max(silhouette_scores)) + 2

# Streamlit app
def main():
    st.title("K-Means Elbow Method Demo")
    
    # Sidebar options
    st.sidebar.title("Parameters")
    n_samples = st.sidebar.slider("Number of samples", min_value=100, max_value=1000, value=200, step=100)
    n_features = st.sidebar.slider("Number of features", min_value=2, max_value=5, value=2, step=1)
    max_k = st.sidebar.slider("Maximum number of clusters", min_value=5, max_value=20, value=10, step=1)
    generate_data = st.sidebar.button("Generate Data")
    
    if generate_data:
        # Generate sample data with a random number of centers between 3 and 6
        n_centers = np.random.randint(3, 7)
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=np.random.randint(100))
        
        # Calculate WCSS and silhouette scores for different values of k
        wcss = calculate_wcss(X, max_k)
        silhouette_scores = calculate_silhouette_scores(X, max_k)
        
        # Plot the elbow curve and silhouette score
        st.subheader("Elbow Curve and Silhouette Score")
        plot_elbow_and_silhouette(wcss, silhouette_scores)
        
        # Optimal number of clusters based on the elbow method
        wcss_optimal_k = find_optimal_k_elbow(wcss)
        st.subheader(f"Optimal Number of Clusters (Elbow Method): {wcss_optimal_k}")
        
        # Optimal number of clusters based on the silhouette score
        silhouette_optimal_k = find_optimal_k_silhouette(silhouette_scores)
        st.subheader(f"Optimal Number of Clusters (Silhouette Score): {silhouette_optimal_k}")
        
        # Choose the best value of k considering both the elbow method and silhouette score
        optimal_k = silhouette_optimal_k if silhouette_optimal_k == wcss_optimal_k else wcss_optimal_k
        st.subheader(f"Best Number of Clusters: {optimal_k}")
        
        # Perform k-means clustering with the optimal k
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        
        # Plot the clustering results
        st.subheader("Clustering Results")
        plot_clusters(X, kmeans)
        
    # Brief explanation of how to use the UI
    st.sidebar.title("How to Use")
    st.sidebar.info(
        "1. Adjust the parameters in the sidebar.\n"
        "2. Click the 'Generate Data' button to generate sample data.\n"
        "3. The elbow curve and silhouette score will be plotted.\n"
        "4. The optimal number of clusters based on the elbow method and silhouette score will be shown.\n"
        "5. The best number of clusters considering both metrics will be displayed.\n"
        "6. The clustering results will be visualized based on the best number of clusters."
    )
    
    # Brief explanation of the concepts involved
    st.sidebar.title("Concepts Involved")
    st.sidebar.info(
        "- K-Means Clustering: An unsupervised learning algorithm that partitions data into k clusters.\n"
        "- Elbow Method: A technique used to determine the optimal number of clusters based on the elbow point in the WCSS curve.\n"
        "- Silhouette Score: A metric that measures the quality of clustering based on the similarity of data points within clusters and dissimilarity between clusters.\n"
        "- Within-Cluster Sum of Squares (WCSS): A measure of the compactness of the clusters.\n"
        "- Optimal Number of Clusters: The value of k that balances the trade-off between WCSS and silhouette score."
    )

if __name__ == '__main__':
    main()