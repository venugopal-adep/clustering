import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Set page title
st.set_page_config(page_title="Gaussian Mixture Model Demo")

# Function to calculate BIC (Bayesian Information Criterion)
def calculate_bic(data, max_k):
    bic = []
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        gmm.fit(data)
        bic.append(gmm.bic(data))
    return bic

# Function to calculate silhouette scores
def calculate_silhouette_scores(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        labels = gmm.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    return silhouette_scores

# Function to visualize the BIC curve and silhouette score
def plot_bic_and_silhouette(bic, silhouette_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(2, len(bic) + 2)), y=bic, mode='lines+markers', name='BIC'))
    fig.add_trace(go.Scatter(x=list(range(2, len(silhouette_scores) + 2)), y=silhouette_scores, mode='lines+markers', name='Silhouette Score', yaxis='y2'))
    fig.update_layout(title='BIC and Silhouette Score',
                      xaxis_title='Number of Components',
                      yaxis=dict(title='BIC'),
                      yaxis2=dict(title='Silhouette Score', overlaying='y', side='right'))
    st.plotly_chart(fig)

# Function to visualize the clusters
def plot_clusters(data, gmm):
    fig = go.Figure()
    for i in range(gmm.n_components):
        cluster_data = data[gmm.predict(data) == i]
        fig.add_trace(go.Scatter(x=cluster_data[:, 0], y=cluster_data[:, 1], mode='markers', name=f'Cluster {i+1}'))
    fig.update_layout(title='Clustering Results',
                      xaxis_title='X',
                      yaxis_title='Y')
    st.plotly_chart(fig)

# Function to find the optimal number of components based on the BIC
def find_optimal_k_bic(bic):
    return bic.index(min(bic)) + 2

# Function to find the optimal number of components based on the silhouette score
def find_optimal_k_silhouette(silhouette_scores):
    return silhouette_scores.index(max(silhouette_scores)) + 2

# Streamlit app
def main():
    st.title("Gaussian Mixture Model Demo")
    
    # Sidebar options
    st.sidebar.title("Parameters")
    n_samples = st.sidebar.slider("Number of samples", min_value=100, max_value=1000, value=200, step=100)
    n_features = st.sidebar.slider("Number of features", min_value=2, max_value=5, value=2, step=1)
    max_k = st.sidebar.slider("Maximum number of components", min_value=5, max_value=20, value=10, step=1)
    generate_data = st.sidebar.button("Generate Data")
    
    if generate_data:
        # Generate sample data with a random number of centers between 3 and 6
        n_centers = np.random.randint(3, 7)
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=np.random.randint(100))
        
        # Calculate BIC and silhouette scores for different numbers of components
        bic = calculate_bic(X, max_k)
        silhouette_scores = calculate_silhouette_scores(X, max_k)
        
        # Plot the BIC curve and silhouette score
        st.subheader("BIC and Silhouette Score")
        plot_bic_and_silhouette(bic, silhouette_scores)
        
        # Optimal number of components based on the BIC
        bic_optimal_k = find_optimal_k_bic(bic)
        st.subheader(f"Optimal Number of Components (BIC): {bic_optimal_k}")
        
        # Optimal number of components based on the silhouette score
        silhouette_optimal_k = find_optimal_k_silhouette(silhouette_scores)
        st.subheader(f"Optimal Number of Components (Silhouette Score): {silhouette_optimal_k}")
        
        # Choose the best number of components considering both BIC and silhouette score
        optimal_k = silhouette_optimal_k if silhouette_optimal_k == bic_optimal_k else bic_optimal_k
        st.subheader(f"Best Number of Components: {optimal_k}")
        
        # Perform GMM clustering with the optimal number of components
        gmm = GaussianMixture(n_components=optimal_k, covariance_type='full', random_state=0)
        gmm.fit(X)
        
        # Plot the clustering results
        st.subheader("Clustering Results")
        plot_clusters(X, gmm)
        
    # Brief explanation of how to use the UI
    st.sidebar.title("How to Use")
    st.sidebar.info(
        "1. Adjust the parameters in the sidebar.\n"
        "2. Click the 'Generate Data' button to generate sample data.\n"
        "3. The BIC curve and silhouette score will be plotted.\n"
        "4. The optimal number of components based on BIC and silhouette score will be shown.\n"
        "5. The best number of components considering both metrics will be displayed.\n"
        "6. The clustering results will be visualized based on the best number of components."
    )
    
    # Brief explanation of the concepts involved
    st.sidebar.title("Concepts Involved")
    st.sidebar.info(
        "- Gaussian Mixture Model (GMM): A probabilistic model that assumes the data is generated from a mixture of Gaussian distributions.\n"
        "- Bayesian Information Criterion (BIC): A criterion used for model selection that balances the goodness of fit and model complexity.\n"
        "- Silhouette Score: A metric that measures the quality of clustering based on the similarity of data points within clusters and dissimilarity between clusters.\n"
        "- Optimal Number of Components: The number of Gaussian components that best represents the underlying structure of the data."
    )

if __name__ == '__main__':
    main()