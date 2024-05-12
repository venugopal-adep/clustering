import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import plotly.express as px

def load_data():
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def plot_clusters(data, labels):
    pca = PCA(n_components=3)  # Updated to 3 components for 3D visualization
    principal_components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = labels
    
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', 
                        labels={'Cluster': 'Cluster'}, title='PCA 3D Cluster Plot')
    fig.update_traces(marker=dict(size=5))
    return fig

def main():
    st.write('## Spectral Clustering and Gaussian Mixture Models')
    st.write('**Developed by : Venugopal Adep**')
    
    data = load_data()
    features = st.multiselect('Select features to cluster', options=data.columns[:-1].tolist(), default=data.columns[:4].tolist())
    
    method = st.selectbox('Select clustering method', options=['Spectral Clustering', 'Gaussian Mixture Models'])

    if method == 'Spectral Clustering':
        n_clusters = st.slider('Select number of clusters', min_value=2, max_value=10, value=3, step=1)
        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    elif method == 'Gaussian Mixture Models':
        n_components = st.slider('Select number of components', min_value=2, max_value=10, value=3, step=1)
        covariance_type = st.selectbox('Select covariance type', ['full', 'tied', 'diag', 'spherical'])
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    
    # Perform clustering and update plot dynamically
    labels = model.fit_predict(data[features])
    fig = plot_clusters(data[features], labels)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
