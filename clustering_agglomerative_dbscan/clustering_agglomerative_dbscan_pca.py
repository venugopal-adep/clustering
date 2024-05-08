import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px

def load_data():
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def plot_clusters(data, labels):
    pca = PCA(n_components=3)  # Using 3 components for 3D plot
    principal_components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = labels
    
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3',
                        color='Cluster', labels={'Cluster': 'Cluster'},
                        title='PCA 3D Cluster Plot')
    fig.update_traces(marker=dict(size=5))
    return fig

def main():
    st.title('Cluster Analysis with Agglomerative Clustering and DBSCAN')
    
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
    
    labels = model.fit_predict(data[features])
    fig = plot_clusters(data[features], labels)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
