import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score


def main():
    st.title('Clustering and PCA Visualization')
    st.subheader('Interactive exploration of clustering and dimensionality reduction')

    # File uploader allows user to add their own CSV
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        with st.sidebar:
            show_data = st.checkbox('Show Dataset', False)
            show_clustering = st.checkbox('Show Clustering', True)
            show_pca = st.checkbox('Show PCA', True)

        if show_data:
            st.subheader('Dataset')
            st.write(data)

        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if show_clustering:
            st.subheader('Clustering')
            selected_columns = st.multiselect("Select columns for clustering", numeric_columns, default=numeric_columns[:2])
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
            
            clustering_algorithm = st.selectbox("Select clustering algorithm", 
                                                ["KMeans", "KMedoids", "SpectralClustering", "GaussianMixture", "DBSCAN"])
            
            if len(selected_columns) < 2:
                st.warning("Please select at least 2 columns for clustering.")
            else:
                X = data[selected_columns]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if clustering_algorithm == "KMeans":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                elif clustering_algorithm == "KMedoids":
                    clusterer = KMedoids(n_clusters=n_clusters, random_state=42)
                elif clustering_algorithm == "SpectralClustering":
                    clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
                elif clustering_algorithm == "GaussianMixture":
                    clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                else:  # DBSCAN
                    eps = st.slider("Select epsilon", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
                    min_samples = st.slider("Select min samples", min_value=2, max_value=10, value=5)
                    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                
                labels = clusterer.fit_predict(X_scaled)
                data['Cluster'] = labels

                if clustering_algorithm != "DBSCAN":
                    st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")

                fig = px.scatter(data, x=selected_columns[0], y=selected_columns[1], color='Cluster', 
                                 title='Clustering Result', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

                if len(selected_columns) >= 3:
                    fig_3d = px.scatter_3d(data, x=selected_columns[0], y=selected_columns[1], z=selected_columns[2], 
                                           color='Cluster', title='3D Clustering Result', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_3d, use_container_width=True)


                cluster_counts = data['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                fig_counts = px.bar(cluster_counts, x='Cluster', y='Count', title='Cluster Sizes', color='Cluster',
                                    color_continuous_scale='Viridis')
                st.plotly_chart(fig_counts, use_container_width=True)

        if show_pca:
            st.subheader('PCA')
            selected_columns = st.multiselect("Select columns for PCA", numeric_columns, default=numeric_columns[:5])
            n_components = st.slider("Select number of components", min_value=2, max_value=3, value=2)

            if len(selected_columns) < 2:
                st.warning("Please select at least 2 columns for PCA.")
            else:
                X = data[selected_columns]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(X_scaled)

                pca_data = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
                pca_data['Cluster'] = labels

                if n_components == 2:
                    fig_pca = px.scatter(pca_data, x='PC1', y='PC2', color='Cluster', title='PCA 2D Visualization', 
                                         color_continuous_scale='Viridis')
                    st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    fig_pca_3d = px.scatter_3d(pca_data, x='PC1', y='PC2', z='PC3', color='Cluster', 
                                               title='PCA 3D Visualization', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_pca_3d, use_container_width=True)

                explained_variance_ratio = pca.explained_variance_ratio_
                explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio)

                fig_explained_variance = px.area(
                    x=range(1, len(explained_variance_ratio) + 1),
                    y=explained_variance_ratio_cumsum,
                    labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'},
                    title='Explained Variance by Components'
                )
                st.plotly_chart(fig_explained_variance, use_container_width=True)

    else:
        st.info('Waiting for CSV file to be uploaded.')

if __name__ == '__main__':
    main()
