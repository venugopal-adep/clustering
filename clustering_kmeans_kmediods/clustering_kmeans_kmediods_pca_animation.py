import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids  # Corrected import
from sklearn.decomposition import PCA
import plotly.express as px

def load_data():
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def plot_clusters(data, labels, n_clusters):
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = labels
    
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', 
                        title=f'PCA 3D Cluster Plot with {n_clusters} Clusters')
    fig.update_traces(marker=dict(size=5))
    return fig

def main():
    st.title('Animated Cluster Analysis with K-means and K-medoids')
    
    if 'n_clusters' not in st.session_state:
        st.session_state.n_clusters = 1  # Initialize session state for cluster count
    
    data = load_data()
    features = st.multiselect('Select features to cluster', options=data.columns[:-1].tolist(), default=data.columns[:4].tolist())
    method = st.selectbox('Select clustering method', options=['K-means', 'K-medoids'])

    # Display the slider to show current number of clusters
    n_clusters = st.slider('Number of Clusters', min_value=1, max_value=10, value=st.session_state.n_clusters)
    
    if method == 'K-means':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'K-medoids':
        model = KMedoids(n_clusters=n_clusters, random_state=0)
    
    labels = model.fit_predict(data[features])
    fig = plot_clusters(data[features], labels, n_clusters)
    st.plotly_chart(fig, use_container_width=True)
    
    # Increment the cluster count
    if n_clusters < 10:
        st.session_state.n_clusters += 1
    else:
        st.session_state.n_clusters = 1  # Reset to 1 after reaching 10

    # Re-run the app to update the plot
    st.button("Click to animate", on_click=lambda: st.experimental_rerun())

if __name__ == "__main__":
    main()
