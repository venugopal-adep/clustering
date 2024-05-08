import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.cluster import AffinityPropagation, Birch
from sklearn.decomposition import PCA
import plotly.express as px

def load_data():
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def plot_clusters(data, labels):
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = labels
    
    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster', labels={'Cluster': 'Cluster'}, title='PCA 3D Cluster Plot')
    fig.update_traces(marker=dict(size=5))
    return fig

def main():
    st.title('Cluster Analysis with Affinity Propagation and Birch')
    
    data = load_data()
    features = st.multiselect('Select features to cluster', options=data.columns[:-1].tolist(), default=data.columns[:4].tolist())
    
    method = st.selectbox('Select clustering method', options=['Affinity Propagation', 'Birch'])

    if method == 'Birch':
        threshold = st.slider('Select threshold for Birch', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    else:
        damping = st.slider('Select damping for Affinity Propagation', min_value=0.5, max_value=1.0, value=0.9, step=0.1)
    
    if method == 'Affinity Propagation':
        model = AffinityPropagation(damping=damping)
    elif method == 'Birch':
        model = Birch(threshold=threshold)
        
    labels = model.fit_predict(data[features])
    fig = plot_clusters(data[features], labels)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
