import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Color palette
COLORS = {
    'primary': '#6C5B7B',
    'secondary': '#C06C84',
    'accent': '#F67280',
    'background': '#F8B195'
}

# Quiz questions and answers
QUIZ_DATA = [
    {
        "question": "What is the main purpose of clustering in data analysis?",
        "options": [
            "A) To predict future outcomes",
            "B) To group similar data points together",
            "C) To reduce the dimensionality of data",
            "D) To visualize data in 3D"
        ],
        "answer": "B",
        "explanation": "Clustering is used to group similar data points together. For example, in a retail business, clustering can be used to group customers with similar purchasing behaviors, allowing for more targeted marketing strategies."
    },
    {
        "question": "What does PCA stand for in the context of data analysis?",
        "options": [
            "A) Predictive Cluster Analysis",
            "B) Principal Component Analysis",
            "C) Programmatic Correlation Assessment",
            "D) Pattern Classification Algorithm"
        ],
        "answer": "B",
        "explanation": "PCA stands for Principal Component Analysis. It's a technique used to reduce the dimensionality of data while preserving as much information as possible. For instance, if you have a dataset with 100 features describing cars, PCA might help you reduce it to 10 principal components that capture the most important aspects of the data."
    },
    {
        "question": "Which of the following is NOT a common clustering algorithm?",
        "options": [
            "A) K-Means",
            "B) DBSCAN",
            "C) Random Forest",
            "D) Gaussian Mixture Model"
        ],
        "answer": "C",
        "explanation": "Random Forest is not a clustering algorithm; it's a machine learning algorithm used for classification and regression. The other options (K-Means, DBSCAN, and Gaussian Mixture Model) are all clustering algorithms. For example, K-Means might be used to group customers into segments based on their purchasing behavior, while Random Forest could be used to predict whether a customer will make a purchase or not."
    }
]

def main():
    st.set_page_config(page_title="Clustering and PCA Visualization", page_icon="📊", layout="wide")
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLORS['background']};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLORS['primary']};
            color: white;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            font-weight: bold;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['secondary']};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('🔍 Clustering and PCA Visualization')
    st.subheader('Interactive exploration of clustering and dimensionality reduction')

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        labels = None

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🧩 Clustering", "🔬 PCA", "❓ Quiz"])

        with tab1:
            st.subheader('Dataset')
            st.dataframe(data.style.background_gradient(cmap='viridis'), height=400)
            st.write(f"Shape of the dataset: {data.shape}")
            st.write("Summary statistics:")
            st.write(data.describe())

        with tab2:
            st.subheader('Clustering')
            col1, col2 = st.columns(2)
            with col1:
                selected_columns = st.multiselect("Select columns for clustering", numeric_columns, default=numeric_columns[:2])
                n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
            with col2:
                clustering_algorithm = st.selectbox("Select clustering algorithm", 
                                                    ["KMeans", "SpectralClustering", "GaussianMixture", "DBSCAN"])
            
            if len(selected_columns) < 2:
                st.warning("Please select at least 2 columns for clustering.")
            else:
                X = data[selected_columns]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if clustering_algorithm == "KMeans":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
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
                    st.metric("Silhouette Score", f"{silhouette_score(X_scaled, labels):.3f}")

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

        with tab3:
            st.subheader('PCA')
            col1, col2 = st.columns(2)
            with col1:
                selected_columns = st.multiselect("Select columns for PCA", numeric_columns, default=numeric_columns[:5])
            with col2:
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
                
                if labels is not None:
                    pca_data['Cluster'] = labels
                    color = 'Cluster'
                else:
                    color = None

                if n_components == 2:
                    fig_pca = px.scatter(pca_data, x='PC1', y='PC2', color=color, title='PCA 2D Visualization', 
                                         color_continuous_scale='Viridis')
                    st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    fig_pca_3d = px.scatter_3d(pca_data, x='PC1', y='PC2', z='PC3', color=color, 
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

        with tab4:
            st.subheader("📝 Quiz")
            st.write("Test your knowledge about clustering and PCA!")

            for i, quiz_item in enumerate(QUIZ_DATA, 1):
                with st.expander(f"Question {i}: {quiz_item['question']}"):
                    for option in quiz_item['options']:
                        st.write(option)
                    
                    if st.button(f"Show Answer for Question {i}"):
                        st.success(f"The correct answer is: {quiz_item['answer']}")
                        st.info(f"Explanation: {quiz_item['explanation']}")

    else:
        st.info('Waiting for CSV file to be uploaded.')

if __name__ == '__main__':
    main()
