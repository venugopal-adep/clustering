import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(layout="wide", page_title="Iris Clustering Explorer", page_icon="🌸")

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .sub-header {
        font-size: 24px !important;
        font-weight: bold;
        color: #4682B4;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .text-content {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .highlight {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #1E90FF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Plot clusters
def plot_clusters(data, labels):
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Cluster'] = labels
    
    # Create a more colorful and customized 3D scatter plot
    fig = go.Figure()

    # Custom color scale
    colors = px.colors.qualitative.Bold

    for i, cluster in enumerate(df_pca['Cluster'].unique()):
        cluster_data = df_pca[df_pca['Cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['PC1'],
            y=cluster_data['PC2'],
            z=cluster_data['PC3'],
            mode='markers',
            marker=dict(
                size=5,
                color=colors[i % len(colors)],
                opacity=0.8,
                symbol='circle'
            ),
            name=f'Cluster {cluster}'
        ))

    # Update layout for a more appealing look
    fig.update_layout(
        title='3D Cluster Visualization (PCA)',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            bgcolor='rgb(240,240,240)',
            xaxis=dict(gridcolor='white', gridwidth=2),
            yaxis=dict(gridcolor='white', gridwidth=2),
            zaxis=dict(gridcolor='white', gridwidth=2)
        ),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig

# Main app (same as before)
def main():
    st.markdown("<h1 class='main-header'>🌸 Iris Clustering Explorer: K-Means vs K-Medoids</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Explore clustering algorithms on the Iris dataset</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding Iris Dataset Clustering</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Iris Dataset</h3>
        <p class='text-content'>
        The Iris dataset is a classic dataset in machine learning, containing measurements for 150 iris flowers from three different species:
        
        1. Setosa
        2. Versicolor
        3. Virginica
        
        Each flower has four features measured: sepal length, sepal width, petal length, and petal width.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Clustering Algorithms</h3>
        <p class='text-content'>
        We'll use two clustering algorithms:
        
        1. K-Means: Groups data points by calculating the mean of each cluster and reassigning points to the nearest mean.
        2. K-Medoids: Similar to K-Means, but uses actual data points as cluster centers instead of calculated means.
        
        Both algorithms aim to group similar iris flowers together based on their measurements.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore Clustering on Iris Dataset</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            data = load_data()
            features = st.multiselect('Select features to cluster', options=data.columns[:-1].tolist(), default=data.columns[:4].tolist())
            
            n_clusters = st.slider('Select number of clusters', min_value=2, max_value=10, value=3, step=1)
            
            method = st.selectbox('Select clustering method', options=['K-means', 'K-medoids'])
            
            if st.button('Run Clustering'):
                if method == 'K-means':
                    model = KMeans(n_clusters=n_clusters)
                elif method == 'K-medoids':
                    model = KMedoids(n_clusters=n_clusters, random_state=0)
                
                labels = model.fit_predict(data[features])
                st.session_state.labels = labels
                st.session_state.features = features
        
        with col2:
            if 'labels' in st.session_state:
                fig = plot_clusters(data[st.session_state.features], st.session_state.labels)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "How many features does the Iris dataset have?",
                "options": ["2", "3", "4", "5"],
                "correct": 2,
                "explanation": "The Iris dataset has 4 features: sepal length, sepal width, petal length, and petal width."
            },
            {
                "question": "What's the main difference between K-Means and K-Medoids?",
                "options": [
                    "K-Means uses the mean as the center, while K-Medoids uses an actual data point",
                    "K-Means is faster, while K-Medoids is more accurate",
                    "K-Means works with categorical data, while K-Medoids doesn't",
                    "There's no difference, they're the same algorithm"
                ],
                "correct": 0,
                "explanation": "K-Means uses the average (mean) of points in a cluster as its center, which might not be an actual data point. K-Medoids always uses an actual data point as the center."
            },
            {
                "question": "Why do we use PCA before plotting the clusters?",
                "options": [
                    "To make the algorithm faster",
                    "To reduce the dimensionality for visualization",
                    "To improve clustering accuracy",
                    "To normalize the data"
                ],
                "correct": 1,
                "explanation": "We use PCA (Principal Component Analysis) to reduce the dimensionality of the data. This allows us to visualize high-dimensional data (like the 4D Iris dataset) in a 3D plot."
            }
        ]

        for i, q in enumerate(questions):
            st.markdown(f"<p class='text-content'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
            user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
            
            if st.button("Check Answer", key=f"check{i}"):
                if q['options'].index(user_answer) == q['correct']:
                    st.success("Correct! 🎉")
                else:
                    st.error("Incorrect. Try again! 🤔")
                st.info(q['explanation'])
            st.markdown("---")

if __name__ == "__main__":
    main()
