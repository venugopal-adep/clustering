import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(layout="wide", page_title="Advanced ML Explorer", page_icon="🚀")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px !important;
        font-weight: bold;
        color: #FF4500;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .sub-header {
        font-size: 36px !important;
        font-weight: bold;
        color: #FF6347;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .text-content {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .highlight {
        background-color: #FFF5EE;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #FF4500;
    }
    .stButton>button {
        background-color: #FF4500;
        color: white;
        font-size: 18px;
        padding: 12px 28px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6347;
        transform: scale(1.05);
    }
    .stSelectbox>div>div>select {
        background-color: #FFF5EE;
        color: #FF4500;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🚀 Supervised vs Unsupervised Learning 🚀</h1>", unsafe_allow_html=True)
st.write('**Developed by : Venugopal Adep**')
# Sidebar
st.sidebar.markdown("<h2 class='sub-header'>Controls</h2>", unsafe_allow_html=True)
num_points = st.sidebar.slider("Number of data points", 100, 1000, 500)
noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.1)
n_features = st.sidebar.slider("Number of features", 2, 10, 3)
n_clusters = st.sidebar.slider("Number of clusters (K-Means)", 2, 10, 3)
plot_dim = st.sidebar.radio("Plot Dimension", ["2D", "3D"])

# Generate data
@st.cache_data
def generate_data(num_points, noise, n_features):
    X, y = make_classification(n_samples=num_points, n_features=n_features, n_informative=n_features-1,
                               n_redundant=0, n_clusters_per_class=1, flip_y=noise,
                               random_state=42)
    return X, y

X, y = generate_data(num_points, noise_level, n_features)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Data Exploration", "🧠 Machine Learning Models", "📊 Model Comparison"])

with tab1:
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<p class='text-content'>Let's explore our generated dataset!</p>", unsafe_allow_html=True)
        
        df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(n_features)])
        df['Label'] = y
        st.dataframe(df.head())
        
        st.write(f"Total data points: {num_points}")
        st.write(f"Number of features: {n_features}")
        st.write(f"Number of classes: 2")
        st.write(f"Class balance: {sum(y)/len(y):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if plot_dim == "2D":
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y,
                             labels={'x': 'PC1', 'y': 'PC2'},
                             title="2D PCA Visualization of Data",
                             color_continuous_scale='viridis')
            fig.update_layout(width=800, height=600)
        else:
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)
            
            fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=y,
                                labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                                title="3D PCA Visualization of Data",
                                color_continuous_scale='viridis')
            fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                              width=800, height=600)
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h2 class='sub-header'>Machine Learning Models</h2>", unsafe_allow_html=True)
    
    model_type = st.radio("Select Learning Type", ["Supervised (Logistic Regression)", "Unsupervised (K-Means Clustering)"])
    
    if model_type == "Supervised (Logistic Regression)":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        #st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write(f"Model: Logistic Regression")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if plot_dim == "2D":
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            model_pca = LogisticRegression()
            model_pca.fit(X_pca, y)
            
            xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 200),
                                 np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 200))
            Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig = go.Figure(data=[
                go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', opacity=0.5, showscale=False),
                go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers',
                           marker=dict(color=y, colorscale='RdBu', size=10))
            ])
            fig.update_layout(title="2D Logistic Regression Decision Boundary",
                              xaxis_title='PC1', yaxis_title='PC2',
                              width=800, height=600)
        else:
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)
            model_pca = LogisticRegression()
            model_pca.fit(X_pca, y)
            
            xx, yy, zz = np.meshgrid(np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 20),
                                     np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 20),
                                     np.linspace(X_pca[:, 2].min(), X_pca[:, 2].max(), 20))
            Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig = go.Figure(data=[
                go.Scatter3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], mode='markers',
                             marker=dict(color=y, colorscale='viridis', size=5)),
                go.Volume(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=Z.flatten(),
                          isomin=0.5, isomax=0.5, opacity=0.1, surface_count=2,
                          colorscale='RdBu')
            ])
            fig.update_layout(title="3D Logistic Regression Decision Boundary",
                              scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                              width=800, height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write(f"Model: K-Means Clustering")
        st.write(f"Number of clusters: {n_clusters}")
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if plot_dim == "2D":
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=cluster_labels,
                             labels={'x': 'PC1', 'y': 'PC2'},
                             title=f"2D K-Means Clustering Results (K={n_clusters})",
                             color_continuous_scale='viridis')
            
            # Add cluster centers
            centers_pca = pca.transform(kmeans.cluster_centers_)
            fig.add_trace(go.Scatter(x=centers_pca[:, 0], y=centers_pca[:, 1],
                                     mode='markers', marker=dict(color='red', size=15, symbol='x')))
            
            fig.update_layout(width=800, height=600)
        else:
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)
            
            fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=cluster_labels,
                                labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                                title=f"3D K-Means Clustering Results (K={n_clusters})",
                                color_continuous_scale='viridis')
            
            # Add cluster centers
            centers_pca = pca.transform(kmeans.cluster_centers_)
            fig.add_trace(go.Scatter3d(x=centers_pca[:, 0], y=centers_pca[:, 1], z=centers_pca[:, 2],
                                       mode='markers', marker=dict(color='red', size=10, symbol='diamond')))
            
            fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                              width=800, height=600)
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)
    
    st.markdown("<p class='text-content'>Let's compare Supervised and Unsupervised learning approaches on our dataset.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3>Supervised Learning (Logistic Regression)</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='highlight'>
        <p><strong>Pros:</strong></p>
        <ul>
            <li>Can make precise predictions</li>
            <li>Provides clear decision boundaries</li>
            <li>Offers interpretable results</li>
        </ul>
        <p><strong>Cons:</strong></p>
        <ul>
            <li>Requires labeled data</li>
            <li>May overfit if not properly regularized</li>
            <li>Assumes linear decision boundary</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3>Unsupervised Learning (K-Means Clustering)</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='highlight'>
        <p><strong>Pros:</strong></p>
        <ul>
            <li>Doesn't require labeled data</li>
            <li>Can discover hidden patterns</li>
            <li>Useful for exploratory data analysis</li>
        </ul>
        <p><strong>Cons:</strong></p>
        <ul>
            <li>Results may be less interpretable</li>
            <li>Sensitive to initial conditions</li>
            <li>Requires specifying number of clusters</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p class='text-content'>The choice between supervised and unsupervised learning depends on your data, problem, and goals. Supervised learning is great when you have labeled data and want to make specific predictions. Unsupervised learning is useful for exploring data structure and finding patterns without predefined labels.</p>", unsafe_allow_html=True)

# Conclusion (continued)
st.markdown("""
<p class='text-content'>
This advanced interactive demo showcases the fundamental differences between supervised and unsupervised machine learning approaches:

1. Supervised learning (Logistic Regression) uses labeled data to learn a decision boundary for classification.
2. Unsupervised learning (K-Means Clustering) finds patterns in data without using labels.
3. Both methods have their strengths and are suited for different types of problems.
4. The choice of method depends on your data, problem context, and specific goals.
5. Dimensionality reduction techniques like PCA can help visualize high-dimensional data in 2D or 3D.

Experiment with different numbers of data points, features, noise levels, and clusters to see how they affect each model's performance!

Key takeaways:
- Supervised learning is powerful when you have labeled data and clear prediction goals.
- Unsupervised learning can reveal hidden patterns and is useful for exploratory data analysis.
- The effectiveness of each method can vary depending on the nature of your data and the problem you're trying to solve.
- Visualizing data and model results in both 2D and 3D can provide valuable insights into the underlying structure of your data and the performance of your models.

Remember, the goal of machine learning is not just to achieve high accuracy, but to gain meaningful insights from your data that can drive informed decision-making.
</p>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr>
<p style='text-align: center;'>
    <i>Supervised vs Unsupservised Learning</i><br>
    Developed by: Venugopal Adep<br>
</p>
""", unsafe_allow_html=True)
