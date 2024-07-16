import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="ML Learning Explorer", page_icon="🤖")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .sub-header {
        font-size: 32px !important;
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
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🤖 Machine Learning Explorer: Supervised vs Unsupervised 🤖</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2 class='sub-header'>Controls</h2>", unsafe_allow_html=True)
num_points = st.sidebar.slider("Number of data points", 50, 500, 200)
noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.1)

# Generate data
@st.cache_data
def generate_data(num_points, noise):
    X, y = make_classification(n_samples=num_points, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, flip_y=noise,
                               random_state=42)
    return X, y

X, y = generate_data(num_points, noise_level)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Data Exploration", "🧠 Machine Learning Models", "📊 Model Comparison"])

with tab1:
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<p class='text-content'>Let's explore our generated dataset!</p>", unsafe_allow_html=True)
        
        df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
        df['Label'] = y
        st.dataframe(df.head())
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write(f"Total data points: {num_points}")
        st.write(f"Number of features: 2")
        st.write(f"Number of classes: 2")
        st.write(f"Class balance: {sum(y)/len(y):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Label', title="Data Distribution")
        st.plotly_chart(fig)

with tab2:
    st.markdown("<h2 class='sub-header'>Machine Learning Models</h2>", unsafe_allow_html=True)
    
    model_type = st.radio("Select Learning Type", ["Supervised (Logistic Regression)", "Unsupervised (K-Means Clustering)"])
    
    if model_type == "Supervised (Logistic Regression)":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write(f"Model: Logistic Regression")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualize decision boundary
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                             np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig = go.Figure(data=[
            go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', opacity=0.5, showscale=False),
            go.Scatter(x=X_test[:, 0], y=X_test[:, 1], mode='markers',
                       marker=dict(color=y_test, colorscale='RdBu', size=10))
        ])
        fig.update_layout(title="Logistic Regression Decision Boundary")
        st.plotly_chart(fig)
        
    else:
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write(f"Model: K-Means Clustering")
        st.write(f"Number of clusters: 2")
        st.markdown("</div>", unsafe_allow_html=True)
        
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=cluster_labels,
                         title="K-Means Clustering Results")
        fig.add_trace(go.Scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
                                 mode='markers', marker=dict(color='black', size=15, symbol='x')))
        st.plotly_chart(fig)

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

# Conclusion
st.markdown("<h2 class='sub-header'>Conclusion</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='text-content'>
This interactive demo showcases the fundamental differences between supervised and unsupervised machine learning approaches:

1. Supervised learning (Logistic Regression) uses labeled data to learn a decision boundary for classification.
2. Unsupervised learning (K-Means Clustering) finds patterns in data without using labels.
3. Both methods have their strengths and are suited for different types of problems.
4. The choice of method depends on your data, problem context, and specific goals.

Experiment with different numbers of data points and noise levels to see how they affect each model's performance!
</p>
""", unsafe_allow_html=True)

st.markdown("<p class='text-content' style='text-align: center; font-style: italic;'>Developed by: Venugopal Adep</p>", unsafe_allow_html=True)
