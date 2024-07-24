import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.graph_objects as go
import plotly.express as px
from sklearn.impute import SimpleImputer


# Set page configuration and custom CSS
st.set_page_config(page_title="Interest Group Clustering Analysis", layout="wide")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 24px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #e6f3ff;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #3498db;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    interest_data = pd.read_csv('kaggle_Interests_group.csv')
    features = [col for col in interest_data.columns if col.startswith('interest')]
    X = interest_data[features]
    
    # Impute NaN values with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled)
    
    return X_3d, features, interest_data, X_scaled

def cluster_data(X, algorithm, n_clusters=None, eps=None, min_samples=None):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == "BIRCH":
        model = Birch(n_clusters=n_clusters)
    else:  # Gaussian Mixture
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    
    labels = model.fit_predict(X)
    
    if algorithm != "DBSCAN":
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        st.sidebar.metric("Silhouette Score", f"{silhouette:.3f}")
        st.sidebar.metric("Calinski-Harabasz Score", f"{calinski:.3f}")
    
    return labels

def create_3d_scatter(X, labels, group_names, feature_names, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.8),
        text=[f"Group: {group}<br>Cluster: {label}" for group, label in zip(group_names, labels)],
        hoverinfo='text'
    )])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", zaxis_title="PCA Component 3"),
        height=700,
        margin=dict(r=0, b=0, l=0, t=40)
    )
    return fig

def interpretation_guide():
    st.markdown("""
    <div style="background-color: #e6f3ff; padding: 20px; border-radius: 10px;">
    <h3 style="color: #2c3e50;">Key Points:</h3>
    <ul>
        <li>Each point represents an interest group from the dataset.</li>
        <li>Colors indicate the cluster assigned by the chosen algorithm.</li>
        <li>Proximity of points suggests similarity in interest patterns.</li>
    </ul>

    <h3 style="color: #2c3e50;">Example:</h3>
    <p>If you see a tight group of blue points in one area, this could represent a cluster of interest groups with similar interest patterns.</p>

    <h3 style="color: #2c3e50;">What to Look For:</h3>
    <ol>
        <li><strong>Well-separated clusters:</strong> Distinct groups of colors might indicate that the algorithm has effectively separated different types of interest groups based on their interest patterns.</li>
        <li><strong>Mixed clusters:</strong> Areas where colors are intermingled could suggest overlapping characteristics between different interest group types.</li>
        <li><strong>Outliers:</strong> Isolated points might represent interest groups with unique interest patterns.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def explain_algorithms():
    st.subheader("K-Means")
    st.markdown("""
    <div style="background-color: #e8f8f5; padding: 20px; border-radius: 10px;">
    <p>K-Means tries to find a specified number of cluster centers and assign each interest group to the nearest center.</p>
    <p><strong>Pros:</strong> Simple, fast, and works well on globular clusters.</p>
    <p><strong>Cons:</strong> Sensitive to initial centroids, assumes spherical clusters, and requires specifying the number of clusters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Agglomerative Clustering")
    st.markdown("""
    <div style="background-color: #fef9e7; padding: 20px; border-radius: 10px;">
    <p>This algorithm starts with each interest group as its own cluster and progressively merges the closest clusters.</p>
    <p><strong>Pros:</strong> Can uncover hierarchical structure in data, doesn't assume cluster shape.</p>
    <p><strong>Cons:</strong> Computationally intensive for large datasets, can be sensitive to noise.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("DBSCAN")
    st.markdown("""
    <div style="background-color: #f4ecf7; padding: 20px; border-radius: 10px;">
    <p>DBSCAN groups together interest groups that are closely packed in the feature space, marking groups in low-density regions as outliers.</p>
    <p><strong>Pros:</strong> Can find arbitrarily shaped clusters, robust to outliers, doesn't require specifying number of clusters.</p>
    <p><strong>Cons:</strong> Sensitive to parameters, struggles with varying density clusters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("BIRCH")
    st.markdown("""
    <div style="background-color: #eaeded; padding: 20px; border-radius: 10px;">
    <p>BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) builds a tree structure to incrementally cluster the data.</p>
    <p><strong>Pros:</strong> Efficient for large datasets, handles outliers well.</p>
    <p><strong>Cons:</strong> May not work well with non-spherical clusters, sensitive to data order.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Gaussian Mixture")
    st.markdown("""
    <div style="background-color: #ebf5fb; padding: 20px; border-radius: 10px;">
    <p>Gaussian Mixture Models assume the data is generated from a mixture of a finite number of Gaussian distributions with unknown parameters.</p>
    <p><strong>Pros:</strong> Flexible, can model complex cluster shapes, provides probabilistic cluster assignments.</p>
    <p><strong>Cons:</strong> Sensitive to initialization, can overfit with too many components.</p>
    </div>
    """, unsafe_allow_html=True)

def cluster_insights(interest_data, labels, feature_names):
    df = interest_data.copy()
    df['Cluster'] = labels
    
    for cluster in sorted(df['Cluster'].unique()):
        st.subheader(f"Cluster {cluster}")
        cluster_data = df[df['Cluster'] == cluster]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Groups", len(cluster_data))
            st.write("Group Types Distribution:")
            st.write(cluster_data['group'].value_counts())
        
        with col2:
            st.write("Top 10 Interests in this Cluster:")
            top_interests = cluster_data[feature_names].mean().nlargest(10)
            for interest, value in top_interests.items():
                st.write(f"{interest}: {value:.2f}")
    
    st.subheader("Conclusions")
    st.write("""
    Based on the cluster analysis, we can draw the following conclusions:
    1. There appear to be distinct groups of interest patterns among the dataset.
    2. Some clusters show higher interest in certain topics, possibly indicating specialized interest groups.
    3. Other clusters exhibit more diverse interest patterns, suggesting general or broad-topic interest groups.
    4. The clustering reveals the diversity of interest groups and their focus areas.
    5. Further investigation into outlier groups could provide insights into unique or niche interest areas.
    """)

def data_quality_check(interest_data):
    st.subheader("Data Quality Check")
    
    # Check for missing values
    missing_values = interest_data.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Warning: Dataset contains missing values")
        st.write("Columns with missing values:")
        st.write(missing_values[missing_values > 0])
        st.write("Missing values have been imputed with column means for analysis.")
    else:
        st.success("No missing values found in the dataset.")
    
    # Check for duplicate rows
    duplicates = interest_data.duplicated().sum()
    if duplicates > 0:
        st.warning(f"Warning: Dataset contains {duplicates} duplicate rows")
    else:
        st.success("No duplicate rows found in the dataset.")

    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(interest_data.describe())

def quiz():
    questions = [
        {
            "question": "Which clustering algorithm is particularly useful for discovering groups of interest patterns without specifying the number of clusters?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN", "BIRCH"],
            "correct": 2,
            "explanation": "DBSCAN is particularly useful for discovering groups without specifying the number of clusters beforehand. It can identify clusters of arbitrary shape and is robust to outliers."
        },
        {
            "question": "What does a high silhouette score indicate in the context of interest group clustering?",
            "options": ["Many outliers", "Well-defined clusters", "Poor clustering quality", "High dimensional data"],
            "correct": 1,
            "explanation": "A high silhouette score indicates well-defined clusters. It measures how similar an object is to its own cluster compared to other clusters, with higher values indicating better-defined clusters."
        },
        {
            "question": "Which algorithm might be most suitable for clustering a very large dataset of interest groups?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN", "BIRCH"],
            "correct": 3,
            "explanation": "BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is particularly suitable for very large datasets as it's designed to minimize I/O costs and can handle large datasets efficiently."
        },
        {
            "question": "What could closely packed points of different colors in the 3D visualization suggest about the interest groups?",
            "options": ["Clear distinct clusters", "Overlapping interests between different group types", "Outlier interest groups", "Uniform interest distribution"],
            "correct": 1,
            "explanation": "Closely packed points of different colors could suggest overlapping interests between different group types. This indicates that while the algorithm has assigned them to different clusters, their interest patterns are similar."
        },
        {
            "question": "Which clustering algorithm assumes that the interest patterns are generated from a mixture of Gaussian distributions?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture"],
            "correct": 3,
            "explanation": "Gaussian Mixture Models assume the data (in this case, interest patterns) is generated from a mixture of a finite number of Gaussian distributions with unknown parameters."
        }
    ]

    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        user_answer = st.radio(q["question"], q["options"])
        if st.button(f"Check Answer {i+1}"):
            if q["options"].index(user_answer) == q["correct"]:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is: {q['options'][q['correct']]}")
            st.markdown(f"<div style='background-color: #e8f8f5; padding: 15px; border-radius: 10px;'><strong>Explanation:</strong> {q['explanation']}</div>", unsafe_allow_html=True)

def dataset_overview(interest_data):
    st.subheader("Sample Data")
    st.dataframe(interest_data.head())

    st.subheader("Dataset Statistics")
    total_groups = len(interest_data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Interest Groups", total_groups)
        st.metric("Average Total Interests", f"{interest_data['grand_tot_interests'].mean():.2f}")

    with col2:
        group_counts = interest_data['group'].value_counts()
        st.metric("Most Common Group Type", f"{group_counts.index[0]} ({group_counts.iloc[0]})")
        st.metric("Number of Interest Categories", len([col for col in interest_data.columns if col.startswith('interest')]))

    st.subheader("Interest Distribution")
    interest_cols = [col for col in interest_data.columns if col.startswith('interest')]
    interest_data_melted = pd.melt(interest_data, id_vars=['group'], value_vars=interest_cols, var_name='Interest', value_name='Value')
    interest_data_melted = interest_data_melted[interest_data_melted['Value'] > 0]  # Filter out zero values
    fig = px.box(interest_data_melted, x='Interest', y='Value', color='group', title="Distribution of Non-Zero Interest Values by Group")
    fig.update_layout(xaxis={'tickangle': 90})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Group Type Distribution")
    fig = px.pie(interest_data, names='group', title="Distribution of Group Types")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Between Interests")
    corr_matrix = interest_data[interest_cols].corr()
    fig = px.imshow(corr_matrix, title="Correlation Heatmap of Interests")
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Interactive Interest Group Clustering Analysis")

    # Sidebar
    with st.sidebar:
        st.header("Clustering Settings")
        algorithm = st.selectbox("Select Clustering Algorithm", 
                                 ["K-Means", "Agglomerative Clustering", "DBSCAN", "BIRCH", "Gaussian Mixture"])
        n_clusters = st.slider("Number of Clusters", 2, 10, 3) if algorithm != "DBSCAN" else None
        eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5) if algorithm == "DBSCAN" else None
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5) if algorithm == "DBSCAN" else None

    # Load and process data
    X_3d, feature_names, interest_data, X_scaled = load_and_process_data()

    # Perform clustering
    labels = cluster_data(X_scaled, algorithm, n_clusters, eps, min_samples)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Visualization", "📘 Interpretation", "🧠 Algorithms", "🔍 Cluster Insights", "🎓 Quiz", "📋 Dataset"])

    with tab1:
        st.header("3D Visualization of Clustering Results")
        fig = create_3d_scatter(X_3d, labels, interest_data['group'], feature_names, f"{algorithm} Clustering")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("How to Interpret the Visualization")
        interpretation_guide()

    with tab3:
        st.header("Understanding the Clustering Algorithms")
        explain_algorithms()

    with tab4:
        st.header("Cluster Insights")
        cluster_insights(interest_data, labels, feature_names)

    with tab5:
        st.header("Test Your Understanding")
        quiz()

    with tab6:
        st.header("Dataset Overview")
        data_quality_check(interest_data)
        dataset_overview(interest_data)

if __name__ == "__main__":
    main()
