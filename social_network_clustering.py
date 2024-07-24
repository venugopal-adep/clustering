#Dataset : https://www.kaggle.com/datasets/zabihullah18/students-social-network-profile-clustering
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from sklearn.impute import SimpleImputer

# Set page configuration and custom CSS
st.set_page_config(page_title="Marketing Clustering Analysis", layout="wide")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 24px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f8ff;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #e6f3ff;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff6e40;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff9e80;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Interactive Marketing Clustering Analysis")

    # Sidebar
    with st.sidebar:
        st.header("Clustering Settings")
        algorithm = st.selectbox("Select Clustering Algorithm", 
                                 ["K-Means", "Agglomerative Clustering", "DBSCAN", "BIRCH", "Gaussian Mixture"])
        n_clusters = st.slider("Number of Clusters", 2, 10, 3) if algorithm != "DBSCAN" else None
        eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5) if algorithm == "DBSCAN" else None
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5) if algorithm == "DBSCAN" else None

    # Load and process data
    X, feature_names, marketing_data = load_and_process_data()

    # Perform clustering
    labels = cluster_data(X, algorithm, n_clusters, eps, min_samples)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Visualization", "📘 Interpretation", "🧠 Algorithms", "🔍 Cluster Insights", "🎓 Quiz", "📋 Dataset"])

    with tab1:
        st.header("3D Visualization of Clustering Results")
        fig = create_3d_scatter(X, labels, marketing_data.index, feature_names, f"{algorithm} Clustering")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("How to Interpret the Visualization")
        interpretation_guide()

    with tab3:
        st.header("Understanding the Clustering Algorithms")
        explain_algorithms()

    with tab4:
        st.header("Cluster Insights")
        cluster_insights(marketing_data, labels, feature_names)

    with tab5:
        st.header("Test Your Understanding")
        quiz()

    with tab6:
        st.header("Dataset Overview")
        dataset_overview(marketing_data)

def load_and_process_data():
    # Load the Marketing dataset
    marketing_data = pd.read_csv('Clustering_Marketing.csv')
    
    # Select features for clustering
    features = ['age', 'NumberOffriends', 'basketball', 'football', 'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading', 'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot', 'kissed', 'dance', 'band', 'marching', 'music', 'rock', 'god', 'church', 'jesus', 'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes', 'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']
    X = marketing_data[features]
    
    # Handle 'age' column separately
    X['age'] = pd.to_datetime(X['age'], format='%d. %b', errors='coerce').dt.month
    
    # Identify numeric columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Impute missing values for numeric columns
    imputer = SimpleImputer(strategy='mean')
    X_imputed = X.copy()
    X_imputed[numeric_features] = imputer.fit_transform(X[numeric_features])
    
    # Standardize the numeric features
    scaler = StandardScaler()
    X_scaled = X_imputed.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X_imputed[numeric_features])
    
    # Apply PCA for visualization
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled[numeric_features])
    
    return X_3d, numeric_features, marketing_data

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
    
    # Calculate silhouette score if applicable
    if algorithm != "DBSCAN":
        score = silhouette_score(X, labels)
        st.sidebar.metric("Silhouette Score", f"{score:.3f}")
    
    return labels

def create_3d_scatter(X, labels, student_ids, feature_names, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.8),
        text=[f"Student ID: {id}<br>Cluster: {label}" for id, label in zip(student_ids, labels)],
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
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
    <h3 style="color: #1e3d59;">Key Points:</h3>
    <ul>
        <li>Each point represents a student from the Marketing dataset.</li>
        <li>Colors indicate the cluster assigned by the chosen algorithm.</li>
        <li>Proximity of points suggests similarity in student interests and behaviors.</li>
    </ul>

    <h3 style="color: #1e3d59;">Example:</h3>
    <p>If you see a tight group of blue points in one area, this could represent a cluster of students with similar interests or social media behaviors.</p>

    <h3 style="color: #1e3d59;">What to Look For:</h3>
    <ol>
        <li><strong>Well-separated clusters:</strong> Distinct groups of colors might indicate that the algorithm has effectively separated different types of students based on their attributes.</li>
        <li><strong>Mixed clusters:</strong> Areas where colors are intermingled could suggest overlapping characteristics between different student groups.</li>
        <li><strong>Outliers:</strong> Isolated points might represent students with unique interests or behaviors.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def explain_algorithms():
    st.subheader("K-Means")
    st.markdown("""
    <div style="background-color: #fff6e9; padding: 20px; border-radius: 10px;">
    <p>K-Means tries to find a specified number of cluster centers and assign each student to the nearest center.</p>
    <p><strong>Pros:</strong> Simple, fast, and works well on globular clusters.</p>
    <p><strong>Cons:</strong> Sensitive to initial centroids, assumes spherical clusters, and requires specifying the number of clusters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Agglomerative Clustering")
    st.markdown("""
    <div style="background-color: #e6f3ff; padding: 20px; border-radius: 10px;">
    <p>This algorithm starts with each student as its own cluster and progressively merges the closest clusters.</p>
    <p><strong>Pros:</strong> Can uncover hierarchical structure in data, doesn't assume cluster shape.</p>
    <p><strong>Cons:</strong> Computationally intensive for large datasets, can be sensitive to noise.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("DBSCAN")
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
    <p>DBSCAN groups together students that are closely packed in the feature space, marking students in low-density regions as outliers.</p>
    <p><strong>Pros:</strong> Can find arbitrarily shaped clusters, robust to outliers, doesn't require specifying number of clusters.</p>
    <p><strong>Cons:</strong> Sensitive to parameters, struggles with varying density clusters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("BIRCH")
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px;">
    <p>BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) builds a tree structure to incrementally cluster the data.</p>
    <p><strong>Pros:</strong> Efficient for large datasets, handles outliers well.</p>
    <p><strong>Cons:</strong> May not work well with non-spherical clusters, sensitive to data order.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Gaussian Mixture")
    st.markdown("""
    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px;">
    <p>Gaussian Mixture Models assume the data is generated from a mixture of a finite number of Gaussian distributions with unknown parameters.</p>
    <p><strong>Pros:</strong> Flexible, can model complex cluster shapes, provides probabilistic cluster assignments.</p>
    <p><strong>Cons:</strong> Sensitive to initialization, can overfit with too many components.</p>
    </div>
    """, unsafe_allow_html=True)

def cluster_insights(marketing_data, labels, feature_names):
    df = marketing_data.copy()
    df['Cluster'] = labels
    
    for cluster in sorted(df['Cluster'].unique()):
        st.subheader(f"Cluster {cluster}")
        cluster_data = df[df['Cluster'] == cluster]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Students", len(cluster_data))
            st.write("Top 5 Student IDs:")
            st.write(", ".join(map(str, cluster_data.index[:5].tolist())))
        
        with col2:
            st.write("Cluster Characteristics:")
            for feature in feature_names:
                mean_value = cluster_data[feature].mean()
                overall_mean = df[feature].mean()
                difference = ((mean_value - overall_mean) / overall_mean) * 100
                emoji = "🔼" if difference > 0 else "🔽"
                st.write(f"{feature}: {emoji} {abs(difference):.1f}% {'above' if difference > 0 else 'below'} average")
    
    st.subheader("Conclusions")
    st.write("""
    Based on the cluster analysis, we can draw the following conclusions:
    1. There appear to be distinct groups of students with similar interests and social media behaviors.
    2. Some clusters show higher engagement in sports and social activities.
    3. Other clusters exhibit more interest in music and cultural activities.
    4. The clustering reveals different student segments that could be targeted with tailored marketing strategies.
    5. Further investigation into outlier students could provide insights into unique interests or potential influencers.
    """)

def quiz():
    questions = [
        {
            "question": "Which feature in the dataset might indicate a student's social media activity?",
            "options": ["age", "NumberOffriends", "basketball", "football"],
            "correct": 1,
            "explanation": "The 'NumberOffriends' feature likely represents a student's social network size, which can be an indicator of social media activity."
        },
        {
            "question": "What type of marketing strategy might be effective for a cluster with high values in 'music', 'band', and 'rock'?",
            "options": ["Sports equipment ads", "Music festival promotions", "Religious event invitations", "Fashion brand campaigns"],
            "correct": 1,
            "explanation": "A cluster with high values in music-related features would likely respond well to music festival promotions or similar music-oriented marketing strategies."
        },
        {
            "question": "How might you interpret a cluster with high values in 'shopping', 'mall', and 'clothes'?",
            "options": ["Sports enthusiasts", "Music lovers", "Fashion-conscious consumers", "Religious group"],
            "correct": 2,
            "explanation": "A cluster with high values in shopping and clothing-related features likely represents fashion-conscious consumers who enjoy shopping."
        },
        {
            "question": "What could be a potential marketing approach for a cluster with high values in 'basketball', 'football', and 'baseball'?",
            "options": ["Promote a new clothing line", "Advertise sports equipment and events", "Market a new music album", "Promote a religious conference"],
            "correct": 1,
            "explanation": "For a cluster showing high interest in various sports, advertising sports equipment and events would be a suitable marketing approach."
        },
        {
            "question": "If a cluster shows high values in 'church', 'god', and 'bible', what type of content might resonate with this group?",
            "options": ["Secular music concerts", "Extreme sports events", "Fashion shows", "Faith-based or spiritual content"],
            "correct": 3,
            "explanation": "A cluster with high values in religious-themed features would likely be more receptive to faith-based or spiritual content in marketing strategies."
        }
    ]

    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        user_answer = st.radio(q["question"], q["options"])
        if st.button(f"Check Answer {i+1}", key=f"btn_{i}"):
            if q["options"].index(user_answer) == q["correct"]:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is: {q['options'][q['correct']]}")
            st.markdown(f"<div style='background-color: #e6f3ff; padding: 15px; border-radius: 10px;'><strong>Explanation:</strong> {q['explanation']}</div>", unsafe_allow_html=True)

def dataset_overview(marketing_data):
    st.subheader("Sample Data")
    st.dataframe(marketing_data.head())

    st.subheader("Dataset Statistics")
    total_students = len(marketing_data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Students", total_students)
        st.metric("Average Age", f"{marketing_data['age'].mean():.2f}" if not pd.isna(marketing_data['age'].mean()) else "N/A")

    with col2:
        st.metric("Average Number of Friends", f"{marketing_data['NumberOffriends'].mean():.2f}" if not pd.isna(marketing_data['NumberOffriends'].mean()) else "N/A")
        st.metric("Most Common Graduation Year", marketing_data['gradyear'].mode().values[0] if not marketing_data['gradyear'].empty else "N/A")

    st.subheader("Feature Distributions")
    numeric_features = ['age', 'NumberOffriends', 'basketball', 'football', 'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading', 'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot', 'kissed', 'dance', 'band', 'marching', 'music', 'rock', 'god', 'church', 'jesus', 'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes', 'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']
    fig = px.box(marketing_data, y=numeric_features, title="Distribution of Student Attributes")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = marketing_data[numeric_features].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap of Features")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gender Distribution")
    gender_counts = marketing_data['gender'].value_counts()
    fig = px.pie(values=gender_counts.values, names=gender_counts.index, title="Distribution of Gender")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Interests")
    interests = numeric_features[2:]  # Exclude age and NumberOffriends
    mean_interests = marketing_data[interests].mean().sort_values(ascending=False)
    fig = px.bar(x=mean_interests.index[:10], y=mean_interests.values[:10], 
                 title="Top 10 Interests", labels={'x': 'Interest', 'y': 'Average Score'})
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
