import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration and custom CSS
st.set_page_config(page_title="20 Newsgroups Clustering", layout="wide")
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

def main():
    st.title("Interactive 20 Newsgroups Clustering Demo")

    # Sidebar
    with st.sidebar:
        st.header("Clustering Settings")
        algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "Agglomerative Clustering", "DBSCAN"])
        n_clusters = st.slider("Number of Clusters", 2, 20, 5) if algorithm != "DBSCAN" else None
        eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5) if algorithm == "DBSCAN" else None
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5) if algorithm == "DBSCAN" else None
        n_documents = st.slider("Number of Documents", 100, 2000, 500)

    # Load and process data
    X_3d, true_labels, target_names, newsgroups = load_and_process_data(n_documents)

    # Perform clustering
    labels = cluster_data(X_3d, algorithm, n_clusters, eps, min_samples)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualization", "📘 Interpretation", "🧠 Algorithms", "🎓 Quiz", "📋 Dataset"])

    with tab1:
        st.header("3D Visualization of Clustering Results")
        fig = create_3d_scatter(X_3d, labels, true_labels, target_names, f"{algorithm} Clustering")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("How to Interpret the Visualization")
        interpretation_guide()

    with tab3:
        st.header("Understanding the Clustering Algorithms")
        explain_algorithms()

    with tab4:
        st.header("Test Your Understanding")
        quiz()

    with tab5:
        st.header("Dataset Overview")
        dataset_overview(newsgroups, target_names)

def load_and_process_data(n_docs):
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(newsgroups.data[:n_docs])
    svd = TruncatedSVD(n_components=3)
    X_3d = svd.fit_transform(X)
    return X_3d, newsgroups.target[:n_docs], newsgroups.target_names, newsgroups

def cluster_data(X, algorithm, n_clusters=None, eps=None, min_samples=None):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:  # DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def create_3d_scatter(X, labels, true_labels, target_names, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.8),
        text=[f"True Category: {target_names[l]}" for l in true_labels],
        hoverinfo='text'
    )])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Component 1", yaxis_title="Component 2", zaxis_title="Component 3"),
        height=700,
        margin=dict(r=0, b=0, l=0, t=40)
    )
    return fig

def interpretation_guide():
    st.markdown("""
    <div style="background-color: #e6f3ff; padding: 20px; border-radius: 10px;">
    <h3 style="color: #2c3e50;">Key Points:</h3>
    <ul>
        <li>Each point represents a news article from the 20 Newsgroups dataset.</li>
        <li>Colors indicate the cluster assigned by the chosen algorithm.</li>
        <li>Hover over points to see the true category of each article.</li>
        <li>Proximity of points suggests similarity in content.</li>
    </ul>

    <h3 style="color: #2c3e50;">Example:</h3>
    <p>Imagine you see a tight group of blue points in one corner of the 3D space. This could represent a cluster of articles all talking about a specific topic, like "space exploration". Even though these articles might come from different newsgroups (e.g., sci.space, sci.astro), the clustering algorithm has determined they're similar based on their content.</p>

    <h3 style="color: #2c3e50;">What to Look For:</h3>
    <ol>
        <li><strong>Well-separated clusters:</strong> If you see distinct groups of colors, it means the algorithm has effectively separated different topics. For instance, articles about computers might form one cluster, while articles about politics form another.</li>
        <li><strong>Mixed clusters:</strong> Areas where colors are intermingled indicate topics that the algorithm found difficult to distinguish. This could happen with closely related topics, like "computers" and "technology".</li>
        <li><strong>Outliers:</strong> Isolated points might represent unique or unusual articles. For example, an article about a rare astronomical event might appear as an outlier in the "space" cluster.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def explain_algorithms():
    st.subheader("K-Means")
    st.markdown("""
    <div style="background-color: #e8f8f5; padding: 20px; border-radius: 10px;">
    <p>K-Means tries to find a specified number of cluster centers and assign each point to the nearest center.</p>
    <p><strong>Analogy:</strong> Imagine organizing a big library. K-Means is like deciding on a number of shelves (clusters) and then putting each book on the shelf it's most similar to. You might have a shelf for science fiction, another for biographies, one for cookbooks, and so on.</p>
    <p><strong>Example:</strong> If we set K=5 for our news articles, K-Means might create clusters for topics like "Sports", "Technology", "Politics", "Science", and "Entertainment", trying to group similar articles together.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Agglomerative Clustering")
    st.markdown("""
    <div style="background-color: #fef9e7; padding: 20px; border-radius: 10px;">
    <p>This algorithm starts with each point as its own cluster and progressively merges the closest clusters.</p>
    <p><strong>Analogy:</strong> Think of building a family tree. We start with individuals, then group siblings, then extend to cousins, and keep going until we have larger family groups.</p>
    <p><strong>Example:</strong> In our news dataset, it might start by grouping very similar articles (like two articles about the same sports event), then gradually combine these small groups into larger topics (like "Basketball", then "Sports").</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("DBSCAN")
    st.markdown("""
    <div style="background-color: #f4ecf7; padding: 20px; border-radius: 10px;">
    <p>DBSCAN groups together points that are closely packed, marking points in low-density regions as outliers.</p>
    <p><strong>Analogy:</strong> Imagine looking at a night sky full of stars. DBSCAN would identify constellations (dense clusters of stars) and ignore the sparse areas between them.</p>
    <p><strong>Example:</strong> In our news articles, DBSCAN might identify dense clusters of articles about popular topics (like a major political event) while labeling articles about rare or unique topics as outliers.</p>
    </div>
    """, unsafe_allow_html=True)

def quiz():
    questions = [
        {
            "question": "What does each point in the 3D visualization represent?",
            "options": ["A cluster center", "A news article", "A word", "A category"],
            "correct": 1,
            "explanation": "Each point in the 3D visualization represents a single news article from the 20 Newsgroups dataset. The position of the point in 3D space is determined by the content of the article, with similar articles appearing closer together."
        },
        {
            "question": "In K-Means clustering, what does 'K' represent?",
            "options": ["Number of iterations", "Number of clusters", "Number of documents", "Number of features"],
            "correct": 1,
            "explanation": "In K-Means clustering, 'K' represents the number of clusters. It's the number of groups you want the algorithm to divide your data into. For example, if K=5, the algorithm will try to organize the news articles into 5 distinct groups or topics."
        },
        {
            "question": "Which clustering algorithm is best for detecting outliers?",
            "options": ["K-Means", "Agglomerative Clustering", "DBSCAN"],
            "correct": 2,
            "explanation": "DBSCAN is particularly good at detecting outliers. Unlike K-Means and Agglomerative Clustering, which assign every point to a cluster, DBSCAN can label points in low-density regions as outliers. This makes it useful for identifying unusual or unique articles in our news dataset."
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

def dataset_overview(newsgroups, target_names):
    st.subheader("Sample Articles")
    sample_size = min(5, len(newsgroups.data))
    samples = np.random.choice(len(newsgroups.data), sample_size, replace=False)
    for i in samples:
        st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                    f"<strong>Category:</strong> {target_names[newsgroups.target[i]]}<br>"
                    f"<strong>Text:</strong> {newsgroups.data[i][:500]}...</div>", unsafe_allow_html=True)

    st.subheader("Dataset Statistics")
    total_articles = len(newsgroups.data)
    avg_length = np.mean([len(text.split()) for text in newsgroups.data])
    category_counts = pd.Series(newsgroups.target).value_counts().sort_index()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Articles", total_articles)
        st.metric("Average Article Length", f"{avg_length:.0f} words")

    with col2:
        st.metric("Number of Categories", len(target_names))
        st.metric("Most Common Category", target_names[category_counts.idxmax()])

    st.subheader("Category Distribution")
    fig = px.bar(x=target_names, y=category_counts.values,
                 labels={'x': 'Category', 'y': 'Number of Articles'},
                 title="Distribution of Articles across Categories")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()