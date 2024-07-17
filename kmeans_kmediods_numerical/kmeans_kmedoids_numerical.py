import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import pairwise_distances

# Set page config
st.set_page_config(layout="wide", page_title="Clustering Explorer", page_icon="🔍")

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

# Data points and clustering functions (same as before)
data_points = np.array([2, 3, 4, 10, 11, 12, 20, 25, 30])

def calculate_new_means(data_points, means):
    clusters = {i: [] for i in range(len(means))}
    for point in data_points:
        closest_mean_index = np.argmin(np.abs(means - point))
        clusters[closest_mean_index].append(point)
    new_means = []
    for cluster in clusters.values():
        new_means.append(np.mean(cluster) if cluster else 0)
    return np.array(new_means), clusters

def calculate_new_medoids(data_points, medoids):
    clusters = {i: [] for i in range(len(medoids))}
    for point in data_points:
        closest_medoid_index = np.argmin(np.abs(medoids - point))
        clusters[closest_medoid_index].append(point)
    new_medoids = []
    for cluster in clusters.values():
        if cluster:
            distances = pairwise_distances(np.array(cluster).reshape(-1, 1))
            new_medoid = cluster[np.argmin(np.sum(distances, axis=1))]
            new_medoids.append(new_medoid)
        else:
            new_medoids.append(0)
    return np.array(new_medoids), clusters

def kmeans_clustering(m1, m2):
    initial_means = np.array([m1, m2])
    means = initial_means
    iterations = 0
    converged = False
    steps = {}
    while not converged and iterations < 10:
        iterations += 1
        new_means, clusters = calculate_new_means(data_points, means)
        steps[iterations] = {'means': means, 'clusters': clusters}
        converged = np.all(new_means == means)
        means = new_means
    return steps

def kmedoids_clustering(m1, m2):
    initial_medoids = np.array([m1, m2])
    medoids = initial_medoids
    iterations = 0
    converged = False
    steps = {}
    while not converged and iterations < 10:
        iterations += 1
        new_medoids, clusters = calculate_new_medoids(data_points, medoids)
        steps[iterations] = {'medoids': medoids, 'clusters': clusters}
        converged = np.all(new_medoids == medoids)
        medoids = new_medoids
    return steps

def plot_clusters(clusters, centroids):
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    for i, points in clusters.items():
        fig.add_trace(go.Scatter(x=points, y=[0] * len(points), mode='markers', marker=dict(color=colors[i]), name=f'Cluster {i+1}'))
    for i, centroid in enumerate(centroids):
        fig.add_trace(go.Scatter(x=[centroid], y=[0], mode='markers', marker=dict(color=colors[i], size=10, symbol='star'), name=f'Centroid {i+1}'))
    fig.update_layout(title='Clustering', xaxis_title='Data Points', yaxis_title='', showlegend=True)
    return fig

# Main app
def main():
    st.markdown("<h1 class='main-header'>🔍 K-Means vs K-Medoids Numerical</h1>", unsafe_allow_html=True)
    st.markdown("<p class='text-content'>Developed by: Venugopal Adep</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧮 Explore", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Understanding K-Means and K-Medoids</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>K-Means Clustering</h3>
        <p class='text-content'>
        K-Means is like organizing a classroom. Imagine you have a bunch of students (data points) and you want to group them into K teams:
        
        1. You start by randomly picking K team captains (initial centroids).
        2. Each student joins the team of the captain closest to them.
        3. The captain's position is moved to the average position of their team.
        4. Students may switch teams based on which captain is now closest.
        5. This continues until no one wants to switch teams anymore.
        
        Example: Grouping customers by age and income for targeted marketing.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>K-Medoids Clustering</h3>
        <p class='text-content'>
        K-Medoids is similar to K-Means, but with a twist. Using the classroom analogy:
        
        1. Again, you start by randomly picking K team captains.
        2. Students join the team of the closest captain.
        3. Instead of moving the captain to the average position, you choose a new captain who's most centrally located within the team.
        4. Students may switch teams based on who the new captains are.
        5. This continues until the captains stop changing.
        
        Example: Grouping cities for regional offices, where the office must be in one of the existing cities.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Explore Clustering Algorithms</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            algorithm = st.selectbox('Select Algorithm', ['K-Means', 'K-Medoids'])
            m1 = st.number_input('Centroid/Medoid 1', value=5)
            m2 = st.number_input('Centroid/Medoid 2', value=15)
            run_clustering = st.button('Run Clustering')

            if run_clustering:
                if algorithm == 'K-Means':
                    steps = kmeans_clustering(m1, m2)
                else:
                    steps = kmedoids_clustering(m1, m2)
                
                st.session_state.steps = steps
                st.session_state.algorithm = algorithm
            
            if 'steps' in st.session_state:
                step = st.slider("Step", 1, len(st.session_state.steps), 1)
                
                details = st.session_state.steps[step]
                centroids = details['means'] if st.session_state.algorithm == 'K-Means' else details['medoids']
                clusters = details['clusters']
                
                st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                st.markdown(f"**Step {step}:**")
                st.markdown(f"Centroids/Medoids: {centroids}")
                for cluster_idx, points in clusters.items():
                    st.markdown(f"Cluster {cluster_idx + 1}: {points}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if 'steps' in st.session_state:
                fig = plot_clusters(clusters, centroids)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge!</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What's the main difference between K-Means and K-Medoids?",
                "options": [
                    "K-Means uses the mean as the center, while K-Medoids uses an actual data point",
                    "K-Means is faster, while K-Medoids is more accurate",
                    "K-Means works with categorical data, while K-Medoids doesn't",
                    "There's no difference, they're the same algorithm"
                ],
                "correct": 0,
                "explanation": "K-Means uses the average (mean) of points in a cluster as its center, which might not be an actual data point. It's like choosing the 'center of gravity' of a group. K-Medoids, on the other hand, always uses an actual data point as the center, like choosing a 'team captain' from the group to represent it."
            },
            {
                "question": "Why might K-Medoids be preferred over K-Means in some situations?",
                "options": [
                    "K-Medoids is always faster",
                    "K-Medoids is less sensitive to outliers",
                    "K-Medoids can only work with 2D data",
                    "K-Medoids always produces better clusters"
                ],
                "correct": 1,
                "explanation": "K-Medoids is often preferred when dealing with outliers because it uses actual data points as cluster centers. Imagine you're choosing a meeting point for a group of friends. If one friend lives very far away (an outlier), K-Means might suggest meeting in the middle of nowhere, while K-Medoids would suggest meeting at someone's actual house, which is usually more practical."
            },
            {
                "question": "In K-Means clustering, how is the centroid of a cluster determined?",
                "options": [
                    "It's always the first point assigned to the cluster",
                    "It's calculated as the average of all points in the cluster",
                    "It's randomly selected from the cluster points",
                    "It's the point furthest from the cluster center"
                ],
                "correct": 1,
                "explanation": "In K-Means, the centroid is calculated as the average (mean) of all points in the cluster. Think of it like finding the balance point of a mobile made from all the data points in the cluster. This average point represents the cluster's center, even if it's not an actual data point itself."
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

if __name__ == '__main__':
    main()
