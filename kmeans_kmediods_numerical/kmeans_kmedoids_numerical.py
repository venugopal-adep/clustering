import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances

# Set page config
st.set_page_config(layout="wide", page_title="Clustering Explorer", page_icon="🔍")

# Custom CSS
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

# Data points
data_points = np.array([2, 3, 4, 10, 11, 12, 20, 25, 30])

def euclidean_distance(a, b):
    return np.abs(a - b)

def kmeans_clustering(k, initial_centroids):
    centroids = initial_centroids.copy()
    steps = []
    iteration = 0
    converged = False

    while not converged and iteration < 10:
        iteration += 1
        step_details = {
            "iteration": iteration,
            "centroids": centroids.copy(),
            "clusters": {},
            "distances": {},
            "new_centroids": None,
        }

        # Measure the distance
        distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in data_points])
        step_details["distances"] = distances

        # Grouping based on minimum distance
        cluster_assignments = np.argmin(distances, axis=1)
        for i in range(k):
            step_details["clusters"][i] = data_points[cluster_assignments == i].tolist()

        # Reposition of centroids
        new_centroids = np.array([np.mean(step_details["clusters"][i]) if step_details["clusters"][i] else centroids[i] for i in range(k)])
        step_details["new_centroids"] = new_centroids

        # Check for convergence
        if np.array_equal(new_centroids, centroids):
            converged = True
        else:
            centroids = new_centroids.copy()

        steps.append(step_details)

    return steps

def kmedoids_clustering(k, initial_medoids):
    medoids = initial_medoids.copy()
    steps = []
    iteration = 0
    converged = False

    while not converged and iteration < 10:
        iteration += 1
        step_details = {
            "iteration": iteration,
            "medoids": medoids.copy(),
            "clusters": {},
            "distances": {},
            "new_medoids": None,
        }

        # Measure the distance
        distances = np.array([[euclidean_distance(point, medoid) for medoid in medoids] for point in data_points])
        step_details["distances"] = distances

        # Grouping based on minimum distance
        cluster_assignments = np.argmin(distances, axis=1)
        for i in range(k):
            step_details["clusters"][i] = data_points[cluster_assignments == i].tolist()

        # Reposition of medoids
        new_medoids = []
        for i in range(k):
            cluster_points = np.array(step_details["clusters"][i])
            if len(cluster_points) > 0:
                cluster_distances = pairwise_distances(cluster_points.reshape(-1, 1))
                total_distances = np.sum(cluster_distances, axis=1)
                new_medoid = cluster_points[np.argmin(total_distances)]
            else:
                new_medoid = medoids[i]
            new_medoids.append(new_medoid)
        
        step_details["new_medoids"] = np.array(new_medoids)

        # Check for convergence
        if np.array_equal(new_medoids, medoids):
            converged = True
        else:
            medoids = np.array(new_medoids)

        steps.append(step_details)

    return steps

def plot_clusters(data_points, centers, cluster_assignments):
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    
    for i, center in enumerate(centers):
        cluster_points = data_points[cluster_assignments == i]
        fig.add_trace(go.Scatter(x=cluster_points, y=[0] * len(cluster_points), mode='markers', 
                                 marker=dict(color=colors[i], size=10), name=f'Cluster {i+1}'))
        fig.add_trace(go.Scatter(x=[center], y=[0], mode='markers', 
                                 marker=dict(color=colors[i], size=15, symbol='star'), name=f'Center {i+1}'))

    fig.update_layout(title='Clustering Result', xaxis_title='Data Points', yaxis_title='', showlegend=True)
    return fig

def main():
    st.markdown("<h1 class='main-header'>🔍 Clustering Explorer</h1>", unsafe_allow_html=True)

    st.markdown("<h2 class='sub-header'>Choose Parameters</h2>", unsafe_allow_html=True)
    algorithm = st.selectbox("Select Algorithm", ["K-means", "K-medoids"])
    k = st.number_input("Number of clusters (K)", min_value=2, max_value=5, value=2)
    initial_centers = np.array([st.number_input(f"Initial Center {i+1}", value=data_points[i*len(data_points)//k]) for i in range(k)])

    if st.button(f"Run {algorithm} Clustering"):
        if algorithm == "K-means":
            steps = kmeans_clustering(k, initial_centers)
        else:
            steps = kmedoids_clustering(k, initial_centers)

        st.markdown(f"<h2 class='sub-header'>{algorithm} Clustering Steps</h2>", unsafe_allow_html=True)
        for step in steps:
            with st.expander(f"Step {step['iteration']}"):
                centers = step['centroids'] if algorithm == "K-means" else step['medoids']
                new_centers = step['new_centroids'] if algorithm == "K-means" else step['new_medoids']
                
                st.write(f"### 1. Current {'Centroids' if algorithm == 'K-means' else 'Medoids'}")
                st.write(f"{'Centroids' if algorithm == 'K-means' else 'Medoids'}: {centers.tolist() if isinstance(centers, np.ndarray) else centers}")

                st.write("### 2. Measure the distance")
                for i, point in enumerate(data_points):
                    st.write(f"Point {point}:")
                    for j, center in enumerate(centers):
                        st.write(f"  Distance to {'Centroid' if algorithm == 'K-means' else 'Medoid'} {j+1}: {step['distances'][i][j]:.2f}")

                st.write("### 3. Grouping based on minimum distance")
                for cluster, points in step['clusters'].items():
                    st.write(f"Cluster {cluster+1}: {points}")

                st.write(f"### 4. Reposition of {'centroids' if algorithm == 'K-means' else 'medoids'}")
                st.write(f"New {'centroids' if algorithm == 'K-means' else 'medoids'}: {new_centers.tolist() if isinstance(new_centers, np.ndarray) else new_centers}")

                # Plot
                cluster_assignments = np.argmin(step['distances'], axis=1)
                fig = plot_clusters(data_points, centers, cluster_assignments)
                st.plotly_chart(fig, use_container_width=True)

                if np.array_equal(new_centers, centers):
                    st.success("Convergence reached! The algorithm has converged to stable centers.")
                else:
                    st.info(f"{'Centroids' if algorithm == 'K-means' else 'Medoids'} have been updated. Moving to the next iteration.")

if __name__ == '__main__':
    main()
