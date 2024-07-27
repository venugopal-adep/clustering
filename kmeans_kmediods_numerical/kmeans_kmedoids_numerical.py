import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Set page config
st.set_page_config(layout="wide", page_title="K-means Clustering Explorer", page_icon="🔍")

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

    while not converged:
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
        if np.all(new_centroids == centroids):
            converged = True
        else:
            centroids = new_centroids

        steps.append(step_details)

    return steps

def plot_clusters(data_points, centroids, cluster_assignments):
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    
    for i, centroid in enumerate(centroids):
        cluster_points = data_points[cluster_assignments == i]
        fig.add_trace(go.Scatter(x=cluster_points, y=[0] * len(cluster_points), mode='markers', 
                                 marker=dict(color=colors[i], size=10), name=f'Cluster {i+1}'))
        fig.add_trace(go.Scatter(x=[centroid], y=[0], mode='markers', 
                                 marker=dict(color=colors[i], size=15, symbol='star'), name=f'Centroid {i+1}'))

    fig.update_layout(title='K-means Clustering', xaxis_title='Data Points', yaxis_title='', showlegend=True)
    return fig

def main():
    st.markdown("<h1 class='main-header'>🔍 K-means Clustering Explorer</h1>", unsafe_allow_html=True)

    st.markdown("<h2 class='sub-header'>Choose Parameters</h2>", unsafe_allow_html=True)
    k = st.number_input("Number of clusters (K)", min_value=2, max_value=5, value=2)
    initial_centroids = np.array([st.number_input(f"Initial Centroid {i+1}", value=data_points[i*len(data_points)//k]) for i in range(k)])

    if st.button("Run K-means Clustering"):
        steps = kmeans_clustering(k, initial_centroids)

        st.markdown("<h2 class='sub-header'>K-means Clustering Steps</h2>", unsafe_allow_html=True)
        for step in steps:
            with st.expander(f"Step {step['iteration']}"):
                st.write("### 1. Current Centroids")
                st.write(f"Centroids: {step['centroids'].tolist()}")

                st.write("### 2. Measure the distance")
                for i, point in enumerate(data_points):
                    st.write(f"Point {point}:")
                    for j, centroid in enumerate(step['centroids']):
                        st.write(f"  Distance to Centroid {j+1}: {step['distances'][i][j]:.2f}")

                st.write("### 3. Grouping based on minimum distance")
                for cluster, points in step['clusters'].items():
                    st.write(f"Cluster {cluster+1}: {points}")

                st.write("### 4. Reposition of centroids")
                st.write(f"New centroids: {step['new_centroids'].tolist()}")

                # Plot
                cluster_assignments = np.argmin(step['distances'], axis=1)
                fig = plot_clusters(data_points, step['centroids'], cluster_assignments)
                st.plotly_chart(fig, use_container_width=True)

                if np.all(step['new_centroids'] == step['centroids']):
                    st.success("Convergence reached! The algorithm has converged to stable centroids.")
                else:
                    st.info("Centroids have been updated. Moving to the next iteration.")

if __name__ == '__main__':
    main()
