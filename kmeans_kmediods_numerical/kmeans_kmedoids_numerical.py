import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Clustering Explorer", page_icon="🔍")

st.markdown("""
<style>
    .main-header { font-size: 36px !important; font-weight: bold; color: #1E90FF; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px #cccccc; }
    .sub-header { font-size: 24px !important; font-weight: bold; color: #4682B4; margin-top: 20px; margin-bottom: 20px; }
    .text-content { font-size: 18px !important; line-height: 1.6; }
    .highlight { background-color: #F0F8FF; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #1E90FF; }
    .stButton>button { background-color: #4CAF50; color: white; font-size: 16px; padding: 10px 24px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

data_points = np.array([2, 3, 4, 10, 11, 12, 20, 25, 30])

def clustering(k, initial_centers, algorithm):
    centers = initial_centers.copy()
    steps = []
    for iteration in range(10):
        distances = {int(point): [round(float(abs(point - center)), 2) for center in centers] for point in data_points}
        cluster_assignments = np.argmin(np.array(list(distances.values())), axis=1)
        clusters = {i: [int(p) for p in data_points[cluster_assignments == i]] for i in range(k)}
        
        if algorithm == "K-means":
            new_centers = np.array([np.mean(clusters[i]) if clusters[i] else centers[i] for i in range(k)])
        else:
            new_centers = []
            for i in range(k):
                cluster_points = np.array(clusters[i])
                if len(cluster_points) > 0:
                    cluster_distances = np.abs(cluster_points[:, np.newaxis] - cluster_points)
                    new_centers.append(cluster_points[np.argmin(np.sum(cluster_distances, axis=1))])
                else:
                    new_centers.append(centers[i])
        
        steps.append({
            "iteration": iteration + 1,
            "centers": [round(float(c), 2) for c in centers],
            "clusters": clusters,
            "distances": distances,
            "new_centers": [round(float(c), 2) for c in new_centers]
        })
        
        if np.array_equal(new_centers, centers):
            break
        centers = np.array(new_centers)
    
    return steps

def plot_clusters(data_points, centers, cluster_assignments):
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    for i, center in enumerate(centers):
        cluster_points = data_points[cluster_assignments == i]
        fig.add_trace(go.Scatter(x=cluster_points, y=[0] * len(cluster_points), mode='markers', marker=dict(color=colors[i], size=10), name=f'Cluster {i+1}'))
        fig.add_trace(go.Scatter(x=[center], y=[0], mode='markers', marker=dict(color=colors[i], size=15, symbol='star'), name=f'Center {i+1}'))
    fig.update_layout(title='Clustering Result', xaxis_title='Data Points', yaxis_title='', showlegend=True)
    return fig

def main():
    st.markdown("<h1 class='main-header'>🔍 Clustering Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Choose Parameters</h2>", unsafe_allow_html=True)
    algorithm = st.selectbox("Select Algorithm", ["K-means", "K-medoids"])
    k = st.number_input("Number of clusters (K)", min_value=2, max_value=5, value=2)
    initial_centers = np.array([st.number_input(f"Initial Center {i+1}", value=float(data_points[i*len(data_points)//k])) for i in range(k)])

    if st.button(f"Run {algorithm} Clustering"):
        steps = clustering(k, initial_centers, algorithm)
        st.markdown(f"<h2 class='sub-header'>{algorithm} Clustering Steps</h2>", unsafe_allow_html=True)
        for step in steps:
            with st.expander(f"Step {step['iteration']}"):
                st.write(f"### 1. Current {'Centroids' if algorithm == 'K-means' else 'Medoids'}")
                st.write(f"{'Centroids' if algorithm == 'K-means' else 'Medoids'}: {step['centers']}")
                
                st.write("### 2. Measure the distance")
                st.write(f"Distances: {step['distances']}")
                
                st.write("### 3. Grouping based on minimum distance")
                for cluster, points in step['clusters'].items():
                    st.write(f"Cluster {cluster+1}: {points}")
                
                st.write(f"### 4. Reposition of {'centroids' if algorithm == 'K-means' else 'medoids'}")
                st.write(f"New {'centroids' if algorithm == 'K-means' else 'medoids'}: {step['new_centers']}")
                
                cluster_assignments = np.argmin(np.array(list(step['distances'].values())), axis=1)
                fig = plot_clusters(data_points, step['centers'], cluster_assignments)
                st.plotly_chart(fig, use_container_width=True)
                
                if step['centers'] == step['new_centers']:
                    st.success("Convergence reached! The algorithm has converged to stable centers.")
                else:
                    st.info(f"{'Centroids' if algorithm == 'K-means' else 'Medoids'} have been updated. Moving to the next iteration.")

if __name__ == '__main__':
    main()
