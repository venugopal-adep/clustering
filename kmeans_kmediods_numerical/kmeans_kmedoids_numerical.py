import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

data_points = np.array([2, 3, 4, 10, 11, 12, 20, 25, 30])

def calculate_new_means(data_points, means):
    clusters = {i: [] for i in range(len(means))}
    for point in data_points:
        closest_mean_index = np.argmin(np.abs(means - point))
        clusters[closest_mean_index].append(point)
    new_means = []
    for cluster in clusters.values():
        new_means.append(np.mean(cluster))
    return np.array(new_means), clusters

def kmeans_clustering(m1, m2):
    initial_means = np.array([m1, m2])
    means = initial_means
    iterations = 0
    converged = False
    steps = {}
    while not converged:
        iterations += 1
        new_means, clusters = calculate_new_means(data_points, means)
        steps[iterations] = {'means': means, 'clusters': clusters}
        converged = np.all(new_means == means)
        means = new_means
    result = ""
    for step, details in steps.items():
        result += f"Step {step}:\n"
        result += f"Centroids: {details['means']}\n"
        for cluster_idx, points in details['clusters'].items():
            result += f"Cluster {cluster_idx + 1}: {points}\n"
        result += "---\n"  # Separator for steps
    return result

def calculate_new_medoids(data_points, medoids):
    clusters = {i: [] for i in range(len(medoids))}
    for point in data_points:
        point_2d = point.reshape(1, -1)  # Reshape point to 2D array
        distances = pairwise_distances(point_2d, medoids.reshape(len(medoids), -1)).flatten()
        closest_medoid_index = np.argmin(distances)
        clusters[closest_medoid_index].append(point)
    new_medoids = []
    for points in clusters.values():
        if points:
            points_2d = np.array(points).reshape(len(points), -1)  # Reshape points to 2D array
            distances = pairwise_distances(points_2d)
            medoid_index = np.argmin(distances.sum(axis=1))
            new_medoids.append(points[medoid_index])
        else:
            new_medoids.append(None)
    return np.array(new_medoids), clusters

def kmedoids_clustering(m1, m2):
    initial_medoids = np.array([m1, m2])
    medoids = initial_medoids
    iterations = 0
    converged = False
    steps = {}
    while not converged:
        iterations += 1
        new_medoids, clusters = calculate_new_medoids(data_points, medoids)
        steps[iterations] = {'medoids': medoids, 'clusters': clusters}
        converged = np.all(new_medoids == medoids)
        medoids = new_medoids
    result = ""
    for step, details in steps.items():
        result += f"Step {step}:\n"
        result += f"Medoids: {details['medoids']}\n"
        for cluster_idx, points in details['clusters'].items():
            result += f"Cluster {cluster_idx + 1}: {points}\n"
        result += "---\n"  # Separator for steps
    return result

def plot_clusters(clusters, centroids):
    data = []
    for i, points in clusters.items():
        data.extend([(point, i+1) for point in points])
    df = pd.DataFrame(data, columns=['Data Points', 'Cluster'])
    fig = px.scatter(df, x='Data Points', y=[0] * len(df), color='Cluster', symbol='Cluster',
                     symbol_map={i+1: 'circle' for i in range(len(centroids))},
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.add_trace(px.scatter(x=centroids, y=[0] * len(centroids), color=[i+1 for i in range(len(centroids))],
                             symbol=[i+1 for i in range(len(centroids))], symbol_map={i+1: 'star' for i in range(len(centroids))},
                             color_discrete_sequence=px.colors.qualitative.Plotly).data[0])
    fig.update_layout(title='Clustering', xaxis_title='Data Points', yaxis_title='', showlegend=False)
    return fig

def main():
    st.title('Clustering Demo')
    algorithm = st.selectbox('Select Algorithm', ['K-Means', 'K-Medoids'])
    m1 = st.number_input('Centroid/Medoid 1', value=5)
    m2 = st.number_input('Centroid/Medoid 2', value=15)
    if st.button('Run Clustering'):
        if algorithm == 'K-Means':
            result = kmeans_clustering(m1, m2)
        else:
            result = kmedoids_clustering(m1, m2)
        steps = [step.strip() for step in result.split('---')]
        for i, step in enumerate(steps):
            if step:
                st.subheader(f'Step {i+1}')
                lines = step.split('\n')
                centroids = np.array([float(x.strip('[]')) for x in lines[1].split(':')[1].split() if x.strip('[]')])
                clusters = {}
                for line in lines[2:]:
                    if line:
                        cluster_idx = int(line.split(':')[0].split()[-1]) - 1
                        points = [float(x.strip('[]')) for x in line.split(':')[1].strip().split(',') if x.strip('[]')]
                        clusters[cluster_idx] = points
                fig = plot_clusters(clusters, centroids)
                st.plotly_chart(fig)
                st.write(step)

if __name__ == '__main__':
    main()
