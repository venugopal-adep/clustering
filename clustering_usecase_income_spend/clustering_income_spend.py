import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load and preprocess the customer data
@st.cache_data
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/venugopal-adep/clustering/main/clustering_usecase_income_spend/customer_data.csv")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[["Annual Income", "Spending Score"]])
    return data, scaled_data

# Perform clustering using KMeans
def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

# Create the Streamlit application
def main():
    st.title("Customer Segmentation using Clustering")
    st.markdown('Venugopal Adep')
    st.write("This application allows you to segment customers based on their annual income and spending score.")

    # Load the customer data
    data, scaled_data = load_data()

    # Display the raw data
    if st.checkbox("Show raw data"):
        st.write(data)

    # Get user input for the number of clusters
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)

    # Perform clustering
    labels = perform_clustering(scaled_data, n_clusters)

    # Add the cluster labels to the data
    data["Cluster"] = labels

    # Display the clustering results
    st.subheader("Clustering Results")
    fig = px.scatter(data, x="Annual Income", y="Spending Score", color="Cluster", hover_data=["CustomerID"])
    st.plotly_chart(fig)

    # Display additional insights
    st.subheader("Cluster Insights")
    for i in range(n_clusters):
        cluster_data = data[data["Cluster"] == i]
        st.write(f"Cluster {i+1}:")
        st.write(f"- Number of customers: {len(cluster_data)}")
        st.write(f"- Average annual income: ${cluster_data['Annual Income'].mean():.2f}")
        st.write(f"- Average spending score: {cluster_data['Spending Score'].mean():.2f}")
        st.write("---")

if __name__ == "__main__":
    main()
