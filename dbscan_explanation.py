import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors

class DBSCANVisualizer:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="DBSCAN Explorer", page_icon="📊")
        self.init_session_state()
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        self.setup_sidebar()
        self.main_layout()

    def init_session_state(self):
        if 'points' not in st.session_state:
            st.session_state.points = np.array([])
        if 'selected_point' not in st.session_state:
            st.session_state.selected_point = None
        if 'show_eps_circles' not in st.session_state:
            st.session_state.show_eps_circles = False

    def setup_sidebar(self):
        st.sidebar.title("🎮 DBSCAN Controls")
        
        st.sidebar.subheader("Algorithm Parameters")
        self.eps = st.sidebar.slider("ε (Epsilon)", 0.1, 2.0, 0.5, 0.1,
                                   help="Maximum distance between two points to be considered neighbors")
        self.min_samples = st.sidebar.slider("MinPts", 2, 10, 3,
                                           help="Minimum number of points to form a dense region")
        
        st.sidebar.subheader("Data Generation")
        point_count = st.sidebar.slider("Number of Points", 5, 50, 10)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Add Random Points"):
                new_points = np.random.rand(point_count, 2) * 4 - 2
                st.session_state.points = (np.vstack([st.session_state.points, new_points]) 
                                         if st.session_state.points.size else new_points)
        with col2:
            if st.button("Clear Points"):
                st.session_state.points = np.array([])
                st.session_state.selected_point = None

        st.sidebar.subheader("Visualization Options")
        st.session_state.show_eps_circles = st.sidebar.checkbox("Show ε-neighborhoods", 
                                                              value=st.session_state.show_eps_circles)

    def calculate_clustering(self):
        if len(st.session_state.points) < self.min_samples:
            return None, None, None, None, None
        
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(st.session_state.points)
        distances = cdist(st.session_state.points, st.session_state.points)
        
        # Calculate point types
        neighbor_counts = np.sum(distances <= self.eps, axis=1)
        core_mask = neighbor_counts >= self.min_samples
        border_mask = (labels != -1) & ~core_mask
        noise_mask = labels == -1
        
        return labels, distances, core_mask, border_mask, neighbor_counts

    def plot_clusters(self, labels, core_mask, border_mask):
        if len(st.session_state.points) == 0:
            fig = go.Figure()
            fig.update_layout(height=600)
            return fig

        df = pd.DataFrame(st.session_state.points, columns=['X', 'Y'])
        df['Cluster'] = labels if labels is not None else -1
        df['Point Type'] = 'Noise'
        df.loc[core_mask, 'Point Type'] = 'Core'
        df.loc[border_mask, 'Point Type'] = 'Border'

        fig = px.scatter(df, x='X', y='Y', color='Cluster',
                        symbol='Point Type',
                        title='DBSCAN Clustering Results',
                        color_continuous_scale='Viridis',
                        symbol_map={'Core': 'circle', 'Border': 'diamond', 'Noise': 'x'})

        # Add epsilon circles if enabled
        if st.session_state.show_eps_circles:
            for point in st.session_state.points:
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = point[0] + self.eps * np.cos(theta)
                circle_y = point[1] + self.eps * np.sin(theta)
                fig.add_trace(go.Scatter(x=circle_x, y=circle_y,
                                       mode='lines', line=dict(dash='dot', width=0.5),
                                       showlegend=False))

        fig.update_layout(
            height=600,
            legend_title_text='Point Types',
            hovermode='closest'
        )
        return fig

    def show_math_explanation(self):
        st.markdown("""
        ### 🔍 DBSCAN Mathematical Concepts
        """)
        
        st.latex(r'''
        \begin{aligned}
        & \textbf{Core Point } p: |N_\varepsilon(p)| \geq \text{MinPts} \\
        & \text{where } N_\varepsilon(p) = \{q \in D \mid \text{dist}(p,q) \leq \varepsilon\} \\
        & \textbf{Border Point}: q \in N_\varepsilon(p) \text{ where p is core point} \\
        & \textbf{Noise Point}: \text{Neither core nor border point} \\
        & \textbf{Distance Metric}: \text{dist}(p,q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
        \end{aligned}
        ''')

    def show_point_analysis(self, distances, core_mask, labels, neighbor_counts):
        if len(st.session_state.points) == 0:
            return
        
        st.session_state.selected_point = st.selectbox(
            "Select point for detailed analysis:",
            range(len(st.session_state.points)),
            format_func=lambda x: f"Point {x}: {st.session_state.points[x]}"
        )
        
        point_idx = st.session_state.selected_point
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Point Statistics")
            st.write(f"**Coordinates:** ({st.session_state.points[point_idx][0]:.2f}, "
                    f"{st.session_state.points[point_idx][1]:.2f})")
            st.write(f"**Number of neighbors:** {neighbor_counts[point_idx]}")
            st.write(f"**Required neighbors (MinPts):** {self.min_samples}")
            st.write(f"**Point type:** {'Core' if core_mask[point_idx] else 'Border' if labels[point_idx] != -1 else 'Noise'}")
            st.write(f"**Cluster Label:** {labels[point_idx] if labels is not None else 'N/A'}")
        
        with col2:
            neighbors_df = pd.DataFrame({
                'Point': range(len(st.session_state.points)),
                'Distance': distances[point_idx],
                'Is Neighbor': distances[point_idx] <= self.eps
            }).sort_values('Distance')
            st.markdown("### 🔍 Neighbor Analysis")
            st.dataframe(neighbors_df.style.highlight_max(subset=['Distance'])
                        .highlight_min(subset=['Distance']))

    def main_layout(self):
        st.title("🔬 Interactive DBSCAN Algorithm Explorer")
        st.markdown("---")
        
        # Calculate clustering
        labels, distances, core_mask, border_mask, neighbor_counts = self.calculate_clustering()
        
        # Main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(self.plot_clusters(labels, core_mask, border_mask), 
                           use_container_width=True)
        
        with col2:
            self.show_math_explanation()
            
            st.markdown("### 📈 Current Parameters")
            st.write(f"**ε (Epsilon):** {self.eps}")
            st.write(f"**MinPts:** {self.min_samples}")
            
            if len(st.session_state.points) > 0:
                st.write(f"**Total Points:** {len(st.session_state.points)}")
                if labels is not None:
                    st.write(f"**Number of Clusters:** {len(np.unique(labels[labels != -1]))}")
                    st.write(f"**Noise Points:** {np.sum(labels == -1)}")
        
        if len(st.session_state.points) > 0:
            st.markdown("---")
            st.subheader("🔎 Detailed Point Analysis")
            self.show_point_analysis(distances, core_mask, labels, neighbor_counts)

if __name__ == "__main__":
    DBSCANVisualizer()
