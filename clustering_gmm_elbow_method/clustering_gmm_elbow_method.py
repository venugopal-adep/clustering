import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(layout="wide", page_title="Gaussian Mixture Model Explorer", page_icon="🎨")

# Custom CSS with aesthetic colors
st.markdown("""
<style>
    .main-header {
        font-size: 40px !important;
        font-weight: bold;
        color: #6A0DAD;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #D8BFD8;
    }
    .sub-header {
        font-size: 28px !important;
        font-weight: bold;
        color: #9370DB;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .text-content {
        font-size: 18px !important;
        line-height: 1.6;
        color: #4B0082;
    }
    .highlight {
        background-color: #E6E6FA;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 2px solid #9370DB;
    }
    .stButton>button {
        background-color: #9370DB;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease 0s;
    }
    .stButton>button:hover {
        background-color: #6A0DAD;
        box-shadow: 0px 15px 20px rgba(154, 85, 255, 0.4);
        transform: translateY(-7px);
    }
    .quiz-option {
        background-color: #E6E6FA;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .quiz-option:hover {
        background-color: #D8BFD8;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def calculate_bic(data, max_k):
    bic = []
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        gmm.fit(data)
        bic.append(gmm.bic(data))
    return bic

def calculate_silhouette_scores(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        labels = gmm.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    return silhouette_scores

def plot_bic_and_silhouette(bic, silhouette_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(2, len(bic) + 2)), y=bic, mode='lines+markers', name='BIC', line=dict(color='#9370DB')))
    fig.add_trace(go.Scatter(x=list(range(2, len(silhouette_scores) + 2)), y=silhouette_scores, mode='lines+markers', name='Silhouette Score', yaxis='y2', line=dict(color='#6A0DAD')))
    fig.update_layout(title='BIC and Silhouette Score',
                      xaxis_title='Number of Components',
                      yaxis=dict(title='BIC', titlefont=dict(color='#9370DB')),
                      yaxis2=dict(title='Silhouette Score', overlaying='y', side='right', titlefont=dict(color='#6A0DAD')),
                      plot_bgcolor='#F8F8FF',
                      paper_bgcolor='#F8F8FF')
    st.plotly_chart(fig)

def plot_clusters(data, gmm):
    fig = go.Figure()
    colors = px.colors.qualitative.Pastel
    for i in range(gmm.n_components):
        cluster_data = data[gmm.predict(data) == i]
        fig.add_trace(go.Scatter(x=cluster_data[:, 0], y=cluster_data[:, 1], mode='markers', name=f'Cluster {i+1}', marker=dict(color=colors[i % len(colors)])))
    fig.update_layout(title='Clustering Results',
                      xaxis_title='X',
                      yaxis_title='Y',
                      plot_bgcolor='#F8F8FF',
                      paper_bgcolor='#F8F8FF')
    st.plotly_chart(fig)

def find_optimal_k_bic(bic):
    return bic.index(min(bic)) + 2

def find_optimal_k_silhouette(silhouette_scores):
    return silhouette_scores.index(max(silhouette_scores)) + 2

# Main app
def main():
    st.markdown("<h1 class='main-header'>🎨 Gaussian Mixture Model Explorer</h1>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🧮 Explore", "📚 Learn", "🧠 Quiz"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Explore Gaussian Mixture Models</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<p class='text-content'>Adjust parameters and generate data:</p>", unsafe_allow_html=True)
            n_samples = st.slider("Number of samples", min_value=100, max_value=1000, value=200, step=100)
            n_features = st.slider("Number of features", min_value=2, max_value=5, value=2, step=1)
            max_k = st.slider("Maximum number of components", min_value=5, max_value=20, value=10, step=1)
            generate_data = st.button("Generate Data")
        
        with col2:
            if generate_data:
                n_centers = np.random.randint(3, 7)
                X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=np.random.randint(100))
                
                bic = calculate_bic(X, max_k)
                silhouette_scores = calculate_silhouette_scores(X, max_k)
                
                st.markdown("<h3 class='sub-header'>BIC and Silhouette Score</h3>", unsafe_allow_html=True)
                plot_bic_and_silhouette(bic, silhouette_scores)
                
                bic_optimal_k = find_optimal_k_bic(bic)
                silhouette_optimal_k = find_optimal_k_silhouette(silhouette_scores)
                optimal_k = silhouette_optimal_k if silhouette_optimal_k == bic_optimal_k else bic_optimal_k
                
                st.markdown(f"<p class='text-content'><strong>Optimal Number of Components (BIC):</strong> {bic_optimal_k}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='text-content'><strong>Optimal Number of Components (Silhouette Score):</strong> {silhouette_optimal_k}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='text-content'><strong>Best Number of Components:</strong> {optimal_k}</p>", unsafe_allow_html=True)
                
                gmm = GaussianMixture(n_components=optimal_k, covariance_type='full', random_state=0)
                gmm.fit(X)
                
                st.markdown("<h3 class='sub-header'>Clustering Results</h3>", unsafe_allow_html=True)
                plot_clusters(X, gmm)

    with tab2:
        st.markdown("<h2 class='sub-header'>Learn about Gaussian Mixture Models</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>What is a Gaussian Mixture Model?</h3>
        <p class='text-content'>
        A Gaussian Mixture Model (GMM) is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. GMMs are commonly used for data clustering, particularly when clusters may have different sizes and correlations between features within each cluster.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='highlight'>
        <h3>Key Concepts</h3>
        <p class='text-content'>
        1. <strong>Components:</strong> Each Gaussian distribution in the mixture.<br>
        2. <strong>BIC (Bayesian Information Criterion):</strong> A criterion for model selection that balances the likelihood of the model with its complexity.<br>
        3. <strong>Silhouette Score:</strong> A measure of how similar an object is to its own cluster compared to other clusters.<br>
        4. <strong>EM Algorithm:</strong> The method used to fit GMMs, alternating between Expectation (E) and Maximization (M) steps.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<h2 class='sub-header'>Test Your Knowledge</h2>", unsafe_allow_html=True)
        
        questions = [
            {
                "question": "What does GMM stand for in the context of this demo?",
                "options": ["General Matrix Multiplication", "Gaussian Mixture Model", "Generalized Maximum Margin", "Grouped Mean Model"],
                "correct": 1,
                "explanation": "GMM stands for Gaussian Mixture Model. Think of it like a recipe for a fruit salad. Just as a fruit salad is a mixture of different fruits, a Gaussian Mixture Model is a mix of different Gaussian distributions. Each Gaussian distribution in the model is like a different type of fruit in the salad, and the model tries to figure out how much of each 'fruit' (distribution) is in the mix."
            },
            {
                "question": "Which of the following is NOT a key concept in GMMs?",
                "options": ["Components", "BIC", "Silhouette Score", "Random Forest"],
                "correct": 3,
                "explanation": "Random Forest is not a key concept in GMMs. It's actually a different machine learning algorithm altogether! The other options are important for GMMs. Components are like the ingredients in our fruit salad analogy. BIC (Bayesian Information Criterion) is like a recipe critic that helps us decide if we have the right number of fruits in our salad. Silhouette Score is like a taste test that tells us how well-mixed our fruit salad is."
            },
            {
                "question": "What does the EM algorithm stand for in the context of GMMs?",
                "options": ["Exemplar Matching", "Expectation-Maximization", "Error Minimization", "Euclidean Measure"],
                "correct": 1,
                "explanation": "EM stands for Expectation-Maximization. It's the method used to 'mix the fruit salad' in our GMM. Imagine you're trying to guess the recipe of a fruit salad while blindfolded. The Expectation step is like taking a bite and guessing what fruits are in it and how much. The Maximization step is like adjusting your guess based on the taste. You keep alternating between these steps until you're confident about the recipe. That's how the EM algorithm works to find the best mix of Gaussian distributions in the data."
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
                st.info(f"The correct answer is: {q['options'][q['correct']]}")
                st.markdown(f"<p class='text-content'><strong>Explanation:</strong> {q['explanation']}</p>", unsafe_allow_html=True)
            st.markdown("---")

if __name__ == '__main__':
    main()
