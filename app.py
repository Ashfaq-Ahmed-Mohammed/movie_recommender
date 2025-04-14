import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(page_title="Movie Recommendation Engine", layout="wide")
st.markdown("""
    <style>
        .main-title {
            font-size: 48px;
            font-weight: bold;
            color: #3366cc;
            text-align: center;
        }
        .subsection {
            font-size: 20px;
            margin-top: 30px;
            font-weight: 600;
        }
        .stButton > button {
            background-color: #cce5ff;
            color: #003366;
            font-weight: bold;
            border-radius: 10px;
            border: 1px solid #99ccff;
            padding: 0.5rem 1rem;
        }
        .stButton > button:focus,
        .stButton > button:active {
            background-color: #99ccff;
            color: #001f4d;
            border: 1px solid #3366cc;
        }
        .stTextInput > div > input {
            border-radius: 10px;
            padding: 0.5rem;
            border: 1px solid #3366cc;
        }
        .stRadio > div {
            background-color: #f4faff;
            padding: 0.5rem;
            border-radius: 10px;
            border: 1px solid #cce5ff;
        }
        .stSlider > div {
            background-color: #f4faff;
            padding: 0.5rem;
            border-radius: 10px;
        }
        .stMarkdown.recommendation-box {
            background-color: #e6f2ff;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 10px;
            border: 1px solid #99ccff;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Data and Models
# ------------------------------
@st.cache_data
def load_resources():
    df = pd.read_csv("models/Data_clustered.csv")
    df['title'] = df['title'].astype(str).str.strip()
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("models/dbscan_model.pkl", "rb") as f:
        dbscan = pickle.load(f)
    with open("models/gmm_model.pkl", "rb") as f:
        gmm = pickle.load(f)
    with open("models/fasttext_vectors.pkl", "rb") as f:
        fasttext_vectors = pickle.load(f)
    with open("models/title_to_index.pkl", "rb") as f:
        title_to_index = pickle.load(f)

    combined_features = np.load("models/combined_features.npy")

    return df, kmeans, dbscan, gmm, fasttext_vectors, title_to_index, combined_features

# Load all resources
df_full, kmeans, dbscan, gmm, fasttext_vectors, title_to_index, combined_features = load_resources()

# ------------------------------
# Persistent State Setup
# ------------------------------
if 'refresh_grid' not in st.session_state:
    st.session_state.refresh_grid = False

if 'grid_movies' not in st.session_state:
    recent_movies = df_full[df_full['year'] >= 2000].dropna(subset=['title', 'popularity']).copy()
    threshold = recent_movies['popularity'].quantile(0.75)
    popular_recent_movies = recent_movies[recent_movies['popularity'] >= threshold]
    st.session_state.grid_movies = popular_recent_movies.sample(n=9, random_state=None).reset_index(drop=True)

if st.session_state.refresh_grid:
    recent_movies = df_full[df_full['year'] >= 2000].dropna(subset=['title', 'popularity']).copy()
    threshold = recent_movies['popularity'].quantile(0.75)
    popular_recent_movies = recent_movies[recent_movies['popularity'] >= threshold]
    st.session_state.grid_movies = popular_recent_movies.sample(n=9, random_state=None).reset_index(drop=True)
    st.session_state.refresh_grid = False

# ------------------------------
if 'selected_title' not in st.session_state:
    st.session_state.selected_title = None

if 'grid_movies' not in st.session_state or st.session_state.get('refresh_grid', False):
    recent_movies = df_full[df_full['year'] >= 2000].dropna(subset=['title', 'popularity']).copy()
    threshold = recent_movies['popularity'].quantile(0.75)
    popular_recent_movies = recent_movies[recent_movies['popularity'] >= threshold]
    st.session_state.grid_movies = popular_recent_movies.sample(n=9, random_state=None).reset_index(drop=True)
    st.session_state.refresh_grid = False

# ------------------------------
# UI
# ------------------------------
st.markdown("<div class='main-title'>🎬 Movie Recommendation Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='subsection'>🎯 Choose a movie you like (popular and post-2000):</div>", unsafe_allow_html=True)

if st.button("🔁 Refresh Movie Grid"):
    st.session_state.refresh_grid = True
    st.experimental_rerun()

selected_from_grid = None
df_grid = st.session_state.grid_movies
cols = st.columns(3)
for i in range(3):
    for j in range(3):
        idx = i * 3 + j
        with cols[j]:
            if st.button(df_grid.loc[idx, 'title'], key=idx):
                selected_from_grid = df_grid.loc[idx, 'title']

st.markdown("---")
st.markdown("<div class='subsection'>✍️ Or enter a custom movie title:</div>", unsafe_allow_html=True)
custom_input = st.text_input("Movie title")

if custom_input:
    st.session_state.selected_title = custom_input
elif selected_from_grid:
    st.session_state.selected_title = selected_from_grid

# ------------------------------
# Recommendation Functions
# ------------------------------
def get_recommendations_kmeans(title, df, title_to_index, combined_features, top_n=10):
    title = title.strip().lower()
    if title not in title_to_index:
        return f"Movie titled '{title}' not found."
    idx = title_to_index[title]
    cluster_label = df.loc[idx, 'cluster']
    cluster_movies = df[df['cluster'] == cluster_label].copy()
    cluster_movies = cluster_movies[cluster_movies.index != idx]
    cluster_indices = cluster_movies.index
    if cluster_movies.empty:
        return f"No other movies found in the same cluster as '{title}'."
    cos_sim = cosine_similarity(combined_features[idx].reshape(1, -1), combined_features[cluster_indices]).flatten()
    cluster_movies['similarity'] = cos_sim
    return cluster_movies.sort_values(by='similarity', ascending=False)[['title', 'director', 'genre_list', 'similarity']].head(top_n)

def get_recommendations_dbscan(title, df, title_to_index, combined_features, top_n=10):
    title = title.strip().lower()
    if title not in title_to_index:
        return f"Movie '{title}' not found."
    idx = title_to_index[title]
    target_cluster = df.loc[idx, 'dbscan_cluster']
    if target_cluster == -1:
        return f"Movie '{title}' is labeled as noise by DBSCAN and has no cluster."
    cluster_movies = df[df['dbscan_cluster'] == target_cluster].copy()
    cluster_movies = cluster_movies[cluster_movies.index != idx]
    cluster_indices = cluster_movies.index
    target_vec = combined_features[idx].reshape(1, -1)
    cos_sim = cosine_similarity(target_vec, combined_features[cluster_indices]).flatten()
    cluster_movies['similarity'] = cos_sim
    return cluster_movies.sort_values(by='similarity', ascending=False)[['title', 'director', 'genre_list', 'similarity']].head(top_n)

def get_recommendations_gmm(title, df, title_to_index, combined_features, top_n=10):
    title = title.strip().lower()
    if title not in title_to_index:
        return f"Movie '{title}' not found in the dataset."
    idx = title_to_index[title]
    target_cluster = df.loc[idx, 'gmm_cluster']
    cluster_movies = df[df['gmm_cluster'] == target_cluster].copy()
    cluster_movies = cluster_movies[cluster_movies.index != idx]
    cluster_indices = cluster_movies.index
    target_vec = combined_features[idx].reshape(1, -1)
    cos_sim = cosine_similarity(target_vec, combined_features[cluster_indices]).flatten()
    cluster_movies['similarity'] = cos_sim
    return cluster_movies.sort_values(by='similarity', ascending=False)[['title', 'director', 'genre_list', 'similarity']].head(top_n)

# ------------------------------
# Recommendation Panel
# ------------------------------
if st.session_state.selected_title:
    selected_title = st.session_state.selected_title
    st.markdown(f"<div class='subsection'>✅ You selected: <b>{selected_title}</b></div>", unsafe_allow_html=True)
    model_choice = st.radio("Select a recommendation model:", ['K-Means', 'DBSCAN', 'GMM', 'FastText'], horizontal=True)
    top_n = st.slider("Number of recommendations:", 5, 15, 10)
    if st.button("🔍 Get Recommendations"):
        st.markdown("### 🎥 You might also like:")
        if model_choice == 'GMM':
            recommendations = get_recommendations_gmm(selected_title, df_full, title_to_index, combined_features, top_n)
        elif model_choice == 'DBSCAN':
            recommendations = get_recommendations_dbscan(selected_title, df_full, title_to_index, combined_features, top_n)
        elif model_choice == 'K-Means':
            recommendations = get_recommendations_kmeans(selected_title, df_full, title_to_index, combined_features, top_n)
        else:
            idx = df_full[df_full['title'].str.strip().str.lower() == selected_title.strip().lower()].index[0]
            sim_scores = cosine_similarity([fasttext_vectors[idx]], fasttext_vectors).flatten()
            sim_indices = sim_scores.argsort()[::-1][1:top_n+1]
            recommendations = df_full.iloc[sim_indices][['title', 'director', 'genre_list']]
        if isinstance(recommendations, str):
            st.warning(recommendations)
        elif recommendations is None or len(recommendations) == 0:
            st.warning("No recommendations found. Try another movie.")
        else:
            for _, row in recommendations.iterrows():
                st.write(f"• **{row['title']}** — 🎬 {row.get('director', 'N/A')}, 🎭 {row.get('genre_list', 'N/A')}")
