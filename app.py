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
            font-size: 24px;
            margin-top: 30px;
            font-weight: 600;
        }
        .stButton > button {
            background-color: #f0f8ff;
            border: none;
            border-radius: 15px;
            padding: 1.5rem;
            width: 100%;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease-in-out;
            font-size: 26px;
            font-weight: bold;
            line-height: 1.3;
            color: #003366;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #d6eaff;
            transform: scale(1.03);
        }
        .refresh-button > button {
            background-color: #d1f2eb;
            color: #00695c;
            font-weight: bold;
            font-size: 16px;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            margin-top: 1rem;
            border: 2px solid #66bb6a;
        }
        .refresh-button > button:hover {
            background-color: #b2dfdb;
            color: #004d40;
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

df_full, kmeans, dbscan, gmm, fasttext_vectors, title_to_index, combined_features = load_resources()

# ------------------------------
# Persistent State Setup (before UI)
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
    st.session_state.refresh_grid = False  # VERY IMPORTANT!

if 'selected_title' not in st.session_state:
    st.session_state.selected_title = None

if 'grid_selected_index' not in st.session_state:
    st.session_state.grid_selected_index = None

if 'custom_input' not in st.session_state:
    st.session_state.custom_input = ""

if 'last_action' not in st.session_state:
    st.session_state.last_action = None

# ------------------------------
# UI Layout
# ------------------------------
st.markdown("<div class='main-title'>🎬 Movie Recommendation Engine</div>", unsafe_allow_html=True)

# ------------------------------
# 🎯 3x3 Movie Grid
st.markdown("<div class='subsection'>🎯 Choose a movie you like:</div>", unsafe_allow_html=True)

df_grid = st.session_state.grid_movies.copy()
df_grid['year'] = df_grid['year'].fillna(0).astype(int).astype(str)

cols = st.columns(3)
for i in range(3):
    for j in range(3):
        idx = i * 3 + j
        with cols[j]:
            movie_title = df_grid.loc[idx, 'title']
            movie_year = df_grid.loc[idx, 'year']

            button_text = f"{movie_title}\n({movie_year})"

            clicked = st.button(button_text, key=f"movie_{idx}")

            if clicked:
                st.session_state.grid_selected_index = idx
                st.session_state.last_action = 'grid'

# ------------------------------
# Refresh Grid Button
st.markdown("### ")
refresh_col = st.columns([1, 6, 1])[1]
with refresh_col:
    if st.button("🔁 Refresh Movie Grid", key="refresh_grid_button", help="Get a new random set of movies"):
        st.session_state.refresh_grid = True
        st.experimental_rerun()

# ------------------------------
# Custom Movie Textbox
st.markdown("---")
st.markdown("<div class='subsection'>✍️ Or select a custom movie title:</div>", unsafe_allow_html=True)

all_titles = df_full['title'].dropna().sort_values().unique().tolist()

st.session_state.custom_input = st.selectbox(
    "Type or select a movie title",
    options=[""] + all_titles,
    index=0
)

if st.session_state.custom_input and st.session_state.custom_input != "":
    st.session_state.last_action = 'textbox'

# ------------------------------
# Final Movie Selection
# ------------------------------
if st.session_state.last_action == 'grid':
    idx = st.session_state.grid_selected_index
    st.session_state.selected_title = df_grid.loc[idx, 'title']
elif st.session_state.last_action == 'textbox':
    st.session_state.selected_title = st.session_state.custom_input

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
            col1, col2 = st.columns(2)
            card_color = "#e6f2ff"

            for idx, row in recommendations.iterrows():
                card_html = f"""
                    <div style="
                        background-color: {card_color};
                        border-radius: 15px;
                        padding: 1.2rem;
                        margin: 0.8rem;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size:22px; font-weight:bold; color:#003366; margin-bottom:0.5rem;">🎬 {row['title']}</div>
                        <div style="font-size:18px; color:#555555;">🎬 Director: {row.get('director', 'N/A')}</div>
                        <div style="font-size:18px; color:#555555;">🎭 Genres: {row.get('genre_list', 'N/A')}</div>
                    </div>
                """
                if idx % 2 == 0:
                    with col1:
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    with col2:
                        st.markdown(card_html, unsafe_allow_html=True)
