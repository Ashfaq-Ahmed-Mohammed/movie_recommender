import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


st.set_page_config(page_title="Movie Recommendation Engine", layout="wide")


st.markdown("""
    <style>
        .stApp {
            background-color: #ADD8E1; 
        }
        .main-title {
            font-size: 48px;
            font-weight: bold;
            color: black;
            text-align: center;
        }
        .subsection {
            font-size: 26px;
            font-weight: bold;
            margin-top: 30px;
            color: black;
        }
        h1, h2, h3, h4, h5, div, span, p, label {
            color: black;
        }
        .stButton > button {
            background-color: white;
            color: #001f3f;
            font-weight: bold;
            border-radius: 15px;
            padding: 1.5rem;
            width: 100%;
            box-shadow: 2px 2px 10px rgba(255,255,255,0.2);
            font-size: 24px;
        }
        .stButton > button:hover {
            background-color: #cce5ff;
            color: #001f3f;
            transform: scale(1.03);
        }
        .refresh-button > button {
            background-color: white;
            color: #001f3f;
            font-weight: bold;
            font-size: 16px;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            border: 2px solid #66bb6a;
        }
        .refresh-button > button:hover {
            background-color: #cce5ff;
            color: #001f3f;
        }
        .recommendation-card {
            background-color: white;
            color: #001f3f;
            border-radius: 15px;
            padding: 1.2rem;
            margin: 0.8rem;
            box-shadow: 2px 2px 8px rgba(255,255,255,0.2);
        }
        
        div[role="radiogroup"] input[type="radio"] {
        width: 20px;
        height: 20px;
        }

        
        div[role="radiogroup"] > label > div {
        color: #001f3f !important; 
        font-weight: bold;
        font-size: 22px; 
        }
        
    </style>
""", unsafe_allow_html=True)


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
if 'recommendation_ready' not in st.session_state:
    st.session_state.recommendation_ready = False


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

if 'selected_title' not in st.session_state:
    st.session_state.selected_title = None

if 'grid_selected_index' not in st.session_state:
    st.session_state.grid_selected_index = None

if 'custom_input' not in st.session_state:
    st.session_state.custom_input = ""

if 'last_action' not in st.session_state:
    st.session_state.last_action = None


st.markdown("<div class='main-title'>Movie Recommendation Engine</div>", unsafe_allow_html=True)


st.markdown("<div class='subsection'>Choose a movie you like:</div>", unsafe_allow_html=True)

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


st.markdown("### ")
refresh_col = st.columns([1, 6, 1])[1]
with refresh_col:
    if st.button("🔁 Refresh Movie Grid", key="refresh_grid_button", help="Get a new random set of movies"):
        st.session_state.refresh_grid = True
        st.session_state.recommendation_ready = False
        st.session_state.recommendations = None
        st.experimental_rerun()


st.markdown("---")
st.markdown("<div class='subsection'>Or select a custom movie title:</div>", unsafe_allow_html=True)

all_titles = df_full['title'].dropna().sort_values().unique().tolist()

st.session_state.custom_input = st.selectbox(
    "Type or select a movie title",
    options=[""] + all_titles,
    index=0
)

if st.session_state.custom_input and st.session_state.custom_input != "":
    st.session_state.last_action = 'textbox'


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


if st.session_state.selected_title:
    selected_title = st.session_state.selected_title
    st.markdown(f"<div class='subsection'>You selected: <b>{selected_title}</b></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:23px; color:black;'>Select a recommendation model:</h3>", unsafe_allow_html=True)
    model_choice = st.radio("placeholder", ['K-Means', 'DBSCAN', 'GMM', 'FastText'], horizontal=True, label_visibility="collapsed")
    st.markdown("<h3 style='font-size:23px; color:black;'>Number of recommendations:</h3>", unsafe_allow_html=True)
    top_n = st.slider("placeholder", 5, 15, 10, label_visibility="collapsed")

    if st.button("🔍 Get Recommendations"):
        
        if model_choice == 'GMM':
            st.session_state.recommendations = recommendations = get_recommendations_gmm(selected_title, df_full, title_to_index, combined_features, top_n)
        elif model_choice == 'DBSCAN':
            st.session_state.recommendations = recommendations = get_recommendations_dbscan(selected_title, df_full, title_to_index, combined_features, top_n)
            
        elif model_choice == 'K-Means':
            st.session_state.recommendations = recommendations = get_recommendations_kmeans(selected_title, df_full, title_to_index, combined_features, top_n)
            
        else:
            idx = df_full[df_full['title'].str.strip().str.lower() == selected_title.strip().lower()].index[0]
            sim_scores = cosine_similarity([fasttext_vectors[idx]], fasttext_vectors).flatten()
            sim_indices = sim_scores.argsort()[::-1][1:top_n+1]
            st.session_state.recommendations = recommendations = df_full.iloc[sim_indices][['title', 'director', 'genre_list']]
        
        st.session_state.recommendation_ready = True

    if st.session_state.get("recommendation_ready") and isinstance(st.session_state.recommendations, pd.DataFrame):
        st.markdown("<h3 style='font-size:23px; color:black;'>You might also like</h3>", unsafe_allow_html=True)

        recommendations = st.session_state.recommendations
        col1, col2 = st.columns(2)

        for idx, row in recommendations.iterrows():
            genres = row.get('genre_list', 'N/A')
            if isinstance(genres, list):
                genres = ", ".join(genres)
            elif isinstance(genres, str):
                genres = genres.replace("[", "").replace("]", "").replace("'", "")

            card_html = f"""
            <div class='recommendation-card'>
                <div style="font-size:22px; font-weight:bold; margin-bottom:0.5rem;">{row['title']}</div>
                <div style="font-size:18px;">Director: {row.get('director', 'N/A')}</div>
                <div style="font-size:18px;">Genres: {genres}</div>
            </div>
            """

            if idx % 2 == 0:
                with col1:
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                with col2:
                    st.markdown(card_html, unsafe_allow_html=True)

    if st.session_state.recommendation_ready:
        st.markdown("<h3 style='font-size:23px; color:black;'>Cluster Visualization</h3>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:18px; color:black;'>Click the button below to visualize the clusters.</h4>", unsafe_allow_html=True)

    if st.button("📊 Show Cluster Visualization"):
        recommendations = st.session_state.recommendations

        
        vectors = fasttext_vectors if model_choice == "FastText" else combined_features

        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(vectors)

        df_full['x'] = coordinates[:, 0]
        df_full['y'] = coordinates[:, 1]

        selected_movie = selected_title.strip().lower()
        selected_index = title_to_index[selected_movie]

        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            recommended_titles = recommendations['title'].tolist()
            recommended_indices = df_full[
                df_full['title'].str.lower().isin([title.lower() for title in recommended_titles])
            ].index

            
            fig, ax = plt.subplots(figsize=(6, 5))
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            legend_font = FontProperties()
            legend_font.set_size('xx-small')
            legend_font.set_weight('normal')

            if model_choice != "FastText":
                
                cluster_column = {
                    "K-Means": "cluster",
                    "DBSCAN": "dbscan_cluster",
                    "GMM": "gmm_cluster"
                }[model_choice]
                selected_cluster_label = df_full.loc[selected_index, cluster_column]
                cluster_movies = df_full[df_full[cluster_column] == selected_cluster_label]

                
                ax.scatter(
                    cluster_movies['x'],
                    cluster_movies['y'],
                    alpha=0.4,
                    s=10,
                    color='gray',
                    label='Same Cluster'
                )
            else:
                
                ax.scatter(
                    df_full['x'],
                    df_full['y'],
                    alpha=0.2,
                    s=10,
                    color='gray',
                    label='All Movies'
                )

            
            ax.scatter(
                df_full.loc[recommended_indices, 'x'],
                df_full.loc[recommended_indices, 'y'],
                color='blue',
                s=10,
                label='Recommended'
            )

            
            ax.scatter(
                df_full.loc[selected_index, 'x'],
                df_full.loc[selected_index, 'y'],
                color='red',
                s=30,
                marker='*',
                label='Selected Movie'
            )

            ax.set_title("FastText Visualization" if model_choice == "FastText" else "Cluster Visualization", fontsize=6, color='black')
            ax.tick_params(labelsize=6, colors='black')
            ax.legend(prop=legend_font)
            ax.set_xlabel("PCA Component 1", fontsize=6, color='black')
            ax.set_ylabel("PCA Component 2", fontsize=6, color='black')
            plt.tight_layout()
            st.pyplot(fig)
