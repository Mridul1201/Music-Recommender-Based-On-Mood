# mood_music_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

# -----------------------------
# Load Model + Dataset
# -----------------------------
@st.cache_data
def load_music_data():
    df = pd.read_csv("music_mood_dataset.csv")
    return df

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_model():
    return joblib.load("mood_knn_model.pkl")


# -----------------------------
# Recommendation Logic
# -----------------------------
def recommend_songs(df, mood, n=5):
    mood_df = df[df['mood'] == mood]

    if mood_df.empty:
        return None

    return mood_df.sample(n)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mood-Based Music Recommender", layout="wide")

st.title("🎧 Mood-Based Music Recommendation System")
st.write("Select your mood and discover songs that match your vibe!")

mood = st.selectbox(
    "How are you feeling today?",
    ["Happy", "Sad", "Energetic", "Calm", "Romantic"]
)

if st.button("Recommend Songs"):
    df = load_music_data()

    results = recommend_songs(df, mood)

    if results is None:
        st.error("No songs found for this mood.")
    else:
        st.success("Here are your recommendations!")
        st.dataframe(results)

        # Display Spotify links if available
        for index, row in results.iterrows():
            if 'spotify_url' in df.columns:
                st.markdown(f"🎵 **[{row['song']}]({row['spotify_url']})** — *{row['artist']}*")

