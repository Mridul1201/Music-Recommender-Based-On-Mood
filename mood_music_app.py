
import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_data():
    return pd.read_csv("music_mood_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("mood_knn_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

def recommend(df, mood):
    return df[df["mood"] == mood].sample(3)

st.title("🎧 Mood-Based Music Recommendation")

mood = st.selectbox("Select Mood:", ["Happy","Sad","Energetic","Calm","Romantic"])

if st.button("Recommend"):
    df = load_data()
    res = recommend(df, mood)
    st.dataframe(res)
