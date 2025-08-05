import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and encoders
model = joblib.load("movie_rating_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load unique values for dropdowns (used during training)
df = pd.read_csv("IMDb Movies India.csv")
df = df[["Genre", "Director", "Actor 1", "Rating"]].dropna()
df.rename(columns={"Actor 1": "Star1", "Rating": "IMDB Rating"}, inplace=True)

genres = sorted(df["Genre"].unique())
directors = sorted(df["Director"].unique())
actors = sorted(df["Star1"].unique())

# Streamlit UI
st.set_page_config(page_title="Movie Rating Predictor", layout="centered")
st.title("🎬 Movie Rating Prediction App")
st.markdown("Predict a movie's **IMDb Rating** based on its genre, director, and lead actor.")

with st.form("rating_form"):
    genre = st.selectbox("🎭 Select Genre:", genres)
    director = st.selectbox("🎬 Select Director:", directors)
    actor = st.selectbox("⭐ Select Lead Actor:", actors)
    submit = st.form_submit_button("Predict Rating")

if submit:
    try:
        # Encode inputs
        genre_enc = label_encoders["Genre"].transform([genre])[0]
        director_enc = label_encoders["Director"].transform([director])[0]
        actor_enc = label_encoders["Star1"].transform([actor])[0]

        features = np.array([[genre_enc, director_enc, actor_enc]])
        prediction = model.predict(features)[0]

        st.success(f"🎯 Predicted IMDb Rating: **{round(prediction, 2)}**")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
