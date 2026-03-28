import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
# importing joblib to load the trained model
import sys
import os

sys.path.append(os.path.abspath("."))

from rag.embed_store import load_vectorstore
from rag.pipeline import answer_question



st.set_page_config(page_title="YouTube Trend Analyzer", layout="wide")

st.title("YouTube Trend Analyzer (India)")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Analytics",
    "💡 Video Idea Generator",
    "🚀 Performance Predictor",
    "📈 View Predictor",
    "📚 RAG Chatbot"
])

st.write("Insights from YouTube trending data to help creators understand what content performs best.")

# Load datasets
categories = pd.read_csv("data/trending_categories_india.csv")
upload_hour = pd.read_csv("data/best_upload_hour_india.csv")
upload_day = pd.read_csv("data/best_upload_day_india.csv")
video_duration = pd.read_csv("data/best_video_duration_india.csv")
keywords = pd.read_csv("data/trending_keywords_india.csv")

# Load model
model = joblib.load("models/preupload_performance_model.pkl")
# loading trained ML model
view_model = joblib.load("models/view_prediction_model.pkl")
view_tfidf = joblib.load("models/view_prediction_tfidf.pkl")
view_columns = joblib.load("models/view_prediction_columns.pkl")
# loading regression model + tfidf + exact training columns

# =========================
# TAB 1 — ANALYTICS
# =========================
with tab1:
    st.header("Trending Video Categories")
    st.bar_chart(categories.set_index(categories.columns[0]))

    st.header("Best Upload Time")
    st.subheader("Best Upload Hour")
    st.bar_chart(upload_hour.set_index(upload_hour.columns[0]))

    st.subheader("Best Upload Day")
    st.bar_chart(upload_day.set_index(upload_day.columns[0]))

    st.header("Best Video Duration")
    st.bar_chart(video_duration.set_index(video_duration.columns[0]))

    st.header("Trending Keywords")
    st.bar_chart(keywords.set_index("keyword"))

# =========================
# TAB 2 — IDEA GENERATOR
# =========================
with tab2:

    st.header("YouTube Video Idea Generator")

    selected_category = st.selectbox(
        "Select a category",
        [
            "Gaming",
            "Music",
            "Entertainment",
            "People & Blogs",
            "Film & Animation",
            "Comedy",
            "Science & Technology",
            "Sports",
            "News & Politics",
            "Howto & Style",
            "Education",
            "Travel & Events",
            "Pets & Animals"
        ]
    )

    st.subheader("Best Strategy Based on Data")

    top_days = upload_day.sort_values(by=upload_day.columns[1], ascending=False).head(3)
    top_hours = upload_hour.sort_values(by=upload_hour.columns[1], ascending=False).head(3)
    top_durations = video_duration.sort_values(by=video_duration.columns[1], ascending=False).head(3)

    st.write("📅 Best Upload Days:")
    for day in top_days.iloc[:, 0]:
        st.write(f"- {day}")

    st.write("⏰ Best Upload Hours:")
    for hour in top_hours.iloc[:, 0]:
        st.write(f"- {hour}:00")

    st.write("⏱️ Best Video Durations:")
    for duration in top_durations.iloc[:, 0]:
        st.write(f"- {duration}")

    st.subheader("Top Trending Keywords")
    top_keywords = keywords.sort_values(by="count", ascending=False).head(10)
    keyword_list = top_keywords["keyword"].tolist()
    st.write(", ".join(keyword_list))

    st.subheader("Suggested Video Titles")

    if selected_category == "Gaming":
        titles = [
            "Minecraft Challenge",
            "GTA Story Mode Gameplay",
            "Horror Game Walkthrough",
            "Pro Player vs Noob",
            "Live Gaming Highlights"
        ]

    elif selected_category == "Music":
        titles = [
            "Official Song Release",
            "Lyrical Video",
            "Music Video Premiere",
            "Live Performance",
            "New Remix Track"
        ]

    elif selected_category == "Entertainment":
        titles = [
            "Best Viral Moments",
            "Reaction to Trending Clip",
            "Behind The Scenes",
            "Must Watch Episode",
            "Top Entertainment Highlights"
        ]

    elif selected_category == "People & Blogs":
        titles = [
            "A Day in My Life",
            "My Honest Experience",
            "Truth About This Trend",
            "Weekend Vlog",
            "What I Learned Recently"
        ]

    elif selected_category == "Film & Animation":
        titles = [
            "Official Trailer Breakdown",
            "Movie Teaser Reaction",
            "Animation Short Film",
            "Film Review",
            "Top Scenes Explained"
        ]

    elif selected_category == "Comedy":
        titles = [
            "Funny Comedy Skit",
            "Expectation vs Reality",
            "Daily Life Parody",
            "Standup Comedy Clip",
            "Funniest Moments Compilation"
        ]

    elif selected_category == "Science & Technology":
        titles = [
            "Latest Tech Explained",
            "AI Tool Review",
            "Best Gadget Comparison",
            "New Technology Update",
            "Is This Tech Worth It?"
        ]

    elif selected_category == "Sports":
        titles = [
            "Match Highlights",
            "Top Sports Moments",
            "Training Session Vlog",
            "Full Match Analysis",
            "Best Player Performance"
        ]

    elif selected_category == "News & Politics":
        titles = [
            "Breaking News Update",
            "Latest Political Analysis",
            "What Happened Today?",
            "Big News Explained",
            "Full Current Affairs Breakdown"
        ]

    elif selected_category == "Howto & Style":
        titles = [
            "How to Style This",
            "Easy Step by Step Tutorial",
            "Beginner Fashion Guide",
            "Simple Daily Routine Tips",
            "Top Style Hacks"
        ]

    elif selected_category == "Education":
        titles = [
            "Topic Explained for Beginners",
            "One Shot Revision",
            "Full Tutorial Class",
            "Easy Learning Session",
            "Practice Questions and Answers"
        ]

    elif selected_category == "Travel & Events":
        titles = [
            "Travel Vlog Experience",
            "Best Places to Visit",
            "Event Highlights",
            "Full Trip Guide",
            "Hidden Gems to Explore"
        ]

    elif selected_category == "Pets & Animals":
        titles = [
            "Cute Pet Moments",
            "Daily Pet Routine",
            "Funny Animal Video",
            "Pet Care Guide",
            "Training Tips for Pets"
        ]

    else:
        titles = [
            "Trending Video Idea 1",
            "Trending Video Idea 2",
            "Trending Video Idea 3"
        ]

    for title in titles:
        st.write(f"- {title}")

    st.subheader("Keyword-Based Title Ideas")
    for kw in keyword_list[:5]:
        st.write(f"- {selected_category} {kw.title()}")

        
# =========================
# TAB 3 — ML PREDICTOR
# =========================
with tab3:
    st.header("Pre-Upload Performance Predictor")
    st.write("Predict whether a video is likely to be high-performing before upload.")

    # user inputs
    video_category = st.selectbox(
        "Select Video Category",
        [
            "Comedy",
            "Education",
            "Entertainment",
            "Film & Animation",
            "Gaming",
            "Howto & Style",
            "Music",
            "News & Politics",
            "People & Blogs",
            "Pets & Animals",
            "Science & Technology",
            "Sports",
            "Travel & Events"
        ]
    )

    video_definition = st.selectbox("Select Video Definition", ["hd", "sd"])

    video_duration_seconds = st.number_input("Enter Video Duration (in seconds)", min_value=1, value=300)

    channel_video_count = st.number_input("Enter Channel Video Count", min_value=0, value=100)

    channel_have_hidden_subscribers = st.selectbox("Hidden Subscribers?", [False, True])

    if st.button("Predict Performance"):

        # create input row
        input_data = pd.DataFrame([{
            "channel_video_count": channel_video_count,
            "channel_have_hidden_subscribers": channel_have_hidden_subscribers,
            "video_duration_seconds": video_duration_seconds,
            "video_category_id_Comedy": 0,
            "video_category_id_Education": 0,
            "video_category_id_Entertainment": 0,
            "video_category_id_Film & Animation": 0,
            "video_category_id_Gaming": 0,
            "video_category_id_Howto & Style": 0,
            "video_category_id_Music": 0,
            "video_category_id_News & Politics": 0,
            "video_category_id_People & Blogs": 0,
            "video_category_id_Pets & Animals": 0,
            "video_category_id_Science & Technology": 0,
            "video_category_id_Sports": 0,
            "video_category_id_Travel & Events": 0,
            "video_definition_sd": 0
        }])

        # set selected category
        category_column = f"video_category_id_{video_category}"
        if category_column in input_data.columns:
            input_data[category_column] = 1

        # set definition
        if video_definition == "sd":
            input_data["video_definition_sd"] = 1

        # FIX (column order)
        input_data = input_data[model.feature_names_in_]

        # prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # result
        if prediction == 1:
            st.success(f"High-performing video likely! Confidence: {probability:.2%}")
        else:
            st.warning(f"Low-performing video likely. Confidence: {probability:.2%}")
# =========================
# TAB 4 — ML PREDICTOR
# =========================  
with tab4:
    st.header("Pre-Upload View Predictor")
    st.write("Estimate how many views a video may get before upload.")

    # user inputs
    view_title = st.text_input("Video Title", placeholder="Enter your video title here")
    view_tags = st.text_input("Video Tags", placeholder="Enter tags separated by commas")
    view_category = st.selectbox(
        "Video Category",
        [
            "Comedy",
            "Education",
            "Entertainment",
            "Film & Animation",
            "Gaming",
            "Howto & Style",
            "Music",
            "News & Politics",
            "People & Blogs",
            "Pets & Animals",
            "Science & Technology",
            "Sports",
            "Travel & Events"
        ],
        key="view_category"
    )
    view_definition = st.selectbox("Video Definition", ["hd", "sd"], key="view_definition")
    view_duration_seconds = st.number_input("Video Duration (seconds)", min_value=1, value=300, key="view_duration_seconds")
    view_channel_subscribers = st.number_input("Channel Subscriber Count", min_value=0, value=100000, key="view_channel_subscribers")
    view_channel_video_count = st.number_input("Channel Video Count", min_value=0, value=100, key="view_channel_video_count")
    view_hidden_subs = st.selectbox("Hidden Subscribers?", [False, True], key="view_hidden_subs")

    if st.button("Predict Views"):
        # combine title + tags like training
        combined_text = f"{view_title} {view_tags}"

        # convert text into tfidf features
        text_features = view_tfidf.transform([combined_text])
        text_df = pd.DataFrame(text_features.toarray(), columns=view_tfidf.get_feature_names_out())

        # base numeric/categorical row
        input_data = pd.DataFrame([{
            "channel_subscriber_count": view_channel_subscribers,
            "channel_video_count": view_channel_video_count,
            "channel_have_hidden_subscribers": view_hidden_subs,
            "video_duration_seconds": view_duration_seconds
        }])

        # category dummies
        category_columns = [
            "video_category_id_Comedy",
            "video_category_id_Education",
            "video_category_id_Entertainment",
            "video_category_id_Film & Animation",
            "video_category_id_Gaming",
            "video_category_id_Howto & Style",
            "video_category_id_Music",
            "video_category_id_News & Politics",
            "video_category_id_People & Blogs",
            "video_category_id_Pets & Animals",
            "video_category_id_Science & Technology",
            "video_category_id_Sports",
            "video_category_id_Travel & Events"
        ]

        for col in category_columns:
            input_data[col] = 0

        selected_category_col = f"video_category_id_{view_category}"
        if selected_category_col in input_data.columns:
            input_data[selected_category_col] = 1

        # video definition dummy
        input_data["video_definition_sd"] = 1 if view_definition == "sd" else 0

        # combine structured inputs + tfidf text inputs
        full_input = pd.concat([input_data.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1)

        # add any missing columns from training
        for col in view_columns:
            if col not in full_input.columns:
                full_input[col] = 0

        # reorder exactly like training
        full_input = full_input[view_columns]

        # predict log views and convert back
        predicted_log_views = view_model.predict(full_input)[0]
        predicted_views = int(np.expm1(predicted_log_views))

        st.success(f"Estimated Views: {predicted_views:,}")          

# =========================
# TAB 5 — RAG CHaTBOT
# =========================        

with tab5:
    st.header("RAG Chatbot")
    st.write("Ask questions based on your documents.")

    # load vectorstore once
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore()

    question = st.text_input("Enter your question", key="rag_input")

    if st.button("Ask", key="rag_button") and question:
        answer, docs = answer_question(st.session_state.vectorstore, question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, doc in enumerate(docs, start=1):
            st.write(f"**Source {i}:** {doc.metadata}")
            st.write(doc.page_content[:300])
            st.write("---") 