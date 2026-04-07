import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt
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
    st.write("Fill in your video details below to find out if it is likely to perform well before you upload.")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Video Details")
        video_category = st.selectbox(
            "Video Category",
            [
                "Comedy", "Education", "Entertainment", "Film & Animation",
                "Gaming", "Howto & Style", "Music", "News & Politics",
                "People & Blogs", "Pets & Animals", "Science & Technology",
                "Sports", "Travel & Events"
            ]
        )
        video_definition = st.selectbox("Video Quality", ["hd", "sd"])
        video_duration_seconds = st.number_input("Video Duration (seconds)", min_value=1, value=300)

    with col_b:
        st.subheader("Channel Details")
        channel_video_count = st.number_input("Total Videos on Channel", min_value=0, value=100)
        channel_have_hidden_subscribers = st.selectbox("Subscribers Hidden?", [False, True])

    st.divider()

    if st.button("Predict Performance", use_container_width=True):

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

        category_column = f"video_category_id_{video_category}"
        if category_column in input_data.columns:
            input_data[category_column] = 1

        if video_definition == "sd":
            input_data["video_definition_sd"] = 1

        input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prediction == 1:
                st.metric(label="Prediction", value="High Performing")
            else:
                st.metric(label="Prediction", value="Low Performing")

        with res_col2:
            st.write("**Confidence Score**")
            st.progress(float(probability))
            st.caption(f"{probability:.1%} confident in this prediction")

        st.divider()
        if prediction == 1:
            st.success("This video has a strong chance of performing well based on its category, quality, and duration.")
        else:
            st.warning("This video may underperform. Consider adjusting the category, duration, or video quality.")

        # --- Confidence chart ---
        fig, ax = plt.subplots(figsize=(5, 1.2))
        bar_color = '#00c853' if probability >= 0.5 else '#ff4b4b'
        ax.barh([''], [probability], color=bar_color, height=0.5)
        ax.barh([''], [1 - probability], left=[probability], color='#e8e8e8', height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='#666', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(f'Performance Confidence: {probability:.1%}', fontsize=10)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- Improvement tips ---
        st.subheader("💡 How to Improve")
        tips = []

        top_dur = video_duration.sort_values(by=video_duration.columns[1], ascending=False).iloc[0, 0]
        if video_duration_seconds < 300:
            tips.append(f"**Duration**: Your video is quite short. Trending videos often perform better at longer lengths — try aiming for **{top_dur}**.")
        elif video_duration_seconds > 1800:
            tips.append(f"**Duration**: Very long videos can lose viewers. Consider trimming to around **{top_dur}**.")

        if video_definition == "sd":
            tips.append("**Quality**: Switch to **HD** — high definition videos consistently outperform SD on YouTube.")

        if channel_video_count < 20:
            tips.append("**Channel Size**: Channels with more videos tend to trend more. Try publishing content more regularly to build momentum.")

        top_category = categories.sort_values(by=categories.columns[1], ascending=False).iloc[0, 0]
        if video_category != top_category:
            tips.append(f"**Category**: **{top_category}** is currently the top-trending category in India. If your content fits, it's worth considering.")

        top_day = upload_day.sort_values(by=upload_day.columns[1], ascending=False).iloc[0, 0]
        top_hour = upload_hour.sort_values(by=upload_hour.columns[1], ascending=False).iloc[0, 0]
        tips.append(f"**Upload Timing**: For best reach, upload on **{top_day}** around **{top_hour}:00**.")

        if not tips:
            tips.append("Your video setup looks great! Keep the quality and duration consistent.")

        for tip in tips:
            st.markdown(f"- {tip}")

# =========================
# TAB 4 — VIEW PREDICTOR
# =========================
with tab4:
    st.header("Pre-Upload View Predictor")
    st.write("Estimate how many views your video may get based on its title, tags, category, and channel details.")

    st.divider()

    st.subheader("Video Content")
    view_title = st.text_input("Video Title", placeholder="Enter your video title here")
    view_tags = st.text_input("Video Tags", placeholder="Enter tags separated by commas")

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        view_category = st.selectbox(
            "Video Category",
            [
                "Comedy", "Education", "Entertainment", "Film & Animation",
                "Gaming", "Howto & Style", "Music", "News & Politics",
                "People & Blogs", "Pets & Animals", "Science & Technology",
                "Sports", "Travel & Events"
            ],
            key="view_category"
        )
        view_definition = st.selectbox("Video Quality", ["hd", "sd"], key="view_definition")

    with vcol2:
        view_duration_seconds = st.number_input("Video Duration (seconds)", min_value=1, value=300, key="view_duration_seconds")

    st.subheader("Channel Details")
    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        view_channel_subscribers = st.number_input("Subscriber Count", min_value=0, value=100000, key="view_channel_subscribers")
    with ccol2:
        view_channel_video_count = st.number_input("Total Videos on Channel", min_value=0, value=100, key="view_channel_video_count")
    with ccol3:
        view_hidden_subs = st.selectbox("Subscribers Hidden?", [False, True], key="view_hidden_subs")

    st.divider()

    if st.button("Predict Views", use_container_width=True):
        combined_text = f"{view_title} {view_tags}"

        text_features = view_tfidf.transform([combined_text])
        text_df = pd.DataFrame(text_features.toarray(), columns=view_tfidf.get_feature_names_out())

        input_data = pd.DataFrame([{
            "channel_subscriber_count": view_channel_subscribers,
            "channel_video_count": view_channel_video_count,
            "channel_have_hidden_subscribers": view_hidden_subs,
            "video_duration_seconds": view_duration_seconds
        }])

        category_columns = [
            "video_category_id_Comedy", "video_category_id_Education",
            "video_category_id_Entertainment", "video_category_id_Film & Animation",
            "video_category_id_Gaming", "video_category_id_Howto & Style",
            "video_category_id_Music", "video_category_id_News & Politics",
            "video_category_id_People & Blogs", "video_category_id_Pets & Animals",
            "video_category_id_Science & Technology", "video_category_id_Sports",
            "video_category_id_Travel & Events"
        ]

        for col in category_columns:
            input_data[col] = 0

        selected_category_col = f"video_category_id_{view_category}"
        if selected_category_col in input_data.columns:
            input_data[selected_category_col] = 1

        input_data["video_definition_sd"] = 1 if view_definition == "sd" else 0

        full_input = pd.concat([input_data.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1)

        for col in view_columns:
            if col not in full_input.columns:
                full_input[col] = 0

        full_input = full_input[view_columns]

        predicted_log_views = view_model.predict(full_input)[0]
        predicted_views = int(np.expm1(predicted_log_views))

        # format views into readable label
        if predicted_views >= 1_000_000:
            views_label = f"{predicted_views / 1_000_000:.1f}M"
        elif predicted_views >= 1_000:
            views_label = f"{predicted_views / 1_000:.1f}K"
        else:
            views_label = str(predicted_views)

        # performance tier
        if predicted_views >= 5_000_000:
            tier = "Viral"
            tier_note = "This video has strong viral potential based on its inputs."
        elif predicted_views >= 1_000_000:
            tier = "Excellent"
            tier_note = "This video is expected to perform very well."
        elif predicted_views >= 200_000:
            tier = "Good"
            tier_note = "This video should get solid engagement."
        elif predicted_views >= 50_000:
            tier = "Average"
            tier_note = "This video may get moderate views. Consider refining the title or tags."
        else:
            tier = "Low"
            tier_note = "This video may struggle. Try a more trending category or stronger keywords."

        st.divider()
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="Estimated Views", value=views_label)
        with m2:
            st.metric(label="Performance Tier", value=tier)

        st.divider()
        if tier in ["Viral", "Excellent"]:
            st.success(tier_note)
        elif tier == "Good":
            st.info(tier_note)
        else:
            st.warning(tier_note)

        # --- View tier chart ---
        tier_labels = ['Low\n<50K', 'Average\n50K–200K', 'Good\n200K–1M', 'Excellent\n1M–5M', 'Viral\n5M+']
        tier_thresholds = [50_000, 200_000, 1_000_000, 5_000_000, 10_000_000]
        tier_colors_list = ['#ff4b4b', '#ff9800', '#ffc107', '#4caf50', '#2196f3']
        tier_map = {"Low": 0, "Average": 1, "Good": 2, "Excellent": 3, "Viral": 4}
        active_idx = tier_map[tier]

        bar_colors = ['#e0e0e0'] * 5
        bar_colors[active_idx] = tier_colors_list[active_idx]

        fig, ax = plt.subplots(figsize=(6, 2.5))
        bars = ax.bar(tier_labels, tier_thresholds, color=bar_colors)
        bars[active_idx].set_edgecolor('black')
        bars[active_idx].set_linewidth(2)
        ax.set_yscale('log')
        ax.set_ylabel('Views', fontsize=9)
        ax.set_title(f'Your Video Falls in: {tier}', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        fig.patch.set_alpha(0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- Improvement tips ---
        st.subheader("💡 How to Improve")
        tips = []

        if len(view_title) < 20:
            tips.append("**Title**: Your title is quite short. Add descriptive keywords — longer titles with clear topics get better click-through.")
        elif len(view_title) > 70:
            tips.append("**Title**: Your title may be too long. Aim for under 70 characters so it doesn't get cut off in search results.")

        tag_count = len([t for t in view_tags.split(',') if t.strip()])
        if tag_count < 5:
            tips.append(f"**Tags**: You only have {tag_count} tag(s). Add more relevant tags (aim for 10–15) to improve discoverability.")

        top_kws = keywords.sort_values(by="count", ascending=False).head(5)["keyword"].tolist()
        title_lower = view_title.lower()
        tags_lower = view_tags.lower()
        missing_kws = [kw for kw in top_kws if kw.lower() not in title_lower and kw.lower() not in tags_lower]
        if missing_kws:
            tips.append(f"**Keywords**: Add trending keywords to your title or tags: **{', '.join(missing_kws[:3])}**")

        if view_duration_seconds < 300:
            tips.append("**Duration**: Videos under 5 minutes may underperform. Try aiming for 8–15 minutes for better retention.")

        if view_definition == "sd":
            tips.append("**Quality**: Upload in **HD** for significantly better performance.")

        if view_channel_subscribers < 10_000:
            tips.append("**Channel**: With fewer subscribers, focus on SEO-heavy titles and trending topics to gain organic reach.")

        top_category_v = categories.sort_values(by=categories.columns[1], ascending=False).iloc[0, 0]
        if view_category != top_category_v:
            tips.append(f"**Category**: **{top_category_v}** is the top-trending category in India right now. Consider it if your content fits.")

        top_day_v = upload_day.sort_values(by=upload_day.columns[1], ascending=False).iloc[0, 0]
        top_hour_v = upload_hour.sort_values(by=upload_hour.columns[1], ascending=False).iloc[0, 0]
        tips.append(f"**Upload Timing**: Upload on **{top_day_v}** at **{top_hour_v}:00** for maximum early traction.")

        for tip in tips:
            st.markdown(f"- {tip}")

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