
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from streamlit_option_menu import option_menu

# Load model
@st.cache_resource
def load_model():
    try:
        pipe_lr = joblib.load(open("C:/Streamlit/App/text_emotion.pkl", "rb"))
        return pipe_lr
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipe_lr = load_model()

# Emotion configuration
emotions_emoji_dict = {
    "anger": "😠", 
    "disgust": "🤮", 
    "fear": "😨😱", 
    "happy": "🤗", 
    "joy": "😂", 
    "neutral": "😐", 
    "sad": "😔",
    "sadness": "😔", 
    "shame": "😳", 
    "surprise": "😮"
}

# Prediction functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Main app
def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            min-height: 200px;
        }
        .result-box {
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .positive { background-color: #d4edda; }
        .neutral { background-color: #e2e3e5; }
        .negative { background-color: #f8d7da; }
        .header-text { font-size: 2.5rem !important; }
        .emoji-large { font-size: 3rem; text-align: center; }
        .confidence-meter {
            height: 20px;
            background: linear-gradient(90deg, #ff7676, #f54ea2, #17ead9, #6078ea);
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Home", "About", "Examples"],
            icons=["house", "info-circle", "collection"],
            default_index=0,
        )

    if selected == "About":
        st.title("About This App")
        st.markdown("""
        This application uses machine learning to detect emotions in text. 
        It can identify the following emotions:
        """)
        
        cols = st.columns(3)
        for i, (emotion, emoji) in enumerate(emotions_emoji_dict.items()):
            with cols[i%3]:
                st.markdown(f"**{emotion.capitalize()}** {emoji}")
        
        st.markdown("""
        ### How it works
        1. Type or paste your text in the input box
        2. Click the 'Analyze' button
        3. View the emotion prediction and probability distribution
        """)
        return
    elif selected == "Examples":
        st.title("Example Texts")
        examples = {
            "Joy": "I'm so excited about our vacation next week! Everything is going perfectly! 😊",
            "Anger": "I can't believe they did this again! This is completely unacceptable!",
            "Sadness": "I've been feeling really down lately. Nothing seems to make me happy anymore.",
            "Surprise": "Oh my goodness! I had no idea you were planning this party for me!"
        }
        
        for emotion, text in examples.items():
            with st.expander(f"{emotion} Example"):
                st.write(text)
                if st.button("Try this example", key=f"btn_{emotion}"):
                    st.session_state.example_text = text
        return

    # Main page
    st.markdown('<p class="header-text">Text Emotion Detector</p>', unsafe_allow_html=True)
    st.markdown("Discover the emotional tone behind any text")
    st.divider()

    # Text input
    with st.form(key='emotion_form'):
        raw_text = st.text_area(
            "Enter your text here:", 
            value=st.session_state.get('example_text', ''),
            placeholder="Type or paste your text to analyze its emotional content..."
        )
        
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            submit_text = st.form_submit_button("Analyze", type="primary")
        with col2:
            if st.form_submit_button("Clear"):
                raw_text = ""
                st.session_state.example_text = ""

    if submit_text and raw_text.strip():
        # Make predictions
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)
        
        # Determine emotion category for styling
        emotion_category = "neutral"
        if prediction in ["happy", "joy", "surprise"]:
            emotion_category = "positive"
        elif prediction in ["anger", "disgust", "fear", "sad", "sadness", "shame"]:
            emotion_category = "negative"
        
        # Display results
        st.divider()
        st.subheader("Analysis Results")
        
        # Emotion prediction box
        with st.container():
            st.markdown(f'<div class="result-box {emotion_category}">', unsafe_allow_html=True)
            
            cols = st.columns([1,3])
            with cols[0]:
                st.markdown(f'<div class="emoji-large">{emotions_emoji_dict[prediction]}</div>', unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"**Predicted Emotion:** {prediction.capitalize()}")
                confidence = np.max(probability)
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.markdown(f'<div class="confidence-meter" style="width: {confidence*100}%"></div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Probability visualization
        with st.expander("Detailed Emotion Probabilities"):
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            proba_df_clean = proba_df_clean.sort_values("probability", ascending=False)
            
            # Add emojis to the dataframe
            proba_df_clean['emoji'] = proba_df_clean['emotions'].map(emotions_emoji_dict)
            proba_df_clean['display'] = proba_df_clean['emoji'] + " " + proba_df_clean['emotions'].str.capitalize()
            
            # Create chart
            chart = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('probability:Q', axis=alt.Axis(format='%'), title='Probability'),
                y=alt.Y('display:N', sort='-x', title='Emotion'),
                color=alt.Color('display:N', legend=None),
                tooltip=['emotions', 'probability']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
            
            # Show raw data
            if st.checkbox("Show raw probability data"):
                st.dataframe(proba_df_clean[['emotions', 'probability']].set_index('emotions'))

if __name__ == '__main__':
    main()







