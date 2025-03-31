# import streamlit as st

# import pandas as pd
# import numpy as np
# import altair as alt

# import joblib

# pipe_lr = joblib.load(open("C:/Streamlit/App/text_emotion.pkl", "rb"))

# emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
#                        "sadness": "😔", "shame": "😳", "surprise": "😮"}


# def predict_emotions(docx):
#     results = pipe_lr.predict([docx])
#     return results[0]


# def get_prediction_proba(docx):
#     results = pipe_lr.predict_proba([docx])
#     return results


# def main():
#     st.title("Text Emotion Detection")
#     st.subheader("Detect Emotions In Text")

#     with st.form(key='my_form'):
#         raw_text = st.text_area("Type Here")
#         submit_text = st.form_submit_button(label='Submit')

#     if submit_text:
#         col1, col2 = st.columns(2)

#         prediction = predict_emotions(raw_text)
#         probability = get_prediction_proba(raw_text)

#         with col1:
#             st.success("Original Text")
#             st.write(raw_text)

#             st.success("Prediction")
#             emoji_icon = emotions_emoji_dict[prediction]
#             st.write("{}:{}".format(prediction, emoji_icon))
#             st.write("Confidence:{}".format(np.max(probability)))

#         with col2:
#             st.success("Prediction Probability")
#             #st.write(probability)
#             proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
#             #st.write(proba_df.T)
#             proba_df_clean = proba_df.T.reset_index()
#             proba_df_clean.columns = ["emotions", "probability"]

#             fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
#             st.altair_chart(fig, use_container_width=True)
            
        
#             fig = alt.Chart(proba_df_clean).mark_arc().encode(theta=alt.Theta(field="probability", type="quantitative"),color=alt.Color(field="emotions", type="nominal"),tooltip=['emotions', 'probability'])

#             st.altair_chart(fig, use_container_width=True)

#             fig = alt.Chart(proba_df_clean).mark_rect().encode( x=alt.X("texts:O", title="Text Samples"),y=alt.Y("emotions:O", title="Emotions"),color=alt.Color("probability:Q", scale=alt.Scale(scheme="viridis")),tooltip=["texts", "emotions", "probability"]).properties( width=600,height=400)

#             st.altair_chart(fig, use_container_width=True)






# if __name__ == '__main__':
#     main()


















# import streamlit as st
# import pandas as pd
# import numpy as np
# import altair as alt
# import joblib
# from streamlit_option_menu import option_menu

# # Load model
# @st.cache_resource
# def load_model():
#     try:
#         pipe_lr = joblib.load(open("C:/Streamlit/App/text_emotion.pkl", "rb"))
#         return pipe_lr
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# pipe_lr = load_model()

# # Emotion configuration
# emotions_emoji_dict = {
#     "anger": "😠", 
#     "disgust": "🤮", 
#     "fear": "😨😱", 
#     "happy": "🤗", 
#     "joy": "😂", 
#     "neutral": "😐", 
#     "sad": "😔",
#     "sadness": "😔", 
#     "shame": "😳", 
#     "surprise": "😮"
# }

# # Prediction functions
# def predict_emotions(docx):
#     results = pipe_lr.predict([docx])
#     return results[0]

# def get_prediction_proba(docx):
#     results = pipe_lr.predict_proba([docx])
#     return results

# # Main app
# def main():
#     # Custom CSS for styling
#     st.markdown("""
#     <style>
#         .main {
#             background-color: #f8f9fa;
#         }
#         .stTextArea textarea {
#             min-height: 200px;
#         }
#         .result-box {
#             border-radius: 10px;
#             padding: 20px;
#             margin-bottom: 20px;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         }
#         .positive { background-color: #d4edda; }
#         .neutral { background-color: #e2e3e5; }
#         .negative { background-color: #f8d7da; }
#         .header-text { font-size: 2.5rem !important; }
#         .emoji-large { font-size: 3rem; text-align: center; }
#         .confidence-meter {
#             height: 20px;
#             background: linear-gradient(90deg, #ff7676, #f54ea2, #17ead9, #6078ea);
#             border-radius: 10px;
#         }
#     </style>
#     """, unsafe_allow_html=True)

#     # Navigation
#     with st.sidebar:
#         selected = option_menu(
#             menu_title="Menu",
#             options=["Home", "About", "Examples"],
#             icons=["house", "info-circle", "collection"],
#             default_index=0,
#         )

#     if selected == "About":
#         st.title("About This App")
#         st.markdown("""
#         This application uses machine learning to detect emotions in text. 
#         It can identify the following emotions:
#         """)
        
#         cols = st.columns(3)
#         for i, (emotion, emoji) in enumerate(emotions_emoji_dict.items()):
#             with cols[i%3]:
#                 st.markdown(f"**{emotion.capitalize()}** {emoji}")
        
#         st.markdown("""
#         ### How it works
#         1. Type or paste your text in the input box
#         2. Click the 'Analyze' button
#         3. View the emotion prediction and probability distribution
#         """)
#         return
#     elif selected == "Examples":
#         st.title("Example Texts")
#         examples = {
#             "Joy": "I'm so excited about our vacation next week! Everything is going perfectly! 😊",
#             "Anger": "I can't believe they did this again! This is completely unacceptable!",
#             "Sadness": "I've been feeling really down lately. Nothing seems to make me happy anymore.",
#             "Surprise": "Oh my goodness! I had no idea you were planning this party for me!"
#         }
        
#         for emotion, text in examples.items():
#             with st.expander(f"{emotion} Example"):
#                 st.write(text)
#                 if st.button("Try this example", key=f"btn_{emotion}"):
#                     st.session_state.example_text = text
#         return

#     # Main page
#     st.markdown('<p class="header-text">Text Emotion Detector</p>', unsafe_allow_html=True)
#     st.markdown("Discover the emotional tone behind any text")
#     st.divider()

#     # Text input
#     with st.form(key='emotion_form'):
#         raw_text = st.text_area(
#             "Enter your text here:", 
#             value=st.session_state.get('example_text', ''),
#             placeholder="Type or paste your text to analyze its emotional content..."
#         )
        
#         col1, col2, col3 = st.columns([1,1,3])
#         with col1:
#             submit_text = st.form_submit_button("Analyze", type="primary")
#         with col2:
#             if st.form_submit_button("Clear"):
#                 raw_text = ""
#                 st.session_state.example_text = ""

#     if submit_text and raw_text.strip():
#         # Make predictions
#         prediction = predict_emotions(raw_text)
#         probability = get_prediction_proba(raw_text)
        
#         # Determine emotion category for styling
#         emotion_category = "neutral"
#         if prediction in ["happy", "joy", "surprise"]:
#             emotion_category = "positive"
#         elif prediction in ["anger", "disgust", "fear", "sad", "sadness", "shame"]:
#             emotion_category = "negative"
        
#         # Display results
#         st.divider()
#         st.subheader("Analysis Results")
        
#         # Emotion prediction box
#         with st.container():
#             st.markdown(f'<div class="result-box {emotion_category}">', unsafe_allow_html=True)
            
#             cols = st.columns([1,3])
#             with cols[0]:
#                 st.markdown(f'<div class="emoji-large">{emotions_emoji_dict[prediction]}</div>', unsafe_allow_html=True)
            
#             with cols[1]:
#                 st.markdown(f"**Predicted Emotion:** {prediction.capitalize()}")
#                 confidence = np.max(probability)
#                 st.markdown(f"**Confidence:** {confidence:.1%}")
#                 st.markdown(f'<div class="confidence-meter" style="width: {confidence*100}%"></div>', unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         # Probability visualization
#         with st.expander("Detailed Emotion Probabilities"):
#             proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
#             proba_df_clean = proba_df.T.reset_index()
#             proba_df_clean.columns = ["emotions", "probability"]
#             proba_df_clean = proba_df_clean.sort_values("probability", ascending=False)
            
#             # Add emojis to the dataframe
#             proba_df_clean['emoji'] = proba_df_clean['emotions'].map(emotions_emoji_dict)
#             proba_df_clean['display'] = proba_df_clean['emoji'] + " " + proba_df_clean['emotions'].str.capitalize()
            
#             # Create chart
#             chart = alt.Chart(proba_df_clean).mark_bar().encode(
#                 x=alt.X('probability:Q', axis=alt.Axis(format='%'), title='Probability'),
#                 y=alt.Y('display:N', sort='-x', title='Emotion'),
#                 color=alt.Color('display:N', legend=None),
#                 tooltip=['emotions', 'probability']
#             ).properties(height=400)
            
#             st.altair_chart(chart, use_container_width=True)
            
#             # Show raw data
#             if st.checkbox("Show raw probability data"):
#                 st.dataframe(proba_df_clean[['emotions', 'probability']].set_index('emotions'))

# if __name__ == '__main__':
#     main()









import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from datetime import datetime

# --- Configuration ---
@st.cache_resource
def load_model():
    try:
        return joblib.load(open("C:/Streamlit/App/text_emotion.pkl", "rb"))
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

pipe_lr = load_model()

emotion_config = {
    "anger": {"emoji": "😠", "color": "#ff6b6b", "type": "negative"},
    "disgust": {"emoji": "🤮", "color": "#94d82d", "type": "negative"},
    "fear": {"emoji": "😨", "color": "#ff922b", "type": "negative"},
    "happy": {"emoji": "🤗", "color": "#51cf66", "type": "positive"},
    "joy": {"emoji": "😂", "color": "#fcc419", "type": "positive"},
    "neutral": {"emoji": "😐", "color": "#adb5bd", "type": "neutral"},
    "sadness": {"emoji": "😔", "color": "#74c0fc", "type": "negative"},
    "shame": {"emoji": "😳", "color": "#b197fc", "type": "negative"},
    "surprise": {"emoji": "😮", "color": "#ff8787", "type": "positive"}
}

# --- UI Functions ---
def render_sidebar():
    with st.sidebar:
        st.title("🎭 Text Emotion Lab")
        st.markdown("---")
        nav_option = st.radio("Explore", ["Analyzer", "Emotion Guide", "History"])
        
        if nav_option == "Emotion Guide":
            st.markdown("### Emotion Spectrum")
            for emotion, config in emotion_config.items():
                st.markdown(
                    f"<div style='background-color:{config['color']}20; padding:10px; border-radius:5px; margin:5px 0;'>"
                    f"{config['emoji']} <b>{emotion.capitalize()}</b>"
                    "</div>", 
                    unsafe_allow_html=True
                )
        return nav_option

def emotion_gauge(confidence):
    gauge = alt.Chart(pd.DataFrame({"value": [confidence]})).mark_arc(
        innerRadius=50,
        outerRadius=70
    ).encode(
        theta="value:Q",
        color=alt.value("#20c997")
    ).properties(width=200, height=200)
    
    text = alt.Chart(pd.DataFrame({"value": [""]})).mark_text(
        size=20,
        fontWeight="bold"
    ).encode(text="value:N")
    
    return (gauge + text).configure_view(strokeWidth=0)

# --- Core Functions ---
def analyze_text(text):
    if not text.strip():
        return None
    
    prediction = pipe_lr.predict([text])[0]
    probabilities = pipe_lr.predict_proba([text])[0]
    
    # Store analysis history
    if "history" not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        "text": text[:100] + ("..." if len(text) > 100 else ""),
        "emotion": prediction,
        "confidence": round(np.max(probabilities), 4),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    
    return prediction, probabilities

# --- Main App ---
def main():
    # Custom CSS Injection
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {background: linear-gradient(180deg, #4a4e69, #22223b);}
        [data-testid="stSidebar"] .st-ae {color: white !important;}
        [data-testid="stSidebar"] .st-cq {color: white;}
        .emotion-card {border-left: 5px solid; border-radius: 5px; padding: 15px; margin: 10px 0;}
        .positive-card {border-color: #51cf66; background-color: #ebfbee;}
        .negative-card {border-color: #ff6b6b; background-color: #fff5f5;}
        .neutral-card {border-color: #adb5bd; background-color: #f8f9fa;}
    </style>
    """, unsafe_allow_html=True)

    # Navigation
    current_page = render_sidebar()

    if current_page == "Emotion Guide":
        st.title("Emotion Reference Guide")
        st.markdown("Understand the emotions our model detects:")
        
        cols = st.columns(3)
        for i, (emotion, config) in enumerate(emotion_config.items()):
            with cols[i%3]:
                with st.container():
                    st.markdown(
                        f"<div class='emotion-card {config['type']}-card'>"
                        f"<h3>{config['emoji']} {emotion.capitalize()}</h3>"
                        f"<div style='color:{config['color']}; font-weight:bold;'>"
                        f"{config['type'].upper()}</div></div>",
                        unsafe_allow_html=True
                    )
        return
    
    elif current_page == "History":
        st.title("Analysis History")
        if "history" not in st.session_state or not st.session_state.history:
            st.info("No analysis history yet. Try analyzing some text!")
            return
            
        for i, entry in enumerate(reversed(st.session_state.history)):
            config = emotion_config[entry["emotion"]]
            st.markdown(
                f"<div class='emotion-card {config['type']}-card'>"
                f"<div style='display:flex; justify-content:space-between;'>"
                f"<b>{config['emoji']} {entry['emotion'].capitalize()}</b>"
                f"<small>{entry['timestamp']}</small></div>"
                f"<p><i>\"{entry['text']}\"</i></p>"
                f"<div style='background:{config['color']}20; height:5px; width:{entry['confidence']*100}%;'></div>"
                f"<div style='text-align:right;'>{entry['confidence']*100:.1f}% confidence</div>"
                "</div>",
                unsafe_allow_html=True
            )
        return

    # Main Analysis Page
    st.title("Text Emotion Analyzer")
    st.caption("Discover the emotional undertones in your text with AI-powered analysis")
    
    with st.form("analysis_form"):
        text_input = st.text_area(
            "Enter your text:", 
            placeholder="Type or paste any text here...",
            height=150
        )
        
        cols = st.columns([1,1,3])
        with cols[0]:
            submitted = st.form_submit_button("Analyze", type="primary")
        with cols[1]:
            if st.form_submit_button("Clear"):
                text_input = ""

    if submitted and text_input.strip():
        with st.spinner("Decoding emotions..."):
            result = analyze_text(text_input)
        
        if result:
            emotion, probabilities = result
            config = emotion_config[emotion]
            
            # Main Result Card
            st.markdown(
                f"<div class='emotion-card {config['type']}-card'>"
                f"<div style='font-size:2.5rem; text-align:center;'>{config['emoji']}</div>"
                f"<h2 style='text-align:center;'>{emotion.capitalize()}</h2>"
                f"<div style='text-align:center; margin-bottom:20px;'>"
                f"<div style='background:{config['color']}; height:8px; "
                f"width:{np.max(probabilities)*100}%; margin:0 auto;'></div>"
                f"{np.max(probabilities)*100:.1f}% confidence</div>"
                "</div>",
                unsafe_allow_html=True
            )
            
            # Probability Visualization
            with st.expander("Detailed Emotion Breakdown"):
                prob_df = pd.DataFrame({
                    "Emotion": pipe_lr.classes_,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=False)
                
                prob_df["Color"] = prob_df["Emotion"].apply(lambda x: emotion_config[x]["color"])
                prob_df["Emoji"] = prob_df["Emotion"].apply(lambda x: emotion_config[x]["emoji"])
                
                chart = alt.Chart(prob_df).mark_bar().encode(
                    x=alt.X("Probability:Q", axis=alt.Axis(format=".0%")),
                    y=alt.Y("Emotion:N", sort="-x", title=None),
                    color=alt.Color("Color:N", scale=None, legend=None),
                    tooltip=["Emotion", "Probability"]
                ).properties(height=400)
                
                st.altair_chart(chart, use_container_width=True)
                
                if st.checkbox("Show numerical data"):
                    st.dataframe(prob_df[["Emoji", "Emotion", "Probability"]].style.format({
                        "Probability": "{:.1%}"
                    }))

if __name__ == "__main__":
    main()