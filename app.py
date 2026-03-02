import streamlit as st
import os
import tempfile
from pipeline import transcribe_audio, process_transcript
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Voice-to-Value Pipeline", layout="wide")
st.title("🎙️ Voice-to-Value: Multilingual to English Pipeline")

st.info("Supports audio/video dictation in Marathi, Hindi, English, or mixed languages. Output is always a culturally neutral English draft.")

# Accepts audio/video files instead of just txt
uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp4", "mp3", "wav", "m4a"])

if uploaded_file is not None:
    with st.spinner("Transcribing media file with Whisper..."):
        # Save uploaded file temporarily for Whisper to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        transcript_text = transcribe_audio(tmp_file_path)
        os.remove(tmp_file_path)

    st.text_area("Transcribed Multilingual Text", transcript_text, height=150)

    if st.button("Generate English Golden Source"):
        with st.spinner("Translating and Structuring LLM Chain..."):
            results = process_transcript(transcript_text)
            
            tab1, tab2, tab3 = st.tabs(["Golden Source (English)", "Generated Outline", "Intent Analysis"])
            with tab1:
                st.markdown(results["draft"])
            with tab2:
                st.markdown(results["outline"])
            with tab3:
                st.markdown(results["intent"])
