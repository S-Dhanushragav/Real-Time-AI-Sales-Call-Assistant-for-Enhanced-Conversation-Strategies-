import streamlit as st
import os
import queue
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from textblob import TextBlob
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time

# --- CONFIGURATION & CLIENT INITIALIZATION ---
load_dotenv()

# Configure Streamlit page
st.set_page_config(layout="wide", page_title="AI Sales Call Assistant")

# Initialize API clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = int(0.05 * SAMPLE_RATE)
SILENCE_THRESHOLD = 0.03
SILENCE_DURATION = 0.8

# --- GOOGLE SHEETS & ANALYSIS FUNCTIONS (Copied from your script) ---
@st.cache_resource
def connect_google_sheet(sheet_name="Sales_Call_Analysis"):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)
        return client.open(sheet_name).sheet1
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return None

sheet = connect_google_sheet()

def analyze_sentiment(text):
    if not text: return "Neutral", 0.0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1: sentiment = "Positive"
    elif polarity < -0.1: sentiment = "Negative"
    else: sentiment = "Neutral"
    return sentiment, round(polarity, 2)

def analyze_text_with_llm(text):
    if not text.strip(): return "No analysis."
    prompt = f"""
    You are a real-time call analysis assistant. Analyze the following user utterance from a sales/support call.
    Provide a concise, one-line summary covering the user's intent and main topic.
    If it's a question, rephrase the question clearly.
    Utterance: "{text}"
    Analysis:
    """
    try:
        response = openrouter_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in LLM analysis: {e}"

def log_to_google_sheet(transcript, sentiment, polarity, llm_analysis):
    if not sheet: return
    try:
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            transcript,
            sentiment.split(" ")[0], # Log only the word (e.g., "Positive")
            polarity,
            llm_analysis
        ]
        sheet.append_row(row)
    except Exception as e:
        st.error(f"Error logging to Google Sheet: {e}")

# --- SESSION STATE INITIALIZATION ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
    st.session_state.full_transcript = ""
    st.session_state.audio_queue = queue.Queue()
    st.session_state.result_queue = queue.Queue()
    st.session_state.stop_event = threading.Event()

# --- AUDIO PROCESSING WORKER (Runs in a background thread) ---
def audio_processing_worker():
    speech_buffer = []
    silence_counter = 0
    max_silence_frames = int(SILENCE_DURATION * (SAMPLE_RATE / BLOCKSIZE))
    is_speaking = False

    while not st.session_state.stop_event.is_set():
        try:
            audio_data = st.session_state.audio_queue.get(timeout=0.1)
            is_silent = np.max(np.abs(audio_data)) < SILENCE_THRESHOLD

            if is_speaking:
                if is_silent:
                    silence_counter += 1
                    if silence_counter > max_silence_frames:
                        full_utterance = np.concatenate(speech_buffer)
                        speech_buffer, is_speaking, silence_counter = [], False, 0
                        # Process the audio chunk
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                            tmpfile_name = tmpfile.name
                            sf.write(tmpfile, full_utterance, SAMPLE_RATE)
                        
                        with open(tmpfile_name, "rb") as audio_file:
                            transcript = groq_client.audio.transcriptions.create(
                                file=audio_file, model="whisper-large-v3", response_format="text"
                            )
                        os.remove(tmpfile_name)
                        
                        st.session_state.result_queue.put(transcript)
                else:
                    silence_counter = 0
                    speech_buffer.append(audio_data)
            elif not is_silent:
                is_speaking = True
                speech_buffer.append(audio_data)
        except queue.Empty:
            continue

# --- STREAMLIT UI LAYOUT ---
st.title("Real Time AI Sales Call Assistant")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Start Assistant", type="primary", disabled=st.session_state.is_running, use_container_width=True):
        st.session_state.is_running = True
        st.session_state.stop_event.clear()
        st.session_state.full_transcript = "" # Clear previous transcript
        
        # Start the audio stream and worker thread
        st.session_state.stream = sd.InputStream(
            callback=lambda indata, frames, time, status: st.session_state.audio_queue.put(indata.copy()),
            samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCKSIZE
        )
        st.session_state.stream.start()
        
        st.session_state.worker_thread = threading.Thread(target=audio_processing_worker)
        st.session_state.worker_thread.start()
        st.rerun()

with col2:
    if st.button("Stop Assistant", disabled=not st.session_state.is_running, use_container_width=True):
        st.session_state.is_running = False
        st.session_state.stop_event.set()
        
        if 'worker_thread' in st.session_state:
            st.session_state.worker_thread.join() # Wait for thread to finish
        if 'stream' in st.session_state:
            st.session_state.stream.stop()
            st.session_state.stream.close()
        st.rerun()

st.divider()

# --- REAL-TIME DISPLAY AREA ---
if st.session_state.is_running:
    st.info("Assistant is running. Speak into your microphone...")
else:
    st.warning("Assistant is stopped.")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Full Conversation Transcript")
    transcript_box = st.text_area("Transcript", value=st.session_state.full_transcript, height=400, key="transcript_box")
    
    st.subheader("Real-time Tracking")
    status_placeholder = st.empty()

with col_right:
    st.subheader("Latest Analysis")
    llm_summary_placeholder = st.empty()


# --- LOGIC TO PROCESS RESULTS AND UPDATE UI ---
while not st.session_state.result_queue.empty():
    transcript = st.session_state.result_queue.get()
    
    if transcript:
        # Perform analyses
        sentiment, polarity = analyze_sentiment(transcript)
        llm_analysis = analyze_text_with_llm(transcript)
        
        # Update UI placeholders
        status_placeholder.metric("Latest Sentiment", sentiment, f"{polarity} Polarity")
        llm_summary_placeholder.info(f"**LLM Summary:**\n\n{llm_analysis}")
        
        # Append to full transcript
        st.session_state.full_transcript += f"Speaker: {transcript}\n\n"
        
        # Log to Google Sheets
        log_to_google_sheet(transcript, sentiment, polarity, llm_analysis)

        st.rerun() # Force an immediate UI update

# Add a small sleep to prevent high CPU usage in the main loop
if st.session_state.is_running:
    time.sleep(0.1)
    st.rerun()