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
from textblob import TextBlob
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import google.generativeai as genai

# --- CONFIGURATION & CLIENT INITIALIZATION ---
load_dotenv()
st.set_page_config(layout="wide", page_title="AI Sales Call Assistant")

# --- CORE SETTINGS ---
CHUNK_DURATION = 10
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = int(0.05 * SAMPLE_RATE)
MAX_BLOCKS_PER_CHUNK = int(CHUNK_DURATION * (SAMPLE_RATE / BLOCKSIZE))

# --- API CLIENTS SETUP ---
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    groq_client = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel('gemini-1.5-flash-latest')
        st.session_state.active_llm = "Gemini Pro"
    except Exception as e:
        st.warning(f"Failed to initialize Gemini client: {e}.")
        st.session_state.active_llm = "N/A"
else:
    st.session_state.active_llm = "N/A"

# --- GOOGLE SHEETS & ANALYSIS FUNCTIONS ---
@st.cache_resource
def connect_google_sheet(sheet_name="Sales_Call_Analysis"):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)
        return client.open(sheet_name).sheet1
    except FileNotFoundError:
        st.warning("`credentials.json` not found. Google Sheets integration disabled.")
        return None
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return None

sheet = connect_google_sheet()

@st.cache_data(ttl=300)
def get_total_calls(_sheet):
    if not _sheet: return 0
    try:
        return len(_sheet.get_all_records())
    except Exception:
        return 0

# <<< FIXED HISTORY FETCHING FUNCTION >>>
@st.cache_data(ttl=300)
def fetch_latest_history_from_sheet(_sheet):
    """Fetches the last 3 logged calls from the sheet for display."""
    if not _sheet: return []
    try:
        # Use get_all_values() for resilience against header issues.
        all_data = _sheet.get_all_values()
        
        # Check for a header row and slice data accordingly
        header = ["Timestamp", "Transcript", "Sentiment", "Polarity", "LLM Summary"]
        if all_data and all_data[0] == header:
            records = all_data[1:]
        else:
            records = all_data

        if not records: return []
        
        history_rows = records[-3:]
        history_list = []
        for row in reversed(history_rows): # Show most recent first
            history_list.append({
                "timestamp": row[0] if len(row) > 0 else "N/A",
                "transcript": row[1] if len(row) > 1 else "N/A",
                "sentiment": row[2] if len(row) > 2 else "N/A",
                "summary": row[4] if len(row) > 4 else "N/A",
            })
        return history_list
    except Exception as e:
        st.warning(f"Could not fetch call history from Google Sheet: {e}")
        return []

def analyze_sentiment(text):
    if not text: return "Neutral", 0.0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1: sentiment = "Positive"
    elif polarity < -0.1: sentiment = "Negative"
    else: sentiment = "Neutral"
    return sentiment, round(polarity, 2)

def get_crm_suggestions(customer_name, product, needs, stage):
    if not customer_name or not product or not needs or not stage:
        return "Please fill in all CRM fields to get suggestions."
    prompt = f"""
    You are an expert sales coach and CRM strategist. Your task is to provide actionable advice for an upcoming sales call.
    **Customer Profile:**
    - **Name:** {customer_name}
    - **Product/Service of Interest:** {product}
    - **Customer's Stated Needs/Pain Points:** {needs}
    - **Current Deal Stage:** {stage}
    **Your Task:**
    Based on the profile above, generate a concise, actionable sales strategy. Structure your response with the following markdown headers:
    ### Key Talking Points:
    - (List 2-3 specific, compelling points to discuss that directly address the customer's needs with the product's benefits.)
    ### Potential Objections to Prepare For:
    - (List 2-3 likely objections the customer might have based on their needs and the deal stage, and suggest a brief counter-argument for each.)
    ### Suggested Next Steps:
    - (Recommend one clear, actionable next step to propose at the end of the call to move the deal forward.)
    """
    if gemini_client:
        try:
            response = gemini_client.generate_content(prompt)
            if response.parts:
                return "".join(part.text for part in response.parts).strip()
            return "CRM analysis from Gemini was empty."
        except Exception as e:
            return f"Gemini API Error: {e}"
    return "LLM client not configured for CRM."

def get_llm_summary(text, is_final_summary=False):
    if not text.strip(): return "No text to analyze."
    prompt_detail = "Provide a concise, overall summary of the entire conversation, covering main topics, user sentiment, and key outcomes." if is_final_summary else "Provide a concise, one-line summary of this part of the conversation."
    prompt = f"You are a real-time call analysis assistant. {prompt_detail}\nUtterance: \"{text}\"\nSummary:"
    if gemini_client:
        try:
            response = gemini_client.generate_content(prompt)
            if response.parts:
                return "".join(part.text for part in response.parts).strip()
            return "Analysis from Gemini was empty."
        except Exception as e:
            return f"Gemini API Error: {e}"
    return "No LLM client configured."

def log_to_google_sheet(full_transcript, final_sentiment, final_polarity, final_summary):
    if not sheet: return
    try:
        header = ["Timestamp", "Transcript", "Sentiment", "Polarity", "LLM Summary"]
        try:
            if sheet.row_values(1) != header:
                 sheet.insert_row(header, 1)
        except gspread.exceptions.APIError: # Sheet is empty
             sheet.insert_row(header, 1)

        row_data = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), full_transcript, final_sentiment, final_polarity, final_summary]
        sheet.append_row(row_data)
        st.toast("✅ Full call summary logged to Google Sheet!")
        get_total_calls.clear()
        fetch_latest_history_from_sheet.clear()
    except Exception as e:
        st.error(f"Error logging to Google Sheet: {e}")

# --- SESSION STATE INITIALIZATION ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
    st.session_state.audio_queue = queue.Queue()
    st.session_state.result_queue = queue.Queue()
    st.session_state.stop_event = threading.Event()
    st.session_state.live_chunks_display = "Waiting to start..."
    st.session_state.full_transcript_parts = []
    st.session_state.final_summary = "Summary will appear here after the call."
    st.session_state.final_sentiment = "Neutral"
    st.session_state.final_polarity = 0.0
    st.session_state.call_history_log = fetch_latest_history_from_sheet(sheet)
    st.session_state.crm_suggestions = ""

# --- AUDIO PROCESSING WORKER ---
def audio_processing_worker(audio_q, result_q, stop_ev):
    speech_buffer = []
    block_counter = 0
    while not stop_ev.is_set():
        try:
            audio_data = audio_q.get(timeout=0.1)
            speech_buffer.append(audio_data)
            block_counter += 1
            if block_counter >= MAX_BLOCKS_PER_CHUNK:
                current_chunk_audio = np.concatenate(speech_buffer)
                speech_buffer, block_counter = [], 0
                tmpfile_name = ""
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        tmpfile_name = tmpfile.name
                        sf.write(tmpfile, current_chunk_audio, SAMPLE_RATE)
                    with open(tmpfile_name, "rb") as audio_file:
                        transcript = groq_client.audio.transcriptions.create(
                            file=audio_file, model="whisper-large-v3", response_format="text")
                    if transcript and transcript.strip():
                        sentiment, _ = analyze_sentiment(transcript)
                        chunk_result = {
                            "type": "chunk", "transcript": transcript,
                            "sentiment": sentiment, "timestamp": datetime.now().strftime("%H:%M:%S")}
                        result_q.put(chunk_result)
                except Exception as e:
                    result_q.put({"type": "error", "message": f"Processing Error: {e}"})
                finally:
                    if tmpfile_name and os.path.exists(tmpfile_name):
                        os.remove(tmpfile_name)
        except queue.Empty:
            continue
        except Exception as e:
            result_q.put({"type": "error", "message": f"Audio Worker Error: {e}"})
            break

# --- STREAMLIT UI ---
with st.sidebar:
    st.title("AI Sales Assistant")
    st.info(f"LLM in use: **{st.session_state.get('active_llm', 'N/A')}**")
    
    st.subheader("Live Call Controls")
    start_button_disabled = st.session_state.is_running or not groq_client
    if st.button("Start Call", disabled=start_button_disabled, use_container_width=True):
        st.session_state.is_running = True
        st.session_state.stop_event.clear()
        st.session_state.live_chunks_display = "🟢 Listening... Speak into your microphone."
        st.session_state.full_transcript_parts = []
        the_audio_queue = st.session_state.audio_queue
        def audio_callback(indata, frames, time, status):
            the_audio_queue.put(indata.copy())
        st.session_state.stream = sd.InputStream(
            callback=audio_callback, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCKSIZE)
        st.session_state.stream.start()
        st.session_state.worker_thread = threading.Thread(
            target=audio_processing_worker,
            args=(st.session_state.audio_queue, st.session_state.result_queue, st.session_state.stop_event))
        st.session_state.worker_thread.start()
        st.rerun()
        
    if st.button("Stop Call", disabled=not st.session_state.is_running, use_container_width=True):
        st.session_state.is_running = False
        st.session_state.stop_event.set()
        if 'worker_thread' in st.session_state and st.session_state.worker_thread.is_alive():
            st.session_state.worker_thread.join(timeout=2)
        if 'stream' in st.session_state:
            st.session_state.stream.stop()
            st.session_state.stream.close()
        with st.spinner("Generating final call summary..."):
            full_transcript = " ".join(st.session_state.full_transcript_parts)
            if full_transcript.strip():
                st.session_state.final_summary = get_llm_summary(full_transcript, is_final_summary=True)
                sentiment, polarity = analyze_sentiment(full_transcript)
                st.session_state.final_sentiment = sentiment
                st.session_state.final_polarity = polarity
                log_to_google_sheet(full_transcript, sentiment, polarity, st.session_state.final_summary)
            else:
                st.session_state.final_summary = "No speech was detected to summarize."
        st.rerun()

    st.divider()

    with st.expander("CRM: AI Strategy Assistant", expanded=False):
        customer_name = st.text_input("Customer/Company Name")
        product = st.text_input("Product/Service of Interest")
        needs = st.text_area("Customer Needs / Pain Points")
        stage = st.selectbox("Deal Stage", ["Initial Contact", "Discovery", "Demo", "Proposal", "Negotiation", "Closing"])
        if st.button("Get AI Suggestions"):
            with st.spinner("Generating sales strategy..."):
                st.session_state.crm_suggestions = get_crm_suggestions(customer_name, product, needs, stage)
        if st.session_state.crm_suggestions:
            st.markdown(st.session_state.crm_suggestions)
            
    st.divider()
    st.subheader("Business Summary")
    st.metric("Total Calls Logged", get_total_calls(sheet))

# --- MAIN CONTENT AREA ---
st.title("Hello, Welcome!")
st.header("Live Call Dashboard")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader(f"Live Call Transcription (Updated every {CHUNK_DURATION}s)")
    st.text_area("Live Feed", value=st.session_state.live_chunks_display, height=300, key="live_feed")
with col2:
    st.subheader("Final Call Summary")
    sentiment_color = "normal"
    if st.session_state.final_sentiment == "Positive": sentiment_color = "inverse"
    elif st.session_state.final_sentiment == "Negative": sentiment_color = "off"
    st.metric("Overall Sentiment", st.session_state.final_sentiment,
              f"{st.session_state.final_polarity} Polarity", delta_color=sentiment_color)
    st.info(f"**LLM Summary:**\n\n{st.session_state.final_summary}")

# --- MAIN POLLING LOOP ---
if st.session_state.is_running:
    try:
        result = st.session_state.result_queue.get(block=False)
        if result["type"] == "chunk":
            st.session_state.full_transcript_parts.append(result["transcript"])
            sentiment_emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}.get(result["sentiment"])
            new_chunk_text = f"[{result['timestamp']}] {sentiment_emoji} {result['transcript']}"
            if st.session_state.live_chunks_display.startswith("🟢 Listening..."):
                 st.session_state.live_chunks_display = new_chunk_text
            else:
                 st.session_state.live_chunks_display = new_chunk_text + "\n\n" + st.session_state.live_chunks_display
        elif result["type"] == "error":
            st.error(result["message"])
        st.rerun()
    except queue.Empty:
        time.sleep(0.5)
        st.rerun()

# --- PERSISTENT HISTORY DISPLAY ---
st.divider()
st.subheader("Recent Call History")
if 'call_history_log' not in st.session_state or not st.session_state.call_history_log:
    st.caption("History of past calls will appear here after you log a call.")
else:
    for call in st.session_state.call_history_log:
        with st.expander(f"**{call['timestamp']}** | Overall Sentiment: **{call['sentiment']}**"):
            st.markdown(f"**LLM Summary:** {call['summary']}")
            st.caption("**Full Transcript:**")
            st.write(f"_{call['transcript']}_")