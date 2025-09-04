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

# CONFIGURATION & CLIENT INITIALIZATION 
# Load environment variables from .env file
load_dotenv()

# Initialize Groq Client for Transcription
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize OpenAI Client to point to OpenRouter for LLM Analysis
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Audio Stream Settings
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = int(0.05 * SAMPLE_RATE)  # 50ms chunks
SILENCE_THRESHOLD = 0.03
SILENCE_DURATION = 0.8

# Global queue to share audio data between threads
audio_queue = queue.Queue()


# GOOGLE SHEETS SETUP 
def connect_google_sheet(sheet_name):
    """Connects to Google Sheets and returns the sheet object."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)
        return client.open(sheet_name).sheet1
    except FileNotFoundError:
        print("Error: 'credentials.json' not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"An error occurred connecting to Google Sheets: {e}")
        return None

# Connect to the sheet globally
sheet = connect_google_sheet("Sales_Call_Analysis")
if sheet:
    print("Successfully connected to Google Sheet 'Sales_Call_Analysis'.")


# ANALYSIS & LOGGING FUNCTIONS

def analyze_sentiment(text):
    """Analyzes sentiment of a given text using TextBlob."""
    if not text:
        return "Neutral", 0.0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, round(polarity, 2)


def analyze_text_with_llm(text):
    """Performs deeper analysis on the text using an LLM via OpenRouter."""
    if not text.strip():
        return "No analysis."
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
    """Appends a new row with the analysis results to the Google Sheet."""
    if not sheet:
        print("Cannot log to Google Sheet, connection not established.")
        return
    try:
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            transcript,
            sentiment,
            polarity,
            llm_analysis
        ]
        sheet.append_row(row)
        print("Logged to Google Sheet")
    except Exception as e:
        print(f"Error logging to Google Sheet: {e}")


# AUDIO PROCESSING 

def audio_callback(indata, frames, time, status):
    """This function is called by the sounddevice stream for each audio block."""
    if status:
        print(status)
    audio_queue.put(indata.copy())


def process_audio_chunk(audio_chunk):
    """Transcribes, analyzes, and logs a single chunk of spoken audio."""
    # Using delete=False is safer on Windows to avoid access errors
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile_name = tmpfile.name
        # Correctly write to the file object itself, not its name
        sf.write(tmpfile, audio_chunk, SAMPLE_RATE)

    try:
        # Transcribe with Groq (Whisper) using the saved filename
        with open(tmpfile_name, "rb") as audio_file:
            transcript = groq_client.audio.transcriptions.create(
                file=audio_file, model="whisper-large-v3", response_format="text"
            )
            print(f"\nUser said: {transcript}")

            # Perform analyses
            sentiment, polarity = analyze_sentiment(transcript)
            llm_analysis = analyze_text_with_llm(transcript)

            # Display Results
            print("Analysis Part")
            print(f"Sentiment: {sentiment} (Polarity: {polarity})")
            print(f"LLM Summary: {llm_analysis}")

            # Log results to Google Sheets
            log_to_google_sheet(transcript, sentiment, polarity, llm_analysis)
            print(".\n.\n")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        # Manually delete the file after processing is complete
        os.remove(tmpfile_name)


def main():
    """Main function to run the real-time assistant."""
    print("Real-Time Call Assistant is running... Press Ctrl+C to stop.")
    print("Listening for your voice...")

    stream = sd.InputStream(
        callback=audio_callback, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCKSIZE
    )
    stream.start()

    is_speaking = False
    speech_buffer = []
    silence_counter = 0
    max_silence_frames = int(SILENCE_DURATION * (SAMPLE_RATE / BLOCKSIZE))

    while True:
        try:
            audio_data = audio_queue.get()
            is_silent = np.max(np.abs(audio_data)) < SILENCE_THRESHOLD

            if is_speaking:
                if is_silent:
                    silence_counter += 1
                    if silence_counter > max_silence_frames:
                        full_utterance = np.concatenate(speech_buffer)
                        threading.Thread(target=process_audio_chunk, args=(full_utterance,)).start()
                        speech_buffer, is_speaking, silence_counter = [], False, 0
                        print("Listening for your voice...")
                else:
                    silence_counter = 0
                    speech_buffer.append(audio_data)
            elif not is_silent:
                print("Speech detected, recording...")
                is_speaking = True
                speech_buffer.append(audio_data)

        except KeyboardInterrupt:
            print("\nStopping assistant.")
            break
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")

    stream.stop()
    stream.close()
    print("Execution completed")


if __name__ == "__main__":
    main()