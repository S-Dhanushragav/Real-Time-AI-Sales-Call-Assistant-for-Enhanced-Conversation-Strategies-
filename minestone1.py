import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
from textblob import TextBlob
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY", "the key"))

# Google Sheets Setup
def connect_google_sheet(sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name).sheet1

sheet = connect_google_sheet("Sales_Call_Analysis")

# Record audio
def record_audio(duration=10, fs=16000):
    print(f"Recording {duration}s...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

# Transcribe using Groq
def transcribe_with_groq(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    return transcription

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)

    if polarity > 0.2:
        sentiment = "Positive"
    elif polarity < -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return polarity, subjectivity, sentiment

# Process call and save to Google Sheets
def process_call():
    audio = record_audio(10)
    sf.write("temp.wav", audio, 16000)
    transcript = transcribe_with_groq("temp.wav")
    polarity, subjectivity, sentiment = analyze_sentiment(transcript)

    print("\n--- Call Analysis ---")
    print("Transcript:", transcript)
    print(f"Sentiment: {sentiment} | Polarity: {polarity} | Subjectivity: {subjectivity}")

    # Log to Google Sheets
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        transcript,
        sentiment,
        polarity,
        subjectivity
    ])
    print("Data logged to Google Sheet successfully!")

if __name__ == "__main__":
    process_call()
