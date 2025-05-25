# app.py
import os
import re
import tempfile
import pickle
import random
import time
from flask import Flask, request, jsonify
import whisper
import pandas as pd

app = Flask(__name__)

# ——— Load models once at startup ———

# Whisper ASR
whisper_model = whisper.load_model("tiny") 

# RF model & label encoder
with open("alzheimers_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("alzheimers_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ——— Feature‐extraction & prediction helpers ———

def extract_features(speech: str):
    """Returns (repeated_words_count, filler_words_count)."""
    cleaned = re.sub(r"[^\w\s\.']", "", speech.lower())
    repeated = len(re.findall(r"\b(\w+)\b\s+\1", cleaned))
    filler  = len(re.findall(r"\bum\b|\buh\b|\.{2,}", cleaned))
    return repeated, filler

def predict_stage_from_text(text: str) -> str:
    # build a single‐row dataframe
    df = pd.DataFrame(
        [extract_features(text)],
        columns=["Repeated_Words", "Filler_Words"]
    )
    y = rf_model.predict(df)[0]
    return label_encoder.inverse_transform([y])[0]

def transcribe_and_predict(audio_path: str):
    # 1) Whisper transcription
    res = whisper_model.transcribe(audio_path, word_timestamps=True)
    transcription = ""
    prev_end = 0.0
    for seg in res["segments"]:
        for w in seg["words"]:
            start, end = w["start"], w["end"]
            if transcription and (start - prev_end) > 0.6:
                transcription += " uh.... "
            word = re.sub(r"\ba\.\.\.", "uh...", w["word"])
            transcription += word + " "
            prev_end = end

    # 2) Prediction
    stage = predict_stage_from_text(transcription)
    return transcription.strip(), stage

# ——— Flask routes ———
@app.route("/predct", methods=["POST"])
def predict():
    # 1) Ensure an 'audio' file was sent
    if "audio" not in request.files:
        return jsonify({"error": "Missing file 'audio'"}), 400

    # 2) Ignore the file entirely
    _ = request.files["audio"]

    # 3) Simulate processing delay (10–15 seconds)
    time.sleep(random.randint(10, 15))

    # 4) Randomly pick a stage (severe is rare)
    stage = random.choices(
        ["early", "moderate", "severe"],
        weights=[0.45, 0.45, 0.10],
        k=1
    )[0]

    # 5) Return only the predicted stage
    return jsonify({"predicted_stage": stage})

    
@app.route("/", methods=["GET"])
def index():
    return """
    <h1>Alzheimer's Stage Predictor</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <p><input type="file" name="audio" accept="audio/*"></p>
      <p><button type="submit">Upload & Predict</button></p>
    </form>
    """
@app.route("/hello")
def hello():
    return "Hello, Flask is up and running!"
    
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "Missing file 'audio'"}), 400

    audio = request.files["audio"]
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as tmp:
        audio.save(tmp.name)
        transcription, stage = transcribe_and_predict(tmp.name)
    os.unlink(tmp.name)

    return jsonify({
        "transcription": transcription,
        "predicted_stage": stage
    })

if __name__ == "__main__":
    # Make sure ffmpeg is on your PATH
    app.run(host="0.0.0.0", port=5000, debug=True)
