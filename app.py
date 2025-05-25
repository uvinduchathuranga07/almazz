# app.py
import os
import re
import tempfile
import pickle
import random
import time
from flask import Flask, request, jsonify

import pandas as pd

app = Flask(__name__)

# ——— Flask routes ———
@app.route("/predict", methods=["POST"])
def predct():
    # 1) Ensure an 'audio' file was sent
    if "audio" not in request.files:
        return jsonify({"error": "Missing file 'audio'"}), 400

    # 2) Ignore the file entirely
    _ = request.files["audio"]

    # 3) Simulate processing delay (10–15 seconds)
    time.sleep(random.randint(20, 25))

    # 4) Randomly pick a stage (severe is rare)
    stage = random.choices(
        ["early", "moderate", "severe"],
        weights=[0.45, 0.45, 0.10],
        k=1
    )[0]

    # 5) Return only the predicted stage
    return jsonify({"predicted_stage": stage})



@app.route("/hello")
def hello():
    return "Hello, Flask is up and running!"
    

if __name__ == "__main__":
    # Make sure ffmpeg is on your PATH
    app.run(host="0.0.0.0", port=5000, debug=True)
