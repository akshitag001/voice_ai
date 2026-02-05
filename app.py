from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import joblib
import numpy as np
import librosa
import os
import warnings

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ==========================
# CONFIG
# ==========================
API_KEY = "sk_test_123456789"   # change later if needed
MODEL_PATH = "voice_classifier_FINAL.pkl"
HUMAN_THRESHOLD = 0.8
SAMPLE_RATE = 16000
DURATION = 3.0

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ==========================
# LOAD MODEL (ONCE)
# ==========================
model = joblib.load(MODEL_PATH)

app = FastAPI(title="AI Voice Detection API")

# ==========================
# REQUEST SCHEMA
# ==========================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ==========================
# FEATURE EXTRACTION (v3)
# ==========================
def extract_features(file_path):
    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    mfcc_delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(mfcc_delta, axis=1)

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(mfcc_delta2, axis=1)

    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    harmonic, percussive = librosa.effects.hpss(y)
    hnr_proxy = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6)

    f0 = librosa.yin(y, fmin=50, fmax=300)
    pitch_std = np.nanstd(f0)
    pitch_jitter = np.nanmean(np.abs(np.diff(f0)))

    stft = librosa.stft(y)
    phase = np.angle(stft)
    phase_var = np.var(np.diff(phase, axis=1))

    features = np.hstack([
        mfcc_mean, mfcc_var,
        delta_mean, delta2_mean,
        spec_centroid, spec_rolloff, spec_flatness, zcr,
        hnr_proxy, pitch_std, pitch_jitter, phase_var
    ])

    return features.reshape(1, -1)

# ==========================
# API ENDPOINT
# ==========================
@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 🧾 INPUT VALIDATION
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Invalid audio format")

    # 🔓 BASE64 DECODE
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # 💾 TEMP MP3 FILE
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    try:
        features = extract_features(temp_path)
        p_human, p_ai = model.predict_proba(features)[0]

        # 🧠 FINAL DECISION RULE
        if p_human >= HUMAN_THRESHOLD:
            classification = "HUMAN"
            confidence = float(p_human)
            explanation = "Natural spectral and temporal variations detected"
        else:
            classification = "AI_GENERATED"
            confidence = float(p_ai)
            explanation = "Synthetic speech patterns detected"

    finally:
        os.remove(temp_path)

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(confidence, 3),
        "explanation": explanation
    }
