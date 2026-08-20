import os
import librosa
import numpy as np
import pandas as pd
import joblib


# ==========================================
# PATHS
# ==========================================

AUDIO_FILE = r"O:\Project\Multimodal-Emotion-Recognition\Test_Audio\test.wav"

MODEL_PATH = r"O:\Project\Multimodal-Emotion-Recognition\Models\best_audio_emotion_model.pkl"

SCALER_PATH = r"O:\Project\Multimodal-Emotion-Recognition\Audio\prepared\audio_scaler.pkl"


# ==========================================
# SETTINGS
# ==========================================

SAMPLE_RATE = 16000


# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

print("Loading model...")

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)


# ==========================================
# LOAD AUDIO
# ==========================================

print("Loading audio...")

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE,
    mono=True
)


# ==========================================
# FEATURE EXTRACTION
# ==========================================

features = []


# ------------------------------------------
# MFCC
# ------------------------------------------

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

features.extend(np.mean(mfcc, axis=1))
features.extend(np.std(mfcc, axis=1))


# ------------------------------------------
# Chroma
# ------------------------------------------

chroma = librosa.feature.chroma_stft(
    y=audio,
    sr=sr
)

features.extend(np.mean(chroma, axis=1))
features.extend(np.std(chroma, axis=1))


# ------------------------------------------
# Mel Spectrogram
# ------------------------------------------

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=40
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)

features.extend(np.mean(mel_db, axis=1))
features.extend(np.std(mel_db, axis=1))


# ------------------------------------------
# Zero Crossing Rate
# ------------------------------------------

zcr = librosa.feature.zero_crossing_rate(
    audio
)

features.append(np.mean(zcr))
features.append(np.std(zcr))


# ------------------------------------------
# Spectral Centroid
# ------------------------------------------

spectral_centroid = librosa.feature.spectral_centroid(
    y=audio,
    sr=sr
)

features.append(np.mean(spectral_centroid))
features.append(np.std(spectral_centroid))


# ------------------------------------------
# Spectral Bandwidth
# ------------------------------------------

spectral_bandwidth = librosa.feature.spectral_bandwidth(
    y=audio,
    sr=sr
)

features.append(np.mean(spectral_bandwidth))
features.append(np.std(spectral_bandwidth))


# ------------------------------------------
# Spectral Rolloff
# ------------------------------------------

spectral_rolloff = librosa.feature.spectral_rolloff(
    y=audio,
    sr=sr
)

features.append(np.mean(spectral_rolloff))
features.append(np.std(spectral_rolloff))


# ==========================================
# CONVERT TO ARRAY
# ==========================================

features = np.array(features)

print("Number of extracted features:", len(features))


# ==========================================
# SCALE FEATURES
# ==========================================

features = features.reshape(1, -1)

features_scaled = scaler.transform(
    features
)


# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(
    features_scaled
)


emotion = prediction[0]


# ==========================================
# CONFIDENCE
# ==========================================

print()
print("================================")
print("AUDIO EMOTION PREDICTION")
print("================================")

print(
    "Predicted Emotion:",
    emotion
)


# Random Forest supports predict_proba
if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(
        features_scaled
    )[0]

    classes = model.classes_

    confidence = np.max(probabilities)

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("\nEmotion probabilities:")

    for emotion_name, probability in sorted(
        zip(classes, probabilities),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{emotion_name:10s}: "
            f"{probability * 100:.2f}%"
        )


print()
print("================================")
print("TEST COMPLETE")
print("================================")