import os
import librosa
import numpy as np
import pandas as pd

# ==========================================
# PATHS
# ==========================================

CLEANED_AUDIO_PATH = r"O:\Project\Multimodal-Emotion-Recognition\Cleaned_Audio"

OUTPUT_PATH = r"O:\Project\Multimodal-Emotion-Recognition\Audio\audio_features.csv"

# Sampling rate used during preprocessing
SAMPLE_RATE = 16000


# ==========================================
# FEATURE EXTRACTION FUNCTION
# ==========================================

def extract_features(file_path):

    # Load audio
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    features = []

    # --------------------------------------
    # 1. MFCC
    # --------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # --------------------------------------
    # 2. Chroma
    # --------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # --------------------------------------
    # 3. Mel Spectrogram
    # --------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=40
    )

    # Convert to log scale
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    # --------------------------------------
    # 4. Zero Crossing Rate
    # --------------------------------------

    zcr = librosa.feature.zero_crossing_rate(audio)

    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # --------------------------------------
    # 5. Spectral Centroid
    # --------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))

    # --------------------------------------
    # 6. Spectral Bandwidth
    # --------------------------------------

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))

    # --------------------------------------
    # 7. Spectral Rolloff
    # --------------------------------------

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    features.append(np.mean(spectral_rolloff))
    features.append(np.std(spectral_rolloff))

    return features


# ==========================================
# MAIN
# ==========================================

data = []

total_files = 0
successful_files = 0
failed_files = 0


print("Starting feature extraction...")
print()


# ==========================================
# LOOP THROUGH ACTOR FOLDERS
# ==========================================

for actor_folder in sorted(os.listdir(CLEANED_AUDIO_PATH)):

    actor_path = os.path.join(
        CLEANED_AUDIO_PATH,
        actor_folder
    )

    if not os.path.isdir(actor_path):
        continue

    # --------------------------------------
    # Loop through audio files
    # --------------------------------------

    for filename in sorted(os.listdir(actor_path)):

        if not filename.endswith(".wav"):
            continue

        total_files += 1

        file_path = os.path.join(
            actor_path,
            filename
        )

        try:

            # Extract features
            features = extract_features(file_path)

            # ----------------------------------
            # Extract emotion from filename
            # ----------------------------------

            parts = filename.split("-")

            emotion_id = int(parts[2])

            emotion_map = {
                1: "neutral",
                2: "calm",
                3: "happy",
                4: "sad",
                5: "angry",
                6: "fear",
                7: "disgust",
                8: "surprise"
            }

            emotion = emotion_map[emotion_id]

            # Actor ID
            actor_id = int(
                parts[6].split(".")[0]
            )

            # ----------------------------------
            # Create row
            # ----------------------------------

            row = {
                "filename": filename,
                "actor": actor_id,
                "emotion": emotion
            }

            # Add numerical features
            for i, value in enumerate(features):

                row[f"feature_{i+1}"] = value

            data.append(row)

            successful_files += 1

            # Progress
            if successful_files % 100 == 0:

                print(
                    f"Processed: "
                    f"{successful_files}/{1428}"
                )

        except Exception as e:

            failed_files += 1

            print(
                f"Failed: {filename}"
            )

            print(
                f"Error: {e}"
            )


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(data)


# ==========================================
# SAVE CSV
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print()
print("======================================")
print("FEATURE EXTRACTION COMPLETE")
print("======================================")

print(
    "Total audio files:",
    total_files
)

print(
    "Successfully processed:",
    successful_files
)

print(
    "Failed:",
    failed_files
)

print(
    "Number of features:",
    len(df.columns) - 3
)

print()
print("Dataset shape:")
print(df.shape)

print()
print("Emotion distribution:")
print(df["emotion"].value_counts())

print()
print("Saved to:")
print(OUTPUT_PATH)