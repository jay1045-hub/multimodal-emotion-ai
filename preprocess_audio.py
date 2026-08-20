import os
import librosa
import soundfile as sf
import pandas as pd
import numpy as np

# ==========================================
# PATHS
# ==========================================

QUALITY_FILE = r"O:\Project\Multimodal-Emotion-Recognition\Audio\audio_quality.csv"

OUTPUT_DIR = r"O:\Project\Multimodal-Emotion-Recognition\Cleaned_Audio"

# Target sampling rate
TARGET_SR = 16000


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# LOAD VALID AUDIO METADATA
# ==========================================

df = pd.read_csv(QUALITY_FILE)

print("Files to preprocess:", len(df))
print()


# ==========================================
# PROCESS EACH AUDIO FILE
# ==========================================

processed = 0
failed = 0

for index, row in df.iterrows():

    input_path = row["path"]
    filename = row["filename"]

    try:

        # ----------------------------------
        # Load audio
        # ----------------------------------

        audio, sr = librosa.load(
            input_path,
            sr=TARGET_SR,
            mono=True
        )

        # ----------------------------------
        # Normalize audio
        # ----------------------------------

        max_value = np.max(np.abs(audio))

        if max_value > 0:
            audio = audio / max_value

        # ----------------------------------
        # Create actor folder
        # ----------------------------------

        actor_folder = os.path.join(
            OUTPUT_DIR,
            f"Actor_{int(row['actor']):02d}"
        )

        os.makedirs(actor_folder, exist_ok=True)

        # ----------------------------------
        # Output path
        # ----------------------------------

        output_path = os.path.join(
            actor_folder,
            filename
        )

        # ----------------------------------
        # Save cleaned audio
        # ----------------------------------

        sf.write(
            output_path,
            audio,
            TARGET_SR
        )

        processed += 1

        if processed % 100 == 0:
            print(f"Processed: {processed}/{len(df)}")

    except Exception as e:

        failed += 1

        print(f"Failed: {filename}")
        print("Error:", e)


# ==========================================
# FINAL RESULT
# ==========================================

print()
print("================================")
print("PREPROCESSING COMPLETE")
print("================================")

print("Original valid files:", len(df))
print("Successfully processed:", processed)
print("Failed:", failed)

print()
print("Cleaned dataset location:")
print(OUTPUT_DIR)