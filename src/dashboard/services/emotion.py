import os

import cv2
import numpy as np


class EmotionDetector:

    EMOTIONS = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral"
    ]

    def __init__(self, model_path=None):

        self.model = None

        if model_path is None:

            model_path = os.path.join(
                "models",
                "emotion_model.keras"
            )

        self.model_path = model_path

        print(
            f"Emotion model path: {self.model_path}"
        )

        # Model will be loaded when
        # Harshal provides the trained file.

        if os.path.exists(self.model_path):

            self.load_model()

        else:

            print(
                "Emotion model not found."
            )

            print(
                "Waiting for emotion_model.keras"
            )


    def load_model(self):

        try:

            from tensorflow.keras.models import load_model

            self.model = load_model(
                self.model_path
            )

            print(
                "Emotion model loaded successfully."
            )

        except Exception as error:

            print(
                "Could not load emotion model:"
            )

            print(error)

            self.model = None


    def preprocess(self, face):

        """
        Prepare detected face for
        the CNN emotion model.
        """

        # Convert BGR → Grayscale

        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )


        # Resize to 48 × 48

        gray = cv2.resize(
            gray,
            (48, 48)
        )


        # Normalize pixel values

        gray = gray.astype(
            np.float32
        ) / 255.0


        # Add dimensions:
        #
        # (48,48)
        #     ↓
        # (1,48,48,1)

        gray = np.expand_dims(
            gray,
            axis=-1
        )

        gray = np.expand_dims(
            gray,
            axis=0
        )


        return gray


    def predict(self, face):

        """
        Predict emotion from a face image.
        """

        if self.model is None:

            return {
                "emotion": "Unknown",
                "confidence": 0.0
            }


        processed_face = self.preprocess(
            face
        )


        predictions = self.model.predict(
            processed_face,
            verbose=0
        )


        probabilities = predictions[0]


        index = int(
            np.argmax(probabilities)
        )


        confidence = float(
            probabilities[index]
        )


        emotion = self.EMOTIONS[index]


        return {
            "emotion": emotion,
            "confidence": confidence
        }