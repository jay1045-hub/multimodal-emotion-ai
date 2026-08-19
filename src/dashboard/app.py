from services.emotion import EmotionDetector

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

from services.camera import FaceDetector


app = Flask(__name__)

# Create face detector
face_detector = FaceDetector()

emotion_detector = EmotionDetector()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/detect-face", methods=["POST"])
def detect_face():
    """
    Receive an image frame from the browser
    and detect faces using OpenCV.
    """

    try:

        # Check whether an image was received
        if "frame" not in request.files:
            return jsonify({
                "success": False,
                "error": "No frame received"
            }), 400


        # Read uploaded image
        file = request.files["frame"]

        image_bytes = file.read()


        # Convert bytes to NumPy array
        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )


        # Decode image using OpenCV
        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:
            return jsonify({
                "success": False,
                "error": "Could not decode image"
            }), 400


        # Detect faces
        faces = face_detector.detect_faces(frame)


        # Convert OpenCV coordinates into JSON
        face_data = []

        for (x, y, w, h) in faces:

            # Crop the detected face
            face = face_detector.crop_face(
            frame,
            (x, y, w, h)
        )

        # Predict emotion
        prediction = emotion_detector.predict(
            face
        )

        face_data.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "emotion": prediction["emotion"],
            "confidence": prediction["confidence"]
        })


        return jsonify({
            "success": True,
            "faces": face_data,
            "count": len(face_data)
        })


    except Exception as e:

        print("Face detection error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)