import cv2

from services.camera import FaceDetector


detector = FaceDetector()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Camera started.")
print("Press Q to quit.")

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame.")
        break

    faces = detector.detect_faces(frame)

    frame = detector.draw_faces(
        frame,
        faces
    )

    cv2.imshow(
        "Real-Time Face Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()

cv2.destroyAllWindows()

print("Camera stopped.")