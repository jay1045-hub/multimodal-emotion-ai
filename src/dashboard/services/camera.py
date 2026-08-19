import cv2


class FaceDetector:
    """
    Handles real-time face detection using OpenCV.
    """

    def __init__(self):

        # Load OpenCV's pre-trained Haar Cascade face detector
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_cascade.empty():

            raise RuntimeError(
                "Could not load Haar Cascade face detector."
            )


    def detect_faces(self, frame):
        """
        Detect faces in a single video frame.

        Returns:
            list of detected face coordinates
        """

        # Convert frame to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )


        return faces


    def crop_face(self, frame, face):
        """
        Crop a detected face from the frame.

        Parameters:
            frame: OpenCV image
            face: (x, y, width, height)

        Returns:
            Cropped face image
        """

        x, y, w, h = face


        # Crop the face
        face_crop = frame[
            y:y + h,
            x:x + w
        ]


        return face_crop


    def draw_faces(self, frame, faces):
        """
        Draw bounding boxes around detected faces.
        """

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        return frame