const video = document.getElementById("camera");

const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");

const cameraMessage =
    document.getElementById("cameraMessage");

const status =
    document.getElementById("status");

const emotion =
    document.getElementById("emotion");

const emotionIcon =
    document.getElementById("emotionIcon");

const confidence =
    document.getElementById("confidence");

const confidenceBar =
    document.getElementById("confidenceBar");


let cameraStream = null;
let detectionInterval = null;


/* ---------------------------------------
   CREATE CANVAS
--------------------------------------- */

const canvas = document.createElement("canvas");

const ctx = canvas.getContext("2d");

canvas.style.position = "absolute";
canvas.style.top = "0";
canvas.style.left = "0";
canvas.style.pointerEvents = "none";


const cameraBox =
    document.querySelector(".camera-box");

if (cameraBox) {
    cameraBox.appendChild(canvas);
}


/* ---------------------------------------
   START CAMERA
--------------------------------------- */

async function startCamera() {

    try {

        cameraStream =
            await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });


        video.srcObject = cameraStream;

        video.style.display = "block";

        cameraMessage.style.display = "none";


        status.textContent =
            "Camera active";


        /*
            Wait until video dimensions
            are available.
        */

        video.onloadedmetadata = () => {

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            startFaceDetection();

        };


        /*
            Temporary emotion display.

            This will later be replaced
            by Harshal's model.
        */

        emotion.textContent =
            "Waiting...";

        emotionIcon.textContent =
            "🧠";

        confidence.textContent =
            "--";

        confidenceBar.style.width =
            "0%";


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        cameraMessage.textContent =
            "Unable to access camera.";


        status.textContent =
            "Camera access denied";


        alert(
            "Please allow camera access in your browser."
        );

    }

}


/* ---------------------------------------
   FACE DETECTION
--------------------------------------- */

function startFaceDetection() {

    /*
        Run detection approximately
        every 200 milliseconds.
    */

    if (detectionInterval) {
        clearInterval(detectionInterval);
    }


    detectionInterval =
        setInterval(
            detectFace,
            200
        );

}


/* ---------------------------------------
   SEND FRAME TO FLASK
--------------------------------------- */

async function detectFace() {

    if (!cameraStream) {
        return;
    }


    if (video.readyState < 2) {
        return;
    }


    try {

        /*
            Draw current video frame
            onto the hidden canvas.
        */

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;


        /*
            We need a separate temporary
            canvas because the visible canvas
            is used for drawing detection boxes.
        */

        const frameCanvas =
            document.createElement("canvas");

        frameCanvas.width =
            video.videoWidth;

        frameCanvas.height =
            video.videoHeight;


        const frameContext =
            frameCanvas.getContext("2d");


        frameContext.drawImage(
            video,
            0,
            0,
            frameCanvas.width,
            frameCanvas.height
        );


        /*
            Convert frame to JPEG.
        */

        const blob =
            await new Promise(resolve => {

                frameCanvas.toBlob(
                    resolve,
                    "image/jpeg",
                    0.7
                );

            });


        if (!blob) {
            return;
        }


        /*
            Prepare form data.
        */

        const formData =
            new FormData();

        formData.append(
            "frame",
            blob,
            "frame.jpg"
        );


        /*
            Send frame to Flask.
        */

        const response =
            await fetch(
                "/api/detect-face",
                {
                    method: "POST",
                    body: formData
                }
            );


        const result =
            await response.json();


        if (!result.success) {

            console.error(
                "Detection error:",
                result.error
            );

            return;

        }


        /*
            Draw detected faces.
        */

        drawFaces(result.faces);


        /*
            Update status.
        */

        if (result.count > 0) {

            status.textContent =
        `       ${result.count} face detected`;

            const firstFace =
                result.faces[0];

            const detectedEmotion =
                firstFace.emotion || "Unknown";

            const detectedConfidence =
                firstFace.confidence || 0;

            const confidencePercent =
                Math.round(
                detectedConfidence * 100
            );

            emotion.textContent =
                detectedEmotion;

            confidence.textContent =
        `       ${confidencePercent}%`;

            confidenceBar.style.width =
                `${confidencePercent}%`;

        } else {

            status.textContent =
                "No face detected";

            emotion.textContent =
                "Waiting...";

            emotionIcon.textContent =
                "⏳";

            confidence.textContent =
                "--";

            confidenceBar.style.width =
                "0%";
}


    } catch (error) {

        console.error(
            "Face detection request failed:",
            error
        );

    }

}


/* ---------------------------------------
   DRAW FACE BOXES
--------------------------------------- */

function drawFaces(faces) {

    /*
        Clear previous boxes.
    */

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    /*
        Calculate scaling between
        video resolution and displayed size.
    */

    const scaleX =
        video.clientWidth /
        video.videoWidth;

    const scaleY =
        video.clientHeight /
        video.videoHeight;


    faces.forEach(face => {

        const x =
            face.x * scaleX;

        const y =
            face.y * scaleY;

        const width =
            face.width * scaleX;

        const height =
            face.height * scaleY;


        /*
            Draw bounding box.
        */

        ctx.strokeStyle =
            "#00ff00";

        ctx.lineWidth = 3;


        ctx.strokeRect(
            x,
            y,
            width,
            height
        );


        /*
            Face label.
        */

        ctx.fillStyle =
            "#00ff00";

        ctx.font =
            "18px Arial";


        const detectedEmotion =
    face.emotion || "Unknown";

    const detectedConfidence =
        face.confidence || 0;

    const confidencePercent =
        Math.round(
            detectedConfidence * 100
        );


    ctx.fillText(
        `${detectedEmotion} ${confidencePercent}%`,
        x,
        Math.max(y - 8, 18)
    );

    });

}


/* ---------------------------------------
   STOP CAMERA
--------------------------------------- */

function stopCamera() {

    /*
        Stop detection.
    */

    if (detectionInterval) {

        clearInterval(
            detectionInterval
        );

        detectionInterval = null;

    }


    /*
        Stop webcam tracks.
    */

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        cameraStream = null;

    }


    video.srcObject = null;

    video.style.display = "none";


    /*
        Clear face boxes.
    */

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    cameraMessage.style.display =
        "block";

    cameraMessage.textContent =
        "Camera is currently off.";


    status.textContent =
        "Camera inactive";


    emotion.textContent =
        "Waiting...";

    emotionIcon.textContent =
        "⏳";

    confidence.textContent =
        "--";

    confidenceBar.style.width =
        "0%";

}


/* ---------------------------------------
   BUTTON EVENTS
--------------------------------------- */

if (startButton) {

    startButton.addEventListener(
        "click",
        startCamera
    );

}


if (stopButton) {

    stopButton.addEventListener(
        "click",
        stopCamera
    );

}