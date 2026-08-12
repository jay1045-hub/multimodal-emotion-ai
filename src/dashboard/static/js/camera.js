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
            Temporary dummy prediction.

            This will later be replaced
            with Harshal's AI model.
        */

        emotion.textContent =
            "Happy";

        emotionIcon.textContent =
            "😊";

        confidence.textContent =
            "92%";

        confidenceBar.style.width =
            "92%";


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
   STOP CAMERA
--------------------------------------- */

function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(track => track.stop());

        cameraStream = null;

    }


    video.srcObject = null;

    video.style.display = "none";

    cameraMessage.style.display = "block";

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