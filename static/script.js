const micButton = document.getElementById("micButton");

const whisperText = document.getElementById("whisper-text");
const whisperInfo = document.getElementById("whisper-info");

const geminiText = document.getElementById("gemini-text");
const geminiInfo = document.getElementById("gemini-info");

const status = document.getElementById("status");

const positive = document.getElementById("positive");
const neutral = document.getElementById("neutral");
const negative = document.getElementById("negative");

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;


function resetSentiment() {
    positive.classList.remove("positive-active");
    neutral.classList.remove("neutral-active");
    negative.classList.remove("negative-active");

    positive.classList.add("inactive");
    neutral.classList.add("inactive");
    negative.classList.add("inactive");
}


async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        audioChunks = [];

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = function() {
            stream.getTracks().forEach(track => track.stop());
            sendAudio();
        };

        mediaRecorder.start();

        isRecording = true;

        micButton.classList.add("recording");
        micButton.textContent = "⏹️";

        status.textContent = "Listening... Speak now";

        whisperText.textContent = "Listening...";
        whisperInfo.textContent =
            "Language: -- | Language confidence: --";

        geminiText.textContent = "Waiting for audio...";
        geminiInfo.textContent =
            "Language: -- | Language confidence: --";

        resetSentiment();

    } catch (error) {
        console.error(error);
        status.textContent = "Microphone access was denied.";
    }
}


function stopRecording() {

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }

    isRecording = false;

    micButton.classList.remove("recording");
    micButton.textContent = "🎤";

    status.textContent = "Processing your voice...";
}


async function sendAudio() {

    const audioBlob = new Blob(
        audioChunks,
        {
            type: "audio/webm"
        }
    );

    const formData = new FormData();

    formData.append(
        "audio",
        audioBlob,
        "voice.webm"
    );


    try {

        const response = await fetch(
            "/analyze",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (data.error) {
            throw new Error(data.error);
        }


        /* Whisper */

        whisperText.textContent =
            data.whisper_text ||
            "No speech detected.";


        let whisperLanguage = "--";

        if (data.language) {
            whisperLanguage =
                data.language.toUpperCase();
        }


        let whisperConfidence = "--";

        if (
            data.language_confidence !== undefined &&
            data.language_confidence !== null
        ) {

            whisperConfidence =
                (
                    data.language_confidence * 100
                ).toFixed(2) + "%";
        }


        whisperInfo.textContent =
            "Language: " +
            whisperLanguage +
            " | Language confidence: " +
            whisperConfidence;



        /* Gemini */

        geminiText.textContent =
            data.gemini_text ||
            (
                data.gemini_error
                    ? "Transcription unavailable."
                    : "No speech detected."
            );


        let geminiLanguage = "--";

        if (data.gemini_language) {
            geminiLanguage =
                data.gemini_language.toUpperCase();
        }


        let geminiConfidence = "--";

        if (
            data.gemini_language_confidence !== undefined &&
            data.gemini_language_confidence !== null
        ) {

            geminiConfidence =
                (
                    data.gemini_language_confidence * 100
                ).toFixed(2) + "%";
        }


        geminiInfo.textContent =
            "Language: " +
            geminiLanguage +
            " | Language confidence: " +
            geminiConfidence;



        /* Sentiment */

        resetSentiment();


        if (data.sentiment === "POSITIVE") {

            positive.classList.remove("inactive");
            positive.classList.add("positive-active");

        }

        else if (data.sentiment === "NEUTRAL") {

            neutral.classList.remove("inactive");
            neutral.classList.add("neutral-active");

        }

        else if (data.sentiment === "NEGATIVE") {

            negative.classList.remove("inactive");
            negative.classList.add("negative-active");

        }


        status.textContent =
            "Analysis complete. Click the microphone to speak again.";

    }


    catch (error) {

        console.error(error);

        whisperText.textContent =
            "Something went wrong.";

        whisperInfo.textContent =
            "Language: -- | Language confidence: --";


        geminiText.textContent =
            "Something went wrong.";

        geminiInfo.textContent =
            "Language: -- | Language confidence: --";


        resetSentiment();


        status.textContent =
            "Could not analyze the audio.";
    }
}


micButton.addEventListener(
    "click",
    function() {

        if (!isRecording) {
            startRecording();
        }

        else {
            stopRecording();
        }

    }
);