import os
import tempfile
import time
import subprocess

import torch
import whisper

from langdetect import detect_langs

from flask import Flask, render_template, request, jsonify

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from google import genai


app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WHISPER_MODEL = "small"
SENTIMENT_MODEL_PATH = "./multilingual_sentiment_model"
GEMINI_MODEL = "gemini-3.5-transcribe"

GEMINI_API_KEY = "YOUR_API_KEY"

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )
    print("Gemini API: Ready")
else:
    print("Gemini API: Not configured")


print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


print("Loading Whisper...")
whisper_model = whisper.load_model(
    WHISPER_MODEL,
    device=DEVICE
)

print("Whisper loaded.")


print("Loading sentiment model...")

tokenizer = AutoTokenizer.from_pretrained(
    SENTIMENT_MODEL_PATH,
    local_files_only=True
)

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL_PATH,
    local_files_only=True
)

sentiment_model.to(DEVICE)
sentiment_model.eval()

print("Sentiment model loaded.")


def transcribe_with_whisper(audio_path):

    audio = whisper.load_audio(audio_path)

    audio_for_detection = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(
        audio_for_detection,
        n_mels=whisper_model.dims.n_mels
    ).to(DEVICE)

    _, language_probs = whisper_model.detect_language(mel)

    language = max(
        language_probs,
        key=language_probs.get
    )

    language_confidence = language_probs[language]

    result = whisper_model.transcribe(
        audio,
        language=language,
        task="transcribe",
        fp16=False
    )

    text = result["text"].strip()

    return (
        text,
        language,
        language_confidence
    )


def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = sentiment_model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        prediction = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()

    sentiment = sentiment_model.config.id2label[
        prediction
    ].upper()

    return sentiment, confidence


from langdetect import detect_langs

def transcribe_with_gemini(audio_path):

    if gemini_client is None:
        return "", None, None, "Gemini API is not configured."

    wav_path = (
        audio_path.rsplit(".", 1)[0]
        + "_gemini.wav"
    )

    try:

        print("Gemini: Converting audio to WAV...")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                wav_path
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print("Gemini: Uploading WAV...")

        audio_file = gemini_client.files.upload(
            file=wav_path
        )

        print("Gemini: Waiting for file...")

        while True:

            state = audio_file.state.name

            if state == "ACTIVE":
                break

            if state == "FAILED":
                return (
                    "",
                    None,
                    None,
                    "Gemini file processing failed."
                )

            time.sleep(1)

            audio_file = gemini_client.files.get(
                name=audio_file.name
            )

        print("Gemini: File is ACTIVE.")

        print("Gemini: Transcribing...")

        interaction = gemini_client.interactions.create(
            model=GEMINI_MODEL,
            input=[
                {
                    "type": "audio",
                    "uri": audio_file.uri,
                    "mime_type": audio_file.mime_type
                }
            ],
            generation_config={
                "transcription_config": {
                    "language_codes": []
                }
            }
        )

        text = interaction.output_text.strip()

        if not text:
            return (
                "",
                None,
                None,
                "Gemini returned empty transcription."
            )

        print("Gemini text:")
        print(text)

        print("Language detection with langdetect...")

        detected = detect_langs(text)

        if detected:
            gemini_language = detected[0].lang
            gemini_language_confidence = 1.0
        else:
            gemini_language = None
            gemini_language_confidence = None

        print("Gemini language:", gemini_language)
        print("Gemini language confidence: 100%")

        return (
            text,
            gemini_language,
            gemini_language_confidence,
            None
        )

    except Exception as e:

        print("Gemini error:")
        print(e)

        return (
            "",
            None,
            None,
            str(e)
        )

    finally:

        if os.path.exists(wav_path):
            os.remove(wav_path)

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/analyze", methods=["POST"])
def analyze():

    if "audio" not in request.files:

        return jsonify({
            "error": "No audio file received."
        }), 400

    audio_file = request.files["audio"]

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".webm"
        ) as temp_file:

            audio_file.save(
                temp_file.name
            )

            temp_path = temp_file.name


        print()
        print("Processing audio...")


        whisper_text, whisper_language, whisper_language_confidence = (
            transcribe_with_whisper(
                temp_path
            )
        )


        print(
            "Whisper:",
            whisper_text
        )

        print(
            "Whisper language:",
            whisper_language,
            whisper_language_confidence
        )


        gemini_text, gemini_language, gemini_language_confidence, gemini_error = (
            transcribe_with_gemini(
                temp_path
            )
        )


        print(
            "Gemini:",
            gemini_text
        )


        if not whisper_text:

            return jsonify({

                "whisper_text": "",

                "gemini_text": gemini_text,

                "gemini_error": gemini_error,

                "language": whisper_language,

                "language_confidence":
                    whisper_language_confidence,

                "gemini_language":
                    gemini_language,

                "gemini_language_confidence":
                    gemini_language_confidence,

                "sentiment": "NONE",

                "confidence": 0

            })


        sentiment, confidence = predict_sentiment(
            whisper_text
        )


        return jsonify({

            "whisper_text": whisper_text,

            "gemini_text": gemini_text,

            "gemini_error": gemini_error,

            "language": whisper_language,

            "language_confidence":
                whisper_language_confidence,

            "gemini_language":
                gemini_language,

            "gemini_language_confidence":
                gemini_language_confidence,

            "sentiment": sentiment,

            "confidence": confidence

        })


    except Exception as e:

        print("Application error:")
        print(e)

        return jsonify({
            "error": str(e)
        }), 500


    finally:

        if temp_path and os.path.exists(
            temp_path
        ):
            os.remove(temp_path)


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )