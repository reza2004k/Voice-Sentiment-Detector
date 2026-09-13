# Voice Sentiment Detector

## Quick Run
```
pip install -r requirements.txt
python app.py
```
If you want to execute your code by GPU you must install Pytorch CUDA 

```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Summery
You press the microphone then say something in which language you want to speak.
Whisper and GoogleGemini to detect your voice and transcribe it.
Sentiment model work on the transcribe of your voice then check the status of your feeling.
Is it Positive Neutral or Negative.
We use Whisper and GoogleGemini for comparing them with each other.
Gemini and Whisper use different approaches and training pipelines for speech recognition. Gemini can achieve strong transcription accuracy, while Whisper is a widely used open-source multilingual ASR model. This project uses both systems to compare their transcription results under the same audio input.
 
# Models

This model is for Whisper to convert speech to text:
## Small

We use this model for sentiment after whisper transcribe your voice:
## multilingual_sentiment_model
You should download this model from HuggingFace

https://huggingface.co/tabularisai/multilingual-sentiment-analysis/tree/main

## How Whisper can convert your speech to text??!!

Whisper is a **Transformer-based neural network** trained on approximately **680,000 hours of multilingual and multitask human speech recordings (equivalent to nearly 78 years of continuous audio without interruption)** collected from the Internet. During training, each audio recording is converted into a **log-Mel spectrogram**, which represents the frequencies and their changes over time. These spectrograms provide the neural network with a visual-like representation of the speech signal.

The training process is similar to teaching a neural network with labeled examples. For example, if an audio recording contains the word **"Hello"**, the corresponding transcript tells the model that this audio represents the text *Hello*. The neural network analyzes the spectrogram and learns patterns between acoustic features, frequencies, sounds, and language.

If the model predicts the wrong text, a **loss function** measures how different the prediction is from the correct transcript. Through **backpropagation**, gradients are calculated and the network's weights are updated using an optimization algorithm. By repeating this process over hundreds of thousands of hours of speech, the network gradually learns how different sounds correspond to words, sentences, and different languages.

When you send a voice recording to Whisper, the process is approximately:

```text
Voice Recording
      ↓
Log-Mel Spectrogram
      ↓
Split into small time-frequency patterns
      ↓
Whisper Small Model
      ↓
Transformer Encoder
      ↓
Speech Representation
      ↓
Transformer Decoder
      ↓
Predicted Tokens
      ↓
Text
```

In this project, we use the **Whisper Small model**. The trained model already contains millions of learned parameters (weights) obtained during training. These weights encode the patterns the neural network learned from speech data.

Therefore, when a new voice is provided, Whisper does not simply search for an identical recording. Instead, it uses the learned weights of its neural network to analyze the new spectrogram, recognize acoustic and linguistic patterns, and generate the most likely transcription **token by token**.

Because Whisper was trained on speech from many languages, speakers, accents, and recording conditions, it can recognize and transcribe a wide range of real-world speech.



