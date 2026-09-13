# Voice Sentiment Detector

# Quick Run
```
pip install -r requirements.txt
python app.py
```
If you want to execute your code by GPU you must install Pytorch CUDA 

```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

# Summery
You press the microphone and say something in the language you want to speak.

Whisper and Google Gemini detect your voice and transcribe it into text.

The sentiment model then analyzes the transcription and determines the sentiment of your speech: **Positive, Neutral, or Negative**.

We use both Whisper and Google Gemini to compare their transcription results using the same audio input.

Gemini and Whisper use different approaches and training pipelines for speech recognition. Gemini can achieve strong transcription accuracy, while Whisper is a widely used open-source multilingual ASR model. This project uses both systems to compare their transcription results under the same audio input.

For Persian (Farsi) voice recordings, Gemini produced more accurate and reliable transcriptions in our experiments, especially when dealing with natural spoken Persian, pronunciation variations, and conversational speech. Therefore, using both systems allows the project to compare their performance and evaluate which transcription is more suitable for the given audio.


# Models

This model is for Whisper to convert speech to text:
## Small

We use this model for sentiment after whisper transcribe your voice:
## multilingual_sentiment_model
You should download this model from HuggingFace

https://huggingface.co/tabularisai/multilingual-sentiment-analysis/tree/main

# How Whisper can convert your speech to text??!!

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

For learning more:

https://arxiv.org/pdf/2212.04356.pdf?utm_source=chatgpt.com

# How Does the Multilingual Sentiment Model Understand Your Feeling?

This project uses a **multilingual Transformer-based sentiment model** to classify text as **Positive, Neutral, or Negative**.

The model does not simply look for positive or negative words. Instead, it uses **Self-Attention** to understand the relationship between words and the context of the entire sentence.

### 1. Text → Tokens → Embeddings

First, the transcribed text is tokenized and converted into numerical vectors called **embeddings**.

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embeddings
  ↓
Transformer
```

### 2. Self-Attention with Q, K, V

For every token, the Transformer creates three vectors:

* **Q (Query):** What information does this word need?
* **K (Key):** What information does this word provide?
* **V (Value):** What information should be passed forward?

The attention mechanism is:

```text
Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V
```

The model compares the Query of a word with the Keys of other words and assigns higher attention to the words that are more relevant.

This allows the model to understand relationships such as:

```text
not → good
film → terrible
I → love
```

Transformers use **Multi-Head Attention**, so different attention heads can learn different types of relationships between words.

### 3. Context Is More Important Than Individual Words

For example:

```text
This film is good. I love it.
→ Positive
```

but:

```text
This film is not good. It is terrible.
→ Negative
```

Both sentences contain the word **"good"**, but the sentiment is different.

The model does not simply learn:

```text
good = positive
```

Instead, it learns that the meaning of a word depends on its surrounding context.

In:

```text
not + good
```

the word **"not"** changes the meaning of **"good"**.

The Transformer combines information from the whole sentence and creates a **contextual representation** before making the final sentiment prediction.

### 4. How Was the Neural Network Trained?

The model learned these patterns from a large amount of text during training.

Training can be simplified as:

```text
Text + Correct Sentiment Label
            ↓
       Transformer
            ↓
        Prediction
            ↓
      Calculate Loss
            ↓
      Backpropagation
            ↓
       Update Weights
            ↓
          Repeat
```

For example, the model can see:

```text
"This film is good. I love it."
→ Positive
```

and:

```text
"This film is terrible. I hate it."
→ Negative
```

If the model predicts the wrong sentiment, the **loss function** measures the error. **Backpropagation** calculates how the neural-network weights contributed to that error, and an optimizer updates the weights.

After seeing many training examples, the model learns useful patterns about language, context, and sentiment.

### 5. Final Sentiment Pipeline

In this project, the complete process is:

```text
Voice
  ↓
Whisper
  ↓
Speech-to-Text
  ↓
Language Detection
  ↓
Multilingual Transformer
  ↓
Token Embeddings
  ↓
Self-Attention (Q, K, V)
  ↓
Contextual Representation
  ↓
Classification Layer
  ↓
Positive / Neutral / Negative
```

The important idea is:

**The model does not understand sentiment by simply searching for positive or negative words. It learns patterns from many training examples and uses Transformer self-attention to understand how words interact with their context.**


