import pickle
import numpy as np
import librosa
import gradio as gr

N_MFCC = 40
MAX_FRAMES = 173
CLASSES = ['angry', 'sad', 'happy', 'neutral', 'fear']

with open("lstm_model.pkl", "rb") as f:
    payload = pickle.load(f)

model = payload['model']
train_mean = payload['mean'].reshape(1, -1)
train_std = payload['std'].reshape(1, -1)

def predict_emotion(audio_path):
    y_audio, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC).T

    if mfcc.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    mfcc = (mfcc - train_mean) / train_std

    probs = model.predict(mfcc).flatten()

    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", label="Upload a speech audio file (.wav)"),
    outputs=gr.Label(num_top_classes=5, label="Emotion Probabilities"),
    title="Speech Emotion Recognition",
    description="Upload a `.wav` file and the model will predict the speaker's emotion: angry, sad, happy, neutral, or fear.",
    examples=[],
)

demo.launch()
