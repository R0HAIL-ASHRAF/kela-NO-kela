import pickle
import numpy as np
import librosa
import gradio as gr

N_MFCC = 40
MAX_FRAMES = 100
CLASSES = ['angry', 'sad', 'happy', 'neutral']

with open("lstm_model.pkl", "rb") as f:
    payload = pickle.load(f)

model = payload['model']
train_mean = payload['mean'].reshape(1, -1)
train_std = payload['std'].reshape(1, -1)

def predict_emotion(audio_path):
    y_audio, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T  # [T, 120]

    if features.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:MAX_FRAMES, :]

    features = (features - train_mean) / train_std

    probs = model.predict(features).flatten()

    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", label="Upload a speech audio file (.wav)"),
    outputs=gr.Label(num_top_classes=4, label="Emotion Probabilities"),
    title="Speech Emotion Recognition",
    description="Upload a `.wav` file and the model will predict the speaker's emotion: angry, sad, happy, or neutral.",
    examples=[],
)

demo.launch()
