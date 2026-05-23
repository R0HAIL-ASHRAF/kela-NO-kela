import os
import numpy as np
import librosa

DATA_DIR = "data"
TARGET_CLASSES = ["angry", "sad", "happy", "neutral"]

N_MFCC = 40
MAX_FRAMES = 100

def extract_feature_vector(y_audio, sr):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T  # [T, 120]

    if features.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:MAX_FRAMES, :]

    return features

def augment_and_extract(y_audio, sr):
    variants = []

    # Original
    variants.append(extract_feature_vector(y_audio, sr))

    # Gaussian noise
    noisy = y_audio + 0.005 * np.random.randn(len(y_audio))
    variants.append(extract_feature_vector(noisy, sr))

    # Pitch shift up (+2 semitones)
    try:
        shifted_up = librosa.effects.pitch_shift(y_audio, sr=sr, n_steps=2)
        variants.append(extract_feature_vector(shifted_up, sr))
    except Exception:
        pass

    # Pitch shift down (-2 semitones)
    try:
        shifted_down = librosa.effects.pitch_shift(y_audio, sr=sr, n_steps=-2)
        variants.append(extract_feature_vector(shifted_down, sr))
    except Exception:
        pass

    return variants

def extract_audio_features():
    print("\nExtracting MFCC + delta + delta² features from dataset folders...")
    print(f"Feature dimensions: {N_MFCC} MFCC + {N_MFCC} delta + {N_MFCC} delta² = {N_MFCC * 3} per frame")

    available_classes = []
    for emotion in TARGET_CLASSES:
        emotion_path = os.path.join(DATA_DIR, emotion)
        if os.path.exists(emotion_path) and os.path.isdir(emotion_path):
            available_classes.append(emotion)
        else:
            print(f"Warning: Folder not found -> {emotion_path} (Skipping class)")

    print(f"Active classes for training: {available_classes}")

    class_mapping = {emotion: i for i, emotion in enumerate(available_classes)}
    print(f"Label Mapping: {class_mapping}\n")

    X_list = []
    y_list = []

    for emotion in available_classes:
        emotion_path = os.path.join(DATA_DIR, emotion)
        files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        print(f"Processing '{emotion}': {len(files)} files found (will be augmented 4x)")

        for idx, filename in enumerate(files):
            file_path = os.path.join(emotion_path, filename)

            try:
                y_audio, sr = librosa.load(file_path, sr=22050)
                variants = augment_and_extract(y_audio, sr)

                for feat in variants:
                    X_list.append(feat)
                    y_list.append(class_mapping[emotion])

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

            if (idx + 1) % 50 == 0:
                print(f"  -> Processed {idx + 1}/{len(files)} files...")

    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=np.int64)

    print("\n--- FEATURE EXTRACTION COMPLETE ---")
    print(f"Raw Matrix X shape: {X_raw.shape}  (samples x frames x features)")
    print(f"Raw Label Vector y shape: {y_raw.shape}")

    np.save(os.path.join(DATA_DIR, "X_raw.npy"), X_raw)
    np.save(os.path.join(DATA_DIR, "y_raw.npy"), y_raw)

    import json
    with open(os.path.join(DATA_DIR, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f)

    print("Unscaled dataset matrices and class maps saved to 'data/'!\n")

if __name__ == "__main__":
    extract_audio_features()
