import os
import numpy as np
import librosa

# =====================================================================
# CONFIGURATION & PARAMETERS
# =====================================================================
DATA_DIR = "data"
TARGET_CLASSES = ["angry", "sad", "happy", "neutral", "fear"]

# Audio Processing Dimensions to match your shape: (Samples, 173, 40)
N_MFCC = 40        # Feature channels (last dimension)
MAX_FRAMES = 173   # Fixed sequence length (middle dimension)

def extract_audio_features():
    print("\nExtracting MFCC features from dataset folders...")
    
    # 1. Dynamically check which folders actually exist
    available_classes = []
    for emotion in TARGET_CLASSES:
        emotion_path = os.path.join(DATA_DIR, emotion)
        if os.path.exists(emotion_path) and os.path.isdir(emotion_path):
            available_classes.append(emotion)
        else:
            print(f"Warning: Folder not found -> {emotion_path} (Skipping class)")
            
    print(f"Active classes for training: {available_classes}")
    
    # Create a clean 0-indexed mapping for active classes
    class_mapping = {emotion: i for i, emotion in enumerate(available_classes)}
    print(f"Label Mapping: {class_mapping}\n")
    
    X_list = []
    y_list = []
    
    # 2. Loop through active directories
    for emotion in available_classes:
        emotion_path = os.path.join(DATA_DIR, emotion)
        files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        print(f"Processing '{emotion}': {len(files)} files found")
        
        for idx, filename in enumerate(files):
            file_path = os.path.join(emotion_path, filename)
            
            try:
                # Load audio (sr=22050 standardizes sample rate across all files)
                y_audio, sr = librosa.load(file_path, sr=22050)
                
                # Extract MFCCs
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
                
                # Transpose to position frames as rows: (frames, n_mfcc)
                mfcc = mfcc.T 
                
                # Padding / Truncating logic to hit exactly MAX_FRAMES (173)
                if mfcc.shape[0] < MAX_FRAMES:
                    pad_width = MAX_FRAMES - mfcc.shape[0]
                    mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
                else:
                    mfcc = mfcc[:MAX_FRAMES, :]
                    
                X_list.append(mfcc)
                y_list.append(class_mapping[emotion])
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
            if (idx + 1) % 50 == 0:
                print(f"  -> Processed {idx + 1}/{len(files)} files...")

    # 3. Convert arrays to structured NumPy matrices
    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=np.int64)
    
    print("\n--- FEATURE EXTRACTION COMPLETE ---")
    print(f"Raw Matrix X shape: {X_raw.shape}")
    print(f"Raw Label Vector y shape: {y_raw.shape}")
    
    # 4. Save the raw assets completely unscaled to break the data leak
    np.save(os.path.join(DATA_DIR, "X_raw.npy"), X_raw)
    np.save(os.path.join(DATA_DIR, "y_raw.npy"), y_raw)
    
    # Save the mapping schema so your test.py script knows the new clean classes
    import json
    with open(os.path.join(DATA_DIR, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f)
        
    print("Unscaled dataset matrices and class maps successfully saved within 'data/'!\n")

if __name__ == "__main__":
    extract_audio_features()