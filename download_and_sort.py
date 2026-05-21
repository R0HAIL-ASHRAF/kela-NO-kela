import os
import shutil
import kagglehub

# 1. Download the latest version of the RAVDESS Speech Audio dataset from Kaggle
print("Connecting to Kaggle to fetch RAVDESS Speech Audio...")
download_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print(f"Dataset securely downloaded to cache at: {download_path}\n")

# 2. Define target output directory and the official RAVDESS numerical mapping
output_dir = "sorted_speech_data"
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprise"
}

# Create output subdirectories for each category
for emotion in emotion_map.values():
    os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)

print("Sorting files into explicit emotion folders...")
move_count = 0

# 3. Walk through the downloaded cache folder structure
for root, _, files in os.walk(download_path):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            
            # Verify the file adheres to the 7-part RAVDESS naming convention
            if len(parts) == 7:
                emotion_code = parts[2]  # The 3rd index determines the emotion category
                emotion_folder = emotion_map.get(emotion_code)
                
                if emotion_folder:
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(output_dir, emotion_folder, file)
                    
                    # Copy the file to its designated folder
                    shutil.copy2(source_file, destination_file)
                    move_count += 1

print("\n--- Processing Complete ---")
print(f"Successfully sorted {move_count} audio files into '{output_dir}/'")

# 4. Print directory breakdown to verify structure balances
for emotion in sorted(os.listdir(output_dir)):
    folder_path = os.path.join(output_dir, emotion)
    if os.path.isdir(folder_path):
        print(f" -> Folder: {emotion:<10} | Files found: {len(os.listdir(folder_path))}")