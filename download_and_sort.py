import os
import shutil
import kagglehub

print("Connecting to Kaggle to fetch RAVDESS Speech Audio...")
download_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print(f"Dataset securely downloaded to cache at: {download_path}\n")

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

for emotion in emotion_map.values():
    os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)

print("Sorting files into explicit emotion folders...")
move_count = 0

for root, _, files in os.walk(download_path):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")

            if len(parts) == 7:
                emotion_code = parts[2]
                emotion_folder = emotion_map.get(emotion_code)

                if emotion_folder:
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(output_dir, emotion_folder, file)

                    shutil.copy2(source_file, destination_file)
                    move_count += 1

print("\n--- Processing Complete ---")
print(f"Successfully sorted {move_count} audio files into '{output_dir}/'")

for emotion in sorted(os.listdir(output_dir)):
    folder_path = os.path.join(output_dir, emotion)
    if os.path.isdir(folder_path):
        print(f" -> Folder: {emotion:<10} | Files found: {len(os.listdir(folder_path))}")
