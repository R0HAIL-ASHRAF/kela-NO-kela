import numpy as np
import pickle
import os

# Define the exact index mapping used during your model's training phase
# (Modify the ordering below if your dataset maps labels differently)
INDEX_TO_EMOTION = {
    0: 'angry',
    1: 'happy',
    2: 'sad',
    3: 'neutral',
    4: 'fear'
}
EMOTION_TO_INDEX = {v: k for k, v in INDEX_TO_EMOTION.items()}

def load_validation_data():
    """
    Simulated placeholder for your validation data loading process.
    Replace this with your actual loading code (e.g., np.load) if necessary.
    """
    # Using the exact structural shapes from your terminal log output
    print("Loaded 101 completely unseen validation samples for testing.")
    
    # Replace these mock shapes with your true validation loading matrices:
    # X_val = np.load("data/X_val.npy")
    # y_val = np.load("data/y_val.npy")
    X_val = np.random.randn(101, 173, 40)
    
    # Simulating standard label inputs containing balanced mixtures of emotion strings
    emotions = ['angry', 'happy', 'sad', 'neutral']
    y_val = np.random.choice(emotions, size=(101,))
    
    print(f"X shape: {X_val.shape}")
    print(f"y shape: {y_val.shape}\n")
    return X_val, y_val

def main():
    model_path = 'lstm_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Please run train.py first!")
        return

    # Load the trained LSTM object instance safely from disk
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")

    # Fetch unseen testing elements
    X_val, y_val = load_validation_data()

    print("Running validation predictions...\n" + "="*50)
    
    correct_predictions = 0
    total_samples = len(X_val)
    
    # Tracking metrics dictionary container for evaluating precise precision/recall bounds
    class_metrics = {emotion: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for emotion in INDEX_TO_EMOTION.values()}

    for i in range(total_samples):
        sample = X_val[i]
        actual = y_val[i]  # Expected format: String label (e.g., 'sad')
        
        # Ensure actual target is processed uniformly as a lowercase string string
        if isinstance(actual, (bytes, np.bytes_)):
            actual = actual.decode('utf-8')
        actual = str(actual).strip().lower()

        # Execute safe, stateless prediction over the sequence
        prob_distribution = model.predict(sample)
        
        # ⚡ FIX: Extract scalar index of highest predicted probability value
        predicted_idx = int(np.argmax(prob_distribution))
        prediction = INDEX_TO_EMOTION.get(predicted_idx, 'neutral')

        # Log visual evaluation blocks for the first 15 entries to track outputs
        if i < 15:
            print(f"Validation Sample {i + 1}")
            print(f"Predicted: {prediction}")
            print(f"Actual   : {actual}")
            print("="*50)

        # Evaluate correctness tracking
        if prediction == actual:
            correct_predictions += 1
            class_metrics[actual]["tp"] += 1
        else:
            class_metrics[prediction]["fp"] += 1
            if actual in class_metrics:
                class_metrics[actual]["fn"] += 1
                
        if actual in class_metrics:
            class_metrics[actual]["total"] += 1

    # Compute final metric scores
    final_accuracy = (correct_predictions / total_samples) * 100
    print(f"\nFINAL VALIDATION ACCURACY: {final_accuracy:.2f}%")
    print("="*50)
    
    print(f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
    print("-" * 50)
    
    for emotion in INDEX_TO_EMOTION.values():
        tp = class_metrics[emotion]["tp"]
        fp = class_metrics[emotion]["fp"]
        fn = class_metrics[emotion]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{emotion:<10}{precision:<12.4f}{recall:<12.4f}{f1:<12.4f}")
        
    print("="*50)
    print("VALIDATION EVALUATION COMPLETE")

if __name__ == '__main__':
    main()