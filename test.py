import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    model_path = 'lstm_model.pkl'

    print("Model loading check...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find trained weights at '{model_path}'. Run train.py first!")

    with open(model_path, 'rb') as f:
        payload = pickle.load(f)

    model = payload['model']
    train_mean = payload['mean']
    train_std = payload['std']
    print("Model and standardization parameters loaded successfully!")

    X_val_path = "data/X_val.npy"
    y_val_path = "data/y_val.npy"

    if not os.path.exists(X_val_path) or not os.path.exists(y_val_path):
        raise FileNotFoundError("Missing test files in 'data/'. Please rerun train.py to generate them.")

    X_val_raw = np.load(X_val_path)
    y_val = np.load(y_val_path)

    print(f"Loaded {X_val_raw.shape[0]} completely unseen validation samples for testing.")
    print(f"X shape: {X_val_raw.shape}")
    print(f"y shape: {y_val.shape}\n")

    X_val = (X_val_raw - train_mean) / train_std

    classes = ['angry', 'sad', 'happy', 'neutral', 'fear']

    y_true_list = []
    y_pred_list = []

    print("Running validation predictions...")
    print("=" * 50)

    for i in range(len(X_val)):
        sample = X_val[i]
        actual_class_idx = int(y_val[i])

        if hasattr(model, 'predict'):
            prob_distribution = model.predict(sample)
        else:
            prob_distribution = model.forward(sample)

        prob_distribution = prob_distribution.flatten()
        predicted_class_idx = np.argmax(prob_distribution)

        y_true_list.append(actual_class_idx)
        y_pred_list.append(predicted_class_idx)

        if i < 15:
            print(f"Validation Sample {i+1}")
            print(f"Predicted: {classes[predicted_class_idx]}")
            print(f"Actual   : {classes[actual_class_idx]}")
            print("-" * 50)

    print("\nFINAL VALIDATION METRICS REPORT")
    print("=" * 50)
    print(f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
    print("-" * 50)

    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)

    total_correct = np.sum(y_true_arr == y_pred_arr)
    final_accuracy = (total_correct / len(y_val)) * 100.0

    for class_idx, class_name in enumerate(classes):
        tp = np.sum((y_true_arr == class_idx) & (y_pred_arr == class_idx))
        fp = np.sum((y_true_arr != class_idx) & (y_pred_arr == class_idx))
        fn = np.sum((y_true_arr == class_idx) & (y_pred_arr != class_idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{class_name:<10}{precision:<12.4f}{recall:<12.4f}{f1_score:<12.4f}")

    print("-" * 50)
    print(f"FINAL VALIDATION ACCURACY: {final_accuracy:.2f}%")
    print("=" * 50)
    print("VALIDATION EVALUATION COMPLETE")

    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_idx, pred_idx in zip(y_true_list, y_pred_list):
        cm[true_idx][pred_idx] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix  (Accuracy: {final_accuracy:.2f}%)')

    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nConfusion matrix saved to '{save_path}'")

if __name__ == '__main__':
    main()
