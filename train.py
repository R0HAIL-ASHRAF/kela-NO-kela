import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from lstm_model import LSTM

def load_dataset():
    print("Loading raw emotion sequence arrays from extractor...")
    if not os.path.exists("data/X_raw.npy") or not os.path.exists("data/y_raw.npy"):
        raise FileNotFoundError("Missing raw matrices. Run extract_features.py first!")

    X_raw = np.load("data/X_raw.npy")
    y_raw = np.load("data/y_raw.npy")

    num_samples = X_raw.shape[0]
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(num_samples * 0.80)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train_raw, y_train = X_raw[train_idx], y_raw[train_idx]
    X_test_raw, y_test = X_raw[test_idx], y_raw[test_idx]

    np.save("data/X_val.npy", X_test_raw)
    np.save("data/y_val.npy", y_test)
    print(f"--> Successfully isolated and saved {X_test_raw.shape[0]} unseen samples to 'data/X_val.npy'")

    print("Standardizing training features...")
    mean = np.mean(X_train_raw, axis=(0, 1), keepdims=True)
    std = np.std(X_train_raw, axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train_raw - mean) / std

    X_val = (X_test_raw - mean) / std
    y_val = y_test

    print(f"\nFinal Training Matrix Shape: {X_train.shape} ({X_train.shape[0]} samples)")
    print(f"Final Validation Matrix Shape: {X_val.shape} ({X_val.shape[0]} samples)")

    return X_train, y_train, X_val, y_val, mean, std

def compute_loss_and_accuracy(model, X_data, y_data):
    total_loss = 0.0
    correct_predictions = 0
    num_samples = X_data.shape[0]

    for i in range(num_samples):
        x_sample = X_data[i]
        true_idx = int(y_data[i])

        logits = model.predict(x_sample)

        logits = logits.flatten()

        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)

        total_loss -= np.log(probs[true_idx] + 1e-15)

        if np.argmax(probs) == true_idx:
            correct_predictions += 1

    avg_loss = total_loss / num_samples
    accuracy_pct = (correct_predictions / num_samples) * 100.0
    return avg_loss, accuracy_pct

def main():
    X_train, y_train, X_val, y_val, train_mean, train_std = load_dataset()

    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    print(f"X_train standardized shape: {X_train.shape}\n")

    input_dim = X_train.shape[2]
    hidden_dim = 128
    output_dim = 5

    model = LSTM(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim, dropout_rate=0.3)

    MAX_EPOCHS = 100
    LEARNING_RATE = 0.001
    L2_LAMBDA = 0.0001

    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print("--- BEGINNING MODEL TRAINING LOOP (ADAM + CONVERGENCE MONITORING) ---")

    for epoch in range(1, MAX_EPOCHS + 1):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        for idx in indices:
            x_sample = X_train[idx]
            y_sample_idx = int(y_train[idx])

            y_pred = model.forward(x_sample, training=True)
            model.backward(y_pred, y_sample_idx, lr=LEARNING_RATE, l2_lambda=L2_LAMBDA)

        train_loss, train_acc = compute_loss_and_accuracy(model, X_train, y_train)
        val_loss, val_acc = compute_loss_and_accuracy(model, X_val, y_val)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} -> "
              f"Train Loss: {train_loss:.4f} [Acc: {train_acc:.2f}%] | "
              f"Val Loss: {val_loss:.4f} [Acc: {val_acc:.2f}%]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = pickle.dumps(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Convergence Reached] Validation loss stopped falling for {patience} epochs.")
                print(f"Halting execution early at Epoch {epoch} to prevent model overfitting.")
                break

    print("\n" + "="*60)
    if best_model_state is not None:
        print("Restoring and exporting absolute lowest validation loss model configuration...")
        model = pickle.loads(best_model_state)

    model_export_path = 'lstm_model.pkl'
    payload = {
        'model': model,
        'mean': train_mean,
        'std': train_std
    }

    with open(model_export_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Model payload successfully exported inside '{model_export_path}'!")
    print("="*60)

    epochs_range = range(1, len(train_loss_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, train_loss_history, label='Train Loss')
    ax1.plot(epochs_range, val_loss_history, label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_acc_history, label='Train Accuracy')
    ax2.plot(epochs_range, val_acc_history, label='Val Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = 'loss_curve.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Loss and accuracy curves saved to '{save_path}'")

if __name__ == '__main__':
    main()
