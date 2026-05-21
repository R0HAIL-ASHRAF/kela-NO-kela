import numpy as np
import os
import pickle
from lstm_model import LSTM

def load_dataset():
    print("Loading raw emotion sequence arrays from extractor...")
    if not os.path.exists("data/X_raw.npy") or not os.path.exists("data/y_raw.npy"):
        raise FileNotFoundError("Missing raw matrices. Run extract_features.py first!")
        
    X_raw = np.load("data/X_raw.npy")  # (864, 173, 40)
    y_raw = np.load("data/y_raw.npy")  # (864,)
    
    # 1. Generate an absolute, reproducible shuffle map
    num_samples = X_raw.shape[0]
    indices = np.arange(num_samples)
    np.random.seed(42)  # Lock the seed so your test split is completely stable
    np.random.shuffle(indices)
    
    # 2. Extract exactly 20% for a permanent Test/Validation split
    split_idx = int(num_samples * 0.80)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train_raw, y_train = X_raw[train_idx], y_raw[train_idx]
    X_test_raw, y_test = X_raw[test_idx], y_raw[test_idx]
    
    # 3. Save the test arrays to disk so test.py can find them later
    np.save("data/X_val.npy", X_test_raw)
    np.save("data/y_val.npy", y_test)
    print(f"--> Successfully isolated and saved {X_test_raw.shape[0]} unseen samples to 'data/X_val.npy'")
    
    # 4. Standardize training channels independently (No Data Leakage)
    print("Standardizing training features...")
    mean = np.mean(X_train_raw, axis=(0, 1), keepdims=True)
    std = np.std(X_train_raw, axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train_raw - mean) / std
    
    # Pre-scale our validation split using the exact training parameters
    X_val = (X_test_raw - mean) / std
    y_val = y_test
    
    print(f"\nFinal Training Matrix Shape: {X_train.shape} ({X_train.shape[0]} samples)")
    print(f"Final Validation Matrix Shape: {X_val.shape} ({X_val.shape[0]} samples)")
    
    # Return everything cleanly to main
    return X_train, y_train, X_val, y_val, mean, std

def compute_loss_and_accuracy(model, X_data, y_data):
    """
    Computes cross-entropy loss and categorical accuracy over an evaluation set.
    """
    total_loss = 0.0
    correct_predictions = 0
    num_samples = X_data.shape[0]
    
    for i in range(num_samples):
        x_sample = X_data[i]
        true_idx = int(y_data[i])
        
        # Forward pass to get logits
        logits = model.forward(x_sample)
        
        # Flatten the output to a clean 1D vector of shape (5,) to protect against index errors
        logits = logits.flatten()
        
        # Compute Stable Softmax Probabilities
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        
        # Cross-Entropy Loss indexing safely from a 1D array
        total_loss -= np.log(probs[true_idx] + 1e-15)
        
        # Track accuracy
        if np.argmax(probs) == true_idx:
            correct_predictions += 1
            
    avg_loss = total_loss / num_samples
    accuracy_pct = (correct_predictions / num_samples) * 100.0
    return avg_loss, accuracy_pct

def main():
    # 1. Gather Data Elements and unpacking scaling matrices properly
    X_train, y_train, X_val, y_val, train_mean, train_std = load_dataset()
    
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    print(f"X_train standardized shape: {X_train.shape}\n")

    # 2. Initialize Model Structure
    input_dim = X_train.shape[2]   # 40 feature tracks
    hidden_dim = 64                # Hidden state capacity
    output_dim = 5                 # 5 target categories
    
    model = LSTM(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
    
    # Hyperparameters
    MAX_EPOCHS = 50
    LEARNING_RATE = 0.0001         # Adjusted to 1e-4 to slow down convergence and help learning
    
    # Convergence Check Tracker States
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5                   # Halt if validation loss stalls for 5 straight runs
    patience_counter = 0

    print("--- BEGINNING MODEL TRAINING LOOP (ADAM + CONVERGENCE MONITORING) ---")
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle training indices every epoch to diversify pattern presentation paths
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        # Core Training Phase Loop
        for idx in indices:
            x_sample = X_train[idx]
            y_sample_idx = int(y_train[idx])
            
            # Forward and backward pass tracking weights optimizations
            y_pred = model.forward(x_sample)
            model.backward(y_pred, y_sample_idx, lr=LEARNING_RATE)

        # Evaluate progress over training and validation checkpoints
        train_loss, train_acc = compute_loss_and_accuracy(model, X_train, y_train)
        val_loss, val_acc = compute_loss_and_accuracy(model, X_val, y_val)
        
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} -> "
              f"Train Loss: {train_loss:.4f} [Acc: {train_acc:.2f}%] | "
              f"Val Loss: {val_loss:.4f} [Acc: {val_acc:.2f}%]")

        # 3. 🛑 Convergence & Early Stopping Verification
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Snapshot deepcopy of the entire optimized model state
            best_model_state = pickle.dumps(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Convergence Reached] Validation loss stopped falling for {patience} epochs.")
                print(f"Halting execution early at Epoch {epoch} to prevent model overfitting.")
                break

    # Save the optimal parameter states found during convergence steps
    print("\n" + "="*60)
    if best_model_state is not None:
        print("Restoring and exporting absolute lowest validation loss model configuration...")
        model = pickle.loads(best_model_state)
        
    # Bundle the model parameters AND our training scaling vectors into one file payload
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

if __name__ == '__main__':
    main()