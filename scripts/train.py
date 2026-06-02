import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import JesterCoordinateDataset
from model import GestureLSTM, GestureGRU  # Ensure GRU is imported
from paths import ROOT

# This function is used to train and validate the gesture models
def train_and_validate(epochs, learning_rate, batch_size, hidden_size, model_type):
    # Use GPU if available, otherwise fallback to use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Booting up {model_type} training engine on: {device}")

    # Directory Routing Setup to save the models and metrics to the correct location
    models_dir = ROOT / "models"
    metrics_dir = ROOT / "metrics"
    models_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    # Load the datasets
    train_dataset = JesterCoordinateDataset(
        coordinates_csv="data/jester_hand_coordinates.csv",
        labels_csv="annotations/jester-v1-train.csv",
        class_list_csv="annotations/jester-v1-labels.csv",
        max_seq_length=30
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = JesterCoordinateDataset(
        coordinates_csv="data/jester_hand_coordinates.csv",
        labels_csv="annotations/jester-v1-validation.csv",
        class_list_csv="annotations/jester-v1-labels.csv",
        max_seq_length=30
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model based on the model type
    if model_type.upper() == 'GRU':
        model = GestureGRU(input_size=63, hidden_size=hidden_size, num_layers=2, num_classes=27).to(device)
    else:
        model = GestureLSTM(input_size=63, hidden_size=hidden_size, num_layers=2, num_classes=27).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # History Tracking & Early Stopping to track the training and validation loss and accuracy
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    # Default best validation loss set to infinity
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0

    print(f"\n--- {model_type} | {epochs} Epochs | LR: {learning_rate} | Batch: {batch_size} | Hidden: {hidden_size} ---")

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        # --- RECORD METRICS ---
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

        # --- EARLY STOPPING & SAVING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            model_path = models_dir / f"best_{model_type.lower()}_h{hidden_size}_lr{learning_rate}.pth"
            torch.save(model.state_dict(), model_path)
            print("  -> Validation loss improved. Saved best model.")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement. Early stopping counter: {epochs_without_improvement}/{patience}")
            if epochs_without_improvement >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}.")
                break 

    # Save the history to the correct location
    history_path = metrics_dir / f"history_{model_type.lower()}_h{hidden_size}_lr{learning_rate}.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
        
    print(f"Training and validation complete. Check /models and /metrics directories.")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'GRU'])
    args = parser.parse_args()
    # Train and validate the gesture models
    train_and_validate(args.epochs, args.lr, args.batch_size, args.hidden_size, args.model_type)