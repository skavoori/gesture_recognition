import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import JesterCoordinateDataset
from model import GestureLSTM
from paths import ROOT, resolve_path

def train_and_validate(epochs, learning_rate, batch_size, hidden_size):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Booting up unified training/validation engine on: {device}")

    # 1. Load BOTH Datasets
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

    # 2. Initialize Model
    model = GestureLSTM(input_size=63, hidden_size=hidden_size, num_layers=2, num_classes=27).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3. History Tracking for Jupyter Notebook Graphs
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    print(f"--- Starting Run: {epochs} Epochs | LR: {learning_rate} | Batch: {batch_size} ---")

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

    # 4. Save the model and the history (project root so notebooks can load them)
    model_path = ROOT / f"gesture_lstm_h{hidden_size}_lr{learning_rate}.pth"
    history_path = ROOT / f"history_h{hidden_size}_lr{learning_rate}.json"
    torch.save(model.state_dict(), model_path)
    with open(history_path, "w") as f:
        json.dump(history, f)
        
    print("Run complete. Weights and history saved.")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=128)
    args = parser.parse_args()
    
    train_and_validate(args.epochs, args.lr, args.batch_size, args.hidden_size)