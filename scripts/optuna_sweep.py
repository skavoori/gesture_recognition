import argparse
from datetime import datetime

import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import JesterCoordinateDataset
from model import GestureLSTM, GestureGRU  # Ensure GRU is imported
from paths import ROOT
from torch.utils.data import DataLoader

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def objective(trial, epochs, log):
    # 1. Optuna Suggests Hyperparameters
    model_type = trial.suggest_categorical("model_type", ["LSTM", "GRU"])
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256])
    # Optuna can search a continuous learning rate space logarithmically!
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True) 
    log(
        f"Trial {trial.number} started | model_type={model_type} | "
        f"hidden_size={hidden_size} | lr={lr:.8f}"
    )

    # 2. Initialize Model
    if model_type == "GRU":
        model = GestureGRU(input_size=63, hidden_size=hidden_size, num_layers=2, num_classes=27).to(DEVICE)
    else:
        model = GestureLSTM(input_size=63, hidden_size=hidden_size, num_layers=2, num_classes=27).to(DEVICE)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Load Datasets
    train_dataset = JesterCoordinateDataset(
        coordinates_csv="data/jester_hand_coordinates.csv",
        labels_csv="annotations/jester-v1-train.csv",
        class_list_csv="annotations/jester-v1-labels.csv",
        max_seq_length=30
    )
    val_dataset = JesterCoordinateDataset(
        coordinates_csv="data/jester_hand_coordinates.csv",
        labels_csv="annotations/jester-v1-validation.csv",
        class_list_csv="annotations/jester-v1-labels.csv",
        max_seq_length=30
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        # --- TRAIN LOOP ---
        model.train()
        train_loss = 0.0
        total_train = 0
        correct_train = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        total_val = 0
        correct_val = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train if total_train else 0.0
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val if total_val else 0.0
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        # Let Optuna prune weak trials early.
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        log(
            f"Trial {trial.number} | Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

    log(
        f"Trial {trial.number} completed | best_val_accuracy={best_val_accuracy:.4f}"
    )
    return best_val_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna sweep for gesture models.")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials to run.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per trial.")
    args = parser.parse_args()

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"optuna_sweep_{timestamp}.log"

    def log(message: str) -> None:
        print(message)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    log("Starting Advanced HPO with Optuna...")
    log(f"Writing detailed training logs to: {log_path}")
    log(f"Device: {DEVICE}")
    log(f"Requested trials: {args.n_trials} | epochs per trial: {args.epochs}")
    # Tree-structured Parzen Estimator (TPE) + Median Pruner for early stopping
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(lambda trial: objective(trial, args.epochs, log), n_trials=args.n_trials)

    log("\n" + "="*50)
    log("🏆 OPTUNA BAYESIAN OPTIMIZATION RESULTS 🏆")
    log("="*50)
    log(f"Best Trial Score: {study.best_value:.4f}")
    log("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        log(f"    {key}: {value}")