import itertools
import subprocess
import time
import sys
from datetime import datetime

# Import your fixed absolute root path
from paths import ROOT 

def run_sweep():
    learning_rates = [0.001, 0.0005, 0.0001]
    hidden_sizes = [128, 256]
    batch_sizes = [32, 64]
    model_types = ['LSTM', 'GRU']
    max_epochs = 50 
    
    experiments = list(itertools.product(learning_rates, hidden_sizes, batch_sizes, model_types))
    total_runs = len(experiments)
    
    # 1. Create the logs directory using your absolute ROOT path
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # 2. Create a timestamped log file for this specific sweep session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = logs_dir / f"sweep_{timestamp}.log"
    
    print(f"Starting Hyperparameter Sweep: {total_runs} total experiments queued.")
    print(f"Detailed epoch logs are being written to: logs/sweep_{timestamp}.log\n")
    
    train_script_path = str(ROOT / "scripts" / "train.py")
    
    # 3. Open the log file
    with open(log_file_path, "w") as log_file:
        log_file.write(f"--- SWEEP INITIATED AT {timestamp} ---\n")
        log_file.write(f"Total Experiments: {total_runs}\n\n")
        
        for idx, (lr, hidden, batch, m_type) in enumerate(experiments):
            msg = f"Experiment {idx+1}/{total_runs} | Model: {m_type} | LR: {lr} | Hidden: {hidden} | Batch: {batch}"
            
            # Print a clean summary to your Mac's Terminal
            print("="*60)
            print(msg + " (Running...)")
            print("="*60)
            
            # Write the header to the log file
            log_file.write("="*80 + "\n")
            log_file.write(msg + "\n")
            log_file.write("="*80 + "\n")
            log_file.flush() # Force write to disk immediately
            
            start_time = time.time()
            
            # 4. Run train.py, but pipe ALL output (stdout and stderr) into the log file!
            subprocess.run([
                sys.executable, train_script_path, 
                "--epochs", str(max_epochs), 
                "--lr", str(lr), 
                "--batch_size", str(batch), 
                "--hidden_size", str(hidden),
                "--model_type", str(m_type)
            ], cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT)
            
            elapsed = (time.time() - start_time) / 60
            finish_msg = f">>> Experiment {idx+1} finished in {elapsed:.2f} minutes.\n"
            
            # Update the Terminal and the Log File
            print(finish_msg)
            log_file.write(f"\n{finish_msg}\n")
            log_file.flush()

if __name__ == "__main__":
    run_sweep()