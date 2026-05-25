import torch
import torch.nn as nn

class GestureLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=27, dropout=0.5):
        super(GestureLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The core LSTM network
        # batch_first=True ensures our tensor shape is (Batch, Sequence, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer to prevent overfitting on the coordinate sequences
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer mapping the hidden state to the 27 gesture classes
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden and cell states dynamically based on batch size and device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the network's understanding at the very last time step 
        # to classify the completed gesture. 
        # out[:, -1, :] slices the tensor to grab the final hidden state.
        out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    # --- Apple Silicon Hardware Verification ---
    # This automatically detects your M5 Max's Metal GPU for accelerated training
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"PyTorch Compute Device: {device}")
    
    # Initialize the model and push it to the GPU
    model = GestureLSTM().to(device)
    
    # Create a dummy tensor representing 1 batch of 30 frames with 63 coordinates
    dummy_input = torch.randn(1, 30, 63).to(device)
    
    # Run a test inference
    dummy_output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}  -> (Batch, Sequence, Features)")
    print(f"Output Shape: {dummy_output.shape} -> (Batch, Classes)")