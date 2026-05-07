"""
Colab-Compatible Training Script for CNN-LSTM Rainfall Prediction
Supports both Google Colab and local machines
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# COLAB DETECTION & PATH SETUP
# ============================================================================
try:
    import google.colab
    IN_COLAB = True
    print("[✓] Running on Google Colab")
except ImportError:
    IN_COLAB = False
    print("[·] Running locally")

# Set base directory
if IN_COLAB:
    BASE_DIR = '/content/cnn_lstm_rainfall_system'
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

# Create directories
os.makedirs(os.path.join(BASE_DIR, 'checkpoints'), exist_ok=True)

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[✓] CUDA available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[·] Using CPU (GPU not available)")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class RainfallCNNLSTM(nn.Module):
    """CNN-LSTM model for rainfall prediction"""
    
    def __init__(self, seq_len=24, n_features=6, n_subdivisions=36):
        super(RainfallCNNLSTM, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        
        # CNN layers
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, 
                           batch_first=True, dropout=0.3)
        
        # Fully connected
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Transpose from (batch, seq_len, features) to (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # CNN
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Transpose back to (batch, features, seq_len) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last output
        
        # FC
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train(epochs=80, batch_size=256, lr=3e-3, weight_decay=1e-4, 
          patience=15, seq_len=24, split_ratio="80:20"):
    """
    Train the CNN-LSTM model
    
    Parameters:
    - epochs: Number of training epochs
    - batch_size: Batch size
    - lr: Learning rate
    - weight_decay: L2 regularization
    - patience: Early stopping patience
    - seq_len: Sequence length
    - split_ratio: Train:test ratio (e.g., "80:20")
    """
    
    print("\n" + "="*80)
    print(f"  COLAB TRAINING - Split Ratio: {split_ratio}")
    print("="*80)
    
    # Import data pipeline
    from colab_data_pipeline import build_dataset
    
    # Load data
    print("\n[·] Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, subdivisions = \
        build_dataset(seq_len=seq_len, train_test_ratio=split_ratio, fit_new_scaler=False)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = RainfallCNNLSTM(seq_len=seq_len, n_features=6, n_subdivisions=len(subdivisions))
    model.to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\n[·] Starting training: {epochs} epochs, batch_size={batch_size}")
    print(f"[·] Patience for early stopping: {patience}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t).item()
        
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'split_ratio': split_ratio
            }
            checkpoint_path = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pt')
            torch.save(checkpoint, checkpoint_path)
        else:
            patience_counter += 1
        
        # Progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n[✓] Early stopping at epoch {epoch+1}")
            break
    
    # Save training history
    history = {
        'epochs': len(train_losses),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'split_ratio': split_ratio
    }
    
    history_path = os.path.join(BASE_DIR, 'checkpoints', 'train_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[✓] Training complete!")
    print(f"[✓] Best model saved to: {checkpoint_path}")
    print(f"[✓] Training history saved to: {history_path}")
    
    return model, history

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN-LSTM rainfall model on Colab')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--split_ratio', type=str, default='80:20', 
                       help='Train:test split ratio')
    
    args = parser.parse_args()
    
    try:
        model, history = train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            split_ratio=args.split_ratio
        )
        print("\n[✓] All done!")
    except Exception as e:
        print(f"\n[✗] Error: {str(e)}")
        import traceback
        traceback.print_exc()
