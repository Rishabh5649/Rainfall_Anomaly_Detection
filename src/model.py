"""
model.py
CNN + LSTM hybrid architecture for monthly rainfall forecasting.

Architecture:
  Input:  (batch, seq_len, n_features)
  ┌──────────────────────────────────────────────────┐
  │  Conv Block 1: Conv1D(64, k=3) + BN + ReLU       │
  │  Conv Block 2: Conv1D(128, k=3) + BN + ReLU      │  ← residual skip
  │  Conv Block 3: Conv1D(256, k=3) + BN + ReLU      │
  └──────────────────────────────────────────────────┘
          ↓
  BiLSTM(256, layers=2, dropout=0.3)
          ↓  (last hidden state)
  FC(256) → ReLU → Dropout(0.2) → FC(128) → ReLU → FC(1)
          ↓
  Output: next-month normalised rainfall (scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """1D Conv → BatchNorm → ReLU with optional residual projection."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, use_residual: bool = True):
        super().__init__()
        pad = kernel_size // 2   # same-padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.use_residual = use_residual
        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity() if use_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, in_channels, seq_len)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out = out + self.shortcut(x)
        return F.relu(out)


class RainSightCNNLSTM(nn.Module):
    """
    CNN + BiLSTM hybrid for rainfall time-series forecasting.

    Parameters
    ----------
    n_features    : number of input features per time-step
    seq_len       : look-back window length (time-steps)
    cnn_channels  : output channels for each CNN block
    lstm_hidden   : BiLSTM hidden size (per direction)
    lstm_layers   : number of stacked LSTM layers
    dropout       : dropout probability used in LSTM + FC head
    """

    def __init__(
        self,
        n_features:   int   = 6,
        seq_len:      int   = 24,
        cnn_channels: tuple = (64, 128, 256),
        lstm_hidden:  int   = 256,
        lstm_layers:  int   = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.n_features  = n_features
        self.lstm_hidden = lstm_hidden

        # ── CNN backbone ──────────────────────────────────────────
        cnn_blocks = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_blocks.append(ConvBlock(in_ch, out_ch, kernel_size=3, use_residual=True))
            cnn_blocks.append(nn.Dropout(dropout * 0.5))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)
        self.cnn_out_channels = cnn_channels[-1]

        # ── BiLSTM ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = self.cnn_out_channels,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_out_size = lstm_hidden * 2   # bidirectional

        # ── Attention over LSTM outputs ───────────────────────────
        self.attention = nn.Linear(lstm_out_size, 1)

        # ── FC regression head ────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        returns : (batch,)  — predicted normalised rainfall
        """
        # ── CNN expects (batch, channels, seq_len) ────────────────
        x_cnn = x.permute(0, 2, 1)           # (B, F, T)
        cnn_out = self.cnn(x_cnn)             # (B, C, T)

        # ── Back to (batch, seq_len, channels) for LSTM ──────────
        lstm_in = cnn_out.permute(0, 2, 1)    # (B, T, C)

        # ── LSTM ──────────────────────────────────────────────────
        lstm_out, _ = self.lstm(lstm_in)      # (B, T, lstm_hidden*2)

        # ── Attention pooling ─────────────────────────────────────
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)                 # (B, lstm_hidden*2)

        # ── Regression head ───────────────────────────────────────
        out = self.head(context).squeeze(-1)   # (B,)
        return out

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Return GPU device if CUDA is available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[✓] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[·] CUDA not available — using CPU")
    return device


def build_model(
    n_features:  int   = 6,
    seq_len:     int   = 24,
    device:      torch.device = None,
) -> RainSightCNNLSTM:
    if device is None:
        device = get_device()
    model = RainSightCNNLSTM(n_features=n_features, seq_len=seq_len)
    model = model.to(device)
    print(f"[✓] Model built | {model.num_params:,} trainable parameters")
    return model


if __name__ == "__main__":
    device = get_device()
    model  = build_model(device=device)
    # Dummy forward pass
    batch = torch.randn(8, 24, 6).to(device)
    out   = model(batch)
    print(f"[✓] Forward pass OK | Output shape: {out.shape}")
