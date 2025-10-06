import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim=32, num_layers=1, dropout=0.0):
        """
        input_dim: feature dimension per time step; represents [x_{t-1}, y_t].
        hidden_dim: LSTM hidden-state dimension.
        num_layers: number of stacked LSTM layers.
        dropout: inter-layer dropout probability when num_layers > 1.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1) # out put x_t

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x) # out: (B, T, hidden_dim)
        return self.fc(out) # (batch, seq_len, 1)

    def forward_step(self, x_t, hidden=None):
        """
        Single-step forward:
        x_t: (batch, 1, input_dim), represents [x_{t-1}^{model}, y_t]
        hidden: previous (h, c) hidden state, or None
        Returns: x_t_pred: (batch, 1, 1) and the new hidden
        """
        out, hidden = self.lstm(x_t, hidden)
        x_t_pred = self.fc(out)
        return x_t_pred, hidden
