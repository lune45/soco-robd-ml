import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim=32, num_layers=1, dropout=0.0):
        """
        input_dim：每个时间步的特征维度。表示 [x_{t-1}, y_t]。
        hidden_dim：LSTM隐状态维度。
        num_layers：LSTM堆叠层数。
        dropout：当 num_layers > 1 时，层间 dropout 概率。
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1) # out put x_t

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x) # out: (B, T, hidden_dim)
        return self.fc(out) # (batch, seq_Len, 1)

    def forward_step(self, x_t, hidden=None):
        """
        单步前向：
        x_t: (batch, 1, input_dim)，表示 [x_{t-1}^{model}, y_t]
        hidden: 上一步的 (h, c) 隐状态，或 None
        返回：x_t_pred: (batch, 1, 1) 以及新的 hidden
        """
        out, hidden = self.lstm(x_t, hidden)
        x_t_pred = self.fc(out)
        return x_t_pred, hidden
