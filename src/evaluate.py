import numpy as np
import torch


def compute_cost_np(x_seq: np.ndarray, y_seq: np.ndarray, m: float = 5.0):
    """
    Compute hitting / switching / total cost per sequence (NumPy version).
    x_seq, y_seq: 1D arrays with equal length.
    Return: hitting, switching, total (scalars)
    """
    x_seq = x_seq.reshape(-1)
    y_seq = y_seq.reshape(-1)
    hitting = (m / 2.0) * np.sum((x_seq - y_seq) ** 2)
    if len(x_seq) > 1:
        switching = 0.5 * np.sum((x_seq[1:] - x_seq[:-1]) ** 2)
    else:
        switching = 0.0
    total = hitting + switching
    return float(hitting), float(switching), float(total)


def predict_autoregressive(model: torch.nn.Module,
                            y_seq: np.ndarray,
                            device: torch.device = torch.device("cpu")):
    """
    Autoregressive prediction over the whole sequence: input per step is [x_{t-1}^{model}, y_t],
    with x_{-1}=0 at t=0. Returns a 1D NumPy array.
    """
    T = len(y_seq)
    model = model.to(device)
    model.eval()
    y = torch.tensor(y_seq.reshape(1, T, 1), dtype=torch.float32, device=device)
    x_tm1 = torch.zeros(1, 1, 1, dtype=torch.float32, device=device)
    hidden = None
    preds = []
    with torch.no_grad():
        for t in range(T):
            y_t = y[:, t:t+1, :]
            inp = torch.cat([x_tm1, y_t], dim=2)
            pred_t, hidden = model.forward_step(inp, hidden)
            preds.append(pred_t)
            x_tm1 = pred_t
    x_pred = torch.cat(preds, dim=1).cpu().numpy().reshape(-1)
    return x_pred
