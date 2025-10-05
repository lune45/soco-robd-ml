import numpy as np
import torch


def compute_cost_np(x_seq: np.ndarray, y_seq: np.ndarray, m: float = 5.0):
    """
    按题意计算单条序列的 hitting / switching / total 成本（numpy 版本）。
    x_seq, y_seq: 一维数组，长度相同。
    返回: hitting, switching, total（标量）
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


def predict_with_teacher_forcing(model: torch.nn.Module,
                                 y_seq: np.ndarray,
                                 x_teacher_seq: np.ndarray,
                                 device: torch.device = torch.device("cpu")):
    """
    使用 teacher forcing 的方式进行整段预测：特征为 [x_{t-1}^{teacher}, y_t]。
    返回预测的 x 序列（numpy，一维）。
    """
    T = len(y_seq)
    x_tf = x_teacher_seq.reshape(-1)
    y_flat = y_seq.reshape(-1)

    # 构造 (1, T, 2) 的输入张量
    X = np.zeros((1, T, 2), dtype=np.float32)
    for t in range(T):
        prev_x = x_tf[t - 1] if t > 0 else 0.0
        X[0, t, 0] = prev_x
        X[0, t, 1] = y_flat[t]

    x_tensor = torch.from_numpy(X).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor)  # (1, T, 1)
    pred_np = pred.cpu().numpy().reshape(-1)
    return pred_np


def predict_autoregressive(model: torch.nn.Module,
                            y_seq: np.ndarray,
                            device: torch.device = torch.device("cpu")):
    """
    自回归整段预测：每步输入 [x_{t-1}^{model}, y_t]，t=0 令 x_{-1}=0。
    返回 numpy 一维预测序列。
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
