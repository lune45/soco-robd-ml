import numpy as np
import torch

from src.train import compute_cost_loss


def mla_robd_single_sequence(y_seq, x_ml_pred, m, la1, la2, la3):
    """
    MLA-ROBD (Algorithm 1) per-sequence calibration:
    x_t = argmin_x [ f(x,y_t) + λ1 c(x,x_{t-1}) + λ2 c(x,v_t) + λ3 c(x, x~_t) ]
      where f(x,y) = (m/2)(x-y)^2, c(a,b)=(a-b)^2, v_t=y_t, and x~_t is the ML prediction.
    Closed form: x_t = ((m+2λ2) y_t + 2λ1 x_{t-1} + 2λ3 x~_t) / (m + 2λ1 + 2λ2 + 2λ3)
    """
    T = len(y_seq)
    x_hybrid = np.zeros(T, dtype=float)
    x_prev = 0.0
    denom = m + 2.0 * la1 + 2.0 * la2 + 2.0 * la3
    for t in range(T):
        y_t = float(y_seq[t])
        x_ml_t = float(x_ml_pred[t])
        x_t = ((m + 2.0 * la2) * y_t + 2.0 * la1 * x_prev + 2.0 * la3 * x_ml_t) / denom
        x_hybrid[t] = x_t
        x_prev = x_t
    return x_hybrid


def mla_robd_rollout_torch(model: torch.nn.Module,
                           y_batch: torch.Tensor,
                           m: float,
                           la1: float,
                           la2: float,
                           la3: float,
                           device: torch.device = torch.device("cpu")):
    """
    Differentiable MLA-ROBD: batch-level autoregression + closed-form calibration, returns (x_ml, x_cal).
    y_batch: (B,T,1)
    """
    model = model.to(device)
    B, T, _ = y_batch.shape
    y_batch = y_batch.to(device)

    x_ml_all = torch.zeros(B, T, 1, dtype=torch.float32, device=device)
    x_cal_all = torch.zeros(B, T, 1, dtype=torch.float32, device=device)

    x_ml_tm1 = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
    x_cal_tm1 = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
    hidden = None
    denom = m + 2.0 * la1 + 2.0 * la2 + 2.0 * la3

    for t in range(T):
        y_t = y_batch[:, t:t+1, :]  # (B,1,1)
        inp = torch.cat([x_ml_tm1, y_t], dim=2)  # (B,1,2)
        x_ml_t, hidden = model.forward_step(inp, hidden)  # (B,1,1)
        x_ml_all[:, t:t+1, :] = x_ml_t

        # v_t = y_t, closed-form calibration
        x_cal_t = ((m + 2.0 * la2) * y_t + 2.0 * la1 * x_cal_tm1 + 2.0 * la3 * x_ml_t) / denom
        x_cal_all[:, t:t+1, :] = x_cal_t

        x_ml_tm1 = x_ml_t
        x_cal_tm1 = x_cal_t

    return x_ml_all, x_cal_all


def train_model_mla_robd(model: torch.nn.Module,
                          train_loader,
                          val_loader=None,
                          *,
                          m: float = 5.0,
                          la1: float = 0.5,
                          la2: float = 0.0,
                          la3: float = 1.0,
                          lr: float = 1e-3,
                          weight_decay: float = 1e-4,
                          grad_clip: float = 1.0,
                          patience: int = 10,
                          min_delta: float = 0.0,
                          use_scheduler: bool = True,
                          device: torch.device = torch.device("cpu")):
    """
    Train-time fusion: optimize model parameters on the task loss (hitting+switching) using calibrated x_cal;
    no fixed epochs — early stopping ends training.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        if val_loader is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(2, patience//3))
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val = float('inf')
    best_state = None
    bad_epochs = 0
    train_losses, val_losses = [], []

    ep = 0
    while True:
        ep += 1
        model.train()
        ep_losses = []
        for (yb,) in train_loader:
            yb = yb.to(device)
            optimizer.zero_grad()
            _, x_cal = mla_robd_rollout_torch(model, yb, m=m, la1=la1, la2=la2, la3=la3, device=device)
            loss = compute_cost_loss(x_cal, yb, m=m)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ep_losses.append(loss.detach().cpu().item())

        train_loss = float(np.mean(ep_losses)) if len(ep_losses) > 0 else 0.0
        train_losses.append(train_loss)

        val_loss = None
        if val_loader is not None:
            model.eval()
            v_losses = []
            with torch.no_grad():
                for (yv,) in val_loader:
                    yv = yv.to(device)
                    _, x_cal_v = mla_robd_rollout_torch(model, yv, m=m, la1=la1, la2=la2, la3=la3, device=device)
                    v_loss = compute_cost_loss(x_cal_v, yv, m=m)
                    v_losses.append(v_loss.detach().cpu().item())
            val_loss = float(np.mean(v_losses)) if len(v_losses) > 0 else None
            if use_scheduler:
                scheduler.step(val_loss)
        else:
            if use_scheduler:
                scheduler.step()

        if val_loss is None:
            print(f"Epoch {ep}, loss={train_loss:.4f}")
        else:
            print(f"Epoch {ep}, loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            val_losses.append(val_loss)

            if best_val - val_loss > min_delta:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


