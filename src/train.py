import torch
import torch.nn as nn

def compute_cost_loss(x_pred, y_true, m=5.0):
    """
    x_pred: model-predicted decisions x, shape (batch, seq_len, 1)
    y_true: observed sequence y (target for hitting), shape (batch, seq_len, 1)
    As specified:
      hitting = (m / 2) * (x_pred - y_true)^2
      switching = 0.5 * (x_t - x_{t-1})^2
    Total loss is the sum (means over batch/time); switching uses time-differences.
    """
    hitting = (m / 2.0) * ((x_pred - y_true) ** 2).mean()
    switching = 0.5 * ((x_pred[:, 1:, :] - x_pred[:, :-1, :]) ** 2).mean()
    return hitting + switching

def train_model(model, train_loader, val_loader=None, epochs=10, lr=1e-3, m=5.0, weight_decay=1e-4, grad_clip=1.0, patience=10, use_scheduler=True):
    """
    Train a model with:
    - Validation and early stopping (patience)
    - Gradient clipping (grad_clip)
    - LR scheduling (ReduceLROnPlateau with val, else StepLR)
    - Weight decay (weight_decay)

    Returns: model, train_losses, val_losses (best weights loaded if val provided)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        if val_loader is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    train_losses, val_losses = [], []
    best_val = float('inf') if val_loader is not None else None
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = compute_cost_loss(pred, yb, m)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        train_losses.append(avg)

        if val_loader is not None:
            model.eval()
            v_total = 0.0
            with torch.no_grad():
                for xvb, yvb in val_loader:
                    v_pred = model(xvb)
                    v_loss = compute_cost_loss(v_pred, yvb, m)
                    v_total += v_loss.item()
            v_avg = v_total / len(val_loader)
            val_losses.append(v_avg)
            print(f"Epoch {ep+1}, loss={avg:.4f}, val_loss={v_avg:.4f}")

            # Track the best validation score
            if v_avg < best_val - 1e-6:
                best_val = v_avg
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {ep+1}")
                    break

            # Scheduler: step based on validation loss
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(v_avg)
        else:
            print(f"Epoch {ep+1}, loss={avg:.4f}")
            # Use StepLR when no validation set is provided
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def train_model_autoregressive(model, train_loader, val_loader=None, epochs=10, lr=1e-3, m=5.0, weight_decay=1e-4, grad_clip=1.0, patience=10, use_scheduler=True):
    """
    Autoregressive training: input is [x_{t-1}^{model}, y_t], with x_{-1}=0 at t=0.
    DataLoader must provide y sequences of shape (B,T,1).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        if val_loader is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    train_losses, val_losses = [], []
    best_val = float('inf') if val_loader is not None else None
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            yb = batch[1] if isinstance(batch, (list, tuple)) and len(batch)==2 else (batch[0] if isinstance(batch, (list, tuple)) else batch)
            B, T, _ = yb.shape
            optimizer.zero_grad()

            hidden = None
            x_tm1 = torch.zeros(B, 1, 1, dtype=yb.dtype, device=yb.device)
            preds = []
            for t in range(T):
                y_t = yb[:, t:t+1, :]
                inp = torch.cat([x_tm1, y_t], dim=2)
                pred_t, hidden = model.forward_step(inp, hidden)
                preds.append(pred_t)
                x_tm1 = pred_t.detach()
            x_pred = torch.cat(preds, dim=1)

            loss = compute_cost_loss(x_pred, yb, m)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        train_losses.append(avg)

        if val_loader is not None:
            model.eval()
            v_total = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    yv = vbatch[1] if isinstance(vbatch, (list, tuple)) and len(vbatch)==2 else (vbatch[0] if isinstance(vbatch, (list, tuple)) else vbatch)
                    B, T, _ = yv.shape
                    hidden = None
                    x_tm1 = torch.zeros(B, 1, 1, dtype=yv.dtype, device=yv.device)
                    preds = []
                    for t in range(T):
                        y_t = yv[:, t:t+1, :]
                        inp = torch.cat([x_tm1, y_t], dim=2)
                        pred_t, hidden = model.forward_step(inp, hidden)
                        preds.append(pred_t)
                        x_tm1 = pred_t
                    x_val = torch.cat(preds, dim=1)
                    v_total += compute_cost_loss(x_val, yv, m).item()
            v_avg = v_total / len(val_loader)
            val_losses.append(v_avg)
            print(f"Epoch {ep+1}, loss={avg:.4f}, val_loss={v_avg:.4f}")
            if v_avg < best_val - 1e-6:
                best_val = v_avg
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {ep+1}")
                    break
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(v_avg)
        else:
            print(f"Epoch {ep+1}, loss={avg:.4f}")
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses