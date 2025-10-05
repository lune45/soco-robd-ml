import os
import numpy as np
import math
from src.preprocess import load_normalize, sequences
from src.robd import robd_single_sequence, compute_cost

# import dataset
data_path = os.path.join("data", "AI_workload.csv")

# load and normolize
y_norm, y_min, y_max = load_normalize(data_path)
print("[Task 1] loaded samples:", len(y_norm))

# cut time sequence
window_size = 24
step = 2
sequence = sequences(y_norm, window_size = window_size, step = step)
print("[Task 1] Number of sequences:", sequence.shape[0])

# R-OBD parameters
m = 5.0
la1 = 2.0 / (1.0 + math.sqrt(1.0 + 4.0 / m)) # 2.0 / (1.0 + math.sqrt(1.0 + (4.0 * beta**2) / (alpha * m))),  α=β=1
la2 = 0

# 注：为统一评估口径，以下“对全体序列运行 OBD 并保存成本”的代码已关闭。
# 现在仅在测试集上计算 OBD 基线成本（见 __main__ 中的评估代码）。
# 如需恢复，请取消以下注释。
#
# # run R-OBD (all sequences)
# costs = []
# for seq in sequence:
#     x = robd_single_sequence(seq, m = m, la1 = la1, la2 = la2)
#     c = compute_cost(x, seq, m = m)
#     costs.append(c)
# costs = np.array(costs)
# print("Average cost:", costs.mean())
# print("Median cost", np.median(costs))
#
# # save costs
# os.makedirs("results", exist_ok = True)
# np.savetxt("results/robd_cost_1.txt", costs)


if __name__ == "__main__":
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from src.ml_model import LSTMModel
    from src.train import train_model_autoregressive

    # 固定随机种子，保证复现性
    torch.manual_seed(42)
    np.random.seed(42)

    split = int(0.8 * len(sequence))
    train_seq = sequence[:split]
    test_seq = sequence[split:]

    # 再从训练中切出验证集（10%）
    val_split = int(0.9 * len(train_seq))
    train_seq_part = train_seq[:val_split]
    val_seq = train_seq[val_split:]

    # 训练集仅保留 y（自回归时运行期构造输入）
    Y_list = []
    for seq in train_seq_part:
        Y_list.append(seq.reshape(-1,1))

    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)

    # 构建验证集张量，仅保留 y
    VY_list = []
    for seq in val_seq:
        VY_list.append(seq.reshape(-1,1))

    VY_tensor = torch.tensor(np.array(VY_list), dtype=torch.float32) if len(VY_list)>0 else torch.empty(0)

    train_data = TensorDataset(Y_tensor)
    val_data = TensorDataset(VY_tensor) if len(VY_list)>0 else None

    # 使用 DataLoader
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False) if val_data is not None else None

    # 模型与训练（增强版）
    model = LSTMModel(input_dim=2, hidden_dim=128, num_layers=2, dropout=0.2)
    trained_model, epoch_losses, val_losses = train_model_autoregressive(
        model, train_loader, val_loader=val_loader, epochs=180, lr=7e-4, m=5.0,
        weight_decay=5e-5, grad_clip=1.0, patience=25, use_scheduler=True
    )

    # （绘图已迁移至 src/plot_results.py）

    # --------------------- 评估与保存结果（test 集） ---------------------
    from src.evaluate import compute_cost_np, predict_autoregressive

    # 使用模型自回归预测并计算成本，同时记录 OBD 的测试成本
    hitting_list, switching_list, total_list = [], [], []
    obd_total_list = []
    example_indices = [0, min(1, len(test_seq)-1)] if len(test_seq)>0 else []
    for i in range(len(test_seq)):
        y_seq = np.array(test_seq[i]).reshape(-1)
        # 模型预测（自回归）
        x_pred = predict_autoregressive(trained_model, y_seq)
        h, s, tot = compute_cost_np(x_pred, y_seq, m=m)
        hitting_list.append(h)
        switching_list.append(s)
        total_list.append(tot)
        # OBD 基线（同一测试集）
        x_teacher = robd_single_sequence(y_seq, m = m, la1 = la1, la2 = la2)
        _, _, obd_tot = compute_cost_np(x_teacher, y_seq, m=m)
        obd_total_list.append(obd_tot)

        # 保存前两个序列的曲线数据，供独立绘图脚本使用
        if i in example_indices:
            os.makedirs("results", exist_ok=True)
            np.savetxt(f"results/seq_{i}_y.txt", y_seq)
            np.savetxt(f"results/seq_{i}_x_ml_prefusion.txt", x_pred)
            np.savetxt(f"results/seq_{i}_x_obd.txt", x_teacher)

    hitting_arr = np.array(hitting_list)
    switching_arr = np.array(switching_list)
    total_arr = np.array(total_list)
    obd_total_arr = np.array(obd_total_list)

    print("[Task 2] Test Average cost:", total_arr.mean())
    print("[Task 2] Test Median cost:", np.median(total_arr))
    print("[Task 2] Test OBD Average cost:", obd_total_arr.mean())
    print("[Task 2] Test OBD Median cost:", np.median(obd_total_arr))

    os.makedirs("results", exist_ok=True)
    np.savetxt("results/test_cost_total.txt", total_arr)
    np.savetxt("results/test_cost_hitting.txt", hitting_arr)
    np.savetxt("results/test_cost_switching.txt", switching_arr)
    np.savetxt("results/test_obd_cost_total.txt", obd_total_arr)

    # （直方图绘制已迁移至 src/plot_results.py）

    # （直方图绘制已迁移至 src/plot_results.py）

    # ==================== Task 4: 可微 MLA-ROBD 训练期融合（论文设定） ====================
    print("\n" + "="*60)
    print("[Task 4] 开始 训练期融合：MLA-ROBD（按论文设定，无搜索）")
    print("="*60)

    from src.hybrid import mla_robd_single_sequence, train_model_mla_robd
    from src.plot_results import generate_plots

    # 论文实验默认参数（按论文设定，不做搜索）。λ2 由你在 main 顶部的设置决定
    la1_train = 0.5
    theta = 0.5
    la3_train = la1_train * theta
    la2_train = la2

    model_fused = LSTMModel(input_dim=2, hidden_dim=128, num_layers=2, dropout=0.2)
    fused_model, fused_train_losses, fused_val_losses = train_model_mla_robd(
        model_fused,
        train_loader,
        val_loader=val_loader,
        m=m,
        la1=float(la1_train),
        la2=float(la2_train),
        la3=float(la3_train),
        lr=7e-4,
        weight_decay=5e-5,
        grad_clip=1.0,
        patience=12,
        min_delta=1e-4,
        use_scheduler=True,
        device=torch.device("cpu")
    )

    # 测试集评估：仅输出训练期融合结果
    fused_total_list = []
    for i in range(len(test_seq)):
        y_seq = np.array(test_seq[i]).reshape(-1)
        x_ml = predict_autoregressive(fused_model, y_seq)
        x_fused = mla_robd_single_sequence(y_seq, x_ml, m, float(la1_train), float(la2_train), float(la3_train))
        _, _, tot = compute_cost_np(x_fused, y_seq, m=m)
        fused_total_list.append(tot)
        if i in example_indices:
            os.makedirs("results", exist_ok=True)
            # 也保存融合后的曲线
            np.savetxt(f"results/seq_{i}_x_mla_robd.txt", x_fused)
    fused_total_arr = np.array(fused_total_list)

    print(f"\n[Task 4] 训练期融合(MLA-ROBD) 测试集：平均={fused_total_arr.mean():.4f}, 中位数={np.median(fused_total_arr):.4f}")

    # 仅保存训练期融合结果
    os.makedirs("results", exist_ok=True)
    np.savetxt("results/test_mla_robd_trainfusion_cost_total.txt", fused_total_arr)

    print(f"\n[Task 4] 训练期融合结果已保存至 results/ 目录")
    # 调用独立绘图脚本，运行 main 时自动出图
    try:
        generate_plots()
        print("[Task 3] 图已生成到 results/ 目录")
    except Exception as e:
        print("[Task 3] 绘图失败:", e)

