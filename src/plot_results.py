import os
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')


def load_or_none(path):
    try:
        return np.loadtxt(path)
    except Exception:
        return None


def plot_hist(data, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=20, alpha=0.85, color='steelblue', edgecolor='black')
    mean_v = float(np.mean(data))
    med_v = float(np.median(data))
    plt.axvline(mean_v, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_v:.3f}')
    plt.axvline(med_v, color='green', linestyle='-.', linewidth=1.5, label=f'Median={med_v:.3f}')
    plt.xlabel('Total Cost (per sequence)')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_plots():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ml_path = os.path.join(RESULTS_DIR, 'test_cost_total.txt')
    obd_path = os.path.join(RESULTS_DIR, 'test_obd_cost_total.txt')
    mla_path = os.path.join(RESULTS_DIR, 'test_mla_robd_trainfusion_cost_total.txt')

    ml = load_or_none(ml_path)
    obd = load_or_none(obd_path)
    mla = load_or_none(mla_path)

    if ml is not None:
        plot_hist(ml, 'Test Cost Distribution (Model)', os.path.join(RESULTS_DIR, 'test_cost_hist.png'))
    if obd is not None:
        plot_hist(obd, 'Test Cost Distribution (OBD baseline)', os.path.join(RESULTS_DIR, 'test_obd_cost_hist.png'))
    if mla is not None:
        plot_hist(mla, 'Test Cost Distribution (MLA-ROBD Train-time)', os.path.join(RESULTS_DIR, 'test_mla_robd_cost_hist.png'))

    if (ml is not None) or (obd is not None) or (mla is not None):
        plt.figure(figsize=(8, 5))
        if obd is not None:
            plt.hist(obd, bins=20, alpha=0.5, color='darkorange', edgecolor='black', label='OBD')
        if ml is not None:
            plt.hist(ml, bins=20, alpha=0.5, color='steelblue', edgecolor='black', label='ML (LSTM)')
        if mla is not None:
            plt.hist(mla, bins=20, alpha=0.5, color='teal', edgecolor='black', label='MLA-ROBD (Train-time)')
        plt.xlabel('Total Cost (per sequence)')
        plt.ylabel('Count')
        plt.title('Test Cost Distribution Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'test_all_methods_comparison.png'), dpi=150)
        plt.close()

    # 逐序列可视化（前两个样例），展示 y / OBD / ML / MLA-ROBD
    for i in [0, 1]:
        y_p = os.path.join(RESULTS_DIR, f'seq_{i}_y.txt')
        ml_p = os.path.join(RESULTS_DIR, f'seq_{i}_x_ml_prefusion.txt')
        obd_p = os.path.join(RESULTS_DIR, f'seq_{i}_x_obd.txt')
        mla_p = os.path.join(RESULTS_DIR, f'seq_{i}_x_mla_robd.txt')
        y = load_or_none(y_p)
        ml_pred = load_or_none(ml_p)
        obd_pred = load_or_none(obd_p)
        mla_pred = load_or_none(mla_p)
        if y is None or ml_pred is None or obd_pred is None:
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(y, label='y (observed)', linewidth=2)
        plt.plot(obd_pred, label='x OBD (baseline)', linestyle='--', alpha=0.8)
        plt.plot(ml_pred, label='x ML (LSTM)', linestyle='-.', alpha=0.8)
        if mla_pred is not None:
            plt.plot(mla_pred, label='x MLA-ROBD (train-time)', linewidth=2)
        plt.xlabel('t')
        plt.ylabel('value')
        plt.title(f'Seq {i} Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'test_seq_{i}_comparison.png'), dpi=150)
        plt.close()

    print('[Plot] 绘图完成，已输出至 results/ 目录。')


