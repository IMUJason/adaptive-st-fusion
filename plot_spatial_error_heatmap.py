import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import os
import pandas as pd
from plot_style import apply_publication_style, model_style, save_png_pdf, add_subfigure_caption

# ================= 配置区 ================= #
RESULT_DIR = './results'
# 统一数据集顺序：PEMS03 → PEMS04 → PEMS07 → PEMS08
DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
# 模型列表已更新
MODELS = ['FC_LSTM', 'STGCN', 'DCRNN', 'APST_Net']

def get_error_by_bin(pred, true, bins=10):
    """根据真实流量大小进行分桶，计算每个区间的平均误差"""
    p_flat = pred.flatten()
    t_flat = true.flatten()
    abs_error = np.abs(p_flat - t_flat)
    df = pd.DataFrame({'Ground Truth': t_flat, 'Error': abs_error})

    # 自动确定最大流量，动态分桶
    max_val = np.max(t_flat)
    # 根据流量规模设置步长
    step = 100
    if max_val > 1000: step = 200  # PEMS07 等大数据集流量可能较大

    bin_edges = np.arange(0, max_val + step, step)
    labels = [f'{int(i)}-{int(i + step)}' for i in bin_edges[:-1]]

    df['Bin'] = pd.cut(df['Ground Truth'], bins=bin_edges, labels=labels)

    # 计算每个桶的平均误差 (MAE)
    grouped = df.groupby('Bin', observed=True)['Error'].mean()
    return grouped


def main():
    apply_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    if sns is not None:
        sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        print(f"Processing Stress Test for {ds}...")

        # 1. 确定分桶的基准 (使用 APST_Net 文件中的 Ground Truth)
        base_path = os.path.join(RESULT_DIR, f'{ds}_APST_Net.npz')
        if not os.path.exists(base_path):
            print(f"   Warning: {base_path} not found, skipping dataset.")
            continue

        # 2. 遍历所有模型并计算分桶误差
        for mod in MODELS:
            path = os.path.join(RESULT_DIR, f'{ds}_{mod}.npz')
            if not os.path.exists(path): continue

            data = np.load(path)
            # 取 60min 预测步长 (Horizon Index 11)
            pred = data['prediction'][:, 11, :, 0]
            true = data['ground_truth'][:, 11, :, 0]

            # 计算各流量区间的误差曲线
            error_series = get_error_by_bin(pred, true)

            # 绘图属性设置：APST_Net 突出显示
            st = model_style(mod)

            # 过滤掉 NaN (排除没有数据落入的流量区间)
            valid_mask = ~np.isnan(error_series.values)
            x_indices = np.arange(len(error_series))[valid_mask]
            y_values = error_series.values[valid_mask]

            ax.plot(x_indices, y_values,
                    label=mod, color=st['color'], marker=st['marker'],
                    linewidth=st['linewidth'], alpha=st['alpha'], zorder=st['zorder'])

            # 设置 X 轴标签 (以 APST_Net 的分桶结果作为刻度标准)
            if mod == 'APST_Net':
                x_labels = error_series.index[valid_mask]
                ax.set_xticks(x_indices)
                ax.set_xticklabels(x_labels, rotation=45, fontsize=9)

        # 图像细节美化：标题下置，避免和 caption 冲突
        add_subfigure_caption(ax, i, ds, y=-0.28)
        ax.set_ylabel('MAE (Error)', fontsize=12)
        if i >= 2:
            ax.set_xlabel('Ground Truth Traffic Volume (veh/5min)', fontsize=12)

        if i == 0:
            ax.legend(frameon=True, fontsize=10, loc='upper left', shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # 输出到论文文件夹，命名为 Figure5.png
    save_path, pdf_path = save_png_pdf(fig, '../elsarticle/Figure5', output_dir='.', dpi=600)
    print(f"\n[Success] Figure 5 saved to:\n- {save_path}\n- {pdf_path}")
    plt.show()


if __name__ == "__main__":
    main()