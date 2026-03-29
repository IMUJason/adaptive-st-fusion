import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import os
import pandas as pd
from plot_style import apply_publication_style, model_style, save_png_pdf, add_subfigure_caption

# ================= 配置 ================= #
RESULT_DIR = './results'
# 统一数据集顺序：PEMS03 → PEMS04 → PEMS07 → PEMS08
DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
# 核心修改：MODELS 列表中加入了 LSTM
MODELS = ['LSTM', 'FC_LSTM', 'STGCN', 'DCRNN', 'APST_Net']

def get_mae_per_step(preds, labels, null_val=0.0):
    """计算每一个预测步长（1-12）的 MAE"""
    steps = preds.shape[1]
    mae_list = []
    for t in range(steps):
        p, l = preds[:, t, :, 0], labels[:, t, :, 0]
        if np.isnan(null_val):
            mask = ~np.isnan(l)
        else:
            mask = (l > null_val)
        mask = mask.astype(float)
        # 避免除以 0
        mean_mask = np.mean(mask)
        if mean_mask > 0:
            mask /= mean_mask

        mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
        mae = np.mean(np.abs(p - l) * mask)
        mae_list.append(mae)
    return mae_list


def main():
    apply_publication_style()

    # 画布设置：2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # 调整子图布局，为顶部图例留出空间
    plt.subplots_adjust(hspace=0.25, wspace=0.22, top=0.88, bottom=0.08)

    # 在图形顶部添加统一图例，水平分布
    handles = []
    labels = []
    legend_order = ['LSTM', 'FC_LSTM', 'STGCN', 'DCRNN', 'APST_Net']
    for mod in legend_order:
        st = model_style(mod)
        handle = plt.Line2D([], [], color=st['color'], linewidth=st['linewidth'],
                            linestyle=st['linestyle'], label=mod, marker=st['marker'])
        handles.append(handle)
        labels.append(mod)
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12, frameon=True,
               shadow=True, bbox_to_anchor=(0.5, 0.95))

    # 风格设置
    if sns is not None:
        sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        print(f"Plotting Horizon Analysis for {ds}...")

        for mod in MODELS:
            path = os.path.join(RESULT_DIR, f'{ds}_{mod}.npz')
            if not os.path.exists(path):
                print(f"  [Skip] {path} not found.")
                continue

            data = np.load(path)
            # 计算 1 到 12 个步长的 MAE
            mae_steps = get_mae_per_step(data['prediction'], data['ground_truth'])

            # X 轴：从 5 分钟到 60 分钟
            x_axis = np.arange(5, 65, 5)

            st = model_style(mod)
            kwargs = {
                'label': mod,
                'color': st['color'],
                'linewidth': st['linewidth'],
                'linestyle': st['linestyle'],
                'alpha': st['alpha'],
                'zorder': st['zorder'],
            }
            if st['marker'] is not None:
                kwargs['marker'] = st['marker']
                kwargs['markersize'] = 6 if mod == 'APST_Net' else 4.5
            ax.plot(x_axis, mae_steps, **kwargs)

        # 子图标题改为下置标签，避免和 TeX caption 重复
        add_subfigure_caption(ax, i, ds, y=-0.18)

        # 坐标轴标签
        if i >= 2:
            ax.set_xlabel('Prediction Horizon (min)', fontsize=16, fontweight='bold')
        if i % 2 == 0:
            ax.set_ylabel('MAE', fontsize=16, fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xticks([5, 15, 30, 45, 60])
        ax.grid(True, alpha=0.5, color='#CBD2D9')

        # 图例已统一放在图形顶部，子图内不再显示

    # 不使用 tight_layout，避免压缩子图大小

    # 输出到论文文件夹，命名为 Figure2.png
    save_path_png, save_path_pdf = save_png_pdf(fig, '../elsarticle/Figure2', output_dir='.', dpi=600)
    print(f"\n[Success] Figure 2 saved as:\n- {save_path_png}\n- {save_path_pdf}")
    plt.show()


if __name__ == "__main__":
    main()