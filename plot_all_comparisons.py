import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import seaborn as sns
except ImportError:
    sns = None
from plot_style import apply_publication_style, model_style, save_png_pdf, add_subfigure_caption

# ================= 论文级配置区 ================= #
RESULT_DIR = './results'
# 统一数据集顺序：PEMS03 → PEMS04 → PEMS07 → PEMS08
DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
# 核心修改：加入了 LSTM 模型
MODELS = ['HA', 'ARIMA', 'DCRNN', 'LSTM', 'FC_LSTM', 'STGCN', 'APST_Net']
HORIZON_IDX = 11  # 对应 60min 预测
SAMPLES = 288  # 展示24小时数据点

def find_best_node(gt):
    """找方差最大的节点进行可视化"""
    num_nodes = gt.shape[2]
    best_n = 0
    max_var = -1
    for n in range(num_nodes):
        var = np.var(gt[:SAMPLES, HORIZON_IDX, n, 0])
        if var > max_var and np.min(gt[:SAMPLES, HORIZON_IDX, n, 0]) > 5:
            max_var = var
            best_n = n
    return best_n


def main():
    apply_publication_style()

    # 设置绘图风格
    if sns is not None:
        sns.set_style("ticks")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5))
    axes = axes.flatten()

    # 在图形顶部添加统一图例，水平分布，避免与子图线条重叠
    handles = []
    labels = []
    legend_order = ['Ground Truth', 'HA', 'ARIMA', 'DCRNN', 'LSTM', 'FC_LSTM', 'STGCN', 'APST_Net']
    for mod in legend_order:
        st = model_style(mod)
        handle = plt.Line2D([], [], color=st['color'], linewidth=st['linewidth'],
                            linestyle=st['linestyle'], label=mod)
        handles.append(handle)
        labels.append(mod)
    fig.legend(handles, labels, loc='upper center', ncol=8, fontsize=10, frameon=True,
               shadow=True, bbox_to_anchor=(0.5, 1.005))

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        print(f"Plotting {ds}...")

        # 1. 加载 APST_Net 结果作为基准
        target_file = f'{ds}_APST_Net.npz'
        base_path = os.path.join(RESULT_DIR, target_file)

        if not os.path.exists(base_path):
            ax.text(0.5, 0.5, f'{target_file} Not Found', ha='center', va='center')
            continue

        base_data = np.load(base_path)
        ground_truth = base_data['ground_truth']
        node_id = find_best_node(ground_truth)

        # 绘制真实值
        y_true = ground_truth[:SAMPLES, HORIZON_IDX, node_id, 0]
        gt_style = model_style('Ground Truth')
        ax.plot(
            y_true,
            label='Ground Truth',
            color=gt_style['color'],
            linewidth=gt_style['linewidth'],
            linestyle=gt_style['linestyle'],
            zorder=gt_style['zorder'],
        )

        # 2. 遍历并绘制所有模型结果 (含 LSTM)
        for mod in MODELS:
            path = os.path.join(RESULT_DIR, f'{ds}_{mod}.npz')
            if not os.path.exists(path):
                continue

            data = np.load(path)
            y_pred = data['prediction'][:SAMPLES, HORIZON_IDX, node_id, 0]

            st = model_style(mod)
            plot_kwargs = {
                'label': mod,
                'color': st['color'],
                'linestyle': st['linestyle'],
                'linewidth': st['linewidth'],
                'zorder': st['zorder'],
                'alpha': st['alpha'],
            }
            if st['marker'] is not None:
                plot_kwargs['marker'] = st['marker']
                plot_kwargs['markevery'] = 24
                plot_kwargs['markersize'] = 4.5

            ax.plot(y_pred, **plot_kwargs)

        # 3. 论文级细节调整：图内不放标题，子图名统一放在下方
        ax.set_xlabel('Time Step (5-min intervals)', fontsize=14)
        ax.set_ylabel('Traffic Flow', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle=':', alpha=0.5, color='#CBD2D9')
        add_subfigure_caption(ax, i, f'{ds} (Node {node_id})', y=-0.24)

        # 图例已统一放在图形顶部，子图内不再显示

    plt.tight_layout(rect=[0, 0.05, 1, 0.98], pad=2.5)

    # 输出到论文文件夹，命名为 Figure4.png
    png_path, pdf_path = save_png_pdf(fig, '../elsarticle/Figure4', output_dir='.', dpi=600)

    print(f"\nSuccess! Figure 4 saved as:\n- {png_path}\n- {pdf_path}")
    # plt.show() # 如果不需要交互窗口可注释掉


if __name__ == "__main__":
    main()