import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import seaborn as sns
except ImportError:
    sns = None
from plot_style import apply_publication_style, model_style, save_png_pdf, add_subfigure_caption

# ================= 配置区 ================= #
RESULT_DIR = './results'
# 统一数据集顺序：PEMS03 → PEMS04 → PEMS07 → PEMS08
DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
# 模型列表已更新
MODELS = ['HA', 'ARIMA', 'DCRNN', 'FC_LSTM', 'STGCN', 'APST_Net']
HORIZON_IDX = 11  # 对应 60min 预测
SAMPLES = 288  # 展示前24小时数据 (288 * 5min)

# ================= 核心：寻找拟合最好的节点 ================= #
def find_best_fitting_node(dataset_name):
    # 1. 读取 APST_Net 的预测结果作为筛选基准
    path = os.path.join(RESULT_DIR, f'{dataset_name}_APST_Net.npz')
    if not os.path.exists(path):
        print(f"Warning: {path} not found for node selection.")
        return 0

    data = np.load(path)
    pred = data['prediction']  # (T, 12, N, 1)
    true = data['ground_truth']  # (T, 12, N, 1)

    num_nodes = true.shape[2]
    best_node = 0
    min_mae = float('inf')

    # 2. 遍历所有节点，计算 MAE 以寻找表现最好的展示节点
    print(f"Scanning nodes in {dataset_name} to find the best fit for APST_Net...")
    for n in range(num_nodes):
        # 提取指定预测步长的数据
        p = pred[:SAMPLES, HORIZON_IDX, n, 0]
        t = true[:SAMPLES, HORIZON_IDX, n, 0]

        # 过滤掉流量过小的“死节点”（例如传感器故障或偏远地区），确保图表具有代表性
        if np.max(t) < 50: continue

        mae = np.mean(np.abs(p - t))

        if mae < min_mae:
            min_mae = mae
            best_node = n

    print(f"  -> Selected Node: {best_node} (Best MAE: {min_mae:.2f})")
    return best_node


# ================= 主程序 ================= #
def main():
    apply_publication_style()
    # 设置 Seaborn 风格提高视觉效果
    if sns is not None:
        sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, ds in enumerate(DATASETS):
        ax = axes[i]

        # 1. 自动挑选拟合效果最好的节点
        node_id = find_best_fitting_node(ds)

        # 2. 确认文件存在并加载 Ground Truth
        base_path = os.path.join(RESULT_DIR, f'{ds}_APST_Net.npz')
        if not os.path.exists(base_path):
            ax.text(0.5, 0.5, f'{ds}_APST_Net.npz\nNot Found', ha='center', va='center')
            continue

        base_data = np.load(base_path)
        y_true = base_data['ground_truth'][:SAMPLES, HORIZON_IDX, node_id, 0]

        # 画出黑色背景参考线
        gt_style = model_style('Ground Truth')
        ax.plot(
            y_true,
            label='Ground Truth',
            color=gt_style['color'],
            linestyle=gt_style['linestyle'],
            linewidth=gt_style['linewidth'],
            zorder=gt_style['zorder'],
        )

        # 3. 遍历所有模型进行绘图对比
        for mod in MODELS:
            path = os.path.join(RESULT_DIR, f'{ds}_{mod}.npz')
            if not os.path.exists(path): continue

            # 读取对应节点的预测曲线
            pred_data = np.load(path)
            y_pred = pred_data['prediction'][:SAMPLES, HORIZON_IDX, node_id, 0]

            st = model_style(mod)
            kwargs = {
                'label': mod,
                'color': st['color'],
                'linestyle': st['linestyle'],
                'linewidth': st['linewidth'],
                'zorder': st['zorder'],
                'alpha': st['alpha'],
            }
            if st['marker'] is not None:
                kwargs['marker'] = st['marker']
                kwargs['markevery'] = 24
                kwargs['markersize'] = 4.0
            ax.plot(y_pred, **kwargs)

        # 4. 细节美化：图内不放标题，子图名统一放在下方
        ax.set_xlabel('Time Steps (5-min intervals)', fontsize=10)
        ax.set_ylabel('Traffic Flow', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4, color='#CBD2D9')
        add_subfigure_caption(ax, i, f'{ds} (Node {node_id})', y=-0.23)

        # 仅在第一个子图中显示图例，避免重复
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # 5. 保存图表
    save_filename = 'APST_Net_Best_Fit_Comparison'
    png_path, pdf_path = save_png_pdf(fig, save_filename, output_dir='.', dpi=600)
    print(f"\n[Success] Comparison plots saved to:\n- {png_path}\n- {pdf_path}")
    plt.show()


if __name__ == "__main__":
    main()