import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import seaborn as sns
except ImportError:
    sns = None
from plot_style import apply_publication_style, model_style, shade_peak_windows, save_png_pdf, add_subfigure_caption

# ================= 配置区 ================= #
RESULT_DIR = './results'
# 统一数据集顺序：PEMS03 → PEMS04 → PEMS07 → PEMS08
DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
# 模型列表已更新
MODELS = ['FC_LSTM', 'STGCN', 'DCRNN', 'APST_Net']

# 定义每个数据集的高峰时段 (Time Steps)
# 假设每天 288 步 (5min间隔)。
# 早高峰: 07:00-10:00 (Step 84-120), 晚高峰: 17:00-20:00 (Step 204-240)
PEAK_WINDOWS = {
    'Morning Peak (07:00-10:00)': (84, 120),
    'Evening Peak (17:00-20:00)': (204, 240)
}


def find_best_node(gt):
    # 找流量/速度最大的节点，通常是主干道，高峰特征最明显
    # 取前24小时数据，第12个预测步长（60min后）
    means = np.mean(gt[:288, 11, :, 0], axis=0)
    return np.argmax(means)


def main():
    apply_publication_style()
    # 创建 4行2列 的大图 (4个数据集 x 2个高峰时段)
    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    plt.subplots_adjust(hspace=0.55, wspace=0.2, top=0.94)

    # 设置绘图风格
    if sns is not None:
        sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

    # 在图形顶部添加统一图例，水平分布，避免与子图线条重叠
    handles = []
    labels = []
    legend_order = ['Ground Truth', 'FC_LSTM', 'STGCN', 'DCRNN', 'APST_Net']
    for mod in legend_order:
        st = model_style(mod)
        handle = plt.Line2D([], [], color=st['color'], linewidth=st['linewidth'],
                            linestyle=st['linestyle'], label=mod)
        handles.append(handle)
        labels.append(mod)
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=11, frameon=True,
               shadow=True, bbox_to_anchor=(0.5, 0.995))

    for row, ds in enumerate(DATASETS):
        print(f"Processing Peak Hour Analysis for {ds}...")

        # 1. 读取 APST_Net 的结果作为基准来确定展示节点
        base_path = os.path.join(RESULT_DIR, f'{ds}_APST_Net.npz')
        if not os.path.exists(base_path):
            print(f"   Warning: {base_path} not found, skipping row.")
            continue

        data = np.load(base_path)
        # 获取真实值：取第一天数据 (288步), 60min预测步长 (idx 11)
        y_true_all = data['ground_truth'][:288, 11, :, 0]
        node_id = find_best_node(data['ground_truth'])

        # 2. 遍历早/晚高峰窗口进行绘图
        for col, (peak_name, (start, end)) in enumerate(PEAK_WINDOWS.items()):
            ax = axes[row, col]

            # 画真实值 (黑色实线)
            y_true_segment = y_true_all[start:end, node_id]
            x_axis = np.arange(start, end)
            gt_style = model_style('Ground Truth')
            ax.plot(
                x_axis,
                y_true_segment,
                label='Ground Truth',
                color=gt_style['color'],
                linewidth=gt_style['linewidth'],
                linestyle=gt_style['linestyle'],
                zorder=gt_style['zorder'],
            )

            # 3. 循环画出各个模型的预测结果
            for mod in MODELS:
                path = os.path.join(RESULT_DIR, f'{ds}_{mod}.npz')
                if not os.path.exists(path): continue

                # 加载预测值
                y_pred = np.load(path)['prediction'][:288, 11, node_id, 0]
                y_pred_segment = y_pred[start:end]

                st = model_style(mod)
                kwargs = {
                    'label': mod,
                    'color': st['color'],
                    'linewidth': st['linewidth'],
                    'linestyle': st['linestyle'],
                    'zorder': st['zorder'],
                    'alpha': st['alpha'],
                }
                if st['marker'] is not None:
                    kwargs['marker'] = st['marker']
                    kwargs['markevery'] = 8
                    kwargs['markersize'] = 3.2
                ax.plot(x_axis, y_pred_segment, **kwargs)

            # 4. 坐标轴美化：图内不放标题
            sub_idx = row * 2 + col
            add_subfigure_caption(ax, sub_idx, f'{ds} {peak_name} (Node {node_id})', y=-0.34)

            # 将 X 轴 Step 转换为 HH:MM 格式
            ticks = np.linspace(start, end, 5)
            labels = [f"{int(t // 12):02d}:{int(t % 12) * 5:02d}" for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Traffic Flow')

            # 注意：由于每个子图已经是高峰时段的特写，不需要再用阴影标记高峰区域
            # shade_peak_windows(ax)  # 已移除

            # 图例已统一放在图形顶部，子图内不再显示

    # 输出到论文文件夹，命名为 Figure6.png
    save_path, pdf_path = save_png_pdf(fig, '../elsarticle/Figure6', output_dir='.', dpi=600)
    print(f"\n[Success] Figure 6 saved to:\n- {save_path}\n- {pdf_path}")
    plt.show()


if __name__ == "__main__":
    main()