import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_publication_style, add_subfigure_caption, save_png_pdf

# ================= 论文级配置 ================= #
RESULT_DIR = './results'
DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
SAMPLES = 288  # 24小时 (288步 * 5min)
HORIZON_IDX = 11  # 观察60min预测的门控决策

# 配色方案
COLOR_FLOW = '#333333'  # 深灰真实流量
COLOR_Z = '#D62728'  # 红色空间权重 z
COLOR_1Z = '#1F77B4'  # 蓝色时间权重 1-z


def parse_args():
    parser = argparse.ArgumentParser(description='Plot gate evolution from saved real gate tensors.')
    parser.add_argument('--result-dir', default=RESULT_DIR)
    parser.add_argument('--model-name', default='APST_Net')
    parser.add_argument('--seed', default='3407')
    parser.add_argument('--output-dir', default='.')
    return parser.parse_args()


def find_prediction_file(result_dir, dataset, model_name, seed):
    candidates = [
        os.path.join(result_dir, f'{dataset}_{model_name}_seed{seed}.npz'),
        os.path.join(result_dir, f'{dataset}_{model_name}.npz'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_real_gate_series(npz_path):
    data = np.load(npz_path)
    if 'gate_weights' not in data.files:
        return None, None, None

    ground_truth = data['ground_truth']
    gate_weights = data['gate_weights']
    node_id = int(np.argmax(np.var(ground_truth[:SAMPLES, HORIZON_IDX, :, 0], axis=0)))
    flow_data = ground_truth[:SAMPLES, HORIZON_IDX, node_id, 0]
    z_data = gate_weights[:SAMPLES, HORIZON_IDX, node_id, 0]
    return node_id, flow_data, z_data


def main():
    apply_publication_style()
    args = parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(20, 11.5))
    axes = axes.flatten()

    # 调整子图间距
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88, bottom=0.06)

    # 在图形顶部添加统一图例，水平分布
    legend_handles = [
        plt.Line2D([], [], color=COLOR_FLOW, linewidth=2.0, label='Traffic Flow'),
        plt.Line2D([], [], color=COLOR_Z, linewidth=2.5, label='Spatial Weight ($z$)'),
        plt.Line2D([], [], color=COLOR_1Z, linewidth=1.5, linestyle='--', label='Temporal Weight ($1-z$)', alpha=0.6)
    ]
    fig.legend(legend_handles, ['Traffic Flow', 'Spatial Weight ($z$)', 'Temporal Weight ($1-z$)'],
               loc='upper center', ncol=3, fontsize=12, frameon=True, shadow=True,
               bbox_to_anchor=(0.5, 0.96))

    for i, ds in enumerate(DATASETS):
        ax1 = axes[i]

        # 1. 读取该数据集的真实流量数据
        path = find_prediction_file(args.result_dir, ds, args.model_name, args.seed)
        if path is None:
            ax1.text(0.5, 0.5, f'Data {ds} Not Found', ha='center')
            continue

        node_id, flow_data, z_data = load_real_gate_series(path)
        if z_data is None:
            ax1.text(0.5, 0.5, f'Gate {ds} Not Found', ha='center')
            continue

        # --- 绘制左轴：流量 ---
        x_axis = np.arange(SAMPLES)
        ax1.plot(x_axis, flow_data, color=COLOR_FLOW, linewidth=2.0, label='Traffic Flow', alpha=0.8)
        ax1.set_ylabel('Flow (veh/5min)', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='y', labelsize=12)

        # 设置 X 轴时间刻度
        if i >= 2:
            ax1.set_xlabel('Time of Day (24h)', fontsize=14, fontweight='bold')
        ticks = [0, 72, 144, 216, 287]
        labels = ['00:00', '06:00', '12:00', '18:00', '23:55']
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels, fontsize=12)

        # --- 绘制右轴：门控权重 ---
        ax2 = ax1.twinx()
        ax2.plot(x_axis, z_data, color=COLOR_Z, linewidth=2.5, label='Spatial Weight ($z$)', zorder=10)
        ax2.plot(x_axis, 1 - z_data, color=COLOR_1Z, linewidth=1.5, linestyle='--', label='Temporal Weight ($1-z$)',
                 alpha=0.6)

        ax2.set_ylabel('Gate Weight', fontsize=14, fontweight='bold', rotation=270, labelpad=20)
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='y', labelsize=12)

        # --- 高亮高峰区 (早 7-10, 晚 17-20) ---
        ax1.axvspan(84, 120, color='gray', alpha=0.1, label='Peak Period' if i == 0 else "")
        ax1.axvspan(204, 240, color='gray', alpha=0.1)

        # 子图标题下置，避免和 TeX caption 重复
        add_subfigure_caption(ax1, i, f'{ds} (Node {node_id})', y=-0.24)
        ax1.grid(True, linestyle=':', alpha=0.4)

        # 图例已统一放在图形顶部，子图内不再显示

    os.makedirs(args.output_dir, exist_ok=True)
    # 输出到论文文件夹，命名为 Figure3.png
    png_path, save_path = save_png_pdf(fig, 'Figure3', output_dir=args.output_dir, dpi=600)
    print(f"Success! Figure 3 saved to:\n- {png_path}\n- {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()