"""
APST-Net Architecture Diagram - Professional Version for ESWA
生成论文 Figure 1: 模型架构图 (顶级期刊风格)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, BoxStyle
import matplotlib.patches as mpatches
import os

def draw_architecture():
    # 使用专业配色和现代设计风格
    fig, ax = plt.subplots(1, 1, figsize=(24, 7))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # 专业配色方案
    temporal_bg = '#EBF5FF'
    spatial_bg = '#EBF9F0'
    fuse_bg = '#FFF8EB'
    output_bg = '#F5EBFF'

    border_colors = {
        'temporal': '#2E75B6',
        'spatial': '#27AE60',
        'fusion': '#D68910',
        'output': '#8E44AD'
    }

    # 字体大小统一
    title_font = {'fontsize': 14, 'fontweight': 'bold'}
    label_font = {'fontsize': 11, 'fontweight': 'bold'}
    text_font = {'fontsize': 10}
    math_font = {'fontsize': 11}

    # ============== 统一布局网格 ==============
    # Y 轴对齐：时间分支 y=5.5, 空间分支 y=2.5, 输出 y=4.0
    # X 轴：Input(1) -> Encoder(5.5) -> Feature(12) -> Fusion(16) -> Output(21)

    # ============== 1. 输入层 ==============
    input_x, input_y = 1.0, 3.5
    input_w, input_h = 2.5, 2.0

    # 输入框阴影
    shadow = FancyBboxPatch(
        (input_x + 0.1, input_y - 0.1), input_w, input_h,
        boxstyle=BoxStyle.Round(pad=0.15, rounding_size=0.2),
        facecolor='#CCCCCC', edgecolor='none', alpha=0.3
    )
    ax.add_patch(shadow)

    # 输入框主体
    input_box = FancyBboxPatch(
        (input_x, input_y), input_w, input_h,
        boxstyle=BoxStyle.Round(pad=0.15, rounding_size=0.2),
        linewidth=2, edgecolor='#333333', facecolor='#FAFAFA'
    )
    ax.add_patch(input_box)

    # 输入标签 (统一字号)
    ax.text(input_x + input_w/2, input_y + input_h - 0.3, 'Input',
            ha='center', va='top', **title_font)
    ax.text(input_x + input_w/2, input_y + input_h - 0.7,
            r'$X_{t-T+1:t} \in \mathbb{R}^{T \times N \times F}$',
            ha='center', va='top', **math_font)
    ax.text(input_x + input_w/2, input_y + 0.6,
            r'$A \in \mathbb{R}^{N \times N}$',
            ha='center', va='top', fontsize=9, color='#555555')

    # ============== 2. 时间分支 ==============
    temp_enc_x, temp_enc_y = 5.5, 5.5
    temp_enc_w, temp_enc_h = 5.0, 2.0

    # 阴影
    shadow = FancyBboxPatch(
        (temp_enc_x + 0.1, temp_enc_y - 0.1), temp_enc_w, temp_enc_h,
        boxstyle=BoxStyle.Round(pad=0.12, rounding_size=0.18),
        facecolor=border_colors['temporal'], edgecolor='none', alpha=0.15
    )
    ax.add_patch(shadow)

    # 时间编码器
    temporal_encoder = FancyBboxPatch(
        (temp_enc_x, temp_enc_y), temp_enc_w, temp_enc_h,
        boxstyle=BoxStyle.Round(pad=0.12, rounding_size=0.18),
        linewidth=2, edgecolor=border_colors['temporal'], facecolor=temporal_bg
    )
    ax.add_patch(temporal_encoder)

    ax.text(temp_enc_x + temp_enc_w/2, temp_enc_y + temp_enc_h - 0.35,
            'Temporal Encoder', ha='center', va='top', **label_font, color=border_colors['temporal'])
    ax.text(temp_enc_x + temp_enc_w/2, temp_enc_y + temp_enc_h - 0.8,
            r'$\mathcal{F}_T$', ha='center', va='top', fontsize=16,
            fontweight='bold', color=border_colors['temporal'], style='italic')
    ax.text(temp_enc_x + temp_enc_w/2, temp_enc_y + temp_enc_h - 1.2,
            'FC-LSTM Stack', ha='center', va='top', **text_font)

    # 时间特征表示
    temp_feat_x, temp_feat_y = 12.0, 5.8
    temp_feat_w, temp_feat_h = 2.8, 1.4

    temp_feat = FancyBboxPatch(
        (temp_feat_x, temp_feat_y), temp_feat_w, temp_feat_h,
        boxstyle=BoxStyle.Round(pad=0.1, rounding_size=0.15),
        linewidth=1.5, edgecolor=border_colors['temporal'],
        facecolor=(200/255, 230/255, 255/255)
    )
    ax.add_patch(temp_feat)

    ax.text(temp_feat_x + temp_feat_w/2, temp_feat_y + temp_feat_h - 0.25,
            'Temporal Feature', ha='center', va='top', fontsize=9)
    ax.text(temp_feat_x + temp_feat_w/2, temp_feat_y + temp_feat_h - 0.55,
            r'$H_T$', ha='center', va='top', fontsize=14,
            fontweight='bold', color=border_colors['temporal'], style='italic')
    ax.text(temp_feat_x + temp_feat_w/2, temp_feat_y + 0.25,
            r'$\mathbb{R}^{H \times N \times d}$', ha='center', va='top', fontsize=8)

    # ============== 3. 空间分支 ==============
    spat_enc_x, spat_enc_y = 5.5, 2.0
    spat_enc_w, spat_enc_h = 5.0, 2.0

    # 阴影
    shadow = FancyBboxPatch(
        (spat_enc_x + 0.1, spat_enc_y - 0.1), spat_enc_w, spat_enc_h,
        boxstyle=BoxStyle.Round(pad=0.12, rounding_size=0.18),
        facecolor=border_colors['spatial'], edgecolor='none', alpha=0.15
    )
    ax.add_patch(shadow)

    # 空间编码器
    spatial_encoder = FancyBboxPatch(
        (spat_enc_x, spat_enc_y), spat_enc_w, spat_enc_h,
        boxstyle=BoxStyle.Round(pad=0.12, rounding_size=0.18),
        linewidth=2, edgecolor=border_colors['spatial'], facecolor=spatial_bg
    )
    ax.add_patch(spatial_encoder)

    ax.text(spat_enc_x + spat_enc_w/2, spat_enc_y + spat_enc_h - 0.35,
            'Spatial Encoder', ha='center', va='top', **label_font, color=border_colors['spatial'])
    ax.text(spat_enc_x + spat_enc_w/2, spat_enc_y + spat_enc_h - 0.8,
            r'$\mathcal{F}_S$', ha='center', va='top', fontsize=16,
            fontweight='bold', color=border_colors['spatial'], style='italic')
    ax.text(spat_enc_x + spat_enc_w/2, spat_enc_y + spat_enc_h - 1.2,
            'STGCN Blocks', ha='center', va='top', **text_font)

    # 空间特征表示
    spat_feat_x, spat_feat_y = 12.0, 2.3
    spat_feat_w, spat_feat_h = 2.8, 1.4

    spat_feat = FancyBboxPatch(
        (spat_feat_x, spat_feat_y), spat_feat_w, spat_feat_h,
        boxstyle=BoxStyle.Round(pad=0.1, rounding_size=0.15),
        linewidth=1.5, edgecolor=border_colors['spatial'],
        facecolor=(200/255, 245/255, 215/255)
    )
    ax.add_patch(spat_feat)

    ax.text(spat_feat_x + spat_feat_w/2, spat_feat_y + spat_feat_h - 0.25,
            'Spatial Feature', ha='center', va='top', fontsize=9)
    ax.text(spat_feat_x + spat_feat_w/2, spat_feat_y + spat_feat_h - 0.55,
            r'$H_S$', ha='center', va='top', fontsize=14,
            fontweight='bold', color=border_colors['spatial'], style='italic')
    ax.text(spat_feat_x + spat_feat_w/2, spat_feat_y + 0.25,
            r'$\mathbb{R}^{H \times N \times d}$', ha='center', va='top', fontsize=8)

    # ============== 4. 融合模块 ==============
    fuse_x, fuse_y = 16.0, 2.8
    fuse_w, fuse_h = 4.5, 3.4

    # 阴影
    shadow = FancyBboxPatch(
        (fuse_x + 0.1, fuse_y - 0.1), fuse_w, fuse_h,
        boxstyle=BoxStyle.Round(pad=0.15, rounding_size=0.2),
        facecolor=border_colors['fusion'], edgecolor='none', alpha=0.15
    )
    ax.add_patch(shadow)

    # 融合模块
    fusion_module = FancyBboxPatch(
        (fuse_x, fuse_y), fuse_w, fuse_h,
        boxstyle=BoxStyle.Round(pad=0.15, rounding_size=0.2),
        linewidth=2, edgecolor=border_colors['fusion'], facecolor=fuse_bg
    )
    ax.add_patch(fusion_module)

    # 模块标题
    ax.text(fuse_x + fuse_w/2, fuse_y + fuse_h - 0.3,
            'Adaptive Gated Fusion Module', ha='center', va='top',
            **label_font, color=border_colors['fusion'])

    # 子模块 1: Coupling
    coupling_x = fuse_x + 0.8
    ax.text(coupling_x, fuse_y + fuse_h - 0.9, 'Coupling',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(coupling_x, fuse_y + fuse_h - 1.15,
            r'$[H_S \Vert H_T]$', ha='center', va='top', fontsize=11)

    # 子模块 2: Gate
    gate_x = fuse_x + fuse_w/2
    ax.text(gate_x, fuse_y + fuse_h - 0.9, 'Gate',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(gate_x, fuse_y + fuse_h - 1.25,
            r'$Z = \sigma(W_g \cdot [\cdot] + b_g)$', ha='center', va='top', fontsize=9)

    # 子模块 3: Fusion
    fusion_x = fuse_x + fuse_w - 0.8
    ax.text(fusion_x, fuse_y + fuse_h - 0.9, 'Fusion',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(fusion_x, fuse_y + fuse_h - 1.2,
            r'$H = Z \odot H_S + (1-Z) \odot H_T$', ha='center', va='top', fontsize=8.5)

    # ============== 5. 输出层 ==============
    out_x, out_y = 21.5, 3.5
    out_w, out_h = 2.5, 2.0

    # 阴影
    shadow = FancyBboxPatch(
        (out_x + 0.1, out_y - 0.1), out_w, out_h,
        boxstyle=BoxStyle.Round(pad=0.15, rounding_size=0.2),
        facecolor=border_colors['output'], edgecolor='none', alpha=0.15
    )
    ax.add_patch(shadow)

    # 输出框
    output_box = FancyBboxPatch(
        (out_x, out_y), out_w, out_h,
        boxstyle=BoxStyle.Round(pad=0.15, rounding_size=0.2),
        linewidth=2, edgecolor=border_colors['output'], facecolor=output_bg
    )
    ax.add_patch(output_box)

    ax.text(out_x + out_w/2, out_y + out_h - 0.3, 'Output',
            ha='center', va='top', **title_font)
    ax.text(out_x + out_w/2, out_y + out_h - 0.7,
            r'$\hat{Y}_{t+1:t+H}$', ha='center', va='top', **math_font)
    ax.text(out_x + out_w/2, out_y + 0.6,
            r'$= W_p H + b_p$', ha='center', va='top', fontsize=9, color='#555555')

    # ============== 6. 连接箭头 (严格对齐) ==============
    # 输入 -> 时间编码器 (水平向上)
    ax.annotate('', xy=(temp_enc_x, temp_enc_y + temp_enc_h/2),
                xytext=(input_x + input_w, temp_enc_y + temp_enc_h/2),
                arrowprops=dict(arrowstyle='->', color=border_colors['temporal'], linewidth=2))

    # 输入 -> 空间编码器 (水平向下)
    ax.annotate('', xy=(spat_enc_x, spat_enc_y + spat_enc_h/2),
                xytext=(input_x + input_w, spat_enc_y + spat_enc_h/2),
                arrowprops=dict(arrowstyle='->', color=border_colors['spatial'], linewidth=2))

    # 时间编码器 -> 时间特征 (水平)
    ax.annotate('', xy=(temp_feat_x, temp_feat_y + temp_feat_h/2),
                xytext=(temp_enc_x + temp_enc_w, temp_feat_y + temp_feat_h/2),
                arrowprops=dict(arrowstyle='->', color=border_colors['temporal'], linewidth=1.8))

    # 空间编码器 -> 空间特征 (水平)
    ax.annotate('', xy=(spat_feat_x, spat_feat_y + spat_feat_h/2),
                xytext=(spat_enc_x + spat_enc_w, spat_feat_y + spat_feat_h/2),
                arrowprops=dict(arrowstyle='->', color=border_colors['spatial'], linewidth=1.8))

    # 时间特征 -> Coupling (斜向下箭头)
    ax.annotate('', xy=(fuse_x + 0.5, fuse_y + fuse_h - 1.0),
                xytext=(temp_feat_x + temp_feat_w, temp_feat_y + temp_feat_h/2),
                arrowprops=dict(arrowstyle='->', color='#666666', linewidth=1.5))

    # 空间特征 -> Coupling (斜向上箭头)
    ax.annotate('', xy=(fuse_x + 0.5, fuse_y + 1.3),
                xytext=(spat_feat_x + spat_feat_w, spat_feat_y + spat_feat_h/2),
                arrowprops=dict(arrowstyle='->', color='#666666', linewidth=1.5))

    # Fusion -> 输出 (水平向右)
    ax.annotate('', xy=(out_x, out_y + out_h/2),
                xytext=(fuse_x + fuse_w, out_y + out_h/2),
                arrowprops=dict(arrowstyle='->', color='#444444', linewidth=2))

    # ============== 7. 分支标签 ==============
    ax.text(temp_enc_x + temp_enc_w/2, temp_enc_y + temp_enc_h + 0.5,
            'Temporal Branch', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=border_colors['temporal'])
    ax.text(spat_enc_x + spat_enc_w/2, spat_enc_y - 0.5,
            'Spatial Branch', ha='center', va='top',
            fontsize=12, fontweight='bold', color=border_colors['spatial'])
    ax.text(fuse_x + fuse_w/2, fuse_y + fuse_h + 0.5,
            'Gated Fusion', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=border_colors['fusion'])

    plt.tight_layout()

    # 保存
    os.makedirs('../elsarticle', exist_ok=True)
    png_path = '../elsarticle/Figure1.png'
    pdf_path = '../elsarticle/Figure1.pdf'
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figure 1 (Architecture) saved to:\n- {png_path}\n- {pdf_path}")
    return png_path, pdf_path

if __name__ == "__main__":
    draw_architecture()
