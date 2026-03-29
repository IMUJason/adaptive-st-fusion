import os

import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:  # optional dependency in minimal training envs
    sns = None


# Unified publication palette: muted, high-contrast, consistent across all figures.
MODEL_COLORS = {
    'Ground Truth': '#1F1F1F',
    'HA': '#A9B0B8',
    'ARIMA': '#7E8792',
    'LSTM': '#8B6F9C',
    'FC_LSTM': '#3C6E8F',
    'STGCN': '#D98F3B',
    'DCRNN': '#5A9A84',
    'APST_Net': '#B13A3A',
    'APST_FixedGate': '#6F8FB5',
    'APST_GlobalGate': '#8D6F53',
    'APST_TemporalThenSpatial': '#577590',
}

LINE_STYLES = {
    'Ground Truth': '-',
    'APST_Net': '-',
}

MARKERS = {
    'LSTM': 'D',
    'FC_LSTM': 'o',
    'STGCN': '^',
    'DCRNN': 'v',
    'APST_Net': 's',
    'APST_FixedGate': 'P',
    'APST_GlobalGate': 'X',
    'APST_TemporalThenSpatial': 'h',
}


def apply_publication_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.92
    plt.rcParams['legend.edgecolor'] = '#D0D5DC'
    plt.rcParams['savefig.dpi'] = 600
    if sns is not None:
        sns.set_theme(style='ticks', context='paper')
    else:
        plt.style.use('default')


def model_style(model_name):
    color = MODEL_COLORS.get(model_name, '#6F7680')
    linestyle = LINE_STYLES.get(model_name, '--')
    marker = MARKERS.get(model_name, None)

    if model_name == 'APST_Net':
        return {
            'color': color,
            'linestyle': linestyle,
            'linewidth': 3.0,
            'alpha': 1.0,
            'zorder': 10,
            'marker': marker,
        }
    if model_name == 'Ground Truth':
        return {
            'color': color,
            'linestyle': '-',
            'linewidth': 2.2,
            'alpha': 1.0,
            'zorder': 8,
            'marker': None,
        }
    return {
        'color': color,
        'linestyle': linestyle,
        'linewidth': 1.8,
        'alpha': 0.88,
        'zorder': 2,
        'marker': marker,
    }


def shade_peak_windows(ax):
    # 5-min interval indexing: morning 07:00-10:00, evening 17:00-20:00
    ax.axvspan(84, 120, color='#D0D5DC', alpha=0.20, lw=0)
    ax.axvspan(204, 240, color='#D0D5DC', alpha=0.20, lw=0)


def save_png_pdf(fig, output_base, output_dir='.', dpi=600):
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, f'{output_base}.png')
    pdf = os.path.join(output_dir, f'{output_base}.pdf')
    fig.savefig(png, dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.15)
    return png, pdf


def add_subfigure_caption(ax, index, caption, y=-0.24):
    tag = chr(97 + index)
    ax.text(
        0.5,
        y,
        f'({tag}) {caption}',
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=12,
        fontweight='bold',
    )
