# APST-Net Reproducibility Package

## Adaptive Parallel Spatio-Temporal Fusion Network for Long-Horizon Traffic Flow Forecasting

This repository contains the code, data, and experimental results for APST-Net, as presented in our paper submitted to Expert Systems with Applications.

## Quick Start

```bash
# Install dependencies
pip install torch numpy pandas scipy scikit-learn matplotlib

# Run experiments
python main_universal.py --model APST_Net --dataset PEMS04 --seed 3407

# Generate figures
python plot_horizon_decay.py
python plot_all_comparisons.py
```

## Repository Structure

```
APST-Net-Reproducibility/
├── README.md
├── requirements.txt
├── main_universal.py      # Main training and evaluation script
├── models.py              # Model definitions
├── utils.py               # Data loading and adjacency matrix construction
├── plot_*.py              # Visualization scripts
├── data/                   # PEMS datasets
│   ├── PEMS03.npz, PEMS03.csv
│   ├── PEMS04.npz, PEMS04.csv
│   ├── PEMS07.npz, PEMS07.csv
│   └── PEMS08.npz, PEMS08.csv
└── models/
    ├── models.py          # Core model implementations
    └── utils.py           # Shared utilities
```

## Supported Models

| Model | Description |
|-------|------------|
| `APST_Net` | Adaptive Parallel Spatio-Temporal Fusion Network |
| `APST_FixedGate` | APST-Net with fixed fusion weight (0.5) |
| `APST_GlobalGate` | APST-Net with global learnable gate |
| `APST_TemporalThenSpatial` | Serial fusion baseline |
| `STGCN` | Spatio-Temporal Graph Convolutional Network |
| `FC_LSTM` | Fully Connected LSTM |
| `DCRNN` | Diffusion Convolutional RNN |
| `LSTM` | Baseline LSTM |
| `HA` | Historical Average |

## Dataset

Four PEMS (California PeMS) subsets are included:
- **PEMS03**: 358 nodes, Sep-Nov 2018
- **PEMS04**: 307 nodes, Jan-Feb 2018
- **PEMS07**: 883 nodes, May-Aug 2017
- **PEMS08**: 170 nodes, Jul-Aug 2018

All datasets are sampled at 5-minute intervals.

## Citation

```bibtex
@article{apstnet2026,
  title={APST-Net: Adaptive Parallel Spatio-Temporal Fusion for Long-Horizon Traffic Flow Forecasting},
  author={Wu, Xin Yue and Cao, Jin Xin},
  journal={Expert Systems with Applications},
  year={2026}
}
```

## Contact

For questions, please contact: imucjx@163.com

## License

MIT License
