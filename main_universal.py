import argparse
import hashlib
import json
import logging
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models import APST_FixedGate, APST_GlobalGate, APST_Net, APST_TemporalThenSpatial, Baseline_LSTM, DCRNN, FC_LSTM, STGCN
from utils import DataLoader, cheb_polynomial, get_adjacency_matrix, masked_mae, masked_mape, masked_rmse, scaled_laplacian


DEFAULT_DATASETS = ['PEMS04', 'PEMS08', 'PEMS03', 'PEMS07']
DEFAULT_MODELS = ['HA', 'ARIMA', 'DCRNN', 'FC_LSTM', 'STGCN', 'APST_Net']

DATASET_CONFIG = {
    'PEMS03': {'num_nodes': 358, 'file': 'PEMS03.npz', 'adj': 'PEMS03.csv'},
    'PEMS04': {'num_nodes': 307, 'file': 'PEMS04.npz', 'adj': 'PEMS04.csv'},
    'PEMS07': {'num_nodes': 883, 'file': 'PEMS07.npz', 'adj': 'PEMS07.csv'},
    'PEMS08': {'num_nodes': 170, 'file': 'PEMS08.npz', 'adj': 'PEMS08.csv'},
}

HORIZON_STEPS = {'5min': 0, '30min': 5, '60min': 11}


def ensure_omp_threads():
    value = os.environ.get('OMP_NUM_THREADS', '').strip()
    if not value.isdigit() or int(value) <= 0:
        os.environ['OMP_NUM_THREADS'] = '1'


def parse_csv_arg(raw_value, cast_fn=str):
    return [cast_fn(item.strip()) for item in raw_value.split(',') if item.strip()]


def setup_logger(log_path):
    logger = logging.getLogger('apst_runner')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested_device):
    if requested_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested_device)


def build_env_snapshot(args, device):
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': sys.version,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
        'cwd': os.getcwd(),
        'args': vars(args),
    }
    if torch.cuda.is_available():
        snapshot['gpu_count'] = torch.cuda.device_count()
        snapshot['gpu_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        snapshot['gpu_total_memory_gb'] = round(props.total_memory / (1024 ** 3), 2)
        snapshot['cudnn_version'] = torch.backends.cudnn.version()
    return snapshot


def build_code_fingerprint(output_path):
    tracked_files = ['main_universal.py', 'models.py', 'utils.py', 'Z.py']
    rows = []
    for file_name in tracked_files:
        abs_path = os.path.abspath(file_name)
        if os.path.exists(abs_path):
            rows.append({'file': file_name, 'sha256': file_sha256(abs_path), 'bytes': os.path.getsize(abs_path)})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def create_experiment_dirs(output_root, exp_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = exp_name.replace(' ', '_') if exp_name else 'run'
    exp_id = f'{timestamp}_{safe_name}'
    base_dir = os.path.join(output_root, exp_id)
    subdirs = {
        'base': base_dir,
        'config': os.path.join(base_dir, 'config'),
        'logs': os.path.join(base_dir, 'logs'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'predictions': os.path.join(base_dir, 'predictions'),
        'plots_data': os.path.join(base_dir, 'plots_data'),
        'env': os.path.join(base_dir, 'env'),
    }
    for directory in subdirs.values():
        os.makedirs(directory, exist_ok=True)
    return exp_id, subdirs


def calc_metrics(preds, labels):
    rows = []
    summary = {}
    for horizon_name, step in HORIZON_STEPS.items():
        pred_step = preds[:, step]
        label_step = labels[:, step]
        mae = masked_mae(pred_step, label_step, 0.0).item()
        rmse = masked_rmse(pred_step, label_step, 0.0).item()
        mape = masked_mape(pred_step, label_step, 10.0).item() * 100.0
        display = f'{mae:.2f}/{rmse:.2f}/{mape:.2f}%'
        rows.append({'horizon': horizon_name, 'step': step, 'mae': round(mae, 4), 'rmse': round(rmse, 4), 'mape': round(mape, 4), 'display': display})
        summary[horizon_name] = display
    return rows, summary


def run_statistical(model_name, loader):
    preds = []
    labels = []
    for x, y in loader.test_loader:
        if model_name == 'HA':
            pred = x.mean(dim=1, keepdim=True).repeat(1, loader.pred_len, 1, 1)
        else:
            pred = x[:, -1:, :, :].repeat(1, loader.pred_len, 1, 1)
        pred = pred * loader.std + loader.mean
        y = y * loader.std + loader.mean
        preds.append(pred)
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


def build_model(model_name, dataset_cfg, cheb_polys, adj_mx, pred_len, k_order):
    num_nodes = dataset_cfg['num_nodes']
    hidden_dim = 64
    if model_name == 'LSTM':
        return Baseline_LSTM(num_nodes, 1, hidden_dim, pred_len)
    if model_name == 'FC_LSTM':
        return FC_LSTM(num_nodes, 1, hidden_dim, pred_len)
    if model_name == 'DCRNN':
        return DCRNN(num_nodes, 1, hidden_dim, pred_len, adj_mx)
    if model_name == 'STGCN':
        return STGCN(num_nodes, 1, hidden_dim, pred_len, k_order, cheb_polys)
    if model_name == 'APST_Net':
        return APST_Net(FC_LSTM(num_nodes, 1, hidden_dim, pred_len), STGCN(num_nodes, 1, hidden_dim, pred_len, k_order, cheb_polys))
    if model_name == 'APST_FixedGate':
        return APST_FixedGate(FC_LSTM(num_nodes, 1, hidden_dim, pred_len), STGCN(num_nodes, 1, hidden_dim, pred_len, k_order, cheb_polys), spatial_weight=0.5)
    if model_name == 'APST_GlobalGate':
        return APST_GlobalGate(FC_LSTM(num_nodes, 1, hidden_dim, pred_len), STGCN(num_nodes, 1, hidden_dim, pred_len, k_order, cheb_polys), init_spatial_weight=0.5)
    if model_name == 'APST_TemporalThenSpatial':
        return APST_TemporalThenSpatial(FC_LSTM(num_nodes, 1, hidden_dim, pred_len), STGCN(num_nodes, 1, hidden_dim, pred_len, k_order, cheb_polys))
    raise ValueError(f'Unsupported model: {model_name}')


def train_eval(dataset, model_name, seed, args, dirs, device, logger):
    cfg = DATASET_CONFIG[dataset]
    data_path = os.path.join(args.data_root, cfg['file'])
    adj_path = os.path.join(args.data_root, cfg['adj'])

    logger.info('running model=%s dataset=%s seed=%s', model_name, dataset, seed)
    loader = DataLoader(data_path, args.batch_size, args.seq_len, args.pred_len, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    if model_name in ['HA', 'ARIMA']:
        preds, labels = run_statistical(model_name, loader)
        metrics_rows, summary = calc_metrics(preds, labels)
        prediction_path = os.path.join(dirs['predictions'], f'{dataset}_{model_name}_seed{seed}.npz')
        np.savez_compressed(prediction_path, prediction=preds.numpy(), ground_truth=labels.numpy())
        return {
            'dataset': dataset,
            'model': model_name,
            'seed': seed,
            'prediction_file': os.path.relpath(prediction_path, dirs['base']),
            'params': 0,
            'train_seconds': 0.0,
            'inference_seconds': 0.0,
            'peak_gpu_memory_mb': 0.0,
            'gate_mean': None,
            'metrics_rows': metrics_rows,
            'summary': summary,
        }

    adj_mx = get_adjacency_matrix(adj_path, cfg['num_nodes'])
    cheb_polys = cheb_polynomial(scaled_laplacian(adj_mx), args.k_order)
    model = build_model(model_name, cfg, cheb_polys, adj_mx, args.pred_len, args.k_order).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    train_history = []

    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_start = time.perf_counter()
        for x, y in loader.train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            model_output = model(x)
            pred = model_output[0] if isinstance(model_output, tuple) else model_output
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        epoch_seconds = time.perf_counter() - epoch_start
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_history.append({'epoch': epoch + 1, 'loss': round(mean_loss, 6), 'seconds': round(epoch_seconds, 4)})
        logger.info('epoch=%s/%s dataset=%s model=%s seed=%s loss=%.6f time=%.2fs', epoch + 1, args.epochs, dataset, model_name, seed, mean_loss, epoch_seconds)
    train_seconds = time.perf_counter() - train_start

    model.eval()
    predictions = []
    ground_truths = []
    gates = []
    gate_means = []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader.test_loader:
            x = x.to(device)
            y = y.to(device)
            model_output = model(x)
            if isinstance(model_output, tuple):
                pred, gate = model_output
                gate_cpu = gate.cpu()
                gates.append(gate_cpu)
                gate_means.append(gate_cpu.mean().item())
            else:
                pred = model_output
            pred = pred * loader.std + loader.mean
            y = y * loader.std + loader.mean
            predictions.append(pred.cpu())
            ground_truths.append(y.cpu())
    inference_seconds = time.perf_counter() - inference_start

    peak_gpu_memory_mb = 0.0
    if torch.cuda.is_available() and device.type == 'cuda':
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    preds = torch.cat(predictions)
    labels = torch.cat(ground_truths)
    gate_array = torch.cat(gates).numpy() if gates else None
    metrics_rows, summary = calc_metrics(preds, labels)

    prediction_payload = {'prediction': preds.numpy(), 'ground_truth': labels.numpy()}
    if gate_array is not None and args.save_gate_tensor:
        prediction_payload['gate_weights'] = gate_array

    prediction_path = os.path.join(dirs['predictions'], f'{dataset}_{model_name}_seed{seed}.npz')
    np.savez_compressed(prediction_path, **prediction_payload)

    history_path = os.path.join(dirs['metrics'], f'train_history_{dataset}_{model_name}_seed{seed}.csv')
    pd.DataFrame(train_history).to_csv(history_path, index=False)

    plot_rows = []
    for metric_row in metrics_rows:
        plot_rows.append({'dataset': dataset, 'model': model_name, 'seed': seed, 'horizon': metric_row['horizon'], 'step': metric_row['step'], 'mae': metric_row['mae'], 'rmse': metric_row['rmse'], 'mape': metric_row['mape']})
    plots_path = os.path.join(dirs['plots_data'], f'horizon_metrics_{dataset}_{model_name}_seed{seed}.csv')
    pd.DataFrame(plot_rows).to_csv(plots_path, index=False)

    gate_mean = float(np.mean(gate_means)) if gate_means else None
    return {
        'dataset': dataset,
        'model': model_name,
        'seed': seed,
        'prediction_file': os.path.relpath(prediction_path, dirs['base']),
        'train_history_file': os.path.relpath(history_path, dirs['base']),
        'plot_data_file': os.path.relpath(plots_path, dirs['base']),
        'params': count_parameters(model),
        'train_seconds': round(train_seconds, 4),
        'inference_seconds': round(inference_seconds, 4),
        'peak_gpu_memory_mb': round(peak_gpu_memory_mb, 2),
        'gate_mean': round(gate_mean, 6) if gate_mean is not None else None,
        'metrics_rows': metrics_rows,
        'summary': summary,
    }


def export_final_tables(records, dirs):
    detail_rows = []
    paper_rows = []
    for record in records:
        detail_row = {'Dataset': record['dataset'], 'Model': record['model'], 'Seed': record['seed'], 'Parameters': record['params'], 'TrainSeconds': record['train_seconds'], 'InferenceSeconds': record['inference_seconds'], 'PeakGpuMemoryMB': record['peak_gpu_memory_mb'], 'GateMean': record['gate_mean']}
        paper_row = {'Dataset': record['dataset'], 'Model': record['model'], 'Seed': record['seed']}
        for metric_row in record['metrics_rows']:
            prefix = metric_row['horizon']
            detail_row[f'{prefix}_MAE'] = metric_row['mae']
            detail_row[f'{prefix}_RMSE'] = metric_row['rmse']
            detail_row[f'{prefix}_MAPE'] = metric_row['mape']
            paper_row[prefix] = metric_row['display']
        detail_rows.append(detail_row)
        paper_rows.append(paper_row)
    pd.DataFrame(detail_rows).to_csv(os.path.join(dirs['metrics'], 'summary_metrics.csv'), index=False)
    pd.DataFrame(paper_rows).to_csv(os.path.join(dirs['metrics'], 'paper_table.csv'), index=False)


def build_download_manifest(dirs, exp_id):
    entries = [
        f'experiment_id: {exp_id}',
        f'config: {os.path.relpath(os.path.join(dirs["config"], "run_config.json"), dirs["base"])}',
        f'env: {os.path.relpath(os.path.join(dirs["env"], "environment.json"), dirs["base"])}',
        f'manifest: {os.path.relpath(os.path.join(dirs["base"], "manifest.json"), dirs["base"])}',
        f'summary_metrics: {os.path.relpath(os.path.join(dirs["metrics"], "summary_metrics.csv"), dirs["base"])}',
        f'paper_table: {os.path.relpath(os.path.join(dirs["metrics"], "paper_table.csv"), dirs["base"])}',
    ]
    with open(os.path.join(dirs['base'], 'DOWNLOAD_MANIFEST.txt'), 'w', encoding='utf-8') as file_obj:
        file_obj.write('\n'.join(entries) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Unified experiment runner for APST traffic forecasting.')
    parser.add_argument('--datasets', default=','.join(DEFAULT_DATASETS))
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS))
    parser.add_argument('--output-root', default='./experiments')
    parser.add_argument('--exp-name', default='baseline_audit')
    parser.add_argument('--data-root', default='../data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seq-len', type=int, default=12)
    parser.add_argument('--pred-len', type=int, default=12)
    parser.add_argument('--k-order', type=int, default=3)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seeds', default='3407')
    parser.add_argument('--save-gate-tensor', action='store_true')
    return parser.parse_args()


def main():
    ensure_omp_threads()
    args = parse_args()
    datasets = parse_csv_arg(args.datasets, str)
    models = parse_csv_arg(args.models, str)
    seeds = parse_csv_arg(args.seeds, int)
    device = resolve_device(args.device)

    exp_id, dirs = create_experiment_dirs(args.output_root, args.exp_name)
    logger = setup_logger(os.path.join(dirs['logs'], 'run.log'))

    config_payload = dict(vars(args))
    config_payload['datasets'] = datasets
    config_payload['models'] = models
    config_payload['seeds'] = seeds
    config_payload['experiment_id'] = exp_id
    config_payload['resolved_device'] = str(device)
    config_payload['raw_command'] = ' '.join(sys.argv)
    save_json(os.path.join(dirs['config'], 'run_config.json'), config_payload)
    save_json(os.path.join(dirs['env'], 'environment.json'), build_env_snapshot(args, device))
    build_code_fingerprint(os.path.join(dirs['config'], 'code_fingerprint.csv'))

    logger.info('experiment_id=%s', exp_id)
    logger.info('datasets=%s', datasets)
    logger.info('models=%s', models)
    logger.info('seeds=%s', seeds)
    logger.info('device=%s', device)

    records = []
    failures = []
    for seed in seeds:
        set_seed(seed)
        for dataset in datasets:
            if dataset not in DATASET_CONFIG:
                logger.error('unknown dataset: %s', dataset)
                failures.append({'dataset': dataset, 'model': None, 'seed': seed, 'error': 'Unknown dataset'})
                continue
            for model_name in models:
                try:
                    records.append(train_eval(dataset, model_name, seed, args, dirs, device, logger))
                except Exception as exc:
                    logger.exception('failed model=%s dataset=%s seed=%s', model_name, dataset, seed)
                    failures.append({'dataset': dataset, 'model': model_name, 'seed': seed, 'error': str(exc)})
                finally:
                    if torch.cuda.is_available() and device.type == 'cuda':
                        torch.cuda.empty_cache()

    export_final_tables(records, dirs)
    save_json(os.path.join(dirs['base'], 'manifest.json'), {'experiment_id': exp_id, 'created_at': datetime.now().isoformat(), 'records': records, 'failures': failures})
    build_download_manifest(dirs, exp_id)
    logger.info('completed records=%s failures=%s', len(records), len(failures))
    logger.info('artifacts saved under %s', dirs['base'])


if __name__ == '__main__':
    main()