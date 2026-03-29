import argparse
import json
import os
from datetime import datetime

import pandas as pd


def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def collect_experiments(experiments_root):
    rows = []
    for name in sorted(os.listdir(experiments_root)):
        exp_dir = os.path.join(experiments_root, name)
        if not os.path.isdir(exp_dir):
            continue
        manifest_path = os.path.join(exp_dir, 'manifest.json')
        summary_csv = os.path.join(exp_dir, 'metrics', 'summary_metrics.csv')
        paper_csv = os.path.join(exp_dir, 'metrics', 'paper_table.csv')
        if os.path.exists(manifest_path):
            manifest = load_manifest(manifest_path)
            recs = manifest.get('records', [])
            failures = manifest.get('failures', [])
        else:
            recs, failures = [], []
        rows.append(
            {
                'experiment_id': name,
                'experiment_dir': exp_dir,
                'records': len(recs),
                'failures': len(failures),
                'has_summary_metrics': os.path.exists(summary_csv),
                'has_paper_table': os.path.exists(paper_csv),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(experiments_root, out_dir):
    detail_frames = []
    paper_frames = []

    for name in sorted(os.listdir(experiments_root)):
        exp_dir = os.path.join(experiments_root, name)
        if not os.path.isdir(exp_dir):
            continue
        summary_csv = os.path.join(exp_dir, 'metrics', 'summary_metrics.csv')
        paper_csv = os.path.join(exp_dir, 'metrics', 'paper_table.csv')

        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            df.insert(0, 'ExperimentID', name)
            detail_frames.append(df)

        if os.path.exists(paper_csv):
            df = pd.read_csv(paper_csv)
            df.insert(0, 'ExperimentID', name)
            paper_frames.append(df)

    all_detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    all_paper = pd.concat(paper_frames, ignore_index=True) if paper_frames else pd.DataFrame()

    if not all_detail.empty:
        all_detail.to_csv(os.path.join(out_dir, 'all_batches_summary_metrics.csv'), index=False)

    if not all_paper.empty:
        all_paper.to_csv(os.path.join(out_dir, 'all_batches_paper_table.csv'), index=False)

    # Build mean/std table by Dataset+Model across seeds if available.
    if not all_detail.empty:
        metric_cols = [
            c
            for c in all_detail.columns
            if c.endswith('_MAE') or c.endswith('_RMSE') or c.endswith('_MAPE')
            or c in ['TrainSeconds', 'InferenceSeconds', 'PeakGpuMemoryMB', 'GateMean']
        ]
        grouped = all_detail.groupby(['ExperimentID', 'Dataset', 'Model'])[metric_cols].agg(['mean', 'std']).reset_index()
        grouped.columns = [
            '_'.join([str(x) for x in col if x]).strip('_') for col in grouped.columns.to_flat_index()
        ]
        grouped.to_csv(os.path.join(out_dir, 'all_batches_mean_std.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='Aggregate all finished experiment batches.')
    parser.add_argument('--experiments-root', default='/root/autodl-tmp/experiments')
    parser.add_argument('--output-dir', default='/root/autodl-tmp/experiments/_global_summary')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    status_df = collect_experiments(args.experiments_root)
    status_df.to_csv(os.path.join(args.output_dir, 'experiment_status.csv'), index=False)

    aggregate_metrics(args.experiments_root, args.output_dir)

    with open(os.path.join(args.output_dir, 'README.txt'), 'w', encoding='utf-8') as file_obj:
        file_obj.write(f'Generated at: {datetime.now().isoformat()}\\n')
        file_obj.write('Files:\\n')
        file_obj.write('- experiment_status.csv\\n')
        file_obj.write('- all_batches_summary_metrics.csv (if available)\\n')
        file_obj.write('- all_batches_paper_table.csv (if available)\\n')
        file_obj.write('- all_batches_mean_std.csv (if available)\\n')

    print('Global aggregation completed:', args.output_dir)


if __name__ == '__main__':
    main()
