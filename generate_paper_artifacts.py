import argparse
import os

import pandas as pd


def markdown_table_from_df(df):
    if df.empty:
        return 'No data available.\n'
    header = '| ' + ' | '.join(df.columns.astype(str)) + ' |\n'
    sep = '| ' + ' | '.join(['---'] * len(df.columns)) + ' |\n'
    rows = ''
    for _, row in df.iterrows():
        rows += '| ' + ' | '.join(str(x) for x in row.values) + ' |\n'
    return header + sep + rows


def main():
    parser = argparse.ArgumentParser(description='Generate manuscript-ready artifacts from experiment summaries.')
    parser.add_argument('--summary-dir', default='/root/autodl-tmp/experiments/_global_summary')
    parser.add_argument('--out-dir', default='/root/autodl-tmp/deliverables')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    detail_path = os.path.join(args.summary_dir, 'all_batches_summary_metrics.csv')
    mean_std_path = os.path.join(args.summary_dir, 'all_batches_mean_std.csv')

    if os.path.exists(detail_path):
        detail_df = pd.read_csv(detail_path)
        # core paper table subset
        core = detail_df[[c for c in detail_df.columns if c in [
            'ExperimentID', 'Dataset', 'Model', 'Seed',
            '5min_MAE', '30min_MAE', '60min_MAE',
            '5min_RMSE', '30min_RMSE', '60min_RMSE',
            '5min_MAPE', '30min_MAPE', '60min_MAPE',
            'TrainSeconds', 'InferenceSeconds', 'PeakGpuMemoryMB', 'GateMean'
        ]]]
        core.to_csv(os.path.join(args.out_dir, 'table_core_metrics.csv'), index=False)

        with open(os.path.join(args.out_dir, 'table_core_metrics.md'), 'w', encoding='utf-8') as file_obj:
            file_obj.write('# Core Metrics Table (Auto-generated)\n\n')
            file_obj.write(markdown_table_from_df(core.head(120)))

    if os.path.exists(mean_std_path):
        mean_std_df = pd.read_csv(mean_std_path)
        mean_std_df.to_csv(os.path.join(args.out_dir, 'table_mean_std.csv'), index=False)
        with open(os.path.join(args.out_dir, 'table_mean_std.md'), 'w', encoding='utf-8') as file_obj:
            file_obj.write('# Mean-Std Table (Auto-generated)\n\n')
            file_obj.write(markdown_table_from_df(mean_std_df.head(120)))

    with open(os.path.join(args.out_dir, 'AUTO_EXPORT_README.md'), 'w', encoding='utf-8') as file_obj:
        file_obj.write('# Auto Export Notes\n\n')
        file_obj.write('This folder contains manuscript-ready tables generated from real experiment outputs.\\n')
        file_obj.write('Regenerate after each batch completion with:\\n\\n')
        file_obj.write('```bash\\n')
        file_obj.write('cd /root/autodl-tmp/code\\n')
        file_obj.write('python generate_paper_artifacts.py --summary-dir /root/autodl-tmp/experiments/_global_summary --out-dir /root/autodl-tmp/deliverables\\n')
        file_obj.write('```\\n')

    print('Generated paper artifacts in', args.out_dir)


if __name__ == '__main__':
    main()
