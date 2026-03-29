import argparse
import csv
import hashlib
import json
import os
from datetime import datetime


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def walk_files(base_dir):
    rows = []
    for root, _, files in os.walk(base_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, base_dir)
            size = os.path.getsize(path)
            rows.append({
                'relative_path': rel,
                'size_bytes': size,
                'sha256': sha256_file(path),
            })
    rows.sort(key=lambda item: item['relative_path'])
    return rows


def write_csv(path, rows):
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(['relative_path', 'size_bytes', 'sha256'])
        return
    with open(path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=['relative_path', 'size_bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_checklist(path, exp_dir, rows):
    buckets = {
        'config': [],
        'env': [],
        'logs': [],
        'metrics': [],
        'predictions': [],
        'plots_data': [],
        'data_catalog': [],
        'other': [],
    }
    for row in rows:
        key = row['relative_path'].split('/')[0]
        buckets[key if key in buckets else 'other'].append(row['relative_path'])

    lines = []
    lines.append(f'# Download Checklist ({os.path.basename(exp_dir)})')
    lines.append('')
    lines.append(f'- generated_at: {datetime.now().isoformat()}')
    lines.append(f'- experiment_dir: {exp_dir}')
    lines.append('')
    lines.append('## Required Artifacts')
    lines.append('')
    for section in ['config', 'env', 'logs', 'metrics', 'predictions', 'plots_data', 'data_catalog', 'other']:
        lines.append(f'### {section}')
        if buckets[section]:
            for item in buckets[section]:
                lines.append(f'- {item}')
        else:
            lines.append('- (none)')
        lines.append('')

    lines.append('## Local Sync Command')
    lines.append('')
    lines.append('```bash')
    lines.append(f"# Example: rsync -avz <server>:{exp_dir}/ ./{os.path.basename(exp_dir)}/")
    lines.append('```')

    with open(path, 'w', encoding='utf-8') as file_obj:
        file_obj.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Finalize experiment artifacts into download-ready indexes.')
    parser.add_argument('--exp-dir', required=True)
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f'Experiment directory not found: {exp_dir}')

    rows = walk_files(exp_dir)
    csv_path = os.path.join(exp_dir, 'artifact_index.csv')
    json_path = os.path.join(exp_dir, 'artifact_index.json')
    md_path = os.path.join(exp_dir, 'DOWNLOAD_CHECKLIST.md')

    write_csv(csv_path, rows)
    with open(json_path, 'w', encoding='utf-8') as file_obj:
        json.dump({'generated_at': datetime.now().isoformat(), 'files': rows}, file_obj, indent=2, ensure_ascii=False)
    write_markdown_checklist(md_path, exp_dir, rows)

    print(f'Wrote {csv_path}')
    print(f'Wrote {json_path}')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()
