import numpy as np
import os
import pandas as pd

# ================= 配置区 ================= #
RESULT_DIR = './results'
DATASETS = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
# 包含你提到的所有 7 个模型
MODELS = ['HA', 'ARIMA', 'DCRNN', 'LSTM', 'FC_LSTM', 'STGCN', 'APST_Net']
OUTPUT_FILE = os.path.join(RESULT_DIR, 'Final_Metrics_Summary.csv')

# 计算指标时使用的阈值（标准做法：MAPE 过滤小于 10 的值以防止数值爆炸）
NULL_VAL = 10.0


def calculate_metrics(pred, real):
    """计算单个步长的 MAE, RMSE, MAPE"""
    # 确保维度一致
    pred = pred.flatten()
    real = real.flatten()

    # 1. MAE (过滤 0 值即可)
    mask_mae = (real > 0)
    mae = np.mean(np.abs(pred[mask_mae] - real[mask_mae]))

    # 2. RMSE (过滤 0 值即可)
    rmse = np.sqrt(np.mean((pred[mask_mae] - real[mask_mae]) ** 2))

    # 3. MAPE (使用阈值 NULL_VAL 过滤，专门解决 PEMS03 这种数据集的异常)
    mask_mape = (real > NULL_VAL)
    if np.any(mask_mape):
        mape = np.mean(np.abs(pred[mask_mape] - real[mask_mape]) / real[mask_mape])
    else:
        mape = 0

    return mae, rmse, mape


def main():
    table = []
    print("Extracting results and calculating metrics...")

    for ds in DATASETS:
        for mod in MODELS:
            # 路径拼接：例如 ./results/PEMS03_APST_Net.npz
            file_path = os.path.join(RESULT_DIR, f'{ds}_{mod}.npz')

            if not os.path.exists(file_path):
                print(f"  [Skip] {ds}_{mod}: File not found.")
                continue

            try:
                data = np.load(file_path)
                # 假设维度为 [Samples, Horizon(12), Nodes, 1]
                preds = data['prediction']
                labels = data['ground_truth']

                # 提取三个时间点：5min(idx 0), 30min(idx 5), 60min(idx 11)
                horizons = [0, 5, 11]
                res_str = []

                for h in horizons:
                    mae, rmse, mape = calculate_metrics(preds[:, h], labels[:, h])
                    # 格式化为：MAE/RMSE/MAPE%
                    res_str.append(f"{mae:.2f}/{rmse:.2f}/{mape * 100:.2f}%")

                table.append({
                    'Dataset': ds,
                    'Model': mod,
                    '5 min': res_str[0],
                    '30 min': res_str[1],
                    '60 min': res_str[2]
                })
                print(f"  [Done] {ds}_{mod}")

            except Exception as e:
                print(f"  [Error] Failed to process {ds}_{mod}: {e}")

    # 保存为 CSV
    df = pd.DataFrame(table)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAll metrics saved to: {OUTPUT_FILE}")

    # 打印预览
    print("\nPreview of the result:")
    print(df.head(10))


if __name__ == "__main__":
    main()