import numpy as np
import torch
import torch.utils.data as data
from scipy.sparse.linalg import eigs
import pandas as pd


class DataLoader(object):
    def __init__(self, data_path, batch_size, seq_len, pred_len, train_ratio=0.7, val_ratio=0.1):
        self.data = np.load(data_path)['data'][:, :, 0]
        self.len_data = self.data.shape[0]
        self.num_nodes = self.data.shape[1]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data = (self.data - self.mean) / self.std
        self.train_size = int(self.len_data * train_ratio)
        self.val_size = int(self.len_data * val_ratio)

        self.train_loader = self._get_loader(0, self.train_size, batch_size, True)
        self.test_loader = self._get_loader(self.train_size + self.val_size, self.len_data, batch_size, False)

    def _get_loader(self, start, end, batch_size, shuffle):
        data_x, data_y = [], []
        for i in range(start, end - self.seq_len - self.pred_len + 1):
            data_x.append(self.data[i: i + self.seq_len, :])
            data_y.append(self.data[i + self.seq_len: i + self.seq_len + self.pred_len, :])
        data_x = torch.tensor(np.array(data_x), dtype=torch.float32).unsqueeze(-1)
        data_y = torch.tensor(np.array(data_y), dtype=torch.float32).unsqueeze(-1)
        dataset = data.TensorDataset(data_x, data_y)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    import csv
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    with open(distance_df_filename, 'r') as f:
        line = f.readline().strip()
        parts = line.split(',')
        if len(parts) > 3:  # 稠密矩阵
            f.seek(0)
            try:
                df = pd.read_csv(f, header=None)
                return df.values[:num_of_vertices, :num_of_vertices].astype(np.float32)
            except:
                return np.eye(num_of_vertices, dtype=np.float32)

        # 边列表
        f.seek(0)
        reader = csv.reader(f)
        try:
            int(next(reader)[0]); f.seek(0)
        except:
            pass

        edges = []
        nodes = set()
        for row in reader:
            if len(row) != 3: continue
            u, v, w = int(row[0]), int(row[1]), float(row[2])
            edges.append((u, v, w))
            nodes.add(u);
            nodes.add(v)

        max_id = max(nodes) if nodes else 0
        if max_id >= num_of_vertices:  # 需要 ID 映射
            sorted_nodes = sorted(list(nodes))
            id2idx = {id: idx for idx, id in enumerate(sorted_nodes)}
            for u, v, w in edges:
                if u in id2idx and v in id2idx:
                    distaneA[id2idx[u], id2idx[v]] = w
                    distaneA[id2idx[v], id2idx[u]] = w
        else:  # 直接使用
            for u, v, w in edges:
                if u < num_of_vertices and v < num_of_vertices:
                    distaneA[u, v] = w
                    distaneA[v, u] = w

    sigma = 10.0
    distaneA[distaneA > 0] = np.exp(-distaneA[distaneA > 0] ** 2 / sigma ** 2)
    return distaneA + np.eye(num_of_vertices, dtype=np.float32)


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0: L[i, j] /= np.sqrt(d[i] * d[j])
    try:
        lambda_max = np.max(np.linalg.eigvalsh(L))
    except:
        lambda_max = 2.0
    return (2 * L / lambda_max - np.eye(n)).astype(np.float32)


def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.eye(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return [torch.from_numpy(i).float() for i in cheb_polynomials]


def masked_mae(preds, labels, null_val=0.0):
    if labels.max() < 200: null_val = 5.0  # SZ-taxi 过滤
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    return torch.mean(loss * mask)


def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mse(preds, labels, null_val=0.0):
    if labels.max() < 200: null_val = 5.0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    return torch.mean(loss * mask)


def masked_mape(preds, labels, null_val=0.0):
    if labels.max() < 200: null_val = 5.0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    if torch.sum(mask) == 0: return torch.tensor(0.0).to(preds.device)

    safe_labels = torch.where(mask.bool(), labels, torch.ones_like(labels))
    loss = torch.abs(preds - labels) / safe_labels
    return torch.sum(loss * mask) / torch.sum(mask)