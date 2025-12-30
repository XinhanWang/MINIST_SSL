import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.manifold import TSNE
import os
from pykeops.torch import LazyTensor

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

def plot_confusion_matrix(y_true, y_pred, save_path, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    
    # Ensure directory exists if save_path contains a folder
    dirpath = os.path.dirname(save_path)
    if (dirpath):
        os.makedirs(dirpath, exist_ok=True)

    # Counts heatmap (use Reds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f'Confusion Matrix {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_path}_cm.png")
    plt.savefig(f"{save_path}_cm.svg")
    plt.close()
    
    # Save CSV (counts)
    np.savetxt(f"{save_path}_cm.csv", cm, delimiter=",", fmt='%d')

    # Row-normalized confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_rownorm = cm.astype(float) / row_sums
        cm_rownorm = np.nan_to_num(cm_rownorm)  # replace nan/inf with 0 for zero-row cases
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_rownorm, annot=True, fmt='.2f', cmap='Reds')
    plt.title(f'Confusion Matrix (Row-normalized) {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_path}_cm_rownorm.png")
    plt.savefig(f"{save_path}_cm_rownorm.svg")
    plt.close()
    np.savetxt(f"{save_path}_cm_rownorm.csv", cm_rownorm, delimiter=",", fmt='%.6f')

    # Zero diagonal
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_no_diag, annot=True, fmt='d', cmap='Reds')
    plt.title(f'Confusion Matrix (No Diagonal) {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_path}_cm_nodiag.png")
    plt.savefig(f"{save_path}_cm_nodiag.svg")
    plt.close()
    np.savetxt(f"{save_path}_cm_nodiag.csv", cm_no_diag, delimiter=",", fmt='%d')

    # Row-normalized zero-diagonal confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums_nodiag = cm_no_diag.sum(axis=1, keepdims=True)
        cm_nodiag_rownorm = cm_no_diag.astype(float) / row_sums_nodiag
        cm_nodiag_rownorm = np.nan_to_num(cm_nodiag_rownorm)  # replace nan/inf with 0 for zero-row cases
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_nodiag_rownorm, annot=True, fmt='.2f', cmap='Reds')
    plt.title(f'Confusion Matrix (No Diagonal, Row-normalized) {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_path}_cm_nodiag_rownorm.png")
    plt.savefig(f"{save_path}_cm_nodiag_rownorm.svg")
    plt.close()
    np.savetxt(f"{save_path}_cm_nodiag_rownorm.csv", cm_nodiag_rownorm, delimiter=",", fmt='%.6f')

def visualize_features(features, labels, save_path, title="t-SNE"):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    # Ensure directory exists if save_path contains a folder
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.savefig(f"{save_path}_tsne.png")
    plt.savefig(f"{save_path}_tsne.svg")
    plt.close()

def construct_knn_graph(features, k=10,sigma=None):
    """
    features: [N, D]
    labeled_indices: [M] (LongTensor)
    labels: [M] (LongTensor) 对应的标签值
    """
    N = features.size(0)
    device = features.device

    print("=== 1. Constructing k-NN Graph ===")
    # --- 使用 KeOps 进行 k-NN ---
    x_i = LazyTensor(features[:, None, :])  # (N, 1, D)
    x_j = LazyTensor(features[None, :, :])  # (1, N, D)
    
    # 计算欧氏距离平方
    d_ij = ((x_i - x_j) ** 2).sum(-1)
    
    # 获取 k+1 个最近邻 (包含自身)
    # indices: [N, k+1], distances: [N, k+1]
    distances, indices = d_ij.Kmin_argKmin(K=k + 1, dim=1) 
    
    # --- 2. 构建权重矩阵 (RBF Kernel) ---
    # 如果没有指定 sigma，可以用距离的平均值估算
    if sigma is None:
        # 取第 k 个邻居的距离作为参考尺度
        sigma = torch.mean(distances[:, -1]) ** 0.5
        print(f"Auto-estimated sigma: {sigma:.4f}")
    
    # W_ij = exp(- dist^2 / (2 * sigma^2))
    # 注意：distances 已经是平方距离
    weights = torch.exp(-distances / (2 * sigma ** 2))
    
    # 展平以构建稀疏矩阵
    row = torch.arange(N).repeat_interleave(k+1).to(device)
    col = indices.flatten()
    data = weights.flatten()
    
    # 构建初始稀疏矩阵 W (N, N)
    # 注意：这里可能会有重复的边 (如果 A 是 B 的邻居，且 B 是 A 的邻居)
    # 为了对称化处理，我们先转为 dense 或者利用 coalesced sparse 处理
    # 鉴于 N 可能很大，我们尽量保持 sparse，但为了演示清晰和对称化方便，
    # 标准做法是 W = (W + W.T) / 2。
    # 下面是一种高效的 Sparse 对称化 + 归一化方法：
    
    indices_sparse = torch.stack([row, col])
    W = torch.sparse_coo_tensor(indices_sparse, data, (N, N)).coalesce()
    
    # 对称化: W = (W + W^t) / 2
    # 注意：PyTorch sparse加法比较耗显存，简单的做法是只保留 kNN 的有向边并进行归一化，
    # 或者在这里做 symmetrization。为了稳定性，推荐 Label Spreading 的对称归一化。
    W = W + W.t()  # 变为对称 (这里近似为相加，省去除2，因为后续会做度归一化)
    
    # --- 3. 归一化 (Symmetric Normalization) ---
    # S = D^(-1/2) * W * D^(-1/2)
    
    # 计算度 D: 按行求和
    W_dense_indices = W.indices()
    W_values = W.values()
    
    # 这是一个 trick 来计算 sparse 矩阵的行和 (Degree)
    # 创建一个全1向量，与 W 相乘得到度
    # 或者直接用 scatter_add
    D = torch.zeros(N).to(device)
    D.scatter_add_(0, W_dense_indices[0], W_values)
    
    # 处理度为0的情况 (防止除0)
    D[D == 0] = 1e-10
    
    # 计算 D^(-1/2)
    D_inv_sqrt = D.pow(-0.5)
    
    # 计算 S 的值: S_ij = D_i^(-1/2) * W_ij * D_j^(-1/2)
    # 利用 row 和 col 索引直接乘
    row_idx, col_idx = W_dense_indices[0], W_dense_indices[1]
    S_values = D_inv_sqrt[row_idx] * W_values * D_inv_sqrt[col_idx]
    
    S = torch.sparse_coo_tensor(W_dense_indices, S_values, (N, N)).coalesce()
    
    return S

