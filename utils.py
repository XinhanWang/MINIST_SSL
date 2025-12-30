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

def _tensor_to_numpy_image(img, mean=(0.5,), std=(0.5,)):
    """
    将 tensor / PIL / numpy 图像转换为 numpy 图像（H,W[,C]），数值范围 [0,1]
    """
    import numpy as _np
    try:
        from PIL import Image as _PILImage
    except Exception:
        _PILImage = None

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 4:
            img = img[0]
        if img.dim() == 3:
            c, h, w = img.shape
            mean_t = torch.tensor(mean).view(c, 1, 1)
            std_t = torch.tensor(std).view(c, 1, 1)
            img = img * std_t + mean_t
            img = img.clamp(0, 1)
            np_img = img.permute(1, 2, 0).numpy()
            if c == 1:
                np_img = np_img[:, :, 0]
            return np_img
        elif img.dim() == 2:
            return img.numpy()
    elif isinstance(img, _np.ndarray):
        return img.astype(_np.float32)
    elif _PILImage is not None and isinstance(img, _PILImage.Image):
        return _np.array(img).astype(_np.float32) / 255.0
    raise TypeError(f"Unsupported image type: {type(img)}")

def visualize_dataset_samples(base_dataset, save_path, num_classes=10, mean=(0.5,), std=(0.5,)):
    """
    为每个类别展示一个样本并保存（save_path 不带后缀）
    """
    import math
    # 获取标签数组
    if hasattr(base_dataset, 'targets'):
        labels = np.array(base_dataset.targets)
    elif hasattr(base_dataset, 'labels'):
        labels = np.array(base_dataset.labels)
    else:
        labels = np.array([int(base_dataset[i][1]) for i in range(len(base_dataset))])

    imgs = []
    titles = []
    for cls in range(num_classes):
        idxs = np.where(labels == cls)[0]
        if idxs.size == 0:
            continue
        idx = int(idxs[0])
        img, _ = base_dataset[idx]
        np_img = _tensor_to_numpy_image(img, mean, std)
        imgs.append(np_img)
        titles.append(f"{cls}")

    if len(imgs) == 0:
        print("No images found for visualization.")
        return

    n = len(imgs)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=(3 * ncols, 3 * nrows))
    for i, img in enumerate(imgs):
        ax = plt.subplot(nrows, ncols, i + 1)
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis('off')

    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png")
    plt.savefig(f"{save_path}.svg")
    plt.close()
    print(f"Saved samples-per-class visualization to {save_path}.(png/svg)")

def visualize_augmentations(base_dataset, consistency_loader, save_path, num_examples=8, mean=(0.5,), std=(0.5,)):
    """
    使用 consistency_loader 的一个 batch，将每个样本的 (原图, weak, strong) 并排可视化并保存
    """
    if consistency_loader is None:
        print("consistency_loader is None; skipping augmentation visualization.")
        return

    # 取一个 batch
    for batch in consistency_loader:
        img_w, img_s, targets, indices = batch
        break

    B = img_w.size(0)
    n = min(num_examples, B)
    plt.figure(figsize=(3 * 3, 3 * n))
    for i in range(n):
        idx = int(indices[i].item()) if isinstance(indices[i], torch.Tensor) else int(indices[i])
        orig_img, _ = base_dataset[idx]
        np_orig = _tensor_to_numpy_image(orig_img, mean, std)
        np_w = _tensor_to_numpy_image(img_w[i], mean, std)
        np_s = _tensor_to_numpy_image(img_s[i], mean, std)
        row_imgs = [np_orig, np_w, np_s]
        titles = ['orig', 'weak', 'strong']
        for j, im in enumerate(row_imgs):
            ax = plt.subplot(n, 3, i * 3 + j + 1)
            if im.ndim == 2:
                ax.imshow(im, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(im)
            if j == 1:
                ax.set_title(f"{titles[j]} idx={idx} target={int(targets[i].item())}")
            else:
                ax.set_title(titles[j])
            ax.axis('off')

    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png")
    plt.savefig(f"{save_path}.svg")
    plt.close()
    print(f"Saved augmentation visualization to {save_path}.(png/svg)")

