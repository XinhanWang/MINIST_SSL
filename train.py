import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from tqdm import tqdm
from dataset import get_dataloaders
from models import ResNet18
from utils import calculate_metrics, plot_confusion_matrix, visualize_features,  visualize_dataset_samples, visualize_augmentations
from itertools import cycle

def train_supervised(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    batch_count = 0
    for data, target, _ in loader:
        batch_count += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if batch_count == 0:
        print("Warning: labeled loader has 0 batches; skipping supervised epoch.")
        return 0.0
    return total_loss / batch_count

def train_pseudo_label(model, labeled_loader, unlabeled_loader, optimizer, criterion, device, threshold=0.9):
    # If no unlabeled data, fallback to supervised training
    if unlabeled_loader is None or len(unlabeled_loader) == 0:
        print("Warning: unlabeled_loader is empty. Falling back to supervised training for this epoch.")
        return train_supervised(model, labeled_loader, optimizer, criterion, device)
    if labeled_loader is None or len(labeled_loader) == 0:
        print("Warning: labeled_loader is empty; skipping pseudo-label epoch.")
        return 0.0
        
    model.train()
    iter_labeled = cycle(labeled_loader)
    total_loss = 0.0
    batch_count = 0
    
    # Change: Iterate over unlabeled_loader (dominant)
    for data_u, _, _ in unlabeled_loader:
        batch_count += 1
        data_l, target_l, _ = next(iter_labeled)
            
        data_l, target_l = data_l.to(device), target_l.to(device)
        data_u = data_u.to(device)
        
        optimizer.zero_grad()
        
        # Supervised loss
        out_l = model(data_l)
        loss_l = criterion(out_l, target_l)
        
        # Pseudo-label loss
        with torch.no_grad():
            out_u = model(data_u)
            probs_u = torch.softmax(out_u, dim=1)
            max_probs, targets_u = torch.max(probs_u, dim=1)
            mask = max_probs.ge(threshold).float()
            
        out_u_train = model(data_u)
        loss_u = (torch.nn.functional.cross_entropy(out_u_train, targets_u, reduction='none') * mask).mean()
        
        loss = loss_l + loss_u
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if batch_count == 0:
        print("Warning: unlabeled loader has 0 batches for pseudo-label; skipping epoch.")
        return 0.0
    return total_loss / batch_count

def train_consistency(model, labeled_loader, consistency_loader, optimizer, criterion, device):
    # If no consistency data, fallback to supervised training
    if consistency_loader is None or len(consistency_loader) == 0:
        print("Warning: consistency_loader is empty. Falling back to supervised training for this epoch.")
        return train_supervised(model, labeled_loader, optimizer, criterion, device)
    if labeled_loader is None or len(labeled_loader) == 0:
        print("Warning: labeled_loader is empty; skipping consistency epoch.")
        return 0.0
        
    model.train()
    iter_labeled = cycle(labeled_loader)
    total_loss = 0.0
    batch_count = 0
    
    # Change: Iterate over consistency_loader (dominant)
    for data_w, data_s, _, _ in consistency_loader:
        batch_count += 1
        data_l, target_l, _ = next(iter_labeled)
            
        data_l, target_l = data_l.to(device), target_l.to(device)
        data_w, data_s = data_w.to(device), data_s.to(device)
        
        optimizer.zero_grad()
        
        # Supervised
        out_l = model(data_l)
        loss_l = criterion(out_l, target_l)
        
        # Consistency (Weak vs Strong)
        with torch.no_grad():
            out_w = model(data_w)
            probs_w = torch.softmax(out_w, dim=1)
            max_probs, targets_w = torch.max(probs_w, dim=1)
            mask = max_probs.ge(0.9).float()
            
        out_s = model(data_s)
        loss_u = (torch.nn.functional.cross_entropy(out_s, targets_w, reduction='none') * mask).mean()
        
        loss = loss_l + loss_u
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if batch_count == 0:
        print("Warning: labeled loader has 0 batches for consistency; skipping epoch.")
        return 0.0
    return total_loss / batch_count

def train_freematch(model, labeled_loader, unlabeled_loader, optimizer, criterion, device, context):
    """
    FreeMatch: Self-adaptive Thresholding.
    context: dictionary to store EMA of probabilities (p_model) and global mean max probability (time_p).
    """
    if unlabeled_loader is None or len(unlabeled_loader) == 0:
        return train_supervised(model, labeled_loader, optimizer, criterion, device)
    if labeled_loader is None or len(labeled_loader) == 0:
        return 0.0
        
    model.train()
    iter_labeled = cycle(labeled_loader)
    total_loss = 0.0
    batch_count = 0
    
    # Initialize context if empty
    if 'p_model' not in context:
        # Assuming 10 classes for MNIST/FashionMNIST
        context['p_model'] = torch.ones(10).to(device) / 10.0 
        context['time_p'] = context['p_model'].mean() 

    for data_u_w, data_u_s, _, _ in unlabeled_loader:
        batch_count += 1
        data_l, target_l, _ = next(iter_labeled)
        
        data_l, target_l = data_l.to(device), target_l.to(device)
        data_u_w, data_u_s = data_u_w.to(device), data_u_s.to(device)
        
        optimizer.zero_grad()
        
        # 1. Supervised Loss
        out_l = model(data_l)
        loss_l = criterion(out_l, target_l)
        
        # 2. FreeMatch Loss
        with torch.no_grad():
            out_u_w = model(data_u_w)
            probs_u_w = torch.softmax(out_u_w, dim=1)
            max_probs, targets_u = torch.max(probs_u_w, dim=1)
            
            # Update EMA statistics
            if batch_count == 1 and context['time_p'].item() == 0.1: # Initial state check
                context['p_model'] = probs_u_w.mean(dim=0)
                context['time_p'] = max_probs.mean()
            else:
                context['p_model'] = 0.999 * context['p_model'] + 0.001 * probs_u_w.mean(dim=0)
                context['time_p'] = 0.999 * context['time_p'] + 0.001 * max_probs.mean()
            
            # Calculate Self-Adaptive Threshold
            p_model = context['p_model']
            max_p_model = p_model.max()
            time_p = context['time_p']
            
            # Formula: tau_c = tau_global * (p_local(c) / max(p_local))
            # Here tau_global is estimated by time_p
            thresholds = time_p * (p_model / torch.max(max_p_model, torch.tensor(1e-6).to(device)))
            
            # Get threshold for each sample based on its predicted class
            batch_thresholds = thresholds[targets_u]
            mask = max_probs.ge(batch_thresholds).float()

        out_u_s = model(data_u_s)
        loss_u = (torch.nn.functional.cross_entropy(out_u_s, targets_u, reduction='none') * mask).mean()
        
        loss = loss_l + loss_u
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / batch_count

def evaluate(model, loader, device, return_features=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_features = []
    
    with torch.no_grad():
        for batch in loader:
            data = batch[0]
            target = batch[1]
            data, target = data.to(device), target.to(device)
            if return_features:
                out, feat = model(data, return_features=True)
                all_features.append(feat.cpu())
            else:
                out = model(data)
            
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    if return_features:
        return np.array(all_targets), np.array(all_preds), torch.cat(all_features)
    return np.array(all_targets), np.array(all_preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--method', type=str, default='Supervised', 
                        choices=['Supervised', 'Consistency', 'PseudoLabel', 'FreeMatch'])
    parser.add_argument('--n_labeled', type=int, default=20, help='Labels per class. -1 for full.')
    parser.add_argument('--unlabeled_mu', type=float, default=7.0, help='Ratio of unlabeled batch size to labeled batch size')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    # 新增：学习率调度器相关参数
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine'], help='LR scheduler to use (default: cosine)')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum LR for cosine scheduler (eta_min)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running {args.method} on {args.dataset} with {args.n_labeled} labels/class")

    # Data
    labeled_loader, unlabeled_loader, consistency_loader, test_loader, train_base = \
        get_dataloaders(args.dataset, args.n_labeled, args.batch_size, args.unlabeled_mu)

    # 可视化：每类一个样本
    viz_dir = os.path.join(args.output_dir, args.dataset, args.method, f"labeled_{args.n_labeled}", "viz")
    os.makedirs(viz_dir, exist_ok=True)
    visualize_dataset_samples(train_base, os.path.join(viz_dir, "samples_per_class"), num_classes=10)

    # 若为半监督（有 unlabeled），可视化弱/强增强（原图 / weak / strong）
    if args.n_labeled != -1 and (consistency_loader is not None):
        visualize_augmentations(train_base, consistency_loader, os.path.join(viz_dir, "augmentations"), num_examples=min(8, args.batch_size))
    else:
        print("No semi-supervised unlabeled data; skipping augmentation visualization.")

    # Model
    model = ResNet18().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # 使用余弦学习率衰减（CosineAnnealingLR），按 epoch 更新
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    # Context for FreeMatch state
    fm_context = {}

    # Standard Iterative Training
    for epoch in tqdm(range(args.epochs)):
        # 在每个 epoch 开始打印当前学习率（便于监控）
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch {epoch+1}/{args.epochs} - LR: {current_lr:.6e}")

        if args.method == 'Supervised':
            train_supervised(model, labeled_loader, optimizer, criterion, device)
        elif args.method == 'PseudoLabel':
            if unlabeled_loader is None or len(unlabeled_loader) == 0:
                print("Warning: unlabeled_loader empty; performing supervised training this epoch.")
                train_supervised(model, labeled_loader, optimizer, criterion, device)
            else:
                train_pseudo_label(model, labeled_loader, unlabeled_loader, optimizer, criterion, device)
        elif args.method == 'Consistency':
            if consistency_loader is None or len(consistency_loader) == 0:
                print("Warning: consistency_loader empty; performing supervised training this epoch.")
                train_supervised(model, labeled_loader, optimizer, criterion, device)
            else:
                train_consistency(model, labeled_loader, consistency_loader, optimizer, criterion, device)
        elif args.method == 'FreeMatch':
            if consistency_loader is None or len(consistency_loader) == 0:
                train_supervised(model, labeled_loader, optimizer, criterion, device)
            else:
                train_freematch(model, labeled_loader, consistency_loader, optimizer, criterion, device, fm_context)

        # 每个 epoch 结束后更新学习率（如果启用）
        if scheduler is not None:
            scheduler.step()

    # Final Evaluation on Test Set
    y_true, y_pred, features = evaluate(model, test_loader, device, return_features=True)
    acc, f1 = calculate_metrics(y_true, y_pred)
    
    # 转成百分比并保留两位小数
    acc_pct = round(acc * 100, 2)
    f1_pct = round(f1 * 100, 2)
    
    print(f"Test Accuracy: {acc_pct:.2f}%, Macro F1: {f1_pct:.2f}%")
    
    # Save Results
    save_path = os.path.join(args.output_dir, args.dataset, args.method, f"labeled_{args.n_labeled}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save Metrics (以百分比形式，保留两位小数)
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc_pct, "f1": f1_pct}, f)
        
    # Save Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, os.path.join(save_path, "confusion"))
    visualize_features(features.numpy(), y_true, os.path.join(save_path, "features"), title=f"{args.method} Features")

if __name__ == '__main__':
    main()
