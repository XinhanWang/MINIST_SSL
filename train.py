import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from tqdm import tqdm
from dataset import get_dataloaders
from models import SimpleCNN
from utils import calculate_metrics, plot_confusion_matrix, visualize_features, construct_knn_graph
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
                        choices=['Supervised', 'Consistency', 'LabelProp'])
    parser.add_argument('--n_labeled', type=int, default=20, help='Labels per class. -1 for full.')
    parser.add_argument('--unlabeled_mu', type=float, default=7.0, help='Ratio of unlabeled batch size to labeled batch size')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
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

    # Model
    model = SimpleCNN().to(device)
    # Use AdamW instead of SGD
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Standard Iterative Training
    for epoch in tqdm(range(args.epochs)):
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
