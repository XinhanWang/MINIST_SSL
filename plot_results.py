import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def load_results(root_dir='./results'):
    data = []
    for dataset in os.listdir(root_dir):
        ds_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(ds_path): continue
        
        for method in os.listdir(ds_path):
            method_path = os.path.join(ds_path, method)
            if not os.path.isdir(method_path): continue
            
            for setting in os.listdir(method_path):
                if not setting.startswith("labeled_"): continue
                n_labeled = int(setting.split("_")[1])
                
                metrics_file = os.path.join(method_path, setting, "metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        data.append({
                            "Dataset": dataset,
                            "Method": method,
                            "Labeled": n_labeled,
                            "Accuracy": metrics['accuracy'],
                            "F1": metrics['f1']
                        })
    return pd.DataFrame(data)

def plot_curves(df):
    datasets = df['Dataset'].unique()
    
    out_dir = './results'
    os.makedirs(out_dir, exist_ok=True)
    
    for ds in datasets:
        ds_df = df[df['Dataset'] == ds]
        
        # Filter out full supervised (-1) for the curve, plot it as a horizontal line maybe?
        # Or just include it if x-axis allows. Usually curves are for 20/50/100.
        curve_df = ds_df[ds_df['Labeled'] != -1].sort_values('Labeled')
        
        # Accuracy Plot
        plt.figure(figsize=(10, 6))
        for method in curve_df['Method'].unique():
            subset = curve_df[curve_df['Method'] == method]
            plt.plot(subset['Labeled'], subset['Accuracy'], marker='o', label=method)
            
        plt.title(f'Learning Curve (Top-1 Accuracy) - {ds}')
        plt.xlabel('Number of Labeled Samples per Class')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        png_path = os.path.join(out_dir, f'{ds}_accuracy_curve.png')
        svg_path = os.path.join(out_dir, f'{ds}_accuracy_curve.svg')
        plt.savefig(png_path)
        plt.savefig(svg_path)
        plt.close()
        
        # F1 Plot
        plt.figure(figsize=(10, 6))
        for method in curve_df['Method'].unique():
            subset = curve_df[curve_df['Method'] == method]
            plt.plot(subset['Labeled'], subset['F1'], marker='o', label=method)
            
        plt.title(f'Learning Curve (Macro-F1) - {ds}')
        plt.xlabel('Number of Labeled Samples per Class')
        plt.ylabel('Macro F1')
        plt.legend()
        plt.grid(True)
        png_path = os.path.join(out_dir, f'{ds}_f1_curve.png')
        svg_path = os.path.join(out_dir, f'{ds}_f1_curve.svg')
        plt.savefig(png_path)
        plt.savefig(svg_path)
        plt.close()

if __name__ == "__main__":
    df = load_results()
    if not df.empty:
        plot_curves(df)
        print("Plots saved to ./results/")
    else:
        print("No results found.")
