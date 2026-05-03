import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score



def find_best_f1_threshold(y_true, y_score):
    """Compute the threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1)
    # thresholds is one element shorter than precision/recall, so adjust index
    best_threshold = thresholds[best_idx - 1] if best_idx > 0 else thresholds[0]
    return best_threshold, f1[best_idx], precision[best_idx], recall[best_idx]

def plot_stacked_confusion_hist(y_true, y_score, threshold, save_path):
    """Visualize score distribution stacked by TP/TN/FP/FN."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)

    counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    results = []
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            cat, key = 'TP (True Positive)', 'TP'
        elif y_true[i] == 0 and y_pred[i] == 0:
            cat, key = 'TN (True Negative)', 'TN'
        elif y_true[i] == 0 and y_pred[i] == 1:
            cat, key = 'FP (False Positive)', 'FP'
        else:
            cat, key = 'FN (False Negative)', 'FN'
        counts[key] += 1
        results.append(cat)
    print(counts)
    label_map = {k: f"{k} n={counts[v]}" for k, v in [
        ('TN (True Negative)', 'TN'), ('FP (False Positive)', 'FP'),
        ('FN (False Negative)', 'FN'), ('TP (True Positive)', 'TP')
    ]}
    
    df = pd.DataFrame({
        'Score': y_score,
        'Category': [label_map[r] for r in results]
    })

    palette = {
        label_map['TP (True Positive)']: '#2ca02c',
        label_map['TN (True Negative)']: '#1f77b4',
        label_map['FP (False Positive)']: '#d62728',
        label_map['FN (False Negative)']: '#ff7f0e'
    }

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=df, x='Score', hue='Category', 
        hue_order=[label_map['TN (True Negative)'], label_map['FP (False Positive)'], 
                   label_map['FN (False Negative)'], label_map['TP (True Positive)']],
        palette=palette, multiple='stack', bins=40, edgecolor='white', alpha=0.8
    )

    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Category Counts")
    plt.title("Score Distribution by Error Category")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

from sklearn.metrics import roc_auc_score
import torch
import numpy as np

def validate(args, model, hidden_list, label_list, idx_layer, device, show_progress=True):
    model.eval()
    
    y_scores = []
    y_trues  = []
    total_loss = 0.0
    count = 0

    indices = range(len(hidden_list))
    if show_progress:
        from tqdm import tqdm
        indices = tqdm(indices, desc="Validating", leave=False)

    with torch.no_grad():
        for i in indices:
            L = len(hidden_list[i])
            # Convert labels to tensor (for loss computation)
            # Adjust shape as needed to match label_list[i]
            labels_tensor = torch.tensor(label_list[i]).to(device)
            
            for m in range(L):
                hidden = hidden_list[i][m % L].to(device)
                
                # Passing labels makes output include 'loss'
                output = model(
                    hidden_states=[hidden],
                    labels=labels_tensor[m:m+1] # Pass the label for this instance
                )
                
                # Store score and ground truth
                y_scores.append(output["scores"].item())
                y_trues.append(label_list[i][m % L])
                
                # Accumulate loss
                if "loss" in output and output["loss"] is not None:
                    total_loss += output["loss"].item()
                    count += 1

    # Compute AUC
    auc = roc_auc_score(y_trues, y_scores)

    # Find threshold that maximizes F1 score
    best_th, _, _, _ = find_best_f1_threshold(y_trues, y_scores)

    # Compute average validation loss
    avg_val_loss = total_loss / count if count > 0 else 0.0

    return auc, best_th, avg_val_loss

def test(args, model, hidden_list, label_list, idx_layer, th, device, show_progress=True):
    model.eval()
    
    groups = {
        "short":  {"margins": [], "trues": [], "scores": []},
        "middle": {"margins": [], "trues": [], "scores": []},
        "long":   {"margins": [], "trues": [], "scores": []}
    }

    all_trues_total = []
    all_scores_total = []
    all_margins_total = []

    indices = range(len(hidden_list))
    if show_progress:
        from tqdm import tqdm
        indices = tqdm(indices, desc="Testing Analysis", leave=False)

    with torch.no_grad():
        for i in indices:

            L_samples = len(hidden_list[i])

            for m in range(L_samples):

                hidden = hidden_list[i][m].to(device)
                current_L = hidden.shape[0]

                output = model(
                    hidden_states=[hidden],
                    labels=None
                )

                score = output["scores"].item()
                logit = output["logits"].item()
                label = label_list[i][m]

                # Convert label to y in {-1, 1}
                y = 1 if label == 1 else -1

                margin = y * logit

                # Group by sequence length
                if current_L <= 5:
                    key = "short"
                elif current_L <= 30:
                    key = "middle"
                else:
                    key = "long"

                groups[key]["margins"].append(margin)
                groups[key]["trues"].append(label)
                groups[key]["scores"].append(score)

                all_trues_total.append(label)
                all_scores_total.append(score)
                all_margins_total.append(margin)

    # ===============================
    # Print results
    # ===============================

    save_dir = os.path.join(args.save_dir, "plots_by_length")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"{'Layer ' + str(idx_layer) + ' Margin Analysis':^80}")
    print("="*80)

    print(f"{'Group':<10} | {'AUC':<8} | {'N':<6} | {'Avg Margin':<12}")
    print("-"*80)

    for key, data in groups.items():

        n_samples = len(data["trues"])
        if n_samples == 0:
            continue

        avg_margin = np.mean(data["margins"])

        auc_val = roc_auc_score(data["trues"], data["scores"]) if len(set(data["trues"])) > 1 else 0.0

        print(f"{key:<10} | {auc_val:<8.4f} | {n_samples:<6} | {avg_margin:<12.4f}")

        plot_stacked_confusion_hist(
            data["trues"],
            data["scores"],
            th,
            os.path.join(save_dir, f"hist_layer{idx_layer}_{key}.png")
        )

    # ===============================
    # Overall statistics
    # ===============================

    total_auc = roc_auc_score(all_trues_total, all_scores_total)

    total_margin = np.mean(all_margins_total)

    print("-"*80)
    print(f"Total Test AUC:     {total_auc:.4f}")
    print(f"Average Margin:     {total_margin:.4f}")
    print("-"*80)

    plot_stacked_confusion_hist(
        all_trues_total,
        all_scores_total,
        th,
        os.path.join(save_dir, f"hist_layer{idx_layer}_TOTAL.png")
    )

    return total_auc, total_margin
