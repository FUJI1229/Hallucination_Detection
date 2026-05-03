import os
import random
import argparse
import json
import numpy as np

import torch
from torch.utils.data import DataLoader

from model import Detection_Model
from train import train
from validate import test
from utils import load_embedding_train, load_embedding_val, CustomDataset


# ===============================
# Feature dimensions by model
# ===============================
FEATURES = {
    'llama3_8b': 4096,
    'llama3_8b_Instruct': 4096,
    'llama3_70b': 8192,
    'mistral_12b': 5120
}

NUM_LAYER = {
    'llama3_8b': 17,
    'llama3_70b': 14,
    'mistral_12b': 11
}


# ===============================
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===============================
def main(args):
    print('--- Now Loading ---')
    init_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    original_train_data = args.data_name
    val_data_name = args.test_data_list[0]

    # ==================== Layer range ====================
    total_layers = NUM_LAYER[args.model_name]
    start_layer = 0
    end_layer = args.end_layer if args.end_layer is not None else total_layers
    interval = args.interval_layer

    print(f"\nLayer Range: {start_layer} ~ {end_layer - 1} (step={interval})")

    val_results = []

    # ==================== Layer Loop ====================
    for idx_layer in range(start_layer, end_layer, interval):
        idx_layer = 7
        print(f"\n==============================")
        print(f">>> Layer {idx_layer} - Start training")
        print(f"==============================")

        # ==================== Train Data ====================
        args.data_name = original_train_data
        hidden_list, label_list, idx_list = load_embedding_train(args, idx_layer)

        print(f"Train samples: {len(hidden_list)}")

        train_dataset = CustomDataset(idx_list, label_list)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )

        # ==================== Validation Data (per layer) ====================
        args.data_name = val_data_name
        val_hidden_list, val_label_list, _, test_hidden_list, test_label_list = load_embedding_val(args, idx_layer)

        print(f"Val samples (layer {idx_layer}): {len(val_hidden_list)}")

        # ==================== Build Model ====================
        model = Detection_Model(
            n_features=FEATURES[args.model_name],
            pooling_method=args.pooling_method,
            reduced_dim=args.reduced_dim,
            p_norm=args.p_norm,
            device=device
        )
        model.to(device)

        # ==================== Train ====================
        model, model_best, val_auc, val_best, best_th = train(
            args,
            hidden_list,
            label_list,
            train_loader,
            model,
            device,
            idx_layer,
            val_hidden_list=val_hidden_list,
            val_label_list=val_label_list
        )

        val_results.append([idx_layer, val_best, val_auc])

        # ==================== Testing ====================
        test_auc, margin_diff = test(
                args,
                model,
                test_hidden_list,
                test_label_list,
                idx_layer,
                best_th,
                device,
                show_progress=False
            )

            # Best model
        test_auc_best, margin_diff_best = test(
                args,
                model_best,
                test_hidden_list,
                test_label_list,
                idx_layer,
                best_th,
                device,
                show_progress=False
        )

        print(
            f"[{args.data_name}] "
                f"Layer {idx_layer} | "
                f"Best_AUC={test_auc_best:.4f}, "
                f"Last_AUC={test_auc:.4f}"
            )

        # Release memory
        del model
        del model_best
        torch.cuda.empty_cache()

    print("\nTraining Complete.")

    # ==================== Val Summary ====================
    print("\nValidation Summary:")
    for layer, best_auc, last_auc in val_results:
        print(f"Layer {layer}: Best={best_auc:.4f}, Last={last_auc:.4f}")


# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--data_name', type=str, default='trivia_qa')
    parser.add_argument('--test_data_list', type=json.loads, default='["trivia_qa"]')
    parser.add_argument('--model_name', type=str, default='llama3_8b')
    parser.add_argument('--val_sample', type=int, default=300)
    parser.add_argument('--used_sample', type=int, default=5)
    parser.add_argument('--pooling_method', type=str, default='max')
    parser.add_argument('--p_norm', type=float, default=10.0)

    # Layer control
    parser.add_argument('--start_layer', type=int, default=0)
    parser.add_argument('--end_layer', type=int, default=None)
    parser.add_argument('--interval_layer', type=int, default=1)

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument('--reduced_dim', type=int, default=512)

    args = parser.parse_args()
    main(args)
