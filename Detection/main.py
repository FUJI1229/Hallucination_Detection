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
    'llama3_8b_Instruct': 17,
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
    # Stored-layer file indices (*_layer{idx}.pkl); --layers overrides the default range
    if args.layers:
        layer_list = [int(x) for x in args.layers.split(',')]
    else:
        layer_list = list(range(NUM_LAYER[args.model_name]))
    print(f"\nLayers: {layer_list}")

    val_results = []
    summary = {}

    # ==================== Layer Loop ====================
    for idx_layer in layer_list:
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
            device=device,
            pos_weight=args.pos_weight
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

        summary[idx_layer] = {"val_auc_best": float(val_best), "test_auc_best": float(test_auc_best),
                              "test_auc_last": float(test_auc)}

        # Release memory
        del model
        del model_best
        torch.cuda.empty_cache()

    print("\nTraining Complete.")

    # ==================== Val Summary ====================
    print("\nValidation Summary:")
    for layer, best_auc, last_auc in val_results:
        print(f"Layer {layer}: Best={best_auc:.4f}, Last={last_auc:.4f}")

    # Layer chosen by validation AUC only; its test AUC is the reported number
    sel = max(summary, key=lambda k: summary[k]["val_auc_best"])
    print(f"\nValidation-selected layer {sel}: val AUC {summary[sel]['val_auc_best']:.4f}, "
          f"test AUC {summary[sel]['test_auc_best']:.4f}")
    out_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{original_train_data}_{args.pooling_method}_s{args.seed}_summary.json"), "w") as f:
        json.dump({"selected_layer": sel, "layers": summary, "pos_weight": args.pos_weight}, f, indent=1)


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
    parser.add_argument('--layers', type=str, default='',
                        help='comma-separated stored-layer indices to search (default: all)')
    parser.add_argument('--mode', type=str, default='none',
                        help="'reg' L2-normalises every hidden state; 'none' keeps them as stored")
    parser.add_argument('--fixed_split', action='store_true',
                        help='split validation/test by id prefix (LongFact++) instead of random sampling')

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument('--reduced_dim', type=int, default=512)
    parser.add_argument('--pos_weight', type=float, default=1.0,
                        help='positive-class (hallucinated) weight in the BCE loss; 1.0 = unweighted')

    args = parser.parse_args()
    main(args)
