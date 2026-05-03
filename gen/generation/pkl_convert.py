import os
import pickle
import torch
import gc
import argparse


def load_pickle_data(path):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle_data(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def split_to_layer_files(data_dir, mode="train"):

    # ===============================
    # Collect part files
    # ===============================
    pkl_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith(f"{mode}_part") and f.endswith(".pkl")
    ])

    if len(pkl_files) == 0:
        raise ValueError("No part pkl files found")

    # ===============================
    # Detect number of layers
    # ===============================
    sample_data = load_pickle_data(os.path.join(data_dir, pkl_files[0]))

    total_layers = None
    for _, example in sample_data.items():
        generations = example.get("generated_answer", [])
        if not generations:
            continue

        h_out = generations[0].get("h_out")
        if h_out is None:
            continue

        if not isinstance(h_out, torch.Tensor):
            h_out = torch.tensor(h_out)

        total_layers = h_out.shape[1]
        break

    del sample_data
    gc.collect()

    print(f"Detected total layers: {total_layers}")

    # ===============================
    # Layer loop
    # ===============================
    for j in range(total_layers):

        print(f"\nProcessing Layer {j}")

        layer_data = {}

        for pkl_name in pkl_files:

            print(f"  Reading {pkl_name}")

            datapath = os.path.join(data_dir, pkl_name)
            loaded_data = load_pickle_data(datapath)

            for tid, example in loaded_data.items():

                generations = example.get("generated_answer", [])
                new_generations = []

                for gen in generations:

                    h_out_raw = gen.get("h_out")
                    if h_out_raw is None:
                        continue

                    if not isinstance(h_out_raw, torch.Tensor):
                        h_out = torch.tensor(h_out_raw)
                    else:
                        h_out = h_out_raw

                    if j >= h_out.shape[1]:
                        continue

                    # Clone only the required layer
                    layer_hidden = h_out[:, j:j+1, :].clone()

                    new_gen = {
                        "h_out": layer_hidden
                    }

                    if "tokens" in gen:
                        new_gen["tokens"] = gen["tokens"]

                    new_generations.append(new_gen)

                if new_generations:
                    layer_data[tid] = {
                        "generated_answer": new_generations
                    }

            del loaded_data
            gc.collect()

        # ===============================
        # Save once per layer
        # ===============================
        save_path = os.path.join(data_dir, f"{mode}_layer{j}.pkl")
        print(f"Saving {save_path}")

        save_pickle_data(layer_data, save_path)

        del layer_data
        gc.collect()

    print("\nFinished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    split_to_layer_files(args.data_dir, args.mode)


if __name__ == "__main__":
    main()
