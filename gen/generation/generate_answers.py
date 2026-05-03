import gc
import os
import random
import json
import re  # Added for regex-based resume parsing
from tqdm import tqdm
import pickle
import torch
from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils

def save_jsonl_append(data_dict, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        for key, value in data_dict.items():
            record = {'id': key}
            record.update(value)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_pickle_data(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle_data(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def detect_total_layers(generations):
    for example in generations.values():
        for gen in example.get('generated_answer', []):
            h_out = gen.get('h_out')
            if h_out is None:
                continue
            if not isinstance(h_out, torch.Tensor):
                h_out = torch.tensor(h_out)
            return h_out.shape[1]
    return 0


def build_layer_entries(generations, layer_idx):
    layer_entries = {}
    for tid, example in generations.items():
        new_generations = []
        for gen in example.get('generated_answer', []):
            h_out_raw = gen.get('h_out')
            if h_out_raw is None:
                continue
            if not isinstance(h_out_raw, torch.Tensor):
                h_out = torch.tensor(h_out_raw)
            else:
                h_out = h_out_raw
            if layer_idx >= h_out.shape[1]:
                continue
            layer_hidden = h_out[:, layer_idx:layer_idx + 1, :].clone()
            new_gen = {'h_out': layer_hidden}
            if 'tokens' in gen:
                new_gen['tokens'] = gen['tokens']
            new_generations.append(new_gen)
        if new_generations:
            layer_entries[tid] = {'generated_answer': new_generations}
    return layer_entries


def update_layer_pickles(generations, data_dir, dataset_split, layer_counts):
    if not generations:
        return

    if dataset_split not in layer_counts:
        total_layers = detect_total_layers(generations)
        layer_counts[dataset_split] = total_layers
    else:
        total_layers = layer_counts[dataset_split]

    if total_layers == 0:
        return

    for layer_idx in range(total_layers):
        layer_entries = build_layer_entries(generations, layer_idx)
        if not layer_entries:
            continue

        save_path = os.path.join(data_dir, f"{dataset_split}_layer{layer_idx}.pkl")
        existing = load_pickle_data(save_path)
        existing.update(layer_entries)
        save_pickle_data(existing, save_path)
        del existing

def main(args):
    default_hf_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    if not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = default_hf_home
    hf_home = os.environ["HF_HOME"]
    os.makedirs(hf_home, exist_ok=True)
    if args.data_name == 'squad':
        args.answerable_only = True

    train_dataset, validation_dataset = load_ds(args, args.data_name, seed=args.random_seed)
    
    # Reorganize datasets for simplified downstream handling
    train_indices, _ = utils.split_dataset(train_dataset)
    train_dataset = [train_dataset[i] for i in train_indices]
    val_indices, _ = utils.split_dataset(validation_dataset)
    validation_dataset = [validation_dataset[i] for i in val_indices]

    BRIEF = 'Answer the following question in a single but complete sentence only.\n'
    model = utils.init_model(args)
    CHUNK_SIZE = 10

    splits_to_process = ['train', 'validation'] if args.split == 'both' else [args.split]
    layer_counts = {}

    for dataset_split in splits_to_process:
        print(f"\n{'='*60}\nStarting: {dataset_split}\n{'='*60}")
        
        if dataset_split == 'train':
            if not args.get_training_set_generations: continue
            dataset = train_dataset
            num_samples = args.train_num_samples
        else:
            dataset = validation_dataset
            num_samples = args.valid_num_samples

        # Create fixed indices for reproducibility
        possible_indices = list(range(len(dataset)))
        random.seed(args.random_seed)
        all_indices = random.sample(possible_indices, min(num_samples, len(dataset)))

        # --- Resume detection logic ---
        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir)

        # Search existing part files
        existing_parts = [f for f in os.listdir(args.data_dir) 
                          if f.startswith(f"{dataset_split}_part") and f.endswith(".pkl")]
        
        if existing_parts:
            # Extract part numbers and get the maximum
            part_nums = [int(re.findall(r'part(\d+)', f)[0]) for f in existing_parts]
            next_file_part = max(part_nums) + 1
            start_idx = next_file_part * CHUNK_SIZE
            print(f"Resume: Found {len(existing_parts)} parts. Starting from part {next_file_part} (index {start_idx})")
        else:
            next_file_part = 0
            start_idx = 0
            # On first run, remove existing jsonl
            save_path_jsonl = f"{args.data_dir}/qa_{dataset_split}.jsonl"
            if os.path.exists(save_path_jsonl):
                os.remove(save_path_jsonl)
            # Also remove related layer pkl files for a clean restart
            for fname in os.listdir(args.data_dir):
                if fname.startswith(f"{dataset_split}_layer") and fname.endswith(".pkl"):
                    os.remove(os.path.join(args.data_dir, fname))

        # Slice target indices to process
        indices_to_run = all_indices[start_idx:]
        if not indices_to_run:
            print(f"Split {dataset_split} is already completed.")
            continue

        save_path_jsonl = f"{args.data_dir}/qa_{dataset_split}.jsonl"
        generations = {}
        qa_generations = {}
        file_part = next_file_part
        count_in_chunk = 0

        # Set initial to synchronize progress bar with resume offset
        for it, index in enumerate(tqdm(indices_to_run, initial=start_idx, total=len(all_indices))):
            example = dataset[index]
            unique_id = f"{example['id']}_{index}"

            # ... [omitted: generation logic unchanged] ...
            question = example["question"]
            context = example['context']
            correct_answer = example['answers']['text']
            generations[unique_id] = {'question': question, 'context': context, 'answer': correct_answer}
            qa_generations[unique_id] = {'question': question, 'context': context, 'answer': correct_answer}

            prompt = BRIEF + (f"Passage: {context[:1500]}\n" if args.use_context else "") + f"Question: {question}\nAnswer:"
            full_responses, qa_responses = [], []

            for i in range(args.num_generations):
                res = model.predict(prompt, args.temperature, args.return_layers)
                pred_ans, pred_tokens, log_lik, entropy, h_in, h_out = res
                
                h_in = h_in.cpu() if h_in is not None else None
                h_out = h_out.cpu() if h_out is not None else None
                if dataset_split == 'train' and i > 0: h_out = None

                full_responses.append({'tokens': pred_tokens.cpu(), 'log_lik': log_lik, 'entropy': entropy.tolist(), 'h_in': h_in, 'h_out': h_out})
                qa_responses.append({'pred_ans': pred_ans, 'log_lik': log_lik})

            generations[unique_id]['generated_answer'] = full_responses
            qa_generations[unique_id]['generated_answer'] = qa_responses
            count_in_chunk += 1

            # Save every CHUNK_SIZE samples
            if count_in_chunk >= CHUNK_SIZE:
                save_path_pkl = f"{args.data_dir}/{dataset_split}_part{file_part}.pkl"
                with open(save_path_pkl, 'wb') as f:
                    pickle.dump(generations, f)
                save_jsonl_append(qa_generations, save_path_jsonl)
                update_layer_pickles(generations, args.data_dir, dataset_split, layer_counts)

                print(f"Saved chunk {file_part}")
                generations, qa_generations = {}, {}
                count_in_chunk = 0
                file_part += 1
                gc.collect()
                torch.cuda.empty_cache()

        # Save remainder
        if len(generations) > 0:
            save_path_pkl = f"{args.data_dir}/{dataset_split}_part{file_part}.pkl"
            with open(save_path_pkl, 'wb') as f:
                pickle.dump(generations, f)
            save_jsonl_append(qa_generations, save_path_jsonl)
            update_layer_pickles(generations, args.data_dir, dataset_split, layer_counts)

    print("Run complete.")

if __name__ == '__main__':
    parser = utils.get_parser()
    args, _ = parser.parse_known_args()
    main(args)
