import os
import json
import datasets

def load_ds(args, dataset_name, seed, add_options=None):
    """Load dataset based on name and sample counts in args."""
    train_dataset, validation_dataset = None, None
    
    # 1. SQuAD v2
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    # 2. Natural Questions (NQ)
    elif dataset_name == 'nq':
        import hashlib
        dataset = datasets.load_dataset("nq_open")
        tmp_train = dataset["train"]
        tmp_valid = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
        reformat = lambda x: {
            'question': x['question'] + '?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }
        train_dataset = [reformat(d) for d in tmp_train]
        validation_dataset = [reformat(d) for d in tmp_valid]

    # 3. TriviaQA
    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        # TriviaQA often has non-standard splits, so split here with a fixed seed
        dataset = dataset.train_test_split(test_size=0.3, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    # 4. BioASQ
    elif dataset_name == "bioasq":
        path = os.getenv("BIOASQ_TRAIN_PATH")
        if not path:
            raise ValueError("BIOASQ dataset requires BIOASQ_TRAIN_PATH environment variable.")
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {"question": [], "answers": [], "id": [], "context": []}

        for question in questions:
            if "exact_answer" not in question: continue
            dataset_dict["question"].append(question["body"])
            exact_answers = question["exact_answer"]
            if not isinstance(exact_answers, list): exact_answers = [exact_answers]
            
            dataset_dict["answers"].append({
                "text": [ans[0] if isinstance(ans, list) else ans for ans in exact_answers],
                "answer_start": [0] * len(exact_answers)
            })
            dataset_dict["id"].append(question["id"])
            dataset_dict["context"].append("")

        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.shuffle(seed=seed)

        # Apply sample limits from args
        t_limit = args.train_num_samples
        v_limit = args.valid_num_samples
        total = len(dataset)

        if t_limit + v_limit > total:
            print(f"⚠️ Warning: Requested {t_limit+v_limit}, but total is {total}.")
            t_limit = min(t_limit, total)
            v_limit = max(0, total - t_limit)

        train_dataset = dataset.select(range(t_limit))
        validation_dataset = dataset.select(range(t_limit, t_limit + v_limit))
    # 5. MATH (Competition Math)    
    elif dataset_name == "math":
        import re
        dataset = datasets.load_dataset("qwedsacf/competition_math")
        
        # --- Fix: split train when a dedicated 'test' split is unavailable ---
        full_data = dataset["train"]
        
        # Set validation split ratio (example: 20% of full data)
        # If args.valid_num_samples is fixed, you can split to match that count instead
        split_data = full_data.train_test_split(test_size=0.5, seed=seed)
        
        tmp_train = split_data["train"]
        tmp_valid = split_data["test"]
        # --------------------------------------------------------

        def extract_boxed_answer(text):
            """Helper to extract content from \boxed{answer} (supports nested braces)."""
            idx = text.rfind("\\boxed{")
            if idx < 0:
                return text.split('\n')[-1].strip()

            idx += 7 
            brace_count = 1
            answer = ""
            
            for char in text[idx:]:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                
                if brace_count == 0:
                    break
                answer += char
                
            return answer
        import hashlib
        reformat = lambda x: {
            'question': x['problem'],
            'answers': {'text': [extract_boxed_answer(x['solution'])]},
            'context': '', 
            'id': hashlib.md5(x['problem'].encode('utf-8')).hexdigest(),
        }

        # Use min() to avoid out-of-range errors in select
        train_indices = range(min(args.train_num_samples, len(tmp_train)))
        valid_indices = range(min(args.valid_num_samples, len(tmp_valid)))

        train_dataset = [reformat(d) for d in tmp_train.shuffle(seed=seed).select(train_indices)]
        validation_dataset = [reformat(d) for d in tmp_valid.shuffle(seed=seed).select(valid_indices)]    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Common note: if needed, also guard SQuAD/NQ/TriviaQA by max list length ---
    # (Since SQuAD etc. are HF dataset objects, this code does not slice them here;
    #  they are subsampled in the caller's main() via random.sample and args.valid_num_samples.)

    return train_dataset, validation_dataset
