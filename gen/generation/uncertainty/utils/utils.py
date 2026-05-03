"""Utility functions."""
import os
import logging
import argparse
import pickle
import json
import numpy as np

# import wandb
# from evaluate import load
STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']
from ..models.huggingface_models import HuggingfaceModel
# from . import openai as oai

BRIEF_PROMPTS = {
    'short': "Answer the following question as briefly as possible.\n",
    'context': "Answer the following question in a single but complete sentence based on the information in the given passage.\n",
    'chat': 'Answer the following question in a single but complete sentence.\n'}

MODELS = {
    'llama3_8b': 'meta-llama/Meta-Llama-3.1-8B',
    'llama3_70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'mistral_12b': 'mistralai/Mistral-Nemo-Instruct-2407'
}


def get_parser(stages=['generate', 'compute']):
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    
    if 'generate' in stages:
        parser.add_argument(
            "--model_name", type=str, default="llama3_8b", help="Model to use for generation.",
        )
        parser.add_argument(
            "--data_name", type=str, default="trivia_qa",
            choices=['trivia_qa', 'squad', 'bioasq', 'nq', 'math'],
            help="Dataset to use")
        parser.add_argument(
            "--train_num_samples", type=int, default=2500,
            help="2000, Number of samples to use")
        parser.add_argument(
            "--valid_num_samples", type=int, default=1500,
            help="Number of samples to use - validation set")
        parser.add_argument(
            "--stop_sequenes",type=str, default='default')
        parser.add_argument(
            "--num_generations", type=int, default=5,
            help="Number of generations to use")
        parser.add_argument(
            "--return_layers", type=str, default='2,2',)
        parser.add_argument(
            "--temperature", type=float, default=0.5,
            help="Temperature")
        parser.add_argument(
            "--get_training_set_generations", default=True,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--use_context", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument('--data_dir', type=str, help='your data directory')
        parser.add_argument(
            "--compute_gt", default=False,
            action=argparse.BooleanOptionalAction,
            help='Trigger evaluation in compute_uncertainty_measures.py')
        parser.add_argument(
            "--compute_se", default=False, help='Trigger entailment measuring in compute_uncertainty_measures.py')
        parser.add_argument(
            "--answerable_only", default=True,
            action=argparse.BooleanOptionalAction,
            help='Exclude unanswerable questions.')
        parser.add_argument(
            "--eval_model", default='gpt-4o-mini', type=str,
            help='Model to use for evaluation.')
        parser.add_argument(
            "--split", type=str, required=True, help='train or validation or both')

    if 'compute' in stages:
        parser.add_argument('--strict_entailment',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_all_generations', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_num_generations', type=int, default=-1)
        parser.add_argument('--entailment_model', default='gpt-3.5', type=str)

    return parser



def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices



def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    reference = {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}
    return reference


def init_model(args, stop_sequences=['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']):
    mn = MODELS[args.model_name]
    model = HuggingfaceModel(
        mn, stop_sequences=stop_sequences,
        max_new_tokens=100)
    return model

    