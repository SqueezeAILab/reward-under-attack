#!/usr/bin/env python3
"""
Script to load and decode all optimized discrete token IDs from experiments.
Prints the decoded tokens and other relevant information in an organized format.
"""

import os
import re
import glob
import torch
from transformers import AutoTokenizer
from collections import defaultdict

# Model paths - set HF_CACHE_PATH env var or pass --cache_path
HF_CACHE_PATH = os.environ.get("HF_CACHE_PATH", "./hf_cache")
MODEL_PATHS = {
    "1.5B": f"{HF_CACHE_PATH}/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
    "7B": f"{HF_CACHE_PATH}/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B",
    "Qwen-7B": "Qwen/Qwen2.5-Math-PRM-7B",
}

# Experiment cache directory
CACHE_DIR = os.environ.get("EXPERIMENT_CACHE_DIR", "./experiment_cache")


def parse_filename(filename: str) -> dict:
    """
    Parse experiment info from filename.
    Expected format: {experiment}_{model}_discrete_{n}tok_{position}_{n}traj_discrete_token_ids.pt
    Or older format: {experiment}_{model}_discrete_{n}tok_{n}traj_discrete_token_ids.pt
    """
    basename = os.path.basename(filename)
    
    # Try format with position (handles both Skywork and Qwen filenames)
    # e.g., batched_1.5B_discrete_1tok_end_8traj_discrete_token_ids.pt
    # e.g., qwen_batched_Qwen-7B_discrete_1tok_middle_8traj_discrete_token_ids.pt
    match = re.match(
        r"(?:qwen_)?(\w+)_([\w\.\-]+B)_discrete_(\d+)tok_(\w+)_(\w+)traj_discrete_token_ids\.pt",
        basename
    )
    if match:
        return {
            "experiment": match.group(1),
            "model": match.group(2),
            "num_tokens": int(match.group(3)),
            "position": match.group(4),
            "num_trajectories": match.group(5),
            "filename": basename,
        }

    return None


def load_tokenizer(model_key: str):
    """Load tokenizer for the specified model."""
    if model_key not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_PATHS.keys())}")
    
    model_path = MODEL_PATHS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


def decode_tokens(token_ids: torch.Tensor, tokenizer) -> dict:
    """Decode token IDs and return detailed information."""
    token_ids_list = token_ids.tolist()
    
    # Decode as a single string
    full_decoded = tokenizer.decode(token_ids_list, skip_special_tokens=False)
    
    # Decode each token individually
    individual_tokens = []
    for tid in token_ids_list:
        decoded = tokenizer.decode([tid], skip_special_tokens=False)
        individual_tokens.append({
            "id": tid,
            "decoded": decoded,
            "repr": repr(decoded),
        })
    
    return {
        "token_ids": token_ids_list,
        "num_tokens": len(token_ids_list),
        "full_decoded": full_decoded,
        "individual_tokens": individual_tokens,
    }


def print_experiment_results(exp_info: dict, decoded_info: dict):
    """Print formatted results for an experiment."""
    print("=" * 80)
    print(f"EXPERIMENT: {exp_info['experiment']} | MODEL: {exp_info['model']} | "
          f"TOKENS: {exp_info['num_tokens']} | POSITION: {exp_info['position']} | "
          f"TRAJECTORIES: {exp_info['num_trajectories']}")
    print("=" * 80)
    print(f"File: {exp_info['filename']}")
    print(f"Number of tokens: {decoded_info['num_tokens']}")
    print()
    
    # Print full decoded string
    print("Full decoded string:")
    print("-" * 40)
    print(decoded_info['full_decoded'])
    print("-" * 40)
    print()
    
    # Print individual tokens (limit to first 20 if many)
    print("Individual tokens:")
    tokens_to_show = decoded_info['individual_tokens']
    if len(tokens_to_show) > 20:
        print(f"(Showing first 20 of {len(tokens_to_show)} tokens)")
        tokens_to_show = tokens_to_show[:20]
    
    for i, tok in enumerate(tokens_to_show):
        print(f"  [{i:3d}] ID: {tok['id']:6d} | {tok['repr']}")
    
    if len(decoded_info['individual_tokens']) > 20:
        print(f"  ... and {len(decoded_info['individual_tokens']) - 20} more tokens")
    
    print()
    
    # Print token ID statistics
    token_ids = decoded_info['token_ids']
    print(f"Token ID range: [{min(token_ids)}, {max(token_ids)}]")
    print(f"Unique tokens: {len(set(token_ids))} / {len(token_ids)}")
    print()


def main():
    # Find all discrete token ID files
    pattern = os.path.join(CACHE_DIR, "*_discrete_token_ids.pt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No discrete token ID files found in {CACHE_DIR}")
        return
    
    print(f"Found {len(files)} discrete token ID files")
    print()
    
    # Group by model to load tokenizer only once per model
    experiments_by_model = defaultdict(list)
    unparsed_files = []
    
    for filepath in sorted(files):
        exp_info = parse_filename(filepath)
        if exp_info:
            exp_info['filepath'] = filepath
            experiments_by_model[exp_info['model']].append(exp_info)
        else:
            unparsed_files.append(filepath)
    
    if unparsed_files:
        print("Warning: Could not parse the following files:")
        for f in unparsed_files:
            print(f"  {os.path.basename(f)}")
        print()
    
    # Process each model's experiments
    for model_key in sorted(experiments_by_model.keys()):
        experiments = experiments_by_model[model_key]
        
        print("#" * 80)
        print(f"# MODEL: {model_key}")
        print(f"# {len(experiments)} experiments")
        print("#" * 80)
        print()
        
        # Load tokenizer for this model
        print(f"Loading tokenizer for {model_key}...")
        tokenizer = load_tokenizer(model_key)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        print()
        
        # Sort experiments by number of tokens, then by position
        experiments_sorted = sorted(
            experiments, 
            key=lambda x: (x['num_tokens'], x.get('position', ''))
        )
        
        for exp_info in experiments_sorted:
            # Load token IDs
            token_ids = torch.load(exp_info['filepath'], map_location='cpu')
            
            # Decode
            decoded_info = decode_tokens(token_ids, tokenizer)
            
            # Print results
            print_experiment_results(exp_info, decoded_info)
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_experiments = sum(len(exps) for exps in experiments_by_model.values())
    print(f"Total experiments processed: {total_experiments}")
    for model_key in sorted(experiments_by_model.keys()):
        print(f"  {model_key}: {len(experiments_by_model[model_key])} experiments")


if __name__ == "__main__":
    main()
