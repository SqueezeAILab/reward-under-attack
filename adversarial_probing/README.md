# Adversarial Probing of Process Reward Models

This repository contains code to reproduce the adversarial token optimization experiments from **Section 5** of our paper. We demonstrate that a small number of adversarial tokens can inflate PRM scores for incorrect solutions, exposing vulnerabilities in current process reward models.

## Overview

We optimize adversarial tokens that, when appended to (or inserted into) incorrect math solutions, cause PRMs to assign high reward scores. Two optimization modes are supported:

- **Continuous**: Optimizes a single embedding vector directly via gradient descent.
- **Discrete**: Uses Gumbel-Softmax relaxation with entropy regularization to find actual vocabulary tokens.

We evaluate three PRMs:
| Model | Key | Size | Reward Signal |
|---|---|---|---|
| [Skywork-o1-Open-PRM-Qwen-2.5-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B) | `1.5B` | 1.5B | Final step reward (sigmoid logit) |
| [Skywork-o1-Open-PRM-Qwen-2.5-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B) | `7B` | 7B | Final step reward (sigmoid logit) |
| [Qwen2.5-Math-PRM-7B](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) | `Qwen-7B` | 7B | Minimum step reward |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU. The 1.5B experiments can run on a single GPU with ~16GB VRAM; the 7B experiments require ~40GB.

### 2. Download Skywork PRM models

The Skywork models must be downloaded manually. Set the `HF_CACHE_PATH` environment variable to point to the directory containing the model checkpoints:

```bash
export HF_CACHE_PATH=/path/to/your/model/cache

# Download using huggingface-cli
huggingface-cli download Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --local-dir $HF_CACHE_PATH/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-1.5B

huggingface-cli download Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B \
    --local-dir $HF_CACHE_PATH/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B
```

The Qwen model (`Qwen/Qwen2.5-Math-PRM-7B`) is loaded directly from HuggingFace Hub and does not require manual download.

### 3. Create output directories

```bash
mkdir -p logs
```

## Running Experiments

### Run all experiments

```bash
bash run_experiments.sh
```

This runs all experiments from Section 5 (Table 3) sequentially. Each experiment takes ~10-30 minutes depending on GPU and model size.

### Run individual experiments

```bash
# Example: Skywork-1.5B, discrete optimization, k=1 adversarial token
python prm_attack_experiments.py \
    --prm_model 1.5B \
    --experiment batched \
    --num_adv_tokens 1 \
    --adv_position end \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 2
```

### Key arguments

| Argument | Description | Default |
|---|---|---|
| `--prm_model` | PRM to attack: `1.5B`, `7B`, or `Qwen-7B` | `1.5B` |
| `--experiment` | `single` (one trajectory) or `batched` (multiple) | `batched` |
| `--continuous` | Use continuous optimization (omit for discrete) | `False` |
| `--num_adv_tokens` | Number of adversarial tokens (k) | `1` |
| `--adv_position` | Token position: `end` (after solution) or `middle` (after question) | `end` |
| `--num_iterations` | Optimization iterations | `1000` |
| `--num_train_trajectories` | Number of training trajectories | All |
| `--num_eval_trajectories` | Number of evaluation trajectories | All |
| `--run_transfer` | Evaluate transfer to AIME 2025 | `False` |
| `--run_plots` | Generate training curve plots | `False` |
| `--run_3d_landscape` | Generate 3D reward landscape | `False` |
| `--force_rerun` | Force rerun even if cached results exist | `False` |
| `--batch_chunk_size` | Chunk size for gradient accumulation (reduce if OOM) | `1` |
| `--cache_dir` | Directory for cached results | `./experiment_cache` |
| `--distributed` | Use distributed training across multiple GPUs | `False` |

### Multi-GPU support

Pass `--distributed` to distribute optimization across multiple GPUs. This can help speed up training or allow running larger experiments that don't fit in a single GPU's memory.

### Notes on PRM types

- **Skywork PRMs** (`1.5B`, `7B`): Adversarial tokens are appended **after** the solution (`--adv_position end`). The attack optimizes the final step reward.
- **Qwen PRM** (`Qwen-7B`): Adversarial tokens are inserted **before** the solution (`--adv_position middle`). The attack optimizes the minimum step reward across all steps.

## Output Structure

All results are cached in `./experiment_cache/` (configurable via `--cache_dir`). The naming convention is:

```
{experiment}_{model}_{mode}_{k}tok_{position}_{n}traj_{type}
```

For Qwen experiments, filenames are prefixed with `qwen_`.

### Saved files per experiment

| File | Description |
|---|---|
| `*_result.pkl` | Full optimization result (rewards, losses, token history) |
| `*_best_token.pt` | Best adversarial token embedding checkpoint |
| `*_discrete_token_ids.pt` | Optimized discrete token IDs (discrete mode only) |
| `*_metrics.pkl` | Evaluation metrics (pre/post attack rewards) |
| `*_config.json` | Experiment configuration |
| `*_transfer.pkl` | Transfer evaluation results on AIME 2025 |
| `*_training_curves.png` | Training reward curves over iterations |

### 3D landscape plots

```
experiment_cache/landscape_3d/{experiment}/{model}/{mode}/{position}/
    single_{model}_{mode}_{k}tok_{position}_adversarial.png
    single_{model}_{mode}_{k}tok_{position}_random.png
```

### Landscape grid data

```
experiment_cache/landscape_cache/
    {experiment}_{model}_{mode}_{k}tok_{position}_{type}_grid.npz
```

## Decoding Optimized Tokens

After running discrete optimization experiments, use the decode script to inspect what tokens were found:

```bash
python decode_optimized_tokens.py
```

This loads all `*_discrete_token_ids.pt` files from `./experiment_cache/` and decodes them using the appropriate tokenizer.

## Datasets

Experiments use math competition problems from:
- **Training**: [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) (AIME 2024)
- **Transfer evaluation**: [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025) (AIME 2025)

Solution trajectories are generated using [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) via vLLM and cached automatically.

## Repository Structure

```
.
├── prm_attack_experiments.py    # Core experiment code (optimization, evaluation, plotting)
├── run_experiments.sh           # Shell script to reproduce all Section 5 experiments
├── decode_optimized_tokens.py   # Decode optimized discrete token IDs
├── requirements.txt             # Python dependencies
├── constants/
│   └── model_constants.py       # Model paths and processor/class mappings
├── utils/
│   ├── io_utils.py              # Input preparation and reward derivation dispatchers
│   ├── processors/
│   │   ├── skywork_o1_open_prm.py   # Skywork PRM input/reward processing
│   │   └── qwen_math_prm.py        # Qwen Math PRM input/reward processing
│   └── models/
│       ├── __init__.py
│       ├── qwen_math_prm/
│       │   └── model.py             # Qwen2ForProcessRewardModel
│       └── skywork_o1_open_prm/
│           ├── model.py             # SkyworkO1OpenPRMForProcessRewardModel
│           └── modeling_base.py     # PreTrainedModelWrapper base class
```
