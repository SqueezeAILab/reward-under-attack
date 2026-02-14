import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import re
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, asdict

from constants.model_constants import MODEL_CLASS_MAP
from utils.io_utils import prepare_input, derive_step_rewards
from utils.processors.skywork_o1_open_prm import STEP_SEP_TOKEN


# ============================================
# ARGUMENT PARSING
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PRM Attack Experiments - Adversarial Token Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model selection
    parser.add_argument(
        "--prm_model", type=str, default="1.5B", choices=["1.5B", "7B", "Qwen-7B"],
        help="Which PRM model to use: 1.5B/7B for Skywork, Qwen-7B for Qwen Math PRM"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run on"
    )
    parser.add_argument(
        "--hf_cache_path", type=str,
        default=os.environ.get("HF_CACHE_PATH", "./hf_cache"),
        help="Path to HuggingFace cache directory for Skywork model checkpoints"
    )
    
    # Experiment type
    parser.add_argument(
        "--experiment", type=str, default="batched", choices=["single", "batched"],
        help="Experiment type: single trajectory or batched trajectories"
    )
    
    # Optimization settings
    parser.add_argument(
        "--continuous", action="store_true", default=False,
        help="Use continuous optimization (no entropy regularization)"
    )
    parser.add_argument(
        "--num_adv_tokens", type=int, default=1,
        help="Number of adversarial tokens to optimize"
    )
    parser.add_argument(
        "--adv_position", type=str, default="end", choices=["end", "middle"],
        help="Position of adversarial tokens: 'end' (after solution) or 'middle' (after question, before solution)"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1000,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1,
        help="Learning rate for optimization"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature for Gumbel-Softmax"
    )
    parser.add_argument(
        "--entropy_weight_start", type=float, default=0.0001,
        help="Starting entropy weight for discrete optimization"
    )
    parser.add_argument(
        "--entropy_weight_end", type=float, default=0.1,
        help="Ending entropy weight for discrete optimization"
    )
    parser.add_argument(
        "--entropy_schedule", type=str, default="cosine",
        choices=["linear", "exponential", "cosine"],
        help="Entropy weight schedule type"
    )
    
    # Dataset settings
    parser.add_argument(
        "--num_train_trajectories", type=int, default=None,
        help="Number of trajectories to use for training (None = use all)"
    )
    parser.add_argument(
        "--num_eval_trajectories", type=int, default=None,
        help="Number of trajectories to use for evaluation (None = use all)"
    )
    parser.add_argument(
        "--single_traj_question_idx", type=int, default=0,
        help="Question index for single trajectory experiment"
    )
    
    # Run controls
    parser.add_argument(
        "--run_optimization", action="store_true", default=True,
        help="Run the optimization"
    )
    parser.add_argument(
        "--run_transfer", action="store_true", default=False,
        help="Run transfer evaluation"
    )
    parser.add_argument(
        "--run_analysis", action="store_true", default=False,
        help="Run analysis and print results"
    )
    parser.add_argument(
        "--run_plots", action="store_true", default=False,
        help="Generate reward plots"
    )
    parser.add_argument(
        "--run_3d_landscape", action="store_true", default=False,
        help="Generate 3D reward landscape visualization"
    )
    parser.add_argument(
        "--landscape_grid_size", type=int, default=50,
        help="Grid size for 3D landscape visualization"
    )
    
    # Caching
    parser.add_argument(
        "--cache_dir", type=str, default="./experiment_cache",
        help="Directory for caching results"
    )
    parser.add_argument(
        "--use_cache", action="store_true", default=True,
        help="Use cached results if available"
    )
    parser.add_argument(
        "--skip_if_exists", action="store_true", default=True,
        help="Skip optimization if checkpoint exists"
    )
    parser.add_argument(
        "--force_rerun", action="store_true", default=False,
        help="Force rerun even if cache exists"
    )
    
    # Distributed training
    parser.add_argument(
        "--distributed", action="store_true", default=False,
        help="Use distributed training"
    )
    
    # Misc
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100,
        help="Logging interval during optimization"
    )
    parser.add_argument(
        "--batch_chunk_size", type=int, default=1,
        help="Chunk size for batched optimization (memory control)"
    )
    
    return parser.parse_args()


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    # Model
    prm_model: str
    device: str
    hf_cache_path: str
    
    # Experiment type
    experiment: str
    
    # Optimization
    continuous: bool
    num_adv_tokens: int
    adv_position: str  # "end" or "middle"
    num_iterations: int
    learning_rate: float
    temperature: float
    entropy_weight_start: float
    entropy_weight_end: float
    entropy_schedule: str
    
    # Dataset
    num_train_trajectories: Optional[int]
    num_eval_trajectories: Optional[int]
    single_traj_question_idx: int
    
    # Run controls
    run_optimization: bool
    run_transfer: bool
    run_analysis: bool
    run_plots: bool
    run_3d_landscape: bool
    landscape_grid_size: int
    
    # Caching
    cache_dir: str
    use_cache: bool
    skip_if_exists: bool
    force_rerun: bool
    
    # Distributed
    distributed: bool
    
    # Misc
    seed: int
    log_interval: int
    batch_chunk_size: int
    
    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name."""
        mode = "continuous" if self.continuous else "discrete"
        n_traj = self.num_train_trajectories or "all"
        pos = self.adv_position  # "end" or "middle"
        # Add "qwen_" prefix for Qwen models
        prm_type = get_prm_type(self.prm_model)
        prm_prefix = "qwen_" if prm_type == "qwen" else ""
        return f"{prm_prefix}{self.experiment}_{self.prm_model}_{mode}_{self.num_adv_tokens}tok_{pos}_{n_traj}traj"
    
    def save(self, path: str):
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        return cls(
            prm_model=args.prm_model,
            device=args.device,
            hf_cache_path=args.hf_cache_path,
            experiment=args.experiment,
            continuous=args.continuous,
            num_adv_tokens=args.num_adv_tokens,
            adv_position=args.adv_position,
            num_iterations=args.num_iterations,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            entropy_weight_start=args.entropy_weight_start,
            entropy_weight_end=args.entropy_weight_end,
            entropy_schedule=args.entropy_schedule,
            num_train_trajectories=args.num_train_trajectories,
            num_eval_trajectories=args.num_eval_trajectories,
            single_traj_question_idx=args.single_traj_question_idx,
            run_optimization=args.run_optimization,
            run_transfer=args.run_transfer,
            run_analysis=args.run_analysis,
            run_plots=args.run_plots,
            run_3d_landscape=args.run_3d_landscape,
            landscape_grid_size=args.landscape_grid_size,
            cache_dir=args.cache_dir,
            use_cache=args.use_cache,
            skip_if_exists=args.skip_if_exists,
            force_rerun=args.force_rerun,
            distributed=args.distributed,
            seed=args.seed,
            log_interval=args.log_interval,
            batch_chunk_size=args.batch_chunk_size,
        )


# Global config (set by main)
CONFIG: Optional[ExperimentConfig] = None


# ============================================
# STATIC CONFIGURATION
# ============================================

# Generator model (for trajectory generation)
GENERATOR_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
GENERATOR_DEVICE = "cuda:0"

# Dataset configs
TRAIN_DATASET = "Maxwell-Jia/AIME_2024"
EVAL_DATASET = "opencompass/AIME2025"

# Random seed for reproducibility
RANDOM_TOKEN_SEED = 42

# Distributed training
DIST_BACKEND = "nccl"
DIST_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))


def get_prm_config(prm_key: str, hf_cache_path: str, device: str) -> Dict:
    """Get PRM model configuration."""
    configs = {
        "1.5B": {
            "name": "Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
            "path": f"{hf_cache_path}/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
            "device": device,
            "prm_type": "skywork",
        },
        "7B": {
            "name": "Skywork-o1-Open-PRM-Qwen-2.5-7B",
            "path": f"{hf_cache_path}/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B",
            "device": device,
            "prm_type": "skywork",
        },
        "Qwen-7B": {
            "name": "Qwen2.5-Math-PRM-7B",
            "path": "Qwen/Qwen2.5-Math-PRM-7B",
            "device": device,
            "prm_type": "qwen",
        },
    }
    return configs[prm_key]


def get_prm_type(prm_key: str) -> str:
    """Return the PRM type: 'skywork' or 'qwen'."""
    if prm_key in ["1.5B", "7B"]:
        return "skywork"
    elif prm_key == "Qwen-7B":
        return "qwen"
    else:
        raise ValueError(f"Unknown PRM key: {prm_key}")


def _format_reward(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _get_optim_mode_suffix(config: ExperimentConfig) -> str:
    """Return suffix indicating optimization mode (continuous vs discrete)."""
    return "continuous" if config.continuous else "discrete"


def _get_experiment_prefix(config: ExperimentConfig) -> str:
    """Return a descriptive prefix for experiment files."""
    mode = _get_optim_mode_suffix(config)
    n_tok = config.num_adv_tokens
    pos = config.adv_position  # "end" or "middle"
    n_traj = config.num_train_trajectories or "all"
    
    # Add "qwen_" prefix for Qwen models to distinguish from Skywork
    # Skywork models don't get prefix to preserve backward compatibility
    prm_type = get_prm_type(config.prm_model)
    prm_prefix = "qwen_" if prm_type == "qwen" else ""
    
    return f"{prm_prefix}{config.experiment}_{config.prm_model}_{mode}_{n_tok}tok_{pos}_{n_traj}traj"


def _result_cache_path(kind: str, config: ExperimentConfig) -> str:
    prefix = _get_experiment_prefix(config)
    return f"{config.cache_dir}/{prefix}_result.pkl"


def _checkpoint_path(kind: str, config: ExperimentConfig) -> str:
    prefix = _get_experiment_prefix(config)
    return f"{config.cache_dir}/{prefix}_best_token.pt"


def _discrete_token_ids_path(kind: str, config: ExperimentConfig) -> str:
    """Path for saving discrete token IDs when using discrete optimization."""
    prefix = _get_experiment_prefix(config)
    return f"{config.cache_dir}/{prefix}_discrete_token_ids.pt"


def _metrics_path(kind: str, config: ExperimentConfig) -> str:
    """Path for saving training metrics."""
    prefix = _get_experiment_prefix(config)
    return f"{config.cache_dir}/{prefix}_metrics.pkl"


def _config_path(config: ExperimentConfig) -> str:
    """Path for saving experiment config."""
    prefix = _get_experiment_prefix(config)
    return f"{config.cache_dir}/{prefix}_config.json"


def _load_result_cache(path: str, use_cache: bool = True) -> Optional[Dict]:
    if not (use_cache and os.path.exists(path)):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_result_cache(path: str, result: Dict, use_cache: bool = True) -> None:
    if not use_cache:
        return
    result_to_save = dict(result)
    if isinstance(result_to_save.get("best_token_coeffs"), torch.Tensor):
        result_to_save["best_token_coeffs"] = result_to_save["best_token_coeffs"].cpu()
    if isinstance(result_to_save.get("best_discrete_token_ids"), torch.Tensor):
        result_to_save["best_discrete_token_ids"] = result_to_save["best_discrete_token_ids"].cpu()
    with open(path, "wb") as f:
        pickle.dump(result_to_save, f)


def _save_metrics(metrics: Dict, path: str) -> None:
    """Save training metrics to file."""
    with open(path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"Saved training metrics to {path}")


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _init_distributed(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(backend=DIST_BACKEND, rank=rank, world_size=world_size)


def _cleanup_distributed() -> None:
    if _is_distributed():
        dist.destroy_process_group()


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{} notation"""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Normalize answer for comparison"""
    if answer is None:
        return None
    answer = answer.replace(" ", "").lower()
    answer = answer.replace(",", "")
    return answer


def parse_steps_from_generation(text: str) -> List[str]:
    """Parse a generated text into individual steps (separated by \\n\\n)"""
    steps = [step.strip() for step in text.split("\n\n") if step.strip()]
    return steps


def calculate_stepwise_rewards(
    model,
    tokenizer,
    problem: str,
    steps: List[str],
    model_path: str,
    device: torch.device,
) -> List[float]:
    """Calculate step-wise rewards for a single trajectory."""
    if len(steps) == 0:
        return []

    input_ids, token_masks = prepare_input(
        model_path,
        problem=problem,
        steps=steps,
        tokenizer=tokenizer,
        device=device,
    )

    with torch.inference_mode():
        logits = model(input_ids.view(1, -1))[-1]

    rewards = derive_step_rewards(model_path, logits, token_masks, tokenizer)
    return rewards[0] if rewards else []


def get_step_token_id(tokenizer: AutoTokenizer) -> int:
    """Return the step separator token id used by PRM inputs."""
    step_token_ids = tokenizer.encode(STEP_SEP_TOKEN, add_special_tokens=False)
    if len(step_token_ids) != 1:
        raise ValueError(f"Expected a single STEP_SEP_TOKEN id, got {step_token_ids}")
    return step_token_ids[0]


def get_step_token_embedding(
    embed_layer, tokenizer: AutoTokenizer, device: torch.device
) -> torch.Tensor:
    """Embedding for the step separator token."""
    step_token_id = get_step_token_id(tokenizer)
    return embed_layer(torch.tensor([step_token_id], device=device))


def get_question_length_skywork(tokenizer: AutoTokenizer, problem: str) -> int:
    """
    Get the length of the question portion for Skywork PRM.
    
    Skywork format: <bos>{problem}\n{step1}\n{step2}\n...
    Question ends at: <bos>{problem}\n
    """
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + STEP_SEP_TOKEN)
    return len(prompt_ids)


def get_question_length_qwen(tokenizer: AutoTokenizer, problem: str) -> int:
    """
    Get the length of the question portion for Qwen PRM.
    
    Qwen format uses chat template:
    <|im_start|>system\n{system_prompt}<|im_end|>\n
    <|im_start|>user\n{problem}<|im_end|>\n
    <|im_start|>assistant\n{step1}<extra_0>{step2}<extra_0>...
    
    Question ends at: the start of assistant's content (after "assistant\n")
    """
    from utils.processors.qwen_math_prm import SYSTEM_PROMPT
    
    # Build the template up to the start of assistant's content
    messages_question_only = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    # Use add_generation_prompt=True to get the assistant start token
    question_str = tokenizer.apply_chat_template(
        messages_question_only,
        tokenize=False,
        add_generation_prompt=True
    )
    question_ids = tokenizer.encode(question_str, add_special_tokens=False)
    return len(question_ids)


def get_question_length(tokenizer: AutoTokenizer, problem: str, prm_type: str) -> int:
    """
    Get the length of the question portion based on PRM type.
    
    Args:
        tokenizer: The tokenizer for the PRM
        problem: The problem/question text
        prm_type: "skywork" or "qwen"
    
    Returns:
        Number of tokens in the question portion
    """
    if prm_type == "skywork":
        return get_question_length_skywork(tokenizer, problem)
    elif prm_type == "qwen":
        return get_question_length_qwen(tokenizer, problem)
    else:
        raise ValueError(f"Unknown PRM type: {prm_type}")


def construct_embeddings_with_adv(
    orig_embeddings: torch.Tensor,
    adv_embeddings: torch.Tensor,
    step_embeddings: torch.Tensor,
    adv_position: str,
    question_length: int,
    prm_type: str = "skywork",
    add_step_after_adv: bool = False,
) -> torch.Tensor:
    """
    Construct full embeddings with adversarial tokens at specified position.
    
    Args:
        orig_embeddings: Original sequence embeddings (question + solution), shape (seq_len, embed_dim)
        adv_embeddings: Adversarial token embeddings, shape (num_adv_tokens, embed_dim)
        step_embeddings: Step token embedding, shape (1, embed_dim)
        adv_position: "end" or "middle"
        question_length: Length of question portion (for "middle" position)
        prm_type: "skywork" or "qwen"
        add_step_after_adv: If True, add a step separator after adv tokens in "middle" position
                           (used for Qwen to treat adv tokens as a scorable step)
    
    Returns:
        Full embeddings tensor
    """
    if adv_position == "end":
        # [question + solution] + [adv tokens] + [step token]
        full_embeddings = torch.cat([orig_embeddings, adv_embeddings, step_embeddings], dim=0)
    elif adv_position == "middle":
        question_embeddings = orig_embeddings[:question_length]
        solution_embeddings = orig_embeddings[question_length:]
        
        if add_step_after_adv:
            # For Qwen: [question] + [adv tokens] + [step sep] + [solution] + [step token]
            # This treats the adversarial tokens as a "step" that can be scored
            full_embeddings = torch.cat([
                question_embeddings, 
                adv_embeddings, 
                step_embeddings,  # Step separator after adv tokens
                solution_embeddings, 
                step_embeddings   # Final step token
            ], dim=0)
        else:
            # For Skywork: [question] + [adv tokens] + [solution] + [step token]
            full_embeddings = torch.cat([
                question_embeddings, 
                adv_embeddings, 
                solution_embeddings, 
                step_embeddings
            ], dim=0)
    else:
        raise ValueError(f"Invalid adv_position: {adv_position}")
    return full_embeddings


def get_embedding_layer(model):
    """
    Get the embedding layer from a model.
    
    Works for both Skywork (pretrained_model.get_input_embeddings) and 
    Qwen (model.get_input_embeddings).
    """
    if hasattr(model, "pretrained_model"):
        # Skywork model
        return model.pretrained_model.get_input_embeddings()
    elif hasattr(model, "model"):
        # Qwen model
        return model.model.embed_tokens
    else:
        return model.get_input_embeddings()


def get_last_hidden_state(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """
    Get last hidden state without forcing full hidden state stack.
    Uses the base model backbone when available to reduce memory.
    """
    base_model = model.pretrained_model if hasattr(model, "pretrained_model") else model
    if hasattr(base_model, "model"):
        outputs = base_model.model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state
    outputs = base_model(
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if getattr(outputs, "hidden_states", None) is not None:
        return outputs.hidden_states[-1]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0]


def compute_step_token_mask(
    tokenizer: AutoTokenizer,
    seq_len: int,
    orig_seq_len: int,
    num_adv_tokens: int,
    adv_position: str,
    question_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute a mask indicating step separator token positions.
    
    For Qwen PRM, step rewards are computed at step separator (<extra_0>) positions.
    This mask accounts for the inserted adversarial tokens.
    
    Args:
        tokenizer: Tokenizer to get step token ID
        seq_len: Total sequence length (with adv tokens and final step token)
        orig_seq_len: Original sequence length (question + solution, without adv tokens)
        num_adv_tokens: Number of adversarial tokens inserted
        adv_position: "end" or "middle"
        question_length: Length of question portion
        device: Device for the tensor
    
    Returns:
        Boolean mask of shape (seq_len,) with True at step positions
    """
    step_token_id = get_step_token_id(tokenizer)
    
    # For now, we'll assume step tokens are only in the original sequence
    # We need to identify where they are in the modified sequence
    # The final token is always a step token (we add it)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[-1] = True  # Final step token we add
    
    return mask


def compute_reward_proxy_skywork(
    model,
    full_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute reward proxy for Skywork PRM.
    
    Skywork PRM is trained to estimate success probability of entire trajectory.
    We use the sigmoid of the value head output at the final token.
    
    Args:
        model: Skywork PRM model (has pretrained_model and v_head)
        full_embeddings: Input embeddings (1, seq_len, embed_dim)
    
    Returns:
        Scalar reward proxy (differentiable)
    """
    last_hidden_state = get_last_hidden_state(model, full_embeddings)
    reward_logits = model.v_head(last_hidden_state).squeeze(-1)
    # Use final token's output
    reward_proxy = torch.sigmoid(reward_logits[0, -1])
    return reward_proxy


def compute_reward_proxy_qwen(
    model,
    full_embeddings: torch.Tensor,
    step_token_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute reward proxy for Qwen Math PRM.
    
    Qwen PRM is trained to locate the first incorrect step.
    We compute rewards at all step positions and maximize the minimum.
    
    torch.min is differentiable - gradients flow to the element that achieved the minimum.
    
    Args:
        model: Qwen PRM model (has model and score)
        full_embeddings: Input embeddings (1, seq_len, embed_dim)
        step_token_positions: Indices of step separator tokens
    
    Returns:
        Tuple of (reward_proxy, all_step_rewards)
        - reward_proxy: Scalar min reward (differentiable via torch.min)
        - all_step_rewards: All step rewards for logging
    """
    # Get hidden states through the model
    outputs = model.model(
        inputs_embeds=full_embeddings,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    
    # Get logits from score head (produces 2 classes: incorrect, correct)
    all_logits = model.score(hidden_states)  # (1, seq_len, 2)
    
    # Extract logits at step token positions
    step_logits = all_logits[0, step_token_positions, :]  # (num_steps, 2)
    
    # Convert to probabilities and get "correct" probability (class 1)
    step_probs = F.softmax(step_logits, dim=-1)  # (num_steps, 2)
    step_rewards = step_probs[:, 1]  # (num_steps,) - probability of being correct
    
    # Use torch.min which is differentiable - gradients flow to the min element
    min_reward = torch.min(step_rewards)
    
    return min_reward, step_rewards


def compute_reward_proxy(
    model,
    full_embeddings: torch.Tensor,
    prm_type: str,
    step_token_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unified function to compute reward proxy based on PRM type.
    
    Args:
        model: PRM model
        full_embeddings: Input embeddings (1, seq_len, embed_dim)
        prm_type: "skywork" or "qwen"
        step_token_positions: Required for Qwen, indices of step tokens
    
    Returns:
        Tuple of (reward_proxy, step_rewards)
        - reward_proxy: Scalar reward to maximize
        - step_rewards: Individual step rewards (only for Qwen, None for Skywork)
    """
    if prm_type == "skywork":
        reward_proxy = compute_reward_proxy_skywork(model, full_embeddings)
        return reward_proxy, None
    elif prm_type == "qwen":
        if step_token_positions is None:
            raise ValueError("step_token_positions required for Qwen PRM")
        reward_proxy, step_rewards = compute_reward_proxy_qwen(
            model, full_embeddings, step_token_positions
        )
        return reward_proxy, step_rewards
    else:
        raise ValueError(f"Unknown PRM type: {prm_type}")


def find_step_token_positions(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    """
    Find positions of step separator tokens in the input sequence.
    
    Args:
        input_ids: Token IDs (can be 1D or 2D with batch dim)
        tokenizer: Tokenizer to get step token ID
    
    Returns:
        1D tensor of positions where step tokens occur
    """
    step_token_id = get_step_token_id(tokenizer)
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    positions = (input_ids == step_token_id).nonzero(as_tuple=True)[0]
    return positions


def load_prm_model(
    prm_key: str, 
    hf_cache_path: str,
    device: str = "cuda:0",
) -> Tuple:
    """Load a PRM model and its tokenizer."""
    prm_config = get_prm_config(prm_key, hf_cache_path, device)
    model_path = prm_config["path"]

    print(f"Loading PRM: {prm_config['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MODEL_CLASS_MAP[model_path].from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model = model.eval()
    print(f"Loaded {prm_config['name']} on {device}")

    return model, tokenizer, model_path, torch.device(device)


def load_aime_dataset(dataset_name: str, split: str = "test") -> Dict:
    """Load AIME dataset."""
    print(f"Loading dataset: {dataset_name}...")

    if dataset_name == "Maxwell-Jia/AIME_2024":
        ds = load_dataset(dataset_name, split=split)
        data = {
            "questions": ds["Problem"],
            "answers": [str(a) for a in ds["Answer"]],
        }
    elif dataset_name == "opencompass/AIME2025":
        ds_I = load_dataset(dataset_name, "AIME2025-I", split=split)
        ds_II = load_dataset(dataset_name, "AIME2025-II", split=split)
        data = {
            "questions": list(ds_I["question"]) + list(ds_II["question"]),
            "answers": [str(a) for a in ds_I["answer"]] + [str(a) for a in ds_II["answer"]],
        }
        print(f"  Loaded {len(ds_I)} questions from AIME2025-I")
        print(f"  Loaded {len(ds_II)} questions from AIME2025-II")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded {len(data['questions'])} questions total")
    return data


def generate_trajectories(
    questions: List[str],
    n_trajectories_per_question: int = 1,
    cache_file: Optional[str] = None,
    force_regenerate: bool = False,
) -> List[List[str]]:
    """
    Generate trajectories for questions using Qwen2.5-Math-7B-Instruct.

    Returns:
        List of lists of generated texts (one list per question)
    """
    if cache_file and os.path.exists(cache_file) and not force_regenerate:
        print(f"Loading cached trajectories from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Generating {n_trajectories_per_question} trajectory(ies) per question...")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=GENERATOR_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        device=GENERATOR_DEVICE,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        n=n_trajectories_per_question,
    )

    prompts = [
        f"""Problem: {q}

Please solve this problem step by step. Separate each step with a blank line (\\n\\n). Provide your final answer in \\boxed{{}}.

Solution:"""
        for q in questions
    ]

    outputs = llm.generate(prompts, sampling_params)

    all_trajectories = []
    for output in outputs:
        question_trajectories = [o.text for o in output.outputs]
        all_trajectories.append(question_trajectories)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump(all_trajectories, f)
        print(f"Saved trajectories to {cache_file}")

    del llm
    torch.cuda.empty_cache()

    return all_trajectories


def optimize_adversarial_tokens_single_trajectory(
    model,
    tokenizer,
    problem: str,
    steps: List[str],
    model_path: str,
    device: torch.device,
    num_adv_tokens: int,
    adv_position: str,
    num_iterations: int,
    learning_rate: float,
    temperature: float,
    entropy_weight_start: float,
    entropy_weight_end: float,
    entropy_schedule: str,
    continuous_optimization: bool,
    prm_type: str = "skywork",
    log_interval: int = 100,
    target_reward: Optional[float] = None,
    distributed_ctx: Optional[Dict] = None,
) -> Dict:
    """
    Optimize adversarial token(s) for a single trajectory.
    
    Args:
        num_adv_tokens: Number of adversarial tokens to optimize
        adv_position: "end" (after solution) or "middle" (after question, before solution)
        continuous_optimization: If True, optimize in continuous space. If False, use
            entropy regularization to push toward discrete tokens.
        prm_type: "skywork" (maximize final step) or "qwen" (maximize min step using torch.min)
    """
    input_ids_orig, _ = prepare_input(
        model_path,
        problem=problem,
        steps=steps,
        tokenizer=tokenizer,
        device=device,
    )

    embed_layer = get_embedding_layer(model)
    vocab_size = embed_layer.weight.shape[0]
    
    # Get question length for "middle" position
    question_length = get_question_length(tokenizer, problem, prm_type)

    # Initialize random logits for adversarial tokens
    # Shape: (num_adv_tokens, vocab_size)
    token_logits = torch.randn(
        num_adv_tokens,
        vocab_size,
        device=device,
        requires_grad=True,
        dtype=torch.float32,
    )
    if distributed_ctx and _is_distributed():
        dist.broadcast(token_logits.data, src=0)

    optimizer = torch.optim.Adam([token_logits], lr=learning_rate)

    orig_embeddings = embed_layer(input_ids_orig).detach()
    step_embeddings = get_step_token_embedding(embed_layer, tokenizer, device).detach()
    
    # For Qwen PRM, find step token positions in original sequence
    # We'll need to adjust these positions based on where adv tokens are inserted
    orig_step_positions = find_step_token_positions(input_ids_orig, tokenizer)
    orig_seq_len = len(input_ids_orig[0]) if input_ids_orig.dim() == 2 else len(input_ids_orig)

    reward_history = []  # Soft rewards (Gumbel-softmax)
    discrete_reward_history = []  # Hard rewards (argmax discrete tokens)
    min_step_reward_history = []  # For Qwen: track min step reward
    entropy_history = []
    entropy_weight_history = []
    best_reward = -float("inf")  # Best soft reward
    best_discrete_reward = -float("inf")  # Best discrete reward
    best_token_coeffs = None
    best_discrete_token_ids = None
    iterations_to_target = None

    is_main = (distributed_ctx is None) or distributed_ctx.get("rank", 0) == 0
    if is_main:
        mode_str = "continuous" if continuous_optimization else "discrete (with entropy regularization)"
        prm_mode_str = "final step" if prm_type == "skywork" else "min step (torch.min)"
        print(f"Starting single trajectory optimization ({mode_str})...")
        print(f"PRM type: {prm_type} (optimizing {prm_mode_str})")
        print(f"Original sequence length: {orig_seq_len}")
        print(f"Optimizing {num_adv_tokens} adversarial token(s) at position '{adv_position}'")
        if adv_position == "middle":
            print(f"Question length: {question_length} tokens")
        if prm_type == "qwen":
            print(f"Original step token positions: {orig_step_positions.tolist()}")
        if not continuous_optimization:
            print(f"Entropy schedule: {entropy_schedule}")
            print(f"  Start weight: {entropy_weight_start:.4f}")
            print(f"  End weight: {entropy_weight_end:.4f}")
        print()

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        progress = iteration / max(num_iterations - 1, 1)
        if continuous_optimization:
            entropy_weight = 0.0
        elif entropy_schedule == "linear":
            entropy_weight = entropy_weight_start + (
                entropy_weight_end - entropy_weight_start
            ) * progress
        elif entropy_schedule == "exponential":
            entropy_weight = entropy_weight_start * (
                entropy_weight_end / max(entropy_weight_start, 1e-9)
            ) ** progress
        elif entropy_schedule == "cosine":
            entropy_weight = entropy_weight_start + (
                entropy_weight_end - entropy_weight_start
            ) * (1 - np.cos(progress * np.pi)) / 2
        else:
            entropy_weight = entropy_weight_start
        entropy_weight_history.append(entropy_weight)

        gumbel_softmax = F.gumbel_softmax(token_logits, tau=temperature, hard=False)

        # Calculate entropy of token distributions
        probs = F.softmax(token_logits, dim=-1)
        log_probs = F.log_softmax(token_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_history.append(entropy.item())

        gumbel_softmax_bf16 = gumbel_softmax.to(embed_layer.weight.dtype)
        # Shape: (num_adv_tokens, embedding_dim)
        new_embeddings = torch.matmul(gumbel_softmax_bf16, embed_layer.weight)
        
        # For Qwen with middle position, add step separator after adv tokens
        add_step_after_adv = (prm_type == "qwen" and adv_position == "middle")

        full_embeddings = construct_embeddings_with_adv(
            orig_embeddings, new_embeddings, step_embeddings, adv_position, question_length,
            prm_type=prm_type, add_step_after_adv=add_step_after_adv
        )
        full_embeddings = full_embeddings.unsqueeze(0)

        with torch.set_grad_enabled(True):
            if prm_type == "skywork":
                # Skywork: maximize final step reward
                if hasattr(model, "pretrained_model"):
                    last_hidden_state = get_last_hidden_state(model, full_embeddings)
                    reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                    reward_proxy = torch.sigmoid(reward_logits[0, -1])
                else:
                    raise NotImplementedError("Skywork model requires pretrained_model attribute")
            elif prm_type == "qwen":
                # Qwen: maximize minimum step reward (using torch.min)
                # Adjust step positions for inserted adversarial tokens
                if adv_position == "end":
                    # Adv tokens after solution, before final step token
                    # Original positions unchanged, plus final step token at the end
                    adjusted_positions = torch.cat([
                        orig_step_positions,
                        torch.tensor([full_embeddings.shape[1] - 1], device=device)
                    ])
                else:  # middle
                    # For Qwen with middle position: [question] + [adv tokens] + [step sep] + [solution] + [step token]
                    # The step separator after adv tokens makes them a scorable "step"
                    # - Position of adv step separator: question_length + num_adv_tokens
                    # - Original solution step positions shift by: num_adv_tokens + 1
                    adv_step_sep_position = question_length + num_adv_tokens
                    # All original step positions (in solution) shift by num_adv_tokens + 1
                    shifted_solution_positions = orig_step_positions + num_adv_tokens + 1
                    adjusted_positions = torch.cat([
                        torch.tensor([adv_step_sep_position], device=device),  # Adv "step" separator
                        shifted_solution_positions,  # Shifted solution step positions
                        torch.tensor([full_embeddings.shape[1] - 1], device=device)  # Final step token
                    ])
                
                reward_proxy, step_rewards = compute_reward_proxy_qwen(
                    model, full_embeddings, adjusted_positions
                )
                min_step_reward_history.append(step_rewards.min().item())
            else:
                raise ValueError(f"Unknown PRM type: {prm_type}")

        loss = -reward_proxy + entropy_weight * entropy

        loss.backward()
        if distributed_ctx and _is_distributed():
            if token_logits.grad is None:
                token_logits.grad = torch.zeros_like(token_logits)
            dist.all_reduce(token_logits.grad, op=dist.ReduceOp.SUM)
            token_logits.grad /= distributed_ctx["world_size"]
        optimizer.step()

        current_reward = reward_proxy.item()
        reward_history.append(current_reward)

        # Evaluate discrete token reward (without gradients)
        with torch.no_grad():
            discrete_token_ids = torch.argmax(token_logits, dim=-1)
            discrete_embeddings = embed_layer(discrete_token_ids)
            full_discrete_embeddings = construct_embeddings_with_adv(
                orig_embeddings, discrete_embeddings, step_embeddings, adv_position, question_length,
                prm_type=prm_type, add_step_after_adv=add_step_after_adv
            ).unsqueeze(0)
            
            if prm_type == "skywork":
                if hasattr(model, "pretrained_model"):
                    last_hidden_state_discrete = get_last_hidden_state(model, full_discrete_embeddings)
                    reward_logits_discrete = model.v_head(last_hidden_state_discrete).squeeze(-1)
                    discrete_reward = torch.sigmoid(reward_logits_discrete[0, -1]).item()
                else:
                    raise NotImplementedError("Skywork model requires pretrained_model attribute")
            elif prm_type == "qwen":
                # For discrete eval, use torch.min
                if adv_position == "end":
                    adjusted_positions_discrete = torch.cat([
                        orig_step_positions,
                        torch.tensor([full_discrete_embeddings.shape[1] - 1], device=device)
                    ])
                else:  # middle with step separator after adv tokens
                    adv_step_sep_pos_discrete = question_length + num_adv_tokens
                    shifted_solution_pos_discrete = orig_step_positions + num_adv_tokens + 1
                    adjusted_positions_discrete = torch.cat([
                        torch.tensor([adv_step_sep_pos_discrete], device=device),
                        shifted_solution_pos_discrete,
                        torch.tensor([full_discrete_embeddings.shape[1] - 1], device=device)
                    ])
                _, step_rewards_discrete = compute_reward_proxy_qwen(
                    model, full_discrete_embeddings, adjusted_positions_discrete
                )
                discrete_reward = step_rewards_discrete.min().item()
            else:
                raise ValueError(f"Unknown PRM type: {prm_type}")
            
            discrete_reward_history.append(discrete_reward)

        if target_reward is not None and iterations_to_target is None:
            check_reward = discrete_reward if not continuous_optimization else current_reward
            if check_reward >= target_reward:
                iterations_to_target = iteration + 1

        # Track best soft reward and coefficients
        if current_reward > best_reward:
            best_reward = current_reward
            best_token_coeffs = gumbel_softmax.detach().cpu()

        # Track best discrete reward and token IDs
        if discrete_reward > best_discrete_reward:
            best_discrete_reward = discrete_reward
            best_discrete_token_ids = discrete_token_ids.detach().cpu()

        if is_main and (iteration + 1) % log_interval == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}", flush=True)
            print(f"  Soft Reward (Gumbel): {current_reward:.4f} | Best: {best_reward:.4f}")
            print(f"  Discrete Reward (Hard): {discrete_reward:.4f} | Best: {best_discrete_reward:.4f}")
            print(f"  Reward Gap (Soft-Hard): {current_reward - discrete_reward:+.4f}")
            if prm_type == "qwen" and min_step_reward_history:
                print(f"  Min Step Reward: {min_step_reward_history[-1]:.4f}")
            if not continuous_optimization:
                print(f"  Entropy Weight: {entropy_weight:.6f}")
            print(f"  Entropy: {entropy.item():.4f} (max possible: {np.log(vocab_size):.4f})")
            decoded_tokens = tokenizer.decode(discrete_token_ids.cpu().tolist())
            print(f"  Current Discrete Tokens: {decoded_tokens[:80]}")
            print()

    if is_main:
        print("\nOptimization Complete!")
        print(f"Best Soft Reward: {best_reward:.4f}")
        print(f"Best Discrete Reward: {best_discrete_reward:.4f}")
        if prm_type == "qwen":
            print(f"PRM Type: Qwen (optimized min step reward)")
        else:
            print(f"PRM Type: Skywork (optimized final step reward)")
        if not continuous_optimization:
            print(f"Final Entropy: {entropy_history[-1]:.4f}")
            print(f"Entropy Reduction: {entropy_history[0] - entropy_history[-1]:.4f}")
        if iterations_to_target is not None:
            print(f"Iterations to reach target {target_reward}: {iterations_to_target}")
        if best_discrete_token_ids is not None:
            decoded = tokenizer.decode(best_discrete_token_ids.tolist())
            print(f"Best Discrete Tokens: {decoded}")

    return {
        "best_token_coeffs": best_token_coeffs,
        "best_discrete_token_ids": best_discrete_token_ids,
        "reward_history": reward_history,
        "discrete_reward_history": discrete_reward_history,
        "min_step_reward_history": min_step_reward_history if prm_type == "qwen" else None,
        "entropy_history": entropy_history,
        "entropy_weight_history": entropy_weight_history,
        "best_reward": best_reward,
        "best_discrete_reward": best_discrete_reward,
        "iterations_to_target": iterations_to_target,
        "continuous_optimization": continuous_optimization,
        "num_adv_tokens": num_adv_tokens,
        "adv_position": adv_position,
        "prm_type": prm_type,
    }


def optimize_adversarial_tokens_batched_trajectories(
    model,
    tokenizer,
    problems: List[str],
    all_steps: List[List[str]],
    model_path: str,
    device: torch.device,
    num_adv_tokens: int,
    adv_position: str,
    num_iterations: int,
    learning_rate: float,
    temperature: float,
    entropy_weight_start: float,
    entropy_weight_end: float,
    entropy_schedule: str,
    chunk_size: int,
    continuous_optimization: bool,
    prm_type: str = "skywork",
    log_interval: int = 100,
    target_reward: Optional[float] = None,
    distributed_ctx: Optional[Dict] = None,
) -> Dict:
    """
    Optimize adversarial token(s) across multiple trajectories (batched).
    
    Args:
        num_adv_tokens: Number of adversarial tokens to optimize
        adv_position: "end" (after solution) or "middle" (after question, before solution)
        continuous_optimization: If True, optimize in continuous space. If False, use
            entropy regularization to push toward discrete tokens.
        prm_type: "skywork" (maximize final step) or "qwen" (maximize min step using torch.min)
    """
    assert len(problems) == len(all_steps), "Number of problems must match number of step lists"

    embed_layer = get_embedding_layer(model)
    vocab_size = embed_layer.weight.shape[0]

    all_input_ids = []
    all_orig_embeddings = []
    all_question_lengths = []  # For "middle" position
    all_step_positions = []  # For Qwen PRM
    step_embeddings = get_step_token_embedding(embed_layer, tokenizer, device).detach()

    print(f"Preparing {len(problems)} trajectories...")
    for problem, steps in zip(problems, all_steps):
        if len(steps) == 0:
            continue
        input_ids, _ = prepare_input(
            model_path,
            problem=problem,
            steps=steps,
            tokenizer=tokenizer,
            device=device,
        )
        all_input_ids.append(input_ids)
        all_orig_embeddings.append(embed_layer(input_ids).detach())
        all_question_lengths.append(get_question_length(tokenizer, problem, prm_type))
        # For Qwen PRM, track step token positions
        if prm_type == "qwen":
            step_positions = find_step_token_positions(input_ids, tokenizer)
            all_step_positions.append(step_positions)

    num_trajectories = len(all_orig_embeddings)
    print(f"Prepared {num_trajectories} valid trajectories")

    if num_trajectories == 0:
        raise ValueError("No valid trajectories to optimize.")
    chunk_size = max(1, min(chunk_size, num_trajectories))

    # Initialize random logits for adversarial tokens
    # Shape: (num_adv_tokens, vocab_size)
    token_logits = torch.randn(
        num_adv_tokens,
        vocab_size,
        device=device,
        requires_grad=True,
        dtype=torch.float32,
    )

    optimizer = torch.optim.Adam([token_logits], lr=learning_rate)

    avg_reward_history = []  # Soft rewards (Gumbel-softmax)
    avg_discrete_reward_history = []  # Hard rewards (argmax discrete tokens)
    avg_min_step_reward_history = []  # For Qwen: track avg min step reward
    per_traj_reward_history = [[] for _ in range(num_trajectories)]
    entropy_history = []
    entropy_weight_history = []
    best_avg_reward = -float("inf")  # Best soft reward
    best_avg_discrete_reward = -float("inf")  # Best discrete reward
    best_token_coeffs = None
    best_discrete_token_ids = None
    iterations_to_target = None

    is_main = (distributed_ctx is None) or distributed_ctx.get("rank", 0) == 0
    if is_main:
        mode_str = "continuous" if continuous_optimization else "discrete (with entropy regularization)"
        prm_mode_str = "final step" if prm_type == "skywork" else "min step (torch.min)"
        print(f"\nStarting batched optimization ({mode_str})...")
        print(f"PRM type: {prm_type} (optimizing {prm_mode_str})")
        print(f"Number of trajectories: {num_trajectories}")
        print(f"Optimizing {num_adv_tokens} shared adversarial token(s) at position '{adv_position}'")
        if not continuous_optimization:
            print(f"Entropy schedule: {entropy_schedule}")
            print(f"  Start weight: {entropy_weight_start:.4f}")
            print(f"  End weight: {entropy_weight_end:.4f}")
        print()

    if distributed_ctx and _is_distributed():
        local_indices = list(range(distributed_ctx["rank"], num_trajectories, distributed_ctx["world_size"]))
    else:
        local_indices = list(range(num_trajectories))

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        progress = iteration / max(num_iterations - 1, 1)
        if continuous_optimization:
            entropy_weight = 0.0
        elif entropy_schedule == "linear":
            entropy_weight = entropy_weight_start + (
                entropy_weight_end - entropy_weight_start
            ) * progress
        elif entropy_schedule == "exponential":
            entropy_weight = entropy_weight_start * (
                entropy_weight_end / max(entropy_weight_start, 1e-9)
            ) ** progress
        elif entropy_schedule == "cosine":
            entropy_weight = entropy_weight_start + (
                entropy_weight_end - entropy_weight_start
            ) * (1 - np.cos(progress * np.pi)) / 2
        else:
            entropy_weight = entropy_weight_start
        entropy_weight_history.append(entropy_weight)

        gumbel_softmax = F.gumbel_softmax(token_logits, tau=temperature, hard=False)

        # Calculate entropy of token distributions
        probs = F.softmax(token_logits, dim=-1)
        log_probs = F.log_softmax(token_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_history.append(entropy.item())

        gumbel_softmax_bf16 = gumbel_softmax.to(embed_layer.weight.dtype)
        # Shape: (num_adv_tokens, embedding_dim)
        new_embeddings = torch.matmul(gumbel_softmax_bf16, embed_layer.weight)

        # Get discrete token embeddings for tracking discrete rewards
        discrete_token_ids = torch.argmax(token_logits, dim=-1)
        discrete_embeddings = embed_layer(discrete_token_ids).detach()

        total_reward_value_local = 0.0
        total_discrete_reward_value_local = 0.0
        total_min_step_reward_local = 0.0  # For Qwen PRM
        rewards_min_local = None
        rewards_max_local = None
        
        # For Qwen with middle position, add step separator after adv tokens
        add_step_after_adv = (prm_type == "qwen" and adv_position == "middle")

        for chunk_start in range(0, len(local_indices), chunk_size):
            chunk_rewards = []
            chunk_end = min(chunk_start + chunk_size, len(local_indices))
            chunk_indices = local_indices[chunk_start:chunk_end]
            for idx in chunk_indices:
                orig_emb = all_orig_embeddings[idx]
                q_len = all_question_lengths[idx]
                full_embeddings = construct_embeddings_with_adv(
                    orig_emb, new_embeddings, step_embeddings, adv_position, q_len,
                    prm_type=prm_type, add_step_after_adv=add_step_after_adv
                )
                full_embeddings = full_embeddings.unsqueeze(0)

                with torch.set_grad_enabled(True):
                    if prm_type == "skywork":
                        # Skywork: maximize final step reward
                        if hasattr(model, "pretrained_model"):
                            last_hidden_state = get_last_hidden_state(model, full_embeddings)
                            reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                            reward = torch.sigmoid(reward_logits[0, -1])
                        else:
                            raise NotImplementedError("Skywork model requires pretrained_model attribute")
                    elif prm_type == "qwen":
                        # Qwen: maximize minimum step reward (using torch.min)
                        orig_step_pos = all_step_positions[idx]
                        if adv_position == "end":
                            adjusted_positions = torch.cat([
                                orig_step_pos,
                                torch.tensor([full_embeddings.shape[1] - 1], device=device)
                            ])
                        else:  # middle with step separator after adv tokens
                            # [question] + [adv tokens] + [step sep] + [solution] + [step token]
                            adv_step_sep_position = q_len + num_adv_tokens
                            shifted_solution_positions = orig_step_pos + num_adv_tokens + 1
                            adjusted_positions = torch.cat([
                                torch.tensor([adv_step_sep_position], device=device),
                                shifted_solution_positions,
                                torch.tensor([full_embeddings.shape[1] - 1], device=device)
                            ])
                        reward, step_rewards = compute_reward_proxy_qwen(
                            model, full_embeddings, adjusted_positions
                        )
                        total_min_step_reward_local += step_rewards.min().item()
                    else:
                        raise ValueError(f"Unknown PRM type: {prm_type}")

                chunk_rewards.append(reward)
                reward_value = reward.item()
                per_traj_reward_history[idx].append(reward_value)
                total_reward_value_local += reward_value
                rewards_min_local = (
                    reward_value if rewards_min_local is None else min(rewards_min_local, reward_value)
                )
                rewards_max_local = (
                    reward_value if rewards_max_local is None else max(rewards_max_local, reward_value)
                )

                # Evaluate discrete reward for this trajectory
                with torch.no_grad():
                    full_discrete_embeddings = construct_embeddings_with_adv(
                        orig_emb, discrete_embeddings, step_embeddings, adv_position, q_len,
                        prm_type=prm_type, add_step_after_adv=add_step_after_adv
                    ).unsqueeze(0)
                    if prm_type == "skywork":
                        if hasattr(model, "pretrained_model"):
                            last_hidden_state_discrete = get_last_hidden_state(model, full_discrete_embeddings)
                            reward_logits_discrete = model.v_head(last_hidden_state_discrete).squeeze(-1)
                            discrete_reward_value = torch.sigmoid(reward_logits_discrete[0, -1]).item()
                    elif prm_type == "qwen":
                        if adv_position == "end":
                            adjusted_positions_discrete = torch.cat([
                                orig_step_pos,
                                torch.tensor([full_discrete_embeddings.shape[1] - 1], device=device)
                            ])
                        else:  # middle with step separator after adv tokens
                            adv_step_sep_pos_discrete = q_len + num_adv_tokens
                            shifted_solution_pos_discrete = orig_step_pos + num_adv_tokens + 1
                            adjusted_positions_discrete = torch.cat([
                                torch.tensor([adv_step_sep_pos_discrete], device=device),
                                shifted_solution_pos_discrete,
                                torch.tensor([full_discrete_embeddings.shape[1] - 1], device=device)
                            ])
                        _, step_rewards_discrete = compute_reward_proxy_qwen(
                            model, full_discrete_embeddings, adjusted_positions_discrete
                        )
                        discrete_reward_value = step_rewards_discrete.min().item()
                    total_discrete_reward_value_local += discrete_reward_value

            chunk_reward_sum = torch.stack(chunk_rewards).sum()
            chunk_loss = -chunk_reward_sum / num_trajectories

            if chunk_end == len(local_indices):
                chunk_loss = chunk_loss + entropy_weight * entropy

            retain_graph = chunk_end < len(local_indices)
            chunk_loss.backward(retain_graph=retain_graph)

        if distributed_ctx and _is_distributed():
            if token_logits.grad is None:
                token_logits.grad = torch.zeros_like(token_logits)
            dist.all_reduce(token_logits.grad, op=dist.ReduceOp.SUM)
            token_logits.grad /= distributed_ctx["world_size"]

        optimizer.step()

        if distributed_ctx and _is_distributed():
            total_reward_tensor = torch.tensor(total_reward_value_local, device=device)
            dist.all_reduce(total_reward_tensor, op=dist.ReduceOp.SUM)
            current_avg_reward = total_reward_tensor.item() / num_trajectories

            total_discrete_reward_tensor = torch.tensor(total_discrete_reward_value_local, device=device)
            dist.all_reduce(total_discrete_reward_tensor, op=dist.ReduceOp.SUM)
            current_avg_discrete_reward = total_discrete_reward_tensor.item() / num_trajectories

            min_tensor = torch.tensor(
                rewards_min_local if rewards_min_local is not None else float("inf"),
                device=device,
            )
            max_tensor = torch.tensor(
                rewards_max_local if rewards_max_local is not None else float("-inf"),
                device=device,
            )
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
            rewards_min = min_tensor.item()
            rewards_max = max_tensor.item()
            
            # For Qwen, also aggregate min step rewards
            if prm_type == "qwen":
                min_step_tensor = torch.tensor(total_min_step_reward_local, device=device)
                dist.all_reduce(min_step_tensor, op=dist.ReduceOp.SUM)
                current_avg_min_step_reward = min_step_tensor.item() / num_trajectories
            else:
                current_avg_min_step_reward = None
        else:
            current_avg_reward = total_reward_value_local / num_trajectories
            current_avg_discrete_reward = total_discrete_reward_value_local / num_trajectories
            rewards_min = rewards_min_local
            rewards_max = rewards_max_local
            current_avg_min_step_reward = total_min_step_reward_local / num_trajectories if prm_type == "qwen" else None
        
        avg_reward_history.append(current_avg_reward)
        avg_discrete_reward_history.append(current_avg_discrete_reward)
        if prm_type == "qwen" and current_avg_min_step_reward is not None:
            avg_min_step_reward_history.append(current_avg_min_step_reward)

        if target_reward is not None and iterations_to_target is None:
            # For discrete optimization, track when discrete reward reaches target
            check_reward = current_avg_discrete_reward if not continuous_optimization else current_avg_reward
            if check_reward >= target_reward:
                iterations_to_target = iteration + 1

        # Track best soft reward and coefficients
        if current_avg_reward > best_avg_reward:
            best_avg_reward = current_avg_reward
            best_token_coeffs = gumbel_softmax.detach().cpu()

        # Track best discrete reward and token IDs
        if current_avg_discrete_reward > best_avg_discrete_reward:
            best_avg_discrete_reward = current_avg_discrete_reward
            best_discrete_token_ids = discrete_token_ids.detach().cpu()

        if is_main and (iteration + 1) % log_interval == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"  Avg Soft Reward (Gumbel): {current_avg_reward:.4f} | Best: {best_avg_reward:.4f}")
            print(f"  Avg Discrete Reward (Hard): {current_avg_discrete_reward:.4f} | Best: {best_avg_discrete_reward:.4f}")
            print(f"  Reward Gap (Soft-Hard): {current_avg_reward - current_avg_discrete_reward:+.4f}")
            if prm_type == "qwen" and current_avg_min_step_reward is not None:
                print(f"  Avg Min Step Reward: {current_avg_min_step_reward:.4f}")
            if not continuous_optimization:
                print(f"  Entropy Weight: {entropy_weight:.6f}")
            print(f"  Entropy: {entropy.item():.4f} (max possible: {np.log(vocab_size):.4f})")
            print(f"  Per-traj soft rewards: min={rewards_min:.4f}, max={rewards_max:.4f}")
            decoded_tokens = tokenizer.decode(discrete_token_ids.cpu().tolist())
            print(f"  Current Discrete Tokens: {decoded_tokens[:80]}")
            print()

    if is_main:
        print("\nBatched Optimization Complete!")
        print(f"Best Avg Soft Reward: {best_avg_reward:.4f}")
        print(f"Best Avg Discrete Reward: {best_avg_discrete_reward:.4f}")
        if prm_type == "qwen":
            print(f"PRM Type: Qwen (optimized min step reward)")
        else:
            print(f"PRM Type: Skywork (optimized final step reward)")
        if not continuous_optimization:
            print(f"Final Entropy: {entropy_history[-1]:.4f}")
            print(f"Entropy Reduction: {entropy_history[0] - entropy_history[-1]:.4f}")
        if iterations_to_target is not None:
            print(f"Iterations to reach target {target_reward}: {iterations_to_target}")
        if best_discrete_token_ids is not None:
            decoded = tokenizer.decode(best_discrete_token_ids.tolist())
            print(f"Best Discrete Tokens: {decoded}")

    return {
        "best_token_coeffs": best_token_coeffs,
        "best_discrete_token_ids": best_discrete_token_ids,
        "avg_reward_history": avg_reward_history,
        "avg_discrete_reward_history": avg_discrete_reward_history,
        "avg_min_step_reward_history": avg_min_step_reward_history if prm_type == "qwen" else None,
        "per_traj_reward_history": per_traj_reward_history,
        "entropy_history": entropy_history,
        "entropy_weight_history": entropy_weight_history,
        "best_avg_reward": best_avg_reward,
        "best_avg_discrete_reward": best_avg_discrete_reward,
        "iterations_to_target": iterations_to_target,
        "continuous_optimization": continuous_optimization,
        "num_adv_tokens": num_adv_tokens,
        "adv_position": adv_position,
        "num_trajectories": num_trajectories,
        "prm_type": prm_type,
    }


def run_single_trajectory_experiment_with_config(
    config: ExperimentConfig,
    train_data: Dict,
    train_trajectories_multiple: List,
    distributed_ctx: Optional[Dict] = None,
) -> Dict:
    """
    Run single trajectory experiment using the config object.
    """
    is_main = (distributed_ctx is None) or distributed_ctx.get("rank", 0) == 0
    mode_str = "continuous" if config.continuous else "discrete"
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Single Trajectory Experiment - {config.prm_model} ({mode_str})")
        print(f"Number of adversarial tokens: {config.num_adv_tokens}")
        print(f"{'='*60}")

    # Set up paths
    result_cache = _result_cache_path("single", config)
    checkpoint_path = _checkpoint_path("single", config)
    discrete_ids_path = _discrete_token_ids_path("single", config)
    metrics_path = _metrics_path("single", config)
    
    os.makedirs(config.cache_dir, exist_ok=True)

    # Check for cached results
    if not config.force_rerun:
        cached = _load_result_cache(result_cache, config.use_cache)
        if cached is not None:
            if is_main:
                print(f"Loading cached result from {result_cache}")
            return cached
        if config.skip_if_exists and os.path.exists(checkpoint_path):
            if is_main:
                print(f"Found checkpoint {checkpoint_path}, skipping optimization.")
            stub = {
                "best_token_coeffs": torch.load(checkpoint_path, map_location="cpu"),
                "best_discrete_token_ids": torch.load(discrete_ids_path, map_location="cpu") if os.path.exists(discrete_ids_path) else None,
                "cached_only": True,
            }
            return stub

    # Load model
    model, tokenizer, model_path, device = load_prm_model(
        config.prm_model, config.hf_cache_path, config.device
    )

    question_idx = config.single_traj_question_idx
    question = train_data["questions"][question_idx]
    answer = train_data["answers"][question_idx]
    trajectories = train_trajectories_multiple[0]

    if is_main:
        print(f"\nQuestion: {question[:200]}...")
        print(f"Ground truth answer: {answer}")
        print(f"Number of trajectories: {len(trajectories)}")

    parsed_generations = [parse_steps_from_generation(t) for t in trajectories]

    if is_main:
        print("\nCalculating rewards for all trajectories...")
    all_rewards = []
    for i, steps in enumerate(tqdm(parsed_generations)):
        try:
            rewards = calculate_stepwise_rewards(
                model, tokenizer, question, steps, model_path, device
            )
            avg_reward = np.mean(rewards) if rewards else 0
            all_rewards.append(avg_reward)
        except Exception as e:
            print(f"Error on trajectory {i}: {e}")
            all_rewards.append(0)

    # Select a trajectory (30th percentile by reward)
    sorted_indices = np.argsort(all_rewards)
    selected_idx = sorted_indices[min(30, len(sorted_indices) - 1)]
    selected_steps = parsed_generations[selected_idx]
    selected_avg_reward = all_rewards[selected_idx]

    if is_main:
        print(f"\nSelected trajectory index: {selected_idx}")
        print(f"Selected trajectory avg reward: {selected_avg_reward:.4f}")
        print(f"Number of steps: {len(selected_steps)}")

    if is_main:
        print("\nStarting adversarial token optimization...")
    
    # Get PRM type from model key
    prm_type = get_prm_type(config.prm_model)
    
    result = optimize_adversarial_tokens_single_trajectory(
        model=model,
        tokenizer=tokenizer,
        problem=question,
        steps=selected_steps,
        model_path=model_path,
        device=device,
        num_adv_tokens=config.num_adv_tokens,
        adv_position=config.adv_position,
        num_iterations=config.num_iterations,
        learning_rate=config.learning_rate,
        temperature=config.temperature,
        entropy_weight_start=config.entropy_weight_start,
        entropy_weight_end=config.entropy_weight_end,
        entropy_schedule=config.entropy_schedule,
        continuous_optimization=config.continuous,
        prm_type=prm_type,
        log_interval=config.log_interval,
        distributed_ctx=distributed_ctx,
    )

    if is_main:
        # Save continuous coefficients
        torch.save(result["best_token_coeffs"], checkpoint_path)
        print(f"\nSaved best token coefficients to {checkpoint_path}")
        
        # Save discrete token IDs
        if result["best_discrete_token_ids"] is not None:
            torch.save(result["best_discrete_token_ids"], discrete_ids_path)
            print(f"Saved best discrete token IDs to {discrete_ids_path}")
        
        # Save training metrics
        metrics = {
            "reward_history": result["reward_history"],
            "discrete_reward_history": result["discrete_reward_history"],
            "entropy_history": result["entropy_history"],
            "entropy_weight_history": result.get("entropy_weight_history", []),
            "config": asdict(config),
        }
        _save_metrics(metrics, metrics_path)

    del model
    torch.cuda.empty_cache()

    result["prm_key"] = config.prm_model
    result["initial_avg_reward"] = selected_avg_reward
    result["config"] = asdict(config)
    
    if is_main:
        _save_result_cache(result_cache, result, config.use_cache)
        # Save config
        config.save(_config_path(config))

    return result


def run_batched_trajectory_experiment_with_config(
    config: ExperimentConfig,
    train_data: Dict,
    train_trajectories_single: List,
    distributed_ctx: Optional[Dict] = None,
) -> Tuple[Dict, Optional[object], Optional[object], Optional[str], Optional[torch.device]]:
    """
    Run batched trajectory experiment using the config object.
    """
    is_main = (distributed_ctx is None) or distributed_ctx.get("rank", 0) == 0
    mode_str = "continuous" if config.continuous else "discrete"
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Batched Trajectory Experiment - {config.prm_model} ({mode_str})")
        print(f"Number of adversarial tokens: {config.num_adv_tokens}")
        n_traj = config.num_train_trajectories or "all"
        print(f"Number of training trajectories: {n_traj}")
        print(f"{'='*60}")

    # Set up paths
    result_cache = _result_cache_path("batched", config)
    checkpoint_path = _checkpoint_path("batched", config)
    discrete_ids_path = _discrete_token_ids_path("batched", config)
    metrics_path = _metrics_path("batched", config)
    
    os.makedirs(config.cache_dir, exist_ok=True)

    # Check for cached results
    if not config.force_rerun:
        cached = _load_result_cache(result_cache, config.use_cache)
        if cached is not None:
            if is_main:
                print(f"Loading cached result from {result_cache}")
            return cached, None, None, None, None
        if config.skip_if_exists and os.path.exists(checkpoint_path):
            if is_main:
                print(f"Found checkpoint {checkpoint_path}, skipping optimization.")
            stub = {
                "best_token_coeffs": torch.load(checkpoint_path, map_location="cpu"),
                "best_discrete_token_ids": torch.load(discrete_ids_path, map_location="cpu") if os.path.exists(discrete_ids_path) else None,
                "cached_only": True,
            }
            return stub, None, None, None, None

    # Load model
    model, tokenizer, model_path, device = load_prm_model(
        config.prm_model, config.hf_cache_path, config.device
    )

    # Prepare trajectories
    problems = train_data["questions"]
    all_steps = [parse_steps_from_generation(t[0]) for t in train_trajectories_single]
    
    # Limit number of trajectories if specified
    if config.num_train_trajectories is not None:
        n_traj = min(config.num_train_trajectories, len(problems))
        problems = problems[:n_traj]
        all_steps = all_steps[:n_traj]

    if is_main:
        print(f"Number of questions: {len(problems)}")
        print(f"Number of trajectories: {len(all_steps)}")

    if is_main:
        print("\nCalculating initial rewards...")
    initial_rewards = []
    for i, (problem, steps) in enumerate(zip(problems, all_steps)):
        if len(steps) == 0:
            initial_rewards.append(0)
            continue
        try:
            rewards = calculate_stepwise_rewards(
                model, tokenizer, problem, steps, model_path, device
            )
            avg_reward = np.mean(rewards) if rewards else 0
            initial_rewards.append(avg_reward)
        except Exception as e:
            print(f"Error on trajectory {i}: {e}")
            initial_rewards.append(0)

    if is_main:
        print(f"Initial avg reward across dataset: {np.mean(initial_rewards):.4f}")

    if is_main:
        print("\nStarting batched adversarial token optimization...")
    
    # Get PRM type from model key
    prm_type = get_prm_type(config.prm_model)
    
    result = optimize_adversarial_tokens_batched_trajectories(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        all_steps=all_steps,
        model_path=model_path,
        device=device,
        num_adv_tokens=config.num_adv_tokens,
        adv_position=config.adv_position,
        num_iterations=config.num_iterations,
        learning_rate=config.learning_rate,
        temperature=config.temperature,
        entropy_weight_start=config.entropy_weight_start,
        entropy_weight_end=config.entropy_weight_end,
        entropy_schedule=config.entropy_schedule,
        chunk_size=config.batch_chunk_size,
        continuous_optimization=config.continuous,
        prm_type=prm_type,
        log_interval=config.log_interval,
        distributed_ctx=distributed_ctx,
    )

    if is_main:
        # Save continuous coefficients
        torch.save(result["best_token_coeffs"], checkpoint_path)
        print(f"\nSaved best token coefficients to {checkpoint_path}")
        
        # Save discrete token IDs
        if result["best_discrete_token_ids"] is not None:
            torch.save(result["best_discrete_token_ids"], discrete_ids_path)
            print(f"Saved best discrete token IDs to {discrete_ids_path}")
        
        # Save training metrics
        metrics = {
            "avg_reward_history": result["avg_reward_history"],
            "avg_discrete_reward_history": result["avg_discrete_reward_history"],
            "per_traj_reward_history": result["per_traj_reward_history"],
            "entropy_history": result["entropy_history"],
            "entropy_weight_history": result.get("entropy_weight_history", []),
            "config": asdict(config),
        }
        _save_metrics(metrics, metrics_path)

    result["prm_key"] = config.prm_model
    result["initial_rewards"] = initial_rewards
    result["initial_avg_reward"] = np.mean(initial_rewards)
    result["config"] = asdict(config)
    
    if is_main:
        _save_result_cache(result_cache, result, config.use_cache)
        # Save config
        config.save(_config_path(config))

    return result, model, tokenizer, model_path, device


def evaluate_transfer(
    model,
    tokenizer,
    model_path: str,
    device: torch.device,
    problems: List[str],
    all_steps: List[List[str]],
    adversarial_token_coeffs: Optional[torch.Tensor] = None,
    adversarial_token_ids: Optional[torch.Tensor] = None,
    use_discrete: bool = False,
    num_eval_trajectories: Optional[int] = None,
    adv_position: str = "end",
    prm_type: str = "skywork",
) -> Dict:
    """
    Evaluate the effect of adversarial token(s) on a different dataset.
    
    Args:
        model: The PRM model
        tokenizer: Tokenizer
        model_path: Path to model
        device: Device to run on
        problems: List of problem statements
        all_steps: List of step lists for each problem
        adversarial_token_coeffs: Continuous token coefficients, shape (num_tokens, vocab_size)
        adversarial_token_ids: Discrete token IDs, shape (num_tokens,)
        use_discrete: If True and adversarial_token_ids is provided, use discrete tokens
        num_eval_trajectories: Limit evaluation to this many trajectories (None = all)
        adv_position: "end" (after solution) or "middle" (after question, before solution)
        prm_type: "skywork" or "qwen"
    """
    embed_layer = get_embedding_layer(model)
    step_embeddings = get_step_token_embedding(embed_layer, tokenizer, device)

    # Limit evaluation trajectories if specified
    if num_eval_trajectories is not None:
        n_eval = min(num_eval_trajectories, len(problems))
        problems = problems[:n_eval]
        all_steps = all_steps[:n_eval]

    # Determine which adversarial embedding to use
    adv_embeddings = None
    using_discrete = False
    num_adv_tokens = 0
    
    if use_discrete and adversarial_token_ids is not None:
        # Use discrete token embeddings
        token_ids = adversarial_token_ids.to(device)
        adv_embeddings = embed_layer(token_ids)  # Shape: (num_tokens, embed_dim)
        using_discrete = True
        num_adv_tokens = len(token_ids)
        print(f"Using {num_adv_tokens} discrete adversarial token(s) at position '{adv_position}': {token_ids.tolist()}")
        print(f"Decoded: {tokenizer.decode(token_ids.tolist())}")
    elif adversarial_token_coeffs is not None:
        # Use continuous token embeddings
        coeffs = adversarial_token_coeffs.to(device).to(embed_layer.weight.dtype)
        adv_embeddings = torch.matmul(coeffs, embed_layer.weight)  # Shape: (num_tokens, embed_dim)
        num_adv_tokens = coeffs.shape[0]
        print(f"Using {num_adv_tokens} continuous adversarial token(s) at position '{adv_position}'")

    rewards_before = []
    rewards_after = []

    for problem, steps in tqdm(zip(problems, all_steps), total=len(problems)):
        if len(steps) == 0:
            rewards_before.append(0)
            rewards_after.append(0)
            continue

        input_ids, _ = prepare_input(
            model_path,
            problem=problem,
            steps=steps,
            tokenizer=tokenizer,
            device=device,
        )
        orig_embeddings = embed_layer(input_ids)
        question_length = get_question_length(tokenizer, problem, prm_type)
        
        # For Qwen with middle position, add step separator after adv tokens
        add_step_after_adv = (prm_type == "qwen" and adv_position == "middle")
        
        # For Qwen, get original step token positions
        orig_step_positions = None
        if prm_type == "qwen":
            orig_step_positions = find_step_token_positions(input_ids, tokenizer)

        with torch.no_grad():
            full_emb_before = torch.cat([orig_embeddings, step_embeddings], dim=0).unsqueeze(0)
            if prm_type == "skywork":
                last_hidden_state = get_last_hidden_state(model, full_emb_before)
                reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                reward_before = torch.sigmoid(reward_logits[0, -1]).item()
            elif prm_type == "qwen":
                # For "before" evaluation, step positions are: original + final step token
                step_positions_before = torch.cat([
                    orig_step_positions,
                    torch.tensor([full_emb_before.shape[1] - 1], device=device)
                ])
                _, step_rewards_before = compute_reward_proxy_qwen(
                    model, full_emb_before, step_positions_before
                )
                reward_before = step_rewards_before.min().item()
            rewards_before.append(reward_before)

        if adv_embeddings is not None:
            with torch.no_grad():
                full_emb_after = construct_embeddings_with_adv(
                    orig_embeddings, adv_embeddings, step_embeddings, adv_position, question_length,
                    prm_type=prm_type, add_step_after_adv=add_step_after_adv
                ).unsqueeze(0)
                if prm_type == "skywork":
                    last_hidden_state = get_last_hidden_state(model, full_emb_after)
                    reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                    reward_after = torch.sigmoid(reward_logits[0, -1]).item()
                elif prm_type == "qwen":
                    # Adjust step positions for adversarial tokens
                    if adv_position == "end":
                        step_positions_after = torch.cat([
                            orig_step_positions,
                            torch.tensor([full_emb_after.shape[1] - 1], device=device)
                        ])
                    else:  # middle with step separator after adv tokens
                        adv_step_sep_position = question_length + num_adv_tokens
                        shifted_solution_positions = orig_step_positions + num_adv_tokens + 1
                        step_positions_after = torch.cat([
                            torch.tensor([adv_step_sep_position], device=device),
                            shifted_solution_positions,
                            torch.tensor([full_emb_after.shape[1] - 1], device=device)
                        ])
                    _, step_rewards_after = compute_reward_proxy_qwen(
                        model, full_emb_after, step_positions_after
                    )
                    reward_after = step_rewards_after.min().item()
                rewards_after.append(reward_after)
        else:
            rewards_after.append(reward_before)

    return {
        "rewards_before": rewards_before,
        "rewards_after": rewards_after,
        "avg_before": np.mean(rewards_before),
        "avg_after": np.mean(rewards_after),
        "improvement": np.mean(rewards_after) - np.mean(rewards_before),
        "used_discrete_tokens": using_discrete,
        "num_adv_tokens": num_adv_tokens,
        "adv_position": adv_position,
        "num_eval_trajectories": len(problems),
    }


def get_best_rewards(results: Dict) -> Dict[str, float]:
    """Extract best reward achieved for each PRM."""
    best_rewards = {}
    for prm_key, result in results.items():
        best_rewards[prm_key] = result.get("best_reward") or result.get("best_avg_reward")
    return best_rewards


def compute_fair_target_rewards(results: Dict, num_targets: int = 10) -> List[float]:
    """
    Compute target rewards that all PRMs can potentially reach.
    Uses the minimum of the maximum rewards achieved across PRMs as the ceiling.
    """
    best_rewards = get_best_rewards(results)
    min_best_reward = min(best_rewards.values())

    min_initial = float("inf")
    for result in results.values():
        history = result.get("reward_history") or result.get("avg_reward_history")
        if history:
            min_initial = min(min_initial, history[0])

    targets = [
        min_initial + (min_best_reward - min_initial) * i / (num_targets - 1)
        for i in range(num_targets)
    ]
    return targets, min_best_reward, best_rewards


def analyze_iterations_to_reward(results: Dict, target_rewards: List[float]) -> Dict:
    """Analyze how many iterations it takes to reach different target rewards."""
    analysis = {}

    for prm_key, result in results.items():
        reward_history = result.get("reward_history") or result.get("avg_reward_history")

        iterations_to_targets = {}
        for target in target_rewards:
            for i, reward in enumerate(reward_history):
                if reward >= target:
                    iterations_to_targets[target] = i + 1
                    break
            else:
                iterations_to_targets[target] = None

        analysis[prm_key] = {
            "iterations_to_targets": iterations_to_targets,
            "best_reward": result.get("best_reward") or result.get("best_avg_reward"),
            "final_reward": reward_history[-1] if reward_history else 0,
        }

    return analysis


def print_iterations_analysis(results: Dict, setup_name: str):
    """Print a complete iterations analysis for a given experiment setup."""
    print("\n" + "=" * 80)
    print(f"ITERATIONS ANALYSIS - {setup_name}")
    print("=" * 80)

    target_rewards, min_best, best_rewards = compute_fair_target_rewards(results)

    print("\nBest rewards achieved by each PRM:")
    for prm_key, best in best_rewards.items():
        print(f"  {prm_key}: {best:.4f}")
    print(f"\nMinimum best reward (fair comparison ceiling): {min_best:.4f}")

    analysis = analyze_iterations_to_reward(results, target_rewards)

    print("\nIterations to reach target rewards:")
    print("-" * 70)
    header = f"{'Target':<12}" + "".join([f"{k:<15}" for k in results.keys()])
    print(header)
    print("-" * 70)
    for target in target_rewards:
        row = f"{target:<12.4f}"
        for prm_key in results.keys():
            iters = analysis.get(prm_key, {}).get("iterations_to_targets", {}).get(target, "N/A")
            row += f"{str(iters):<15}"
        print(row)

    print(f"\n{'='*70}")
    print(f"Iterations to reach the fair comparison point ({min_best:.4f}):")
    print("-" * 70)
    for prm_key in results.keys():
        result = results[prm_key]
        history = result.get("reward_history") or result.get("avg_reward_history")
        iters = None
        for i, reward in enumerate(history):
            if reward >= min_best:
                iters = i + 1
                break
        print(f"  {prm_key}: {iters if iters else 'Never reached'} iterations")

    return analysis, target_rewards


def plot_3d_reward_landscape(
    model,
    tokenizer,
    model_path: str,
    device: torch.device,
    problem: str,
    steps: List[str],
    adversarial_token_coeffs: torch.Tensor,
    n_points: int = 25,
    prm_name: str = "PRM",
):
    """
    Plot 3D reward landscape around the adversarial token.
    """
    print(f"\nGenerating 3D reward landscape for {prm_name}...")

    embed_layer = get_embedding_layer(model)
    vocab_size = embed_layer.weight.shape[0]

    input_ids, _ = prepare_input(
        model_path, problem=problem, steps=steps, tokenizer=tokenizer, device=device
    )
    orig_embeddings = embed_layer(input_ids).unsqueeze(0)
    step_embeddings = get_step_token_embedding(embed_layer, tokenizer, device).unsqueeze(0)

    coeffs = adversarial_token_coeffs.to(device).to(embed_layer.weight.dtype)
    v_adv = torch.matmul(coeffs, embed_layer.weight).squeeze(0)
    v_norm = torch.norm(v_adv)

    random_weights_1 = torch.rand(vocab_size, device=device, dtype=torch.float32)
    random_weights_1 = random_weights_1 / random_weights_1.sum()
    random_weights_1 = random_weights_1.to(torch.bfloat16)
    d1_raw = (random_weights_1 @ embed_layer.weight).detach()
    d1_unit = d1_raw / torch.norm(d1_raw)

    random_weights_2 = torch.rand(vocab_size, device=device, dtype=torch.float32)
    random_weights_2 = random_weights_2 / random_weights_2.sum()
    random_weights_2 = random_weights_2.to(torch.bfloat16)
    d2_raw = (random_weights_2 @ embed_layer.weight).detach()
    d2_raw = d2_raw - (torch.dot(d2_raw.float(), d1_unit.float()) * d1_unit.float()).to(
        torch.bfloat16
    )
    d2_unit = d2_raw / torch.norm(d2_raw)

    print(
        f"Direction orthogonality check (should be ~0): "
        f"{torch.dot(d1_unit.float(), d2_unit.float()).item():.6f}"
    )

    d1_scaled = d1_unit * v_norm
    d2_scaled = d2_unit * v_norm

    epsilons = np.linspace(-1.0, 1.0, n_points)
    eps1_grid, eps2_grid = np.meshgrid(epsilons, epsilons)
    rewards_grid = np.zeros((n_points, n_points))

    print(f"Grid: {n_points}x{n_points} = {n_points**2} evaluations")

    for i, eps1 in enumerate(tqdm(epsilons)):
        for j, eps2 in enumerate(epsilons):
            perturbation = eps1 * d1_scaled + eps2 * d2_scaled
            v_perturbed = v_adv + perturbation

            v_perturbed_reshaped = v_perturbed.view(1, 1, -1)
            full_embeddings = torch.cat(
                [orig_embeddings, v_perturbed_reshaped, step_embeddings], dim=1
            )

            with torch.no_grad():
                last_hidden_state = get_last_hidden_state(model, full_embeddings)
                reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                reward = torch.sigmoid(reward_logits[0, -1]).item()

            rewards_grid[j, i] = reward

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        eps1_grid, eps2_grid, rewards_grid, cmap="viridis", edgecolor="none", alpha=0.8
    )

    center_idx = n_points // 2
    center_reward = rewards_grid[center_idx, center_idx]
    ax.scatter(
        [0],
        [0],
        [center_reward],
        color="red",
        s=100,
        marker="*",
        label=f"Adversarial Token (reward={center_reward:.4f})",
        zorder=5,
    )

    ax.set_xlabel(r"$\epsilon_1$ (Direction 1)", fontsize=11, labelpad=10)
    ax.set_ylabel(r"$\epsilon_2$ (Direction 2)", fontsize=11, labelpad=10)
    ax.set_zlabel("Reward", fontsize=11, labelpad=10)
    ax.set_title(f"3D Reward Landscape: {prm_name}", fontsize=13, fontweight="bold")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Reward")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(f"{CACHE_DIR}/landscape_3d_{prm_name.replace(' ', '_')}.png", dpi=150)
    plt.show()

    print(f"Reward range: [{rewards_grid.min():.4f}, {rewards_grid.max():.4f}]")
    print(f"Center reward: {center_reward:.4f}")

    return rewards_grid


def plot_2d_perturbation_analysis(
    model,
    tokenizer,
    model_path: str,
    device: torch.device,
    problem: str,
    steps: List[str],
    adversarial_token_coeffs: torch.Tensor,
    n_points: int = 50,
    prm_name: str = "PRM",
):
    """
    1D perturbation analysis: sweep epsilon and plot reward.
    """
    print(f"\nRunning 2D perturbation analysis for {prm_name}...")

    embed_layer = get_embedding_layer(model)
    vocab_size = embed_layer.weight.shape[0]

    input_ids, _ = prepare_input(
        model_path, problem=problem, steps=steps, tokenizer=tokenizer, device=device
    )
    orig_embeddings = embed_layer(input_ids).unsqueeze(0)
    step_embeddings = get_step_token_embedding(embed_layer, tokenizer, device).unsqueeze(0)

    coeffs = adversarial_token_coeffs.to(device).to(embed_layer.weight.dtype)
    v_adv = torch.matmul(coeffs, embed_layer.weight).squeeze(0)
    v_norm = torch.norm(v_adv)

    random_weights = torch.rand(vocab_size, device=device, dtype=torch.float32)
    random_weights = random_weights / random_weights.sum()
    random_weights = random_weights.to(torch.bfloat16)
    d_raw = (random_weights @ embed_layer.weight).detach()
    d_unit = d_raw / torch.norm(d_raw)
    d_scaled = d_unit * v_norm

    epsilons = np.linspace(-1.0, 1.0, n_points)
    rewards = []

    for eps in tqdm(epsilons):
        perturbation = eps * d_scaled
        v_perturbed = v_adv + perturbation

        v_perturbed_reshaped = v_perturbed.view(1, 1, -1)
        full_embeddings = torch.cat(
            [orig_embeddings, v_perturbed_reshaped, step_embeddings], dim=1
        )

        with torch.no_grad():
            last_hidden_state = get_last_hidden_state(model, full_embeddings)
            reward_logits = model.v_head(last_hidden_state).squeeze(-1)
            reward = torch.sigmoid(reward_logits[0, -1]).item()

        rewards.append(reward)

    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, rewards, marker="o", markersize=4, linewidth=2, color="royalblue")
    plt.axvline(x=0, color="red", linestyle="--", label="Adversarial Token (eps=0)")
    plt.xlabel(r"Epsilon ($\epsilon$)", fontsize=12, fontweight="bold")
    plt.ylabel("Reward", fontsize=12, fontweight="bold")
    plt.title(f"Reward Landscape Perturbation Analysis - {prm_name}", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CACHE_DIR}/perturbation_2d_{prm_name.replace(' ', '_')}.png", dpi=150)
    plt.show()

    return epsilons, rewards


def _stable_seed(label: str, base: int = RANDOM_TOKEN_SEED) -> int:
    return base + sum(ord(c) for c in label)


def sample_random_token_id(
    vocab_size: int, exclude_ids: Optional[List[int]] = None, rng: Optional[np.random.Generator] = None
) -> int:
    exclude_set = set(exclude_ids or [])
    rng = rng or np.random.default_rng(RANDOM_TOKEN_SEED)
    while True:
        token_id = int(rng.integers(0, vocab_size))
        if token_id not in exclude_set:
            return token_id


def build_unit_directions(
    embed_layer, vocab_size: int, device: torch.device, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Build a single pair of orthogonal unit directions in embedding space."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    random_weights_1 = torch.rand(vocab_size, device=device, dtype=torch.float32, generator=gen)
    random_weights_1 = random_weights_1 / random_weights_1.sum()
    random_weights_1 = random_weights_1.to(embed_layer.weight.dtype)
    d1_raw = (random_weights_1 @ embed_layer.weight).detach()
    d1_unit = d1_raw / torch.norm(d1_raw)

    random_weights_2 = torch.rand(vocab_size, device=device, dtype=torch.float32, generator=gen)
    random_weights_2 = random_weights_2 / random_weights_2.sum()
    random_weights_2 = random_weights_2.to(embed_layer.weight.dtype)
    d2_raw = (random_weights_2 @ embed_layer.weight).detach()
    d2_raw = d2_raw - (torch.dot(d2_raw.float(), d1_unit.float()) * d1_unit.float()).to(
        embed_layer.weight.dtype
    )
    d2_unit = d2_raw / torch.norm(d2_raw)

    ortho = torch.dot(d1_unit.float(), d2_unit.float()).item()
    return d1_unit, d2_unit, ortho


def build_unit_directions_per_token(
    embed_layer, vocab_size: int, device: torch.device, seed: int, num_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build per-token orthogonal unit directions in embedding space.
    
    Returns:
        d1_units: (num_tokens, embed_dim) - first direction for each token
        d2_units: (num_tokens, embed_dim) - second direction for each token (orthogonal to d1)
        avg_ortho: Average orthogonality across all token pairs
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    embed_dim = embed_layer.weight.shape[1]
    d1_units = torch.zeros(num_tokens, embed_dim, device=device, dtype=embed_layer.weight.dtype)
    d2_units = torch.zeros(num_tokens, embed_dim, device=device, dtype=embed_layer.weight.dtype)
    ortho_values = []
    
    for t in range(num_tokens):
        # Generate first direction for token t
        random_weights_1 = torch.rand(vocab_size, device=device, dtype=torch.float32, generator=gen)
        random_weights_1 = random_weights_1 / random_weights_1.sum()
        random_weights_1 = random_weights_1.to(embed_layer.weight.dtype)
        d1_raw = (random_weights_1 @ embed_layer.weight).detach()
        d1_unit = d1_raw / torch.norm(d1_raw)
        
        # Generate second direction for token t (orthogonal to first)
        random_weights_2 = torch.rand(vocab_size, device=device, dtype=torch.float32, generator=gen)
        random_weights_2 = random_weights_2 / random_weights_2.sum()
        random_weights_2 = random_weights_2.to(embed_layer.weight.dtype)
        d2_raw = (random_weights_2 @ embed_layer.weight).detach()
        d2_raw = d2_raw - (torch.dot(d2_raw.float(), d1_unit.float()) * d1_unit.float()).to(
            embed_layer.weight.dtype
        )
        d2_unit = d2_raw / torch.norm(d2_raw)
        
        d1_units[t] = d1_unit
        d2_units[t] = d2_unit
        ortho_values.append(torch.dot(d1_unit.float(), d2_unit.float()).item())
    
    avg_ortho = np.mean(ortho_values)
    return d1_units, d2_units, avg_ortho


def compute_landscape_volume(rewards_grid: np.ndarray, epsilons: np.ndarray) -> Dict[str, float]:
    """
    Compute the volume under the reward landscape surface using 2D trapezoidal integration.
    
    This gives a combined measure of how wide/flat the landscape is and how tall it is.
    Higher volume = larger rewards spread over more area.
    
    Args:
        rewards_grid: 2D array of reward values
        epsilons: 1D array of epsilon values used for the grid
        
    Returns:
        Dictionary with volume metrics
    """
    if rewards_grid.shape[0] < 2 or rewards_grid.shape[1] < 2:
        return {
            "volume": float("nan"),
            "volume_normalized": float("nan"),
            "mean_reward": float("nan"),
            "peak_reward": float("nan"),
            "peak_location": (float("nan"), float("nan")),
            "center_reward": float("nan"),
            "grid_area": float("nan"),
        }
    
    h = float(epsilons[1] - epsilons[0])  # Grid spacing
    n_points = len(epsilons)
    
    # 2D Trapezoidal rule for volume integration
    # Interior points get weight 1, edges get weight 0.5, corners get weight 0.25
    weights = np.ones_like(rewards_grid)
    weights[0, :] *= 0.5
    weights[-1, :] *= 0.5
    weights[:, 0] *= 0.5
    weights[:, -1] *= 0.5
    
    # Volume = sum of (reward * weight * dx * dy)
    volume = float(np.sum(rewards_grid * weights) * h * h)
    
    # Grid area (total area covered by the grid)
    grid_span = epsilons[-1] - epsilons[0]
    grid_area = grid_span * grid_span
    
    # Normalized volume (volume per unit area, essentially mean reward)
    volume_normalized = volume / grid_area if grid_area > 0 else float("nan")
    
    # Mean reward across all grid points
    mean_reward = float(np.mean(rewards_grid))
    
    # Peak reward and its location
    peak_idx = np.unravel_index(np.argmax(rewards_grid), rewards_grid.shape)
    peak_reward = float(rewards_grid[peak_idx])
    peak_location = (float(epsilons[peak_idx[1]]), float(epsilons[peak_idx[0]]))
    
    # Center reward (at epsilon = 0, 0)
    center = n_points // 2
    center_reward = float(rewards_grid[center, center])
    
    return {
        "volume": volume,
        "volume_normalized": volume_normalized,
        "mean_reward": mean_reward,
        "peak_reward": peak_reward,
        "peak_location": peak_location,
        "center_reward": center_reward,
        "grid_area": grid_area,
    }


def compute_curvature_metric(rewards_grid: np.ndarray, epsilons: np.ndarray) -> Dict[str, float]:
    if rewards_grid.shape[0] < 3 or rewards_grid.shape[1] < 3:
        return {
            "curvature_metric": float("nan"),
            "dxx": float("nan"),
            "dyy": float("nan"),
            "dxy": float("nan"),
            "eig1": float("nan"),
            "eig2": float("nan"),
            "top_eigenvalue": float("nan"),
            "max_eig_abs": float("nan"),
        }
    h = float(epsilons[1] - epsilons[0])
    center = rewards_grid.shape[0] // 2
    dxx = (
        rewards_grid[center, center + 1]
        - 2 * rewards_grid[center, center]
        + rewards_grid[center, center - 1]
    ) / (h * h)
    dyy = (
        rewards_grid[center + 1, center]
        - 2 * rewards_grid[center, center]
        + rewards_grid[center - 1, center]
    ) / (h * h)
    dxy = (
        rewards_grid[center + 1, center + 1]
        - rewards_grid[center + 1, center - 1]
        - rewards_grid[center - 1, center + 1]
        + rewards_grid[center - 1, center - 1]
    ) / (4 * h * h)
    curvature_metric = float(np.sqrt(dxx ** 2 + dyy ** 2 + 2 * dxy ** 2))
    trace = dxx + dyy
    det = dxx * dyy - dxy * dxy
    disc = max(trace * trace - 4 * det, 0.0)
    sqrt_disc = float(np.sqrt(disc))
    eig1 = float(0.5 * (trace + sqrt_disc))
    eig2 = float(0.5 * (trace - sqrt_disc))
    # Top eigenvalue is the maximum (most positive) eigenvalue
    top_eigenvalue = max(eig1, eig2)
    max_eig_abs = float(max(abs(eig1), abs(eig2)))
    return {
        "curvature_metric": curvature_metric,
        "dxx": float(dxx),
        "dyy": float(dyy),
        "dxy": float(dxy),
        "eig1": eig1,
        "eig2": eig2,
        "top_eigenvalue": top_eigenvalue,
        "max_eig_abs": max_eig_abs,
    }


def compute_reward_grid_single(
    model,
    orig_embeddings: torch.Tensor,
    step_embeddings: torch.Tensor,
    v_center: torch.Tensor,
    d1_scaled: torch.Tensor,
    d2_scaled: torch.Tensor,
    epsilons: np.ndarray,
    adv_position: str = "end",
    question_length: int = 0,
    prm_type: str = "skywork",
    orig_step_positions: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Compute reward grid for a single trajectory with single token embedding."""
    n_points = len(epsilons)
    rewards_grid = np.zeros((n_points, n_points))
    add_step_after_adv = (prm_type == "qwen" and adv_position == "middle")
    num_adv_tokens = 1  # Single token
    
    for i, eps1 in enumerate(tqdm(epsilons)):
        for j, eps2 in enumerate(epsilons):
            perturbation = eps1 * d1_scaled + eps2 * d2_scaled
            v_perturbed = v_center + perturbation
            v_perturbed_reshaped = v_perturbed.view(1, -1)
            full_embeddings = construct_embeddings_with_adv(
                orig_embeddings, v_perturbed_reshaped, step_embeddings, adv_position, question_length,
                prm_type=prm_type, add_step_after_adv=add_step_after_adv
            ).unsqueeze(0)
            with torch.no_grad():
                if prm_type == "skywork":
                    last_hidden_state = get_last_hidden_state(model, full_embeddings)
                    reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                    reward = torch.sigmoid(reward_logits[0, -1]).item()
                elif prm_type == "qwen":
                    # Compute step positions for Qwen
                    if adv_position == "end":
                        step_positions = torch.cat([
                            orig_step_positions,
                            torch.tensor([full_embeddings.shape[1] - 1], device=full_embeddings.device)
                        ])
                    else:  # middle
                        adv_step_sep_position = question_length + num_adv_tokens
                        shifted_positions = orig_step_positions + num_adv_tokens + 1
                        step_positions = torch.cat([
                            torch.tensor([adv_step_sep_position], device=full_embeddings.device),
                            shifted_positions,
                            torch.tensor([full_embeddings.shape[1] - 1], device=full_embeddings.device)
                        ])
                    _, step_rewards = compute_reward_proxy_qwen(model, full_embeddings, step_positions)
                    reward = step_rewards.min().item()
            rewards_grid[j, i] = reward
    return rewards_grid


def compute_reward_grid_single_multi_token(
    model,
    orig_embeddings: torch.Tensor,
    step_embeddings: torch.Tensor,
    v_centers: torch.Tensor,  # Shape: (num_tokens, embed_dim)
    d1_scaled: torch.Tensor,  # Shape: (num_tokens, embed_dim) - per-token directions
    d2_scaled: torch.Tensor,  # Shape: (num_tokens, embed_dim) - per-token directions
    epsilons: np.ndarray,
    adv_position: str = "end",
    question_length: int = 0,
    prm_type: str = "skywork",
    orig_step_positions: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute reward grid for a single trajectory with multiple token embeddings.
    
    Each token has its own pair of perturbation directions, but the same epsilon
    values are used across all tokens.
    
    Args:
        v_centers: (num_tokens, embed_dim) - the embeddings of all adversarial tokens
        d1_scaled: (num_tokens, embed_dim) - first scaled direction for each token
        d2_scaled: (num_tokens, embed_dim) - second scaled direction for each token
        epsilons: Array of epsilon values to sweep
        adv_position: "end" (after solution) or "middle" (after question, before solution)
        question_length: Length of question portion (for "middle" position)
        prm_type: "skywork" or "qwen"
        orig_step_positions: Step token positions for Qwen PRM
    """
    n_points = len(epsilons)
    rewards_grid = np.zeros((n_points, n_points))
    add_step_after_adv = (prm_type == "qwen" and adv_position == "middle")
    num_adv_tokens = v_centers.shape[0]
    
    for i, eps1 in enumerate(tqdm(epsilons)):
        for j, eps2 in enumerate(epsilons):
            # Per-token perturbations: each token gets its own direction pair
            # but same epsilon scalars
            perturbations = eps1 * d1_scaled + eps2 * d2_scaled  # (num_tokens, embed_dim)
            v_perturbed = v_centers + perturbations  # (num_tokens, embed_dim)
            
            full_embeddings = construct_embeddings_with_adv(
                orig_embeddings, v_perturbed, step_embeddings, adv_position, question_length,
                prm_type=prm_type, add_step_after_adv=add_step_after_adv
            ).unsqueeze(0)
            with torch.no_grad():
                if prm_type == "skywork":
                    last_hidden_state = get_last_hidden_state(model, full_embeddings)
                    reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                    reward = torch.sigmoid(reward_logits[0, -1]).item()
                elif prm_type == "qwen":
                    # Compute step positions for Qwen
                    if adv_position == "end":
                        step_positions = torch.cat([
                            orig_step_positions,
                            torch.tensor([full_embeddings.shape[1] - 1], device=full_embeddings.device)
                        ])
                    else:  # middle
                        adv_step_sep_position = question_length + num_adv_tokens
                        shifted_positions = orig_step_positions + num_adv_tokens + 1
                        step_positions = torch.cat([
                            torch.tensor([adv_step_sep_position], device=full_embeddings.device),
                            shifted_positions,
                            torch.tensor([full_embeddings.shape[1] - 1], device=full_embeddings.device)
                        ])
                    _, step_rewards = compute_reward_proxy_qwen(model, full_embeddings, step_positions)
                    reward = step_rewards.min().item()
            rewards_grid[j, i] = reward
    return rewards_grid


def compute_reward_grid_batched(
    model,
    orig_embeddings_list: List[torch.Tensor],
    step_embeddings: torch.Tensor,
    v_center: torch.Tensor,
    d1_scaled: torch.Tensor,
    d2_scaled: torch.Tensor,
    epsilons: np.ndarray,
    adv_position: str = "end",
    question_lengths: Optional[List[int]] = None,
    prm_type: str = "skywork",
    all_step_positions: Optional[List[torch.Tensor]] = None,
) -> np.ndarray:
    """Compute reward grid for batched trajectories with single token embedding."""
    n_points = len(epsilons)
    rewards_grid = np.zeros((n_points, n_points))
    if question_lengths is None:
        question_lengths = [0] * len(orig_embeddings_list)
    add_step_after_adv = (prm_type == "qwen" and adv_position == "middle")
    num_adv_tokens = 1  # Single token
    
    for i, eps1 in enumerate(tqdm(epsilons)):
        for j, eps2 in enumerate(epsilons):
            perturbation = eps1 * d1_scaled + eps2 * d2_scaled
            v_perturbed = v_center + perturbation
            v_perturbed_reshaped = v_perturbed.view(1, -1)
            total = 0.0
            count = 0
            for idx, orig_embeddings in enumerate(orig_embeddings_list):
                q_len = question_lengths[idx]
                full_embeddings = construct_embeddings_with_adv(
                    orig_embeddings, v_perturbed_reshaped, step_embeddings, adv_position, q_len,
                    prm_type=prm_type, add_step_after_adv=add_step_after_adv
                ).unsqueeze(0)
                with torch.no_grad():
                    if prm_type == "skywork":
                        last_hidden_state = get_last_hidden_state(model, full_embeddings)
                        reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                        reward = torch.sigmoid(reward_logits[0, -1]).item()
                    elif prm_type == "qwen":
                        orig_step_pos = all_step_positions[idx]
                        if adv_position == "end":
                            step_positions = torch.cat([
                                orig_step_pos,
                                torch.tensor([full_embeddings.shape[1] - 1], device=full_embeddings.device)
                            ])
                        else:  # middle
                            adv_step_sep_position = q_len + num_adv_tokens
                            shifted_positions = orig_step_pos + num_adv_tokens + 1
                            step_positions = torch.cat([
                                torch.tensor([adv_step_sep_position], device=full_embeddings.device),
                                shifted_positions,
                                torch.tensor([full_embeddings.shape[1] - 1], device=full_embeddings.device)
                            ])
                        _, step_rewards = compute_reward_proxy_qwen(model, full_embeddings, step_positions)
                        reward = step_rewards.min().item()
                total += reward
                count += 1
            rewards_grid[j, i] = total / max(count, 1)
    return rewards_grid


def compute_reward_grid_batched_multi_token(
    model,
    orig_embeddings_list: List[torch.Tensor],
    step_embeddings: torch.Tensor,
    v_centers: torch.Tensor,  # Shape: (num_tokens, embed_dim)
    d1_scaled: torch.Tensor,  # Shape: (num_tokens, embed_dim) - per-token directions
    d2_scaled: torch.Tensor,  # Shape: (num_tokens, embed_dim) - per-token directions
    epsilons: np.ndarray,
    adv_position: str = "end",
    question_lengths: Optional[List[int]] = None,
    prm_type: str = "skywork",
    all_step_positions: Optional[List[torch.Tensor]] = None,
) -> np.ndarray:
    """
    Compute reward grid for batched trajectories with multiple token embeddings.
    
    Each token has its own pair of perturbation directions, but the same epsilon
    values are used across all tokens.
    
    Args:
        v_centers: (num_tokens, embed_dim) - the embeddings of all adversarial tokens
        d1_scaled: (num_tokens, embed_dim) - first scaled direction for each token
        d2_scaled: (num_tokens, embed_dim) - second scaled direction for each token
        epsilons: Array of epsilon values to sweep
        adv_position: "end" (after solution) or "middle" (after question, before solution)
        question_lengths: List of question lengths for each trajectory (for "middle" position)
        prm_type: "skywork" or "qwen"
        all_step_positions: List of step positions for each trajectory (for Qwen PRM)
    """
    n_points = len(epsilons)
    rewards_grid = np.zeros((n_points, n_points))
    if question_lengths is None:
        question_lengths = [0] * len(orig_embeddings_list)
    add_step_after_adv = (prm_type == "qwen" and adv_position == "middle")
    num_adv_tokens = v_centers.shape[0]
    
    for i, eps1 in enumerate(tqdm(epsilons)):
        for j, eps2 in enumerate(epsilons):
            # Per-token perturbations: each token gets its own direction pair
            # but same epsilon scalars
            perturbations = eps1 * d1_scaled + eps2 * d2_scaled  # (num_tokens, embed_dim)
            v_perturbed = v_centers + perturbations  # (num_tokens, embed_dim)
            
            total = 0.0
            count = 0
            for idx, orig_embeddings in enumerate(orig_embeddings_list):
                q_len = question_lengths[idx]
                full_embeddings = construct_embeddings_with_adv(
                    orig_embeddings, v_perturbed, step_embeddings, adv_position, q_len,
                    prm_type=prm_type, add_step_after_adv=add_step_after_adv
                ).unsqueeze(0)
                with torch.no_grad():
                    if prm_type == "skywork":
                        last_hidden_state = get_last_hidden_state(model, full_embeddings)
                        reward_logits = model.v_head(last_hidden_state).squeeze(-1)
                        reward = torch.sigmoid(reward_logits[0, -1]).item()
                    elif prm_type == "qwen":
                        orig_step_pos = all_step_positions[idx]
                        if adv_position == "end":
                            step_positions = torch.cat([
                                orig_step_pos,
                                torch.tensor([full_embeddings.shape[1] - 1], device=full_embeddings.device)
                            ])
                        else:  # middle
                            adv_step_sep_position = q_len + num_adv_tokens
                            shifted_positions = orig_step_pos + num_adv_tokens + 1
                            step_positions = torch.cat([
                                torch.tensor([adv_step_sep_position], device=full_embeddings.device),
                                shifted_positions,
                                torch.tensor([full_embeddings.shape[1] - 1], device=full_embeddings.device)
                            ])
                        _, step_rewards = compute_reward_proxy_qwen(model, full_embeddings, step_positions)
                        reward = step_rewards.min().item()
                total += reward
                count += 1
            rewards_grid[j, i] = total / max(count, 1)
    return rewards_grid


def sample_random_token_ids_multi(
    vocab_size: int,
    num_tokens: int,
    exclude_ids: List[int],
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Sample multiple random token IDs, excluding specified IDs.
    
    Args:
        vocab_size: Size of vocabulary
        num_tokens: Number of tokens to sample
        exclude_ids: Token IDs to exclude from sampling
        rng: Random number generator
        
    Returns:
        List of sampled token IDs
    """
    if rng is None:
        rng = np.random.default_rng()
    
    available_ids = [i for i in range(vocab_size) if i not in exclude_ids]
    if len(available_ids) < num_tokens:
        raise ValueError(f"Not enough available token IDs. Need {num_tokens}, have {len(available_ids)}")
    
    return list(rng.choice(available_ids, size=num_tokens, replace=False))


def get_landscape_cache_path(
    cache_dir: str,
    experiment: str,  # "single" or "batched"
    prm_model: str,
    mode: str,  # "continuous" or "discrete"
    num_adv_tokens: int,
    adv_position: str,  # "end" or "middle"
    landscape_type: str,  # "adversarial" or "random"
) -> str:
    """
    Generate a cache file path for reward landscape grid data.
    
    Naming convention: {experiment}_{prm_model}_{mode}_{num_tokens}tok_{position}_{type}_grid.npz
    """
    prm_prefix = "qwen_" if get_prm_type(prm_model) == "qwen" else ""
    filename = f"{prm_prefix}{experiment}_{prm_model}_{mode}_{num_adv_tokens}tok_{adv_position}_{landscape_type}_grid.npz"
    
    # Create landscape_cache subdirectory
    landscape_cache_dir = os.path.join(cache_dir, "landscape_cache")
    os.makedirs(landscape_cache_dir, exist_ok=True)
    
    return os.path.join(landscape_cache_dir, filename)


def save_landscape_cache(
    cache_path: str,
    rewards_grid: np.ndarray,
    epsilons: np.ndarray,
    curvature_info: Dict[str, float],
    volume_info: Dict[str, float],
) -> None:
    """
    Save reward landscape grid and metadata to cache file.
    """
    # Handle peak_location which is a tuple
    peak_location = volume_info.get("peak_location", (0.0, 0.0))
    if isinstance(peak_location, tuple):
        peak_location_arr = np.array(peak_location)
    else:
        peak_location_arr = np.array([0.0, 0.0])
    
    np.savez(
        cache_path,
        rewards_grid=rewards_grid,
        epsilons=epsilons,
        curvature_metric=curvature_info.get("curvature_metric", 0.0),
        top_eigenvalue=curvature_info.get("top_eigenvalue", 0.0),
        volume=volume_info.get("volume", 0.0),
        volume_normalized=volume_info.get("volume_normalized", 0.0),
        mean_reward=volume_info.get("mean_reward", 0.0),
        peak_reward=volume_info.get("peak_reward", 0.0),
        peak_location=peak_location_arr,
        center_reward=volume_info.get("center_reward", 0.0),
        grid_area=volume_info.get("grid_area", 0.0),
    )
    print(f"Cached landscape grid to {cache_path}")


def load_landscape_cache(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]]:
    """
    Load reward landscape grid and metadata from cache file.
    
    Returns:
        Tuple of (rewards_grid, epsilons, curvature_info, volume_info) if cache exists, None otherwise.
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        data = np.load(cache_path)
        rewards_grid = data["rewards_grid"]
        epsilons = data["epsilons"]
        curvature_info = {
            "curvature_metric": float(data["curvature_metric"]),
            "top_eigenvalue": float(data["top_eigenvalue"]),
        }
        # Handle peak_location which is stored as an array
        peak_location_arr = data.get("peak_location", np.array([0.0, 0.0]))
        if hasattr(peak_location_arr, '__iter__') and len(peak_location_arr) >= 2:
            peak_location = (float(peak_location_arr[0]), float(peak_location_arr[1]))
        else:
            peak_location = (0.0, 0.0)
        
        volume_info = {
            "volume": float(data["volume"]),
            "volume_normalized": float(data["volume_normalized"]),
            "mean_reward": float(data["mean_reward"]),
            "peak_reward": float(data["peak_reward"]),
            "peak_location": peak_location,
            "center_reward": float(data["center_reward"]),
            "grid_area": float(data["grid_area"]),
        }
        print(f"Loaded cached landscape grid from {cache_path}")
        return rewards_grid, epsilons, curvature_info, volume_info
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}")
        return None


def plot_reward_landscape_from_grid(
    rewards_grid: np.ndarray,
    epsilons: np.ndarray,
    title: str,
    output_path: str,
    curvature_info: Dict[str, float],
    volume_info: Dict[str, float],
    zlim: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Plot 3D reward landscape.
    
    Args:
        rewards_grid: 2D array of reward values
        epsilons: Array of epsilon values for axes
        title: Plot title (currently unused, kept for compatibility)
        output_path: Path to save plot
        curvature_info: Dictionary with curvature metrics
        volume_info: Dictionary with volume metrics
        zlim: Optional (min, max) tuple for z-axis limits. If None, auto-determined.
        
    Returns:
        Tuple of (zmin, zmax) used for the plot (useful for matching across plots)
    """
    eps1_grid, eps2_grid = np.meshgrid(epsilons, epsilons)
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        eps1_grid, eps2_grid, rewards_grid, cmap="viridis", edgecolor="none", alpha=0.8
    )

    # Axis labels with larger font
    ax.set_xlabel(r"$\epsilon_1$ (Direction 1)", fontsize=24, labelpad=18)
    ax.set_ylabel(r"$\epsilon_2$ (Direction 2)", fontsize=24, labelpad=18)
    ax.set_zlabel("Reward", fontsize=24, labelpad=18)
    
    # Larger tick labels
    ax.tick_params(axis='x', labelsize=18, pad=8)
    ax.tick_params(axis='y', labelsize=18, pad=8)
    ax.tick_params(axis='z', labelsize=18, pad=8)

    # Apply z-axis limits if specified
    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])
        zmin, zmax = zlim
    else:
        zmin, zmax = rewards_grid.min(), rewards_grid.max()

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Reward", fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Use close instead of show for non-interactive use
    
    return (zmin, zmax)


def load_or_generate_data(require_cached: bool = False) -> None:
    """Load datasets and trajectories, optionally requiring cached trajectories."""
    global train_data, eval_data, train_trajectories_single, train_trajectories_multiple
    global eval_trajectories_single

    train_data = load_aime_dataset(TRAIN_DATASET, split="train")
    eval_data = load_aime_dataset(EVAL_DATASET, split="test")

    train_single_cache = f"{CACHE_DIR}/aime2024_trajectories_single.pkl"
    train_multiple_cache = f"{CACHE_DIR}/aime2024_q{SINGLE_TRAJ_QUESTION_IDX}_100traj.pkl"
    eval_single_cache = f"{CACHE_DIR}/aime2025_trajectories_single.pkl"

    if require_cached:
        missing = [
            path
            for path in [train_single_cache, train_multiple_cache, eval_single_cache]
            if not os.path.exists(path)
        ]
        if missing:
            raise RuntimeError(
                "Missing trajectory caches for distributed training. "
                "Run once with USE_DISTRIBUTED=False to generate:\n"
                + "\n".join(missing)
            )

    if os.path.exists(train_single_cache):
        print(f"Loading cached trajectories from {train_single_cache}")
        with open(train_single_cache, "rb") as f:
            train_trajectories_single = pickle.load(f)
        print(f"Loaded {len(train_trajectories_single)} trajectory sets")
    else:
        train_trajectories_single = generate_trajectories(
            train_data["questions"],
            n_trajectories_per_question=1,
            cache_file=train_single_cache,
        )

    if os.path.exists(train_multiple_cache):
        print(f"Loading cached trajectories from {train_multiple_cache}")
        with open(train_multiple_cache, "rb") as f:
            train_trajectories_multiple = pickle.load(f)
        print(
            f"Loaded {len(train_trajectories_multiple)} trajectory sets with "
            f"{len(train_trajectories_multiple[0])} trajectories each"
        )
    else:
        train_trajectories_multiple = generate_trajectories(
            [train_data["questions"][SINGLE_TRAJ_QUESTION_IDX]],
            n_trajectories_per_question=100,
            cache_file=train_multiple_cache,
        )

    if os.path.exists(eval_single_cache):
        print(f"Loading cached trajectories from {eval_single_cache}")
        with open(eval_single_cache, "rb") as f:
            eval_trajectories_single = pickle.load(f)
        print(f"Loaded {len(eval_trajectories_single)} trajectory sets")
    else:
        eval_trajectories_single = generate_trajectories(
            eval_data["questions"],
            n_trajectories_per_question=1,
            cache_file=eval_single_cache,
        )


def _distributed_worker(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    _init_distributed(rank, world_size)
    load_or_generate_data(require_cached=True)

    distributed_ctx = {"rank": rank, "world_size": world_size}

    if RUN_SINGLE_TRAJ:
        if USE_DISTRIBUTED_SINGLE:
            for prm_key in SINGLE_TRAJ_PRM_KEYS:
                run_single_trajectory_experiment(
                    prm_key,
                    distributed_ctx=distributed_ctx,
                    force_rerun=False,
                    device_override=f"cuda:{rank}",
                )
        elif rank == 0:
            for prm_key in SINGLE_TRAJ_PRM_KEYS:
                run_single_trajectory_experiment(
                    prm_key,
                    distributed_ctx=None,
                    force_rerun=False,
                    device_override=f"cuda:{rank}",
                )

    if RUN_BATCHED:
        if USE_DISTRIBUTED_BATCHED:
            for prm_key in BATCHED_PRM_KEYS:
                run_batched_trajectory_experiment(
                    prm_key,
                    distributed_ctx=distributed_ctx,
                    force_rerun=False,
                    load_model_for_transfer=False,
                    device_override=f"cuda:{rank}",
                )
        elif rank == 0:
            for prm_key in BATCHED_PRM_KEYS:
                run_batched_trajectory_experiment(
                    prm_key,
                    distributed_ctx=None,
                    force_rerun=False,
                    load_model_for_transfer=False,
                    device_override=f"cuda:{rank}",
                )

    dist.barrier()
    _cleanup_distributed()


def main():
    use_distributed = USE_DISTRIBUTED and (
        USE_DISTRIBUTED_SINGLE or USE_DISTRIBUTED_BATCHED
    )
    if USE_DISTRIBUTED and "RANK" not in os.environ:
        available = DIST_WORLD_SIZE or torch.cuda.device_count()
        if available <= 1:
            print("Distributed enabled but only one GPU available. Running locally.")
            use_distributed = False

    load_or_generate_data(require_cached=use_distributed)

    if use_distributed:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            _distributed_worker(rank, world_size)
            if rank != 0:
                return
        else:
            world_size = DIST_WORLD_SIZE or torch.cuda.device_count()
            mp.spawn(_distributed_worker, nprocs=world_size, args=(world_size,))

        # After distributed training, load cached results
        single_traj_results = {}
        for prm_key in SINGLE_TRAJ_PRM_KEYS:
            cached = _load_result_cache(_result_cache_path("single_traj", prm_key))
            if cached is None and os.path.exists(_checkpoint_path("single_traj", prm_key)):
                discrete_ids_path = _discrete_token_ids_path("single_traj", prm_key)
                cached = {
                    "best_token_coeffs": torch.load(
                        _checkpoint_path("single_traj", prm_key), map_location="cpu"
                    ),
                    "best_discrete_token_ids": torch.load(discrete_ids_path, map_location="cpu") if os.path.exists(discrete_ids_path) else None,
                    "reward_history": [],
                    "discrete_reward_history": [],
                    "entropy_history": [],
                    "best_reward": None,
                    "best_discrete_reward": None,
                    "iterations_to_target": None,
                    "initial_avg_reward": None,
                    "prm_key": prm_key,
                    "continuous_optimization": CONTINUOUS_OPTIMIZATION,
                    "cached_only": True,
                }
            if cached is not None:
                single_traj_results[prm_key] = cached

        batched_results = {}
        for prm_key in BATCHED_PRM_KEYS:
            cached = _load_result_cache(_result_cache_path("batched", prm_key))
            if cached is None and os.path.exists(_checkpoint_path("batched", prm_key)):
                discrete_ids_path = _discrete_token_ids_path("batched", prm_key)
                cached = {
                    "best_token_coeffs": torch.load(
                        _checkpoint_path("batched", prm_key), map_location="cpu"
                    ),
                    "best_discrete_token_ids": torch.load(discrete_ids_path, map_location="cpu") if os.path.exists(discrete_ids_path) else None,
                    "avg_reward_history": [],
                    "avg_discrete_reward_history": [],
                    "per_traj_reward_history": [],
                    "entropy_history": [],
                    "best_avg_reward": None,
                    "best_avg_discrete_reward": None,
                    "iterations_to_target": None,
                    "initial_avg_reward": None,
                    "prm_key": prm_key,
                    "continuous_optimization": CONTINUOUS_OPTIMIZATION,
                    "cached_only": True,
                }
            if cached is not None:
                batched_results[prm_key] = cached
        # No model refs in this path; will load on demand
    else:
        single_traj_results: Dict[str, Dict] = {}
        if RUN_SINGLE_TRAJ:
            for prm_key in SINGLE_TRAJ_PRM_KEYS:
                single_traj_results[prm_key] = run_single_trajectory_experiment(prm_key)
        else:
            print("Skipping single trajectory experiments.")

        batched_results: Dict[str, Dict] = {}
        if RUN_BATCHED:
            for prm_key in BATCHED_PRM_KEYS:
                (
                    batched_results[prm_key],
                    model,
                    tokenizer,
                    model_path,
                    device,
                ) = run_batched_trajectory_experiment(prm_key)
                batched_results[prm_key]["model_refs"] = (
                    model,
                    tokenizer,
                    model_path,
                    device,
                )
        else:
            print("Skipping batched trajectory experiments.")

    transfer_results: Dict[str, Dict] = {}
    if RUN_TRANSFER:
        if not batched_results:
            print("Skipping transfer evaluation: no batched results.")
        else:
            eval_problems = eval_data["questions"]
            eval_steps = [parse_steps_from_generation(t[0]) for t in eval_trajectories_single]

            for prm_key in BATCHED_PRM_KEYS:
                if prm_key not in batched_results:
                    continue
                mode_str = "discrete" if not CONTINUOUS_OPTIMIZATION else "continuous"
                print(f"\n{'='*60}")
                print(f"Transfer Evaluation - {prm_key} ({mode_str})")
                print("Train: AIME 2024 -> Test: AIME 2025")
                print(f"{'='*60}")

                loaded_for_transfer = False
                if "model_refs" in batched_results[prm_key]:
                    model, tokenizer, model_path, device = batched_results[prm_key]["model_refs"]
                else:
                    model, tokenizer, model_path, device = load_prm_model(prm_key)
                    loaded_for_transfer = True
                
                adv_token_coeffs = batched_results[prm_key].get("best_token_coeffs")
                adv_token_ids = batched_results[prm_key].get("best_discrete_token_ids")
                
                # For discrete optimization, use discrete token IDs for transfer evaluation
                use_discrete = not CONTINUOUS_OPTIMIZATION

                transfer_results[prm_key] = evaluate_transfer(
                    model,
                    tokenizer,
                    model_path,
                    device,
                    eval_problems,
                    eval_steps,
                    adversarial_token_coeffs=adv_token_coeffs,
                    adversarial_token_ids=adv_token_ids,
                    use_discrete=use_discrete,
                    adv_position=result.get("adv_position", "end"),
                    prm_type=get_prm_type(prm_key),
                )

                print("\nResults:")
                print(
                    f"  Avg reward before adversarial token: "
                    f"{transfer_results[prm_key]['avg_before']:.4f}"
                )
                print(
                    f"  Avg reward after adversarial token: "
                    f"{transfer_results[prm_key]['avg_after']:.4f}"
                )
                print(f"  Improvement: {transfer_results[prm_key]['improvement']:.4f}")
                print(f"  Used discrete tokens: {transfer_results[prm_key]['used_discrete_tokens']}")
                if loaded_for_transfer:
                    del model
                    torch.cuda.empty_cache()

    if RUN_ANALYSIS:
        single_for_analysis = {
            k: v for k, v in single_traj_results.items() if v.get("reward_history")
        }
        batched_for_analysis = {
            k: v for k, v in batched_results.items() if v.get("avg_reward_history")
        }
        if single_for_analysis:
            _single_analysis, _single_targets = print_iterations_analysis(
                single_for_analysis, "Single Trajectory Setup"
            )
        else:
            print("Skipping single-trajectory analysis: no results.")
        if batched_for_analysis:
            _batched_analysis, _batched_targets = print_iterations_analysis(
                batched_for_analysis, "Batched Trajectory Setup"
            )
        else:
            print("Skipping batched analysis: no results.")

    if RUN_PLOTS:
        plots = []
        if single_traj_results:
            plots.append("single")
        if batched_results:
            plots.append("batched")

        if plots:
            ncols = len(plots)
            fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
            if ncols == 1:
                axes = [axes]

            ax_idx = 0
            if single_traj_results:
                ax = axes[ax_idx]
                for prm_key, result in single_traj_results.items():
                    reward_history = result.get("reward_history") or []
                    if not reward_history:
                        continue
                    ax.plot(reward_history, label=f"{prm_key}", linewidth=2)
                ax.set_xlabel("Iteration", fontsize=12)
                ax.set_ylabel("Reward", fontsize=12)
                ax.set_title(
                    "Single Trajectory: Reward vs Iterations",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.legend()
                ax.grid(alpha=0.3)
                ax_idx += 1

            if batched_results:
                ax = axes[ax_idx]
                for prm_key, result in batched_results.items():
                    reward_history = result.get("avg_reward_history") or []
                    if not reward_history:
                        continue
                    ax.plot(reward_history, label=f"{prm_key}", linewidth=2)
                ax.set_xlabel("Iteration", fontsize=12)
                ax.set_ylabel("Average Reward", fontsize=12)
                ax.set_title(
                    "Batched Trajectories: Avg Reward vs Iterations",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.legend()
                ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{CACHE_DIR}/reward_comparison.png", dpi=150)
            plt.show()
        else:
            print("Skipping reward curve plots: no results.")

    if RUN_3D_LANDSCAPE:
        question_idx = SINGLE_TRAJ_QUESTION_IDX
        question = train_data["questions"][question_idx]
        trajectories = train_trajectories_multiple[0]
        parsed_generations = [parse_steps_from_generation(t) for t in trajectories]

        for prm_key in BATCHED_PRM_KEYS:
            print(f"\n{'='*60}")
            print(f"Reward Landscape (Single + Batched) - {prm_key}")
            print(f"{'='*60}")

            model, tokenizer, model_path, device = load_prm_model(prm_key)
            embed_layer = get_embedding_layer(model)
            vocab_size = embed_layer.weight.shape[0]

            # ---- Single trajectory selection ----
            all_rewards = []
            for steps in parsed_generations:
                try:
                    rewards = calculate_stepwise_rewards(
                        model, tokenizer, question, steps, model_path, device
                    )
                    all_rewards.append(np.mean(rewards) if rewards else 0)
                except Exception:
                    all_rewards.append(0)

            sorted_indices = np.argsort(all_rewards)
            selected_idx = sorted_indices[min(30, len(sorted_indices) - 1)]
            selected_steps = parsed_generations[selected_idx]

            input_ids, _ = prepare_input(
                model_path, problem=question, steps=selected_steps, tokenizer=tokenizer, device=device
            )
            orig_embeddings_single = embed_layer(input_ids).detach()
            step_embeddings_single = get_step_token_embedding(embed_layer, tokenizer, device).detach()

            # ---- Batched embeddings ----
            batched_steps = [parse_steps_from_generation(t[0]) for t in train_trajectories_single]
            orig_embeddings_list = []
            for problem, steps in zip(train_data["questions"], batched_steps):
                if not steps:
                    continue
                input_ids, _ = prepare_input(
                    model_path, problem=problem, steps=steps, tokenizer=tokenizer, device=device
                )
                orig_embeddings_list.append(embed_layer(input_ids).detach())
            step_embeddings_batched = step_embeddings_single

            # ---- Shared directions for single setting ----
            d1_unit_single, d2_unit_single, ortho_single = build_unit_directions(
                embed_layer,
                vocab_size,
                device,
                _stable_seed(f"{prm_key}-single-directions"),
            )
            print(f"Single directions orthogonality: {ortho_single:.6f}")

            # ---- Adversarial token (single) ----
            single_ckpt = _checkpoint_path("single_traj", prm_key)
            if os.path.exists(single_ckpt):
                adv_coeffs = torch.load(single_ckpt, map_location=device).to(embed_layer.weight.dtype)
                v_adv = torch.matmul(adv_coeffs.to(device), embed_layer.weight).squeeze(0)
                v_norm = torch.norm(v_adv)
                d1_scaled = d1_unit_single * v_norm
                d2_scaled = d2_unit_single * v_norm
                epsilons = np.linspace(-1.0, 1.0, 25)
                rewards_grid = compute_reward_grid_single(
                    model,
                    orig_embeddings_single,
                    step_embeddings_single,
                    v_adv,
                    d1_scaled,
                    d2_scaled,
                    epsilons,
                )
                curvature = compute_curvature_metric(rewards_grid, epsilons)
                print(
                    f"Single adv curvature: {curvature['curvature_metric']:.4e} "
                    f"(dxx={curvature['dxx']:.4e}, dyy={curvature['dyy']:.4e}, "
                    f"dxy={curvature['dxy']:.4e}, "
                    f"top_eig={curvature['top_eigenvalue']:.4e}, max|eig|={curvature['max_eig_abs']:.4e})"
                )
                out_dir = os.path.join(CACHE_DIR, "landscape_3d", "single", prm_key)
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, f"single_{prm_key}_adversarial.png")
                title = f"Single Trajectory - {prm_key} - Adversarial Token"
                volume = compute_landscape_volume(rewards_grid, epsilons)
                plot_reward_landscape_from_grid(
                    rewards_grid, epsilons, title, output_path, curvature, volume
                )
            else:
                print(f"Single adv checkpoint not found: {single_ckpt}")

            # ---- Random token (single) ----
            rng = np.random.default_rng(_stable_seed(f"{prm_key}-single"))
            exclude_ids = []
            if os.path.exists(single_ckpt):
                adv_coeffs = torch.load(single_ckpt, map_location="cpu")
                exclude_ids = [int(torch.argmax(adv_coeffs).item())]
            random_token_id = sample_random_token_id(vocab_size, exclude_ids, rng=rng)
            v_random = embed_layer(
                torch.tensor([random_token_id], device=device)
            ).squeeze(0)
            v_norm = torch.norm(v_random)
            d1_scaled = d1_unit_single * v_norm
            d2_scaled = d2_unit_single * v_norm
            epsilons = np.linspace(-1.0, 1.0, 25)
            rewards_grid = compute_reward_grid_single(
                model,
                orig_embeddings_single,
                step_embeddings_single,
                v_random,
                d1_scaled,
                d2_scaled,
                epsilons,
            )
            curvature = compute_curvature_metric(rewards_grid, epsilons)
            print(
                f"Single random curvature (token {random_token_id}): "
                f"{curvature['curvature_metric']:.4e} "
                f"(dxx={curvature['dxx']:.4e}, dyy={curvature['dyy']:.4e}, "
                f"dxy={curvature['dxy']:.4e}, "
                f"top_eig={curvature['top_eigenvalue']:.4e}, max|eig|={curvature['max_eig_abs']:.4e})"
            )
            out_dir = os.path.join(CACHE_DIR, "landscape_3d", "single", prm_key)
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(
                out_dir, f"single_{prm_key}_random_{random_token_id}.png"
            )
            title = f"Single Trajectory - {prm_key} - Random Token {random_token_id}"
            volume = compute_landscape_volume(rewards_grid, epsilons)
            plot_reward_landscape_from_grid(
                rewards_grid, epsilons, title, output_path, curvature, volume
            )

            # ---- Shared directions for batched setting ----
            d1_unit_batched, d2_unit_batched, ortho_batched = build_unit_directions(
                embed_layer,
                vocab_size,
                device,
                _stable_seed(f"{prm_key}-batched-directions"),
            )
            print(f"Batched directions orthogonality: {ortho_batched:.6f}")

            # ---- Adversarial token (batched) ----
            batched_ckpt = _checkpoint_path("batched", prm_key)
            if os.path.exists(batched_ckpt):
                adv_coeffs = torch.load(batched_ckpt, map_location=device).to(embed_layer.weight.dtype)
                v_adv = torch.matmul(adv_coeffs.to(device), embed_layer.weight).squeeze(0)
                v_norm = torch.norm(v_adv)
                d1_scaled = d1_unit_batched * v_norm
                d2_scaled = d2_unit_batched * v_norm
                epsilons = np.linspace(-1.0, 1.0, 25)
                rewards_grid = compute_reward_grid_batched(
                    model,
                    orig_embeddings_list,
                    step_embeddings_batched,
                    v_adv,
                    d1_scaled,
                    d2_scaled,
                    epsilons,
                )
                curvature = compute_curvature_metric(rewards_grid, epsilons)
                print(
                    f"Batched adv curvature: {curvature['curvature_metric']:.4e} "
                    f"(dxx={curvature['dxx']:.4e}, dyy={curvature['dyy']:.4e}, "
                    f"dxy={curvature['dxy']:.4e}, "
                    f"top_eig={curvature['top_eigenvalue']:.4e}, max|eig|={curvature['max_eig_abs']:.4e})"
                )
                out_dir = os.path.join(CACHE_DIR, "landscape_3d", "batched", prm_key)
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, f"batched_{prm_key}_adversarial.png")
                title = f"Batched Trajectories - {prm_key} - Adversarial Token (Avg Reward)"
                volume = compute_landscape_volume(rewards_grid, epsilons)
                plot_reward_landscape_from_grid(
                    rewards_grid, epsilons, title, output_path, curvature, volume
                )
            else:
                print(f"Batched adv checkpoint not found: {batched_ckpt}")

            # ---- Random token (batched) ----
            rng = np.random.default_rng(_stable_seed(f"{prm_key}-batched"))
            exclude_ids = []
            if os.path.exists(batched_ckpt):
                adv_coeffs = torch.load(batched_ckpt, map_location="cpu")
                exclude_ids = [int(torch.argmax(adv_coeffs).item())]
            random_token_id = sample_random_token_id(vocab_size, exclude_ids, rng=rng)
            v_random = embed_layer(
                torch.tensor([random_token_id], device=device)
            ).squeeze(0)
            v_norm = torch.norm(v_random)
            d1_scaled = d1_unit_batched * v_norm
            d2_scaled = d2_unit_batched * v_norm
            epsilons = np.linspace(-1.0, 1.0, 25)
            rewards_grid = compute_reward_grid_batched(
                model,
                orig_embeddings_list,
                step_embeddings_batched,
                v_random,
                d1_scaled,
                d2_scaled,
                epsilons,
            )
            curvature = compute_curvature_metric(rewards_grid, epsilons)
            print(
                f"Batched random curvature (token {random_token_id}): "
                f"{curvature['curvature_metric']:.4e} "
                f"(dxx={curvature['dxx']:.4e}, dyy={curvature['dyy']:.4e}, "
                f"dxy={curvature['dxy']:.4e}, "
                f"top_eig={curvature['top_eigenvalue']:.4e}, max|eig|={curvature['max_eig_abs']:.4e})"
            )
            out_dir = os.path.join(CACHE_DIR, "landscape_3d", "batched", prm_key)
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(
                out_dir, f"batched_{prm_key}_random_{random_token_id}.png"
            )
            title = f"Batched Trajectories - {prm_key} - Random Token {random_token_id} (Avg Reward)"
            volume = compute_landscape_volume(rewards_grid, epsilons)
            plot_reward_landscape_from_grid(
                rewards_grid, epsilons, title, output_path, curvature, volume
            )

            del model
            torch.cuda.empty_cache()

    if RUN_2D_PERTURBATION:
        question_idx = SINGLE_TRAJ_QUESTION_IDX
        question = train_data["questions"][question_idx]
        trajectories = train_trajectories_multiple[0]
        parsed_generations = [parse_steps_from_generation(t) for t in trajectories]

        for prm_key in BATCHED_PRM_KEYS:
            model, tokenizer, model_path, device = load_prm_model(prm_key)

            all_rewards = []
            for steps in parsed_generations:
                try:
                    rewards = calculate_stepwise_rewards(
                        model, tokenizer, question, steps, model_path, device
                    )
                    all_rewards.append(np.mean(rewards) if rewards else 0)
                except Exception:
                    all_rewards.append(0)

            sorted_indices = np.argsort(all_rewards)
            selected_idx = sorted_indices[min(30, len(sorted_indices) - 1)]
            selected_steps = parsed_generations[selected_idx]

            checkpoint_path = f"{CACHE_DIR}/single_traj_{prm_key}_best_token.pt"
            if os.path.exists(checkpoint_path):
                adv_token = torch.load(checkpoint_path)
                plot_2d_perturbation_analysis(
                    model,
                    tokenizer,
                    model_path,
                    device,
                    question,
                    selected_steps,
                    adv_token,
                    n_points=50,
                    prm_name=f"Skywork {prm_key}",
                )
            else:
                print(f"Checkpoint not found: {checkpoint_path}")

            del model
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    mode_str = "Continuous" if CONTINUOUS_OPTIMIZATION else "Discrete"
    print(f"Optimization Mode: {mode_str}")
    print("=" * 80)

    if single_traj_results:
        print("\n--- Single Trajectory Setup ---")
        for prm_key, result in single_traj_results.items():
            print(f"\n{prm_key}:")
            init_reward = result.get("initial_avg_reward")
            best_soft_reward = result.get("best_reward")
            best_discrete_reward = result.get("best_discrete_reward")
            print(f"  Initial reward: {_format_reward(init_reward)}")
            print(f"  Best soft reward (Gumbel): {_format_reward(best_soft_reward)}")
            print(f"  Best discrete reward (Hard): {_format_reward(best_discrete_reward)}")
            # For discrete optimization, show improvement based on discrete reward
            if not CONTINUOUS_OPTIMIZATION and best_discrete_reward is not None and init_reward is not None:
                print(f"  Discrete improvement: {best_discrete_reward - init_reward:.4f}")
            elif best_soft_reward is not None and init_reward is not None:
                print(f"  Soft improvement: {best_soft_reward - init_reward:.4f}")
            else:
                print("  Improvement: N/A")

    if batched_results:
        print("\n--- Batched Trajectory Setup ---")
        for prm_key, result in batched_results.items():
            print(f"\n{prm_key}:")
            init_avg = result.get("initial_avg_reward")
            best_soft_avg = result.get("best_avg_reward")
            best_discrete_avg = result.get("best_avg_discrete_reward")
            print(f"  Initial avg reward: {_format_reward(init_avg)}")
            print(f"  Best avg soft reward (Gumbel): {_format_reward(best_soft_avg)}")
            print(f"  Best avg discrete reward (Hard): {_format_reward(best_discrete_avg)}")
            # For discrete optimization, show improvement based on discrete reward
            if not CONTINUOUS_OPTIMIZATION and best_discrete_avg is not None and init_avg is not None:
                print(f"  Discrete improvement: {best_discrete_avg - init_avg:.4f}")
            elif best_soft_avg is not None and init_avg is not None:
                print(f"  Soft improvement: {best_soft_avg - init_avg:.4f}")
            else:
                print("  Improvement: N/A")

    if transfer_results:
        print("\n--- Cross-Dataset Transfer (AIME 2024 -> AIME 2025) ---")
        for prm_key, result in transfer_results.items():
            print(f"\n{prm_key}:")
            print(f"  Avg reward before: {result['avg_before']:.4f}")
            print(f"  Avg reward after: {result['avg_after']:.4f}")
            print(f"  Transfer improvement: {result['improvement']:.4f}")
            print(f"  Used discrete tokens: {result.get('used_discrete_tokens', False)}")

    print("\n" + "=" * 80)
    print("Experiments complete!")
    print(f"Results saved in: {CACHE_DIR}")
    print("=" * 80)

    all_results = {
        "single_trajectory": {
            k: {kk: vv for kk, vv in v.items() if kk != "model_refs"}
            for k, v in single_traj_results.items()
        },
        "batched": {
            k: {kk: vv for kk, vv in v.items() if kk != "model_refs"}
            for k, v in batched_results.items()
        },
        "transfer": transfer_results,
        "config": {
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "num_iterations": NUM_ITERATIONS,
            "entropy_schedule": ENTROPY_SCHEDULE,
            "continuous_optimization": CONTINUOUS_OPTIMIZATION,
            "entropy_weight_start": ENTROPY_WEIGHT_START,
            "entropy_weight_end": ENTROPY_WEIGHT_END,
            "train_dataset": TRAIN_DATASET,
            "eval_dataset": EVAL_DATASET,
        },
    }

    with open(f"{CACHE_DIR}/all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"All results saved to {CACHE_DIR}/all_results.pkl")


def _distributed_optimization_worker(
    rank: int,
    world_size: int,
    config: ExperimentConfig,
    train_data: Dict,
    train_trajectories_single: List,
    train_trajectories_multiple: Optional[List],
):
    """
    Worker function for distributed optimization.
    Each worker handles a subset of trajectories.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    # Initialize distributed
    torch.cuda.set_device(rank)
    dist.init_process_group(backend=DIST_BACKEND, rank=rank, world_size=world_size)
    
    # Set random seed (different for each rank to avoid correlation, but reproducible)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    
    distributed_ctx = {"rank": rank, "world_size": world_size}
    device_override = f"cuda:{rank}"
    
    is_main = rank == 0
    if is_main:
        print(f"\n[Rank {rank}] Starting distributed optimization with {world_size} GPUs")
    
    # Create a modified config with the device for this rank
    from dataclasses import replace
    local_config = replace(config, device=device_override)
    
    result = None
    if config.experiment == "single":
        if train_trajectories_multiple is None:
            raise ValueError("Need multiple trajectories for single experiment")
        result = run_single_trajectory_experiment_with_config(
            config=local_config,
            train_data=train_data,
            train_trajectories_multiple=train_trajectories_multiple,
            distributed_ctx=distributed_ctx,
        )
    elif config.experiment == "batched":
        result, model, tokenizer, model_path, device = run_batched_trajectory_experiment_with_config(
            config=local_config,
            train_data=train_data,
            train_trajectories_single=train_trajectories_single,
            distributed_ctx=distributed_ctx,
        )
        # Clean up model after optimization
        if model is not None:
            del model
            torch.cuda.empty_cache()
    
    # Wait for all workers to complete
    dist.barrier()
    
    # Clean up distributed
    dist.destroy_process_group()
    
    return result


def main_with_args():
    """Main function that uses command-line arguments."""
    args = parse_args()
    config = ExperimentConfig.from_args(args)
    
    # Set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    print("=" * 80)
    print("PRM ATTACK EXPERIMENT")
    print("=" * 80)
    print(f"Experiment: {config.experiment}")
    print(f"Model: {config.prm_model}")
    print(f"Mode: {'continuous' if config.continuous else 'discrete'}")
    print(f"Num adversarial tokens: {config.num_adv_tokens}")
    print(f"Adv position: {config.adv_position}")
    print(f"Num iterations: {config.num_iterations}")
    print(f"Num train trajectories: {config.num_train_trajectories or 'all'}")
    print(f"Num eval trajectories: {config.num_eval_trajectories or 'all'}")
    print(f"Distributed: {config.distributed}")
    print(f"Cache dir: {config.cache_dir}")
    print("=" * 80)
    
    # Create cache directory
    os.makedirs(config.cache_dir, exist_ok=True)
    
    # Check for distributed training
    use_distributed = config.distributed
    world_size = torch.cuda.device_count()
    if use_distributed:
        if world_size <= 1:
            print("Warning: Distributed training requested but only 1 GPU available. Running on single GPU.")
            use_distributed = False
        else:
            print(f"Using {world_size} GPUs for distributed training")
    
    # Load datasets and trajectories
    print("\nLoading datasets...")
    train_data = load_aime_dataset(TRAIN_DATASET, split="train")
    eval_data = load_aime_dataset(EVAL_DATASET, split="test")
    
    # Load or generate trajectories
    train_single_cache = f"{config.cache_dir}/aime2024_trajectories_single.pkl"
    train_multiple_cache = f"{config.cache_dir}/aime2024_q{config.single_traj_question_idx}_100traj.pkl"
    eval_single_cache = f"{config.cache_dir}/aime2025_trajectories_single.pkl"
    
    if os.path.exists(train_single_cache):
        print(f"Loading cached trajectories from {train_single_cache}")
        with open(train_single_cache, "rb") as f:
            train_trajectories_single = pickle.load(f)
    else:
        train_trajectories_single = generate_trajectories(
            train_data["questions"],
            n_trajectories_per_question=1,
            cache_file=train_single_cache,
        )
    
    train_trajectories_multiple = None
    if config.experiment == "single":
        if os.path.exists(train_multiple_cache):
            print(f"Loading cached trajectories from {train_multiple_cache}")
            with open(train_multiple_cache, "rb") as f:
                train_trajectories_multiple = pickle.load(f)
        else:
            train_trajectories_multiple = generate_trajectories(
                [train_data["questions"][config.single_traj_question_idx]],
                n_trajectories_per_question=100,
                cache_file=train_multiple_cache,
            )
    
    eval_trajectories_single = None
    if config.run_transfer:
        if os.path.exists(eval_single_cache):
            print(f"Loading cached trajectories from {eval_single_cache}")
            with open(eval_single_cache, "rb") as f:
                eval_trajectories_single = pickle.load(f)
        else:
            eval_trajectories_single = generate_trajectories(
                eval_data["questions"],
                n_trajectories_per_question=1,
                cache_file=eval_single_cache,
            )
    
    result = None
    model_refs = None
    
    # Run optimization
    if config.run_optimization:
        if use_distributed:
            # Run distributed optimization using mp.spawn
            print(f"\nStarting distributed optimization with {world_size} GPUs...")
            mp.spawn(
                _distributed_optimization_worker,
                args=(world_size, config, train_data, train_trajectories_single, train_trajectories_multiple),
                nprocs=world_size,
                join=True,
            )
            
            # Load cached result after distributed training
            result_cache = _result_cache_path("", config)
            if os.path.exists(result_cache):
                print(f"\nLoading result from distributed training: {result_cache}")
                with open(result_cache, "rb") as f:
                    result = pickle.load(f)
            else:
                print(f"Warning: No cached result found at {result_cache}")
        else:
            # Run single-GPU optimization
            if config.experiment == "single":
                if train_trajectories_multiple is None:
                    raise ValueError("Need to generate multiple trajectories for single experiment")
                result = run_single_trajectory_experiment_with_config(
                    config=config,
                    train_data=train_data,
                    train_trajectories_multiple=train_trajectories_multiple,
                )
            elif config.experiment == "batched":
                result, model, tokenizer, model_path, device = run_batched_trajectory_experiment_with_config(
                    config=config,
                    train_data=train_data,
                    train_trajectories_single=train_trajectories_single,
                )
                if model is not None:
                    model_refs = (model, tokenizer, model_path, device)
    
    # Run transfer evaluation
    transfer_result = None
    if config.run_transfer and result is not None:
        print(f"\n{'='*60}")
        print("Transfer Evaluation - AIME 2024 -> AIME 2025")
        print(f"{'='*60}")
        
        eval_problems = eval_data["questions"]
        eval_steps = [parse_steps_from_generation(t[0]) for t in eval_trajectories_single]
        
        # Load model if not already loaded
        if model_refs is None:
            model, tokenizer, model_path, device = load_prm_model(
                config.prm_model, config.hf_cache_path, config.device
            )
        else:
            model, tokenizer, model_path, device = model_refs
        
        adv_token_coeffs = result.get("best_token_coeffs")
        adv_token_ids = result.get("best_discrete_token_ids")
        use_discrete = not config.continuous
        
        transfer_result = evaluate_transfer(
            model,
            tokenizer,
            model_path,
            device,
            eval_problems,
            eval_steps,
            adversarial_token_coeffs=adv_token_coeffs,
            adversarial_token_ids=adv_token_ids,
            use_discrete=use_discrete,
            num_eval_trajectories=config.num_eval_trajectories,
            adv_position=config.adv_position,
            prm_type=get_prm_type(config.prm_model),
        )
        
        print("\nTransfer Results:")
        print(f"  Avg reward before: {transfer_result['avg_before']:.4f}")
        print(f"  Avg reward after: {transfer_result['avg_after']:.4f}")
        print(f"  Improvement: {transfer_result['improvement']:.4f}")
        print(f"  Used discrete tokens: {transfer_result['used_discrete_tokens']}")
        
        # Save transfer results
        transfer_result_path = f"{config.cache_dir}/{_get_experiment_prefix(config)}_transfer.pkl"
        with open(transfer_result_path, "wb") as f:
            pickle.dump(transfer_result, f)
        print(f"Saved transfer results to {transfer_result_path}")
    
    # Run plots
    if config.run_plots and result is not None:
        print("\nGenerating plots...")
        
        plot_path = f"{config.cache_dir}/{_get_experiment_prefix(config)}_training_curves.png"
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Soft vs Discrete Reward
        ax = axes[0]
        if config.experiment == "single":
            soft_rewards = result.get("reward_history", [])
            discrete_rewards = result.get("discrete_reward_history", [])
        else:
            soft_rewards = result.get("avg_reward_history", [])
            discrete_rewards = result.get("avg_discrete_reward_history", [])
        
        if soft_rewards:
            ax.plot(soft_rewards, label="Soft (Gumbel)", linewidth=2)
        if discrete_rewards:
            ax.plot(discrete_rewards, label="Discrete (Hard)", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.set_title("Reward vs Iteration")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Entropy
        ax = axes[1]
        entropy_history = result.get("entropy_history", [])
        if entropy_history:
            ax.plot(entropy_history, linewidth=2, color="green")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Entropy")
        ax.set_title("Entropy vs Iteration")
        ax.grid(alpha=0.3)
        
        # Plot 3: Entropy Weight (for discrete optimization)
        ax = axes[2]
        entropy_weight_history = result.get("entropy_weight_history", [])
        if entropy_weight_history:
            ax.plot(entropy_weight_history, linewidth=2, color="purple")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Entropy Weight")
        ax.set_title("Entropy Weight Schedule")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved training curves to {plot_path}")
    
    # Run 3D landscape visualization
    if config.run_3d_landscape and result is not None:
        print(f"\n{'='*60}")
        print("3D Reward Landscape Visualization")
        print(f"{'='*60}")
        
        # Load model if not available
        if model_refs is None:
            model, tokenizer, model_path, device = load_prm_model(
                config.prm_model, config.hf_cache_path, config.device
            )
            loaded_for_landscape = True
        else:
            model, tokenizer, model_path, device = model_refs
            loaded_for_landscape = False
        
        embed_layer = get_embedding_layer(model)
        vocab_size = embed_layer.weight.shape[0]
        
        # Create output directory
        mode_str = "continuous" if config.continuous else "discrete"
        pos_str = config.adv_position  # "end" or "middle"
        prm_type = get_prm_type(config.prm_model)
        prm_prefix = "qwen_" if prm_type == "qwen" else ""
        out_dir = os.path.join(config.cache_dir, "landscape_3d", f"{prm_prefix}{config.experiment}", config.prm_model, mode_str, pos_str)
        os.makedirs(out_dir, exist_ok=True)
        
        # Grid settings
        epsilons = np.linspace(-1.0, 1.0, config.landscape_grid_size)
        
        # Get adversarial token embeddings
        adv_token_coeffs = result.get("best_token_coeffs")
        adv_token_ids = result.get("best_discrete_token_ids")
        num_adv_tokens = config.num_adv_tokens
        
        # Determine which embeddings to use for adversarial tokens
        use_discrete_for_landscape = not config.continuous and adv_token_ids is not None
        
        if use_discrete_for_landscape:
            # Use discrete token embeddings
            token_ids = adv_token_ids.to(device)
            v_adv = embed_layer(token_ids)  # Shape: (num_tokens, embed_dim)
            print(f"Using {num_adv_tokens} discrete adversarial token(s): {token_ids.tolist()}")
            print(f"Decoded: {tokenizer.decode(token_ids.tolist())}")
            exclude_ids = token_ids.cpu().tolist()
        elif adv_token_coeffs is not None:
            # Use continuous token embeddings
            coeffs = adv_token_coeffs.to(device).to(embed_layer.weight.dtype)
            v_adv = torch.matmul(coeffs, embed_layer.weight)  # Shape: (num_tokens, embed_dim)
            print(f"Using {num_adv_tokens} continuous adversarial token(s)")
            # For exclusion, use argmax of coefficients
            exclude_ids = [int(torch.argmax(coeffs[i]).item()) for i in range(coeffs.shape[0])]
        else:
            print("No adversarial tokens found in result. Skipping landscape.")
            v_adv = None
        
        if v_adv is not None:
            # Build unit directions for perturbation
            # For multi-token, use per-token directions
            if num_adv_tokens == 1:
                d1_unit, d2_unit, ortho = build_unit_directions(
                    embed_layer,
                    vocab_size,
                    device,
                    _stable_seed(f"{config.prm_model}-{config.experiment}-landscape-adv"),
                )
                print(f"Direction orthogonality: {ortho:.6f}")
                # Scale directions by norm of adversarial embedding
                v_norm = torch.norm(v_adv)
                d1_scaled = d1_unit * v_norm
                d2_scaled = d2_unit * v_norm
            else:
                d1_units, d2_units, avg_ortho = build_unit_directions_per_token(
                    embed_layer,
                    vocab_size,
                    device,
                    _stable_seed(f"{config.prm_model}-{config.experiment}-landscape-adv"),
                    num_adv_tokens,
                )
                print(f"Per-token direction avg orthogonality: {avg_ortho:.6f}")
                # Scale each token's directions by its own embedding norm
                v_norms = torch.norm(v_adv, dim=-1, keepdim=True)  # (num_tokens, 1)
                d1_scaled = d1_units * v_norms  # (num_tokens, embed_dim)
                d2_scaled = d2_units * v_norms  # (num_tokens, embed_dim)
            
            # Prepare trajectories for computing rewards
            if config.experiment == "single":
                # Single trajectory setup
                question_idx = config.single_traj_question_idx
                question = train_data["questions"][question_idx]
                trajectories = train_trajectories_multiple[0]
                parsed_generations = [parse_steps_from_generation(t) for t in trajectories]
                
                # Select a trajectory (same as optimization)
                all_rewards = []
                for steps in parsed_generations:
                    try:
                        rewards = calculate_stepwise_rewards(
                            model, tokenizer, question, steps, model_path, device
                        )
                        all_rewards.append(np.mean(rewards) if rewards else 0)
                    except Exception:
                        all_rewards.append(0)
                sorted_indices = np.argsort(all_rewards)
                selected_idx = sorted_indices[min(30, len(sorted_indices) - 1)]
                selected_steps = parsed_generations[selected_idx]
                
                input_ids, _ = prepare_input(
                    model_path, problem=question, steps=selected_steps, tokenizer=tokenizer, device=device
                )
                orig_embeddings = embed_layer(input_ids).detach()
                step_embeddings = get_step_token_embedding(embed_layer, tokenizer, device).detach()
                prm_type = get_prm_type(config.prm_model)
                question_length = get_question_length(tokenizer, question, prm_type)
                
                # For Qwen, get step token positions
                orig_step_positions = None
                if prm_type == "qwen":
                    orig_step_positions = find_step_token_positions(input_ids, tokenizer)
                
                # Compute adversarial landscape (with caching)
                print("\nComputing adversarial token landscape (single trajectory)...")
                print(f"Adversarial token position: {config.adv_position}")
                
                adv_cache_path = get_landscape_cache_path(
                    config.cache_dir, "single", config.prm_model, mode_str,
                    num_adv_tokens, config.adv_position, "adversarial"
                )
                cached_adv = load_landscape_cache(adv_cache_path)
                
                if cached_adv is not None:
                    rewards_grid_adv, epsilons, curvature_adv, volume_adv = cached_adv
                else:
                    if num_adv_tokens == 1:
                        rewards_grid_adv = compute_reward_grid_single(
                            model, orig_embeddings, step_embeddings, v_adv.squeeze(0), d1_scaled, d2_scaled, epsilons,
                            adv_position=config.adv_position, question_length=question_length, prm_type=prm_type,
                            orig_step_positions=orig_step_positions
                        )
                    else:
                        rewards_grid_adv = compute_reward_grid_single_multi_token(
                            model, orig_embeddings, step_embeddings, v_adv, d1_scaled, d2_scaled, epsilons,
                            adv_position=config.adv_position, question_length=question_length, prm_type=prm_type,
                            orig_step_positions=orig_step_positions
                        )
                    
                    curvature_adv = compute_curvature_metric(rewards_grid_adv, epsilons)
                    volume_adv = compute_landscape_volume(rewards_grid_adv, epsilons)
                    save_landscape_cache(adv_cache_path, rewards_grid_adv, epsilons, curvature_adv, volume_adv)
                
                title_adv = f"Single Trajectory - {config.prm_model} - Adversarial ({mode_str}, {num_adv_tokens} tokens, {pos_str})"
                adv_plot_path = os.path.join(out_dir, f"single_{config.prm_model}_{mode_str}_{num_adv_tokens}tok_{pos_str}_adversarial.png")
                adv_zlim = plot_reward_landscape_from_grid(rewards_grid_adv, epsilons, title_adv, adv_plot_path, curvature_adv, volume_adv)
                print(f"Saved adversarial landscape to {adv_plot_path}")
                print(f"Adversarial - Volume: {volume_adv['volume']:.4f}, Peak: {volume_adv['peak_reward']:.4f}, Mean: {volume_adv['mean_reward']:.4f}")
                print(f"Adversarial - Curvature: {curvature_adv['curvature_metric']:.4e}, Top eigenvalue: {curvature_adv['top_eigenvalue']:.4e}")
                
                # Compute random baseline landscape (with caching)
                print("\nComputing random token landscape (single trajectory)...")
                
                random_cache_path = get_landscape_cache_path(
                    config.cache_dir, "single", config.prm_model, mode_str,
                    num_adv_tokens, config.adv_position, "random"
                )
                cached_random = load_landscape_cache(random_cache_path)
                
                if cached_random is not None:
                    rewards_grid_random, _, curvature_random, volume_random = cached_random
                else:
                    rng = np.random.default_rng(_stable_seed(f"{config.prm_model}-single-random"))
                    random_token_ids = sample_random_token_ids_multi(vocab_size, num_adv_tokens, exclude_ids, rng)
                    v_random = embed_layer(torch.tensor(random_token_ids, device=device))
                    
                    # Build separate random directions for the random baseline
                    if num_adv_tokens == 1:
                        d1_unit_rand, d2_unit_rand, _ = build_unit_directions(
                            embed_layer,
                            vocab_size,
                            device,
                            _stable_seed(f"{config.prm_model}-{config.experiment}-landscape-random"),
                        )
                        v_norm_rand = torch.norm(v_random)
                        d1_scaled_rand = d1_unit_rand * v_norm_rand
                        d2_scaled_rand = d2_unit_rand * v_norm_rand
                        rewards_grid_random = compute_reward_grid_single(
                            model, orig_embeddings, step_embeddings, v_random.squeeze(0), d1_scaled_rand, d2_scaled_rand, epsilons,
                            adv_position=config.adv_position, question_length=question_length, prm_type=prm_type,
                            orig_step_positions=orig_step_positions
                        )
                    else:
                        d1_units_rand, d2_units_rand, _ = build_unit_directions_per_token(
                            embed_layer,
                            vocab_size,
                            device,
                            _stable_seed(f"{config.prm_model}-{config.experiment}-landscape-random"),
                            num_adv_tokens,
                        )
                        v_norms_rand = torch.norm(v_random, dim=-1, keepdim=True)
                        d1_scaled_rand = d1_units_rand * v_norms_rand
                        d2_scaled_rand = d2_units_rand * v_norms_rand
                        rewards_grid_random = compute_reward_grid_single_multi_token(
                            model, orig_embeddings, step_embeddings, v_random, d1_scaled_rand, d2_scaled_rand, epsilons,
                            adv_position=config.adv_position, question_length=question_length, prm_type=prm_type,
                            orig_step_positions=orig_step_positions
                        )
                    
                    curvature_random = compute_curvature_metric(rewards_grid_random, epsilons)
                    volume_random = compute_landscape_volume(rewards_grid_random, epsilons)
                    save_landscape_cache(random_cache_path, rewards_grid_random, epsilons, curvature_random, volume_random)
                
                title_random = f"Single Trajectory - {config.prm_model} - Random Tokens ({num_adv_tokens} tokens, {pos_str})"
                random_plot_path = os.path.join(out_dir, f"single_{config.prm_model}_{mode_str}_{num_adv_tokens}tok_{pos_str}_random.png")
                # Use z-limits from 0 to max of adversarial plot
                random_zlim = (0.0, adv_zlim[1])
                plot_reward_landscape_from_grid(rewards_grid_random, epsilons, title_random, random_plot_path, curvature_random, volume_random, zlim=random_zlim)
                print(f"Saved random baseline landscape to {random_plot_path}")
                print(f"Random - Volume: {volume_random['volume']:.4f}, Peak: {volume_random['peak_reward']:.4f}, Mean: {volume_random['mean_reward']:.4f}")
                print(f"Random - Curvature: {curvature_random['curvature_metric']:.4e}, Top eigenvalue: {curvature_random['top_eigenvalue']:.4e}")
                print(f"Z-axis: [0, {adv_zlim[1]:.4f}] (max from adversarial plot)")
                
            else:  # batched
                # Batched trajectories setup
                problems = train_data["questions"]
                batched_steps = [parse_steps_from_generation(t[0]) for t in train_trajectories_single]
                
                # Limit trajectories if specified
                if config.num_train_trajectories is not None:
                    n_traj = min(config.num_train_trajectories, len(problems))
                    problems = problems[:n_traj]
                    batched_steps = batched_steps[:n_traj]
                
                # Build embeddings list, question lengths, and step positions (for Qwen)
                orig_embeddings_list = []
                question_lengths = []
                all_step_positions = []
                for problem, steps in zip(problems, batched_steps):
                    if not steps:
                        continue
                    input_ids, _ = prepare_input(
                        model_path, problem=problem, steps=steps, tokenizer=tokenizer, device=device
                    )
                    orig_embeddings_list.append(embed_layer(input_ids).detach())
                    question_lengths.append(get_question_length(tokenizer, problem, prm_type))
                    if prm_type == "qwen":
                        all_step_positions.append(find_step_token_positions(input_ids, tokenizer))
                step_embeddings = get_step_token_embedding(embed_layer, tokenizer, device).detach()
                
                # For non-Qwen, set to None
                if prm_type != "qwen":
                    all_step_positions = None
                
                # Compute adversarial landscape (with caching)
                print(f"\nComputing adversarial token landscape (batched, {len(orig_embeddings_list)} trajectories)...")
                print(f"Adversarial token position: {config.adv_position}")
                
                adv_cache_path = get_landscape_cache_path(
                    config.cache_dir, "batched", config.prm_model, mode_str,
                    num_adv_tokens, config.adv_position, "adversarial"
                )
                cached_adv = load_landscape_cache(adv_cache_path)
                
                if cached_adv is not None:
                    rewards_grid_adv, epsilons, curvature_adv, volume_adv = cached_adv
                else:
                    if num_adv_tokens == 1:
                        rewards_grid_adv = compute_reward_grid_batched(
                            model, orig_embeddings_list, step_embeddings, v_adv.squeeze(0), d1_scaled, d2_scaled, epsilons,
                            adv_position=config.adv_position, question_lengths=question_lengths, prm_type=prm_type,
                            all_step_positions=all_step_positions
                        )
                    else:
                        rewards_grid_adv = compute_reward_grid_batched_multi_token(
                            model, orig_embeddings_list, step_embeddings, v_adv, d1_scaled, d2_scaled, epsilons,
                            adv_position=config.adv_position, question_lengths=question_lengths, prm_type=prm_type,
                            all_step_positions=all_step_positions
                        )
                    
                    curvature_adv = compute_curvature_metric(rewards_grid_adv, epsilons)
                    volume_adv = compute_landscape_volume(rewards_grid_adv, epsilons)
                    save_landscape_cache(adv_cache_path, rewards_grid_adv, epsilons, curvature_adv, volume_adv)
                
                title_adv = f"Batched - {config.prm_model} - Adversarial ({mode_str}, {num_adv_tokens} tokens, {pos_str})"
                adv_plot_path = os.path.join(out_dir, f"batched_{config.prm_model}_{mode_str}_{num_adv_tokens}tok_{pos_str}_adversarial.png")
                adv_zlim = plot_reward_landscape_from_grid(rewards_grid_adv, epsilons, title_adv, adv_plot_path, curvature_adv, volume_adv)
                print(f"Saved adversarial landscape to {adv_plot_path}")
                print(f"Adversarial - Volume: {volume_adv['volume']:.4f}, Peak: {volume_adv['peak_reward']:.4f}, Mean: {volume_adv['mean_reward']:.4f}")
                print(f"Adversarial - Curvature: {curvature_adv['curvature_metric']:.4e}, Top eigenvalue: {curvature_adv['top_eigenvalue']:.4e}")
                
                # Compute random baseline landscape (with caching)
                print(f"\nComputing random token landscape (batched, {len(orig_embeddings_list)} trajectories)...")
                
                random_cache_path = get_landscape_cache_path(
                    config.cache_dir, "batched", config.prm_model, mode_str,
                    num_adv_tokens, config.adv_position, "random"
                )
                cached_random = load_landscape_cache(random_cache_path)
                
                if cached_random is not None:
                    rewards_grid_random, _, curvature_random, volume_random = cached_random
                else:
                    rng = np.random.default_rng(_stable_seed(f"{config.prm_model}-batched-random"))
                    random_token_ids = sample_random_token_ids_multi(vocab_size, num_adv_tokens, exclude_ids, rng)
                    v_random = embed_layer(torch.tensor(random_token_ids, device=device))
                    
                    # Build separate random directions for the random baseline
                    if num_adv_tokens == 1:
                        d1_unit_rand, d2_unit_rand, _ = build_unit_directions(
                            embed_layer,
                            vocab_size,
                            device,
                            _stable_seed(f"{config.prm_model}-{config.experiment}-landscape-random"),
                        )
                        v_norm_rand = torch.norm(v_random)
                        d1_scaled_rand = d1_unit_rand * v_norm_rand
                        d2_scaled_rand = d2_unit_rand * v_norm_rand
                        rewards_grid_random = compute_reward_grid_batched(
                            model, orig_embeddings_list, step_embeddings, v_random.squeeze(0), d1_scaled_rand, d2_scaled_rand, epsilons,
                            adv_position=config.adv_position, question_lengths=question_lengths, prm_type=prm_type,
                            all_step_positions=all_step_positions
                        )
                    else:
                        d1_units_rand, d2_units_rand, _ = build_unit_directions_per_token(
                            embed_layer,
                            vocab_size,
                            device,
                            _stable_seed(f"{config.prm_model}-{config.experiment}-landscape-random"),
                            num_adv_tokens,
                        )
                        v_norms_rand = torch.norm(v_random, dim=-1, keepdim=True)
                        d1_scaled_rand = d1_units_rand * v_norms_rand
                        d2_scaled_rand = d2_units_rand * v_norms_rand
                        rewards_grid_random = compute_reward_grid_batched_multi_token(
                            model, orig_embeddings_list, step_embeddings, v_random, d1_scaled_rand, d2_scaled_rand, epsilons,
                            adv_position=config.adv_position, question_lengths=question_lengths, prm_type=prm_type,
                            all_step_positions=all_step_positions
                        )
                    
                    curvature_random = compute_curvature_metric(rewards_grid_random, epsilons)
                    volume_random = compute_landscape_volume(rewards_grid_random, epsilons)
                    save_landscape_cache(random_cache_path, rewards_grid_random, epsilons, curvature_random, volume_random)
                
                title_random = f"Batched - {config.prm_model} - Random Tokens ({num_adv_tokens} tokens, {pos_str})"
                random_plot_path = os.path.join(out_dir, f"batched_{config.prm_model}_{mode_str}_{num_adv_tokens}tok_{pos_str}_random.png")
                # Use z-limits from 0 to max of adversarial plot
                random_zlim = (0.0, adv_zlim[1])
                plot_reward_landscape_from_grid(rewards_grid_random, epsilons, title_random, random_plot_path, curvature_random, volume_random, zlim=random_zlim)
                print(f"Saved random baseline landscape to {random_plot_path}")
                print(f"Random - Volume: {volume_random['volume']:.4f}, Peak: {volume_random['peak_reward']:.4f}, Mean: {volume_random['mean_reward']:.4f}")
                print(f"Random - Curvature: {curvature_random['curvature_metric']:.4e}, Top eigenvalue: {curvature_random['top_eigenvalue']:.4e}")
                print(f"Z-axis: [0, {adv_zlim[1]:.4f}] (max from adversarial plot)")
        
        # Clean up if we loaded the model just for landscape
        if loaded_for_landscape:
            del model
            torch.cuda.empty_cache()
            model_refs = None
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Experiment: {config.get_experiment_name()}")
    
    if result is not None:
        if config.experiment == "single":
            print(f"  Initial reward: {_format_reward(result.get('initial_avg_reward'))}")
            print(f"  Best soft reward: {_format_reward(result.get('best_reward'))}")
            print(f"  Best discrete reward: {_format_reward(result.get('best_discrete_reward'))}")
        else:
            print(f"  Initial avg reward: {_format_reward(result.get('initial_avg_reward'))}")
            print(f"  Best avg soft reward: {_format_reward(result.get('best_avg_reward'))}")
            print(f"  Best avg discrete reward: {_format_reward(result.get('best_avg_discrete_reward'))}")
    
    if transfer_result is not None:
        print(f"\nTransfer (AIME 2024 -> AIME 2025):")
        print(f"  Avg before: {transfer_result['avg_before']:.4f}")
        print(f"  Avg after: {transfer_result['avg_after']:.4f}")
        print(f"  Improvement: {transfer_result['improvement']:.4f}")
    
    print("\n" + "=" * 80)
    print("Experiment complete!")
    print(f"Results saved in: {config.cache_dir}")
    print("=" * 80)
    
    # Clean up
    if model_refs is not None:
        del model_refs
        torch.cuda.empty_cache()
    
    return result, transfer_result


if __name__ == "__main__":
    main_with_args()
