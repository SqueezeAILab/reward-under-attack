import torch
import numpy as np
from scipy.special import expit
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from typing import Optional

STEP_SEP_TOKEN = "\n"

def prepare_input(problem: Optional[str], steps: list[str], tokenizer: PreTrainedTokenizerBase):
    """
    This function prepares the input for the PRM model.
    It takes a problem and a list of steps, and returns the input ids and the reward flags.
    """
    if problem is None:
        problem = ""
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + STEP_SEP_TOKEN)
    response_ids = []
    token_masks = [0] * len(prompt_ids)
    step_token_id = tokenizer.encode(STEP_SEP_TOKEN, add_special_tokens=False)[0]
    for _, step in enumerate(steps):
        if step != "":
            step_ids = tokenizer.encode(step, add_special_tokens=False)
            step_ids += [step_token_id]
            flag = [0] * len(step_ids)
            flag[-1] = 1
            response_ids.extend(step_ids)
            token_masks.extend(flag)

    input_ids = prompt_ids + response_ids
    input_ids = torch.from_numpy(np.array([input_ids]))
    token_masks = torch.from_numpy(np.array([token_masks]))
    return input_ids, token_masks


def derive_last_reward(logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    rewards = torch.sigmoid(logits) * token_masks
    
    B, T = rewards.shape
    device = rewards.device

    nonzero = rewards != 0
    idx = torch.arange(T, device=device).expand(B, T)

    # index of last non-zero per row, -1 if all zero
    last_idx = idx.masked_fill(~nonzero, -1).max(dim=1).values

    # gather safely
    last_idx_safe = last_idx.clamp(min=0)
    last_rewards = rewards.gather(1, last_idx_safe.unsqueeze(1)).squeeze(1)

    # zero out rows that had no non-zero rewards
    last_rewards = torch.where(
        last_idx == -1,
        torch.zeros_like(last_rewards),
        last_rewards,
    )

    return last_rewards


def derive_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    # Per-step reward at each step-separator position, zero elsewhere. Shape (B, T).
    return torch.sigmoid(logits) * token_masks
