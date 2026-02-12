import torch
import numpy as np
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from typing import Optional
STEP_SEP_TOKEN = "ки"
CANDIDATE_TOKENS = ["+", "-"]

def prepare_input(problem: str, steps: list[str], tokenizer: PreTrainedTokenizerBase):
    """
    This function prepares the input for the PRM model.
    It takes a problem and a list of steps, and returns the input ids and the token masks.
    """
    ## Generate input ids
    if problem is None:
        problem = ""
    output = f" {STEP_SEP_TOKEN}\n".join(steps) + f" {STEP_SEP_TOKEN}"
    input_ids = tokenizer.encode(f"{problem} {output}", return_tensors="pt")

    ## Calculate token masks
    step_sep_id = tokenizer.encode(STEP_SEP_TOKEN, add_special_tokens=False)[0]
    token_masks = (np.array(input_ids) == step_sep_id)

    token_masks = torch.from_numpy(token_masks)
    return input_ids, token_masks


def derive_last_reward(logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    candidate_tokens = tokenizer.encode(f"{CANDIDATE_TOKENS[0]} {CANDIDATE_TOKENS[1]}", add_special_tokens=False)

    probabilities = F.softmax(logits[..., candidate_tokens], dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    rewards = probabilities[:, 0]

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
