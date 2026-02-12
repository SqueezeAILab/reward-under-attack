import torch
import numpy as np
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from typing import Optional

STEP_SEP_TOKEN = "<extra_0>"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def prepare_input(problem: Optional[str], steps: list[str], tokenizer: PreTrainedTokenizerBase):
    """
    This function prepares the input for the PRM model.
    It takes a problem and a list of steps, and returns the input ids and the token masks.
    """
    ## Generate input ids
    if problem is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},  
            {"role": "user", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},  
            {"role": "user", "content": problem},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    input_ids = tokenizer.encode(conversation_str, return_tensors="pt")

    ## Calculate token masks
    step_sep_id = tokenizer.encode(STEP_SEP_TOKEN, add_special_tokens=False)[0]
    token_mask = (np.array(input_ids) == step_sep_id)

    token_mask = torch.from_numpy(token_mask[0])
    return input_ids, token_mask


def derive_last_reward(logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    rewards = probabilities[:, 1]

    nonzero = rewards != 0

    # Replace zeros with inf so they don't affect the min
    rewards_for_min = torch.where(nonzero, rewards, torch.full_like(rewards, float('inf')))

    # Get minimum per row
    min_rewards = rewards_for_min.min(dim=1).values

    # Zero out rows that had no non-zero rewards (min would be inf)
    min_rewards = torch.where(
        min_rewards == float('inf'),
        torch.zeros_like(min_rewards),
        min_rewards,
    )

    return min_rewards
