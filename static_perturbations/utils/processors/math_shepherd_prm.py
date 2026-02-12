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

def derive_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    """
    This function derives the step rewards for the PRM model.
    It takes the logits, the token masks, and the tokenizer, and returns the step rewards.
    """
    candidate_tokens = tokenizer.encode(f"{CANDIDATE_TOKENS[0]} {CANDIDATE_TOKENS[1]}", add_special_tokens=False)

    probabilities = F.softmax(logits[..., candidate_tokens], dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    batch_size = probabilities.size(0)

    all_scores_res = []
    for i in range(batch_size):
        sample = probabilities[i] # seq_len, 2
        positive_probs = sample[sample != 0].view(-1, 2)[:, 0]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res