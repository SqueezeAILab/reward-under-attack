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
            ## Fixed
            step_ids = tokenizer.encode(step, add_special_tokens=False)
            step_ids += [step_token_id]
            flag = [0] * len(step_ids)
            flag[-1] = 1
            response_ids.extend(step_ids)
            token_masks.extend(flag)
   
    input_ids = prompt_ids + response_ids

    input_ids = torch.from_numpy(np.array([input_ids]))
    token_masks = torch.from_numpy(np.array(token_masks))
    return input_ids, token_masks

def derive_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    """
    This function derives the step rewards for the PRM model.
    It takes the rewards, the token masks, and the tokenizer, and returns the step rewards.
    """
    probabilities = F.sigmoid(logits)
    probabilities = probabilities * token_masks.unsqueeze(0)
    batch_size = probabilities.size(0)

    all_scores_res = []
    for i in range(batch_size):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1)
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

def derive_step_rewards_vllm(logits, token_masks, tokenizer):
    batch_size = len(logits.data)
    res = []
    for idx in range(batch_size):
        token_mask = token_masks[idx].cpu().numpy()
        sample_prob = expit(logits.data[idx].embedding)
        sample_prob = sample_prob * token_mask
        res.append(sample_prob[sample_prob != 0].tolist())
    return res