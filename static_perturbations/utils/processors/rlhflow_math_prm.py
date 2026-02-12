import torch
import numpy as np
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from typing import Optional
STEP_SEP_TOKEN = "+"
CANDIDATE_TOKENS = ["+", "-"]

def prepare_input(problem: Optional[str], steps: list[str], tokenizer):
    """
    This function prepares the input for the PRM model.
    It takes a problem and a list of steps, and returns the input ids and the token masks.
    """

    tokenizer.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""
    if problem is None:
        problem = ""
    ## Generate input ids
    messages = [
        {"role": "user", "content": f"{problem} {steps[0]}"},
        {"role": "assistant", "content": STEP_SEP_TOKEN},
    ]
    for step in steps[1:]:
        messages.append({"role": "user", "content": step})
        messages.append({"role": "assistant", "content": STEP_SEP_TOKEN})

    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    input_ids = tokenizer.encode(conversation_str, return_tensors="pt", add_special_tokens=False)
    
    ## Calculate token masks for each response
    token_masks = np.zeros(len(input_ids[0]), dtype=bool)
    
    current_position = 0
    for message in messages:
        tokenized_input = tokenizer.apply_chat_template([message])[1:]
        if message["role"] == "assistant":
            token_masks[current_position+len(tokenized_input)+1 -3] = True
        current_position += len(tokenized_input)

    token_masks = torch.from_numpy(token_masks)
    assert all(input_ids[0][token_masks] == 271), "The +/- is predicted by the position before +"
    return input_ids, token_masks

def derive_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    """
    This function derives the step rewards for the PRM model.
    It takes the rewards, the token masks, and the tokenizer, and returns the step rewards.
    """
    candidate_tokens = [
                        tokenizer.encode(CANDIDATE_TOKENS[0], add_special_tokens=False)[0], 
                        tokenizer.encode(CANDIDATE_TOKENS[1], add_special_tokens=False)[0]
                       ]
    probabilities = F.softmax(logits[..., candidate_tokens], dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    batch_size = probabilities.size(0)

    all_scores_res = []
    for i in range(batch_size):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 0]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res