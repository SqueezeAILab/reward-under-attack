import torch
from transformers import PreTrainedTokenizerBase
from constants.model_constants import PREPARE_INPUT_MAP, DERIVE_STEP_REWARDS_MAP, DERIVE_STEP_REWARDS_VLLM_MAP

def prepare_input(model_name: str, 
                  problem: str, 
                  steps: list[str], 
                  tokenizer: PreTrainedTokenizerBase,
                  convert_to_list = False,
                  device="cuda"):
    prepare_input_fn = PREPARE_INPUT_MAP[model_name]
    input_ids, token_masks = prepare_input_fn(problem, steps, tokenizer)
    input_ids = input_ids[0]
    if convert_to_list:
        input_ids = input_ids.cpu().tolist()
        token_masks = token_masks.cpu().tolist()
    else:
        input_ids = input_ids.to(device)
        token_masks = token_masks.to(device)
    return input_ids, token_masks

def derive_step_rewards(model_name: str, logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    derive_step_rewards_fn = DERIVE_STEP_REWARDS_MAP[model_name]
    return derive_step_rewards_fn(logits, token_masks, tokenizer)

def derive_step_rewards_vllm(model_name, logits, token_masks, tokenizer):
    derive_step_rewards_fn = DERIVE_STEP_REWARDS_VLLM_MAP[model_name]
    return derive_step_rewards_fn(logits, token_masks, tokenizer)

def prepare_batch_input_for_model(input_ids, token_masks, pad_token_id=0):
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        [ids if isinstance(ids, torch.Tensor) else torch.LongTensor(ids) for ids in input_ids], 
        batch_first=True,
        padding_value=pad_token_id
    )
    padded_token_masks = torch.nn.utils.rnn.pad_sequence(
        [token_mask if isinstance(token_mask, torch.Tensor) else torch.LongTensor(token_mask) for token_mask in token_masks], 
        batch_first=True,
        padding_value=0
    )
    return padded_input_ids, padded_token_masks