# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_rm.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
"""Inference-only Qwen2-RM model compatible with HuggingFace weights."""
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn

from vllm.config import VllmConfig

from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP, SupportsV0Only
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

class ValueHead(nn.Module):
    """Custom ValueHead for weight loading compatibility"""

    def __init__(self, config, **kwargs):
        super().__init__()

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if hidden_states.dtype != self.summary.weight.dtype:
            hidden_states = hidden_states.to(self.summary.weight.dtype)
        output = self.summary(hidden_states)
        return output

class Qwen2ForPrmModel(nn.Module, SupportsLoRA, SupportsPP, SupportsV0Only):
    """Custom PRM model with ValueHead for weight loading compatibility"""
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # Use ValueHead for weight loading compatibility
        self.v_head = ValueHead(self.config)
        
        # Setup pooler for embeddings API compatibility
        pooler_config = vllm_config.model_config.pooler_config
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.STEP,
            normalize=False,
            softmax=False,
            step_tag_id=198,
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        logits = self.v_head(hidden_states).squeeze()
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["lm_head."])
        return loader.load_weights(weights)

def register():
    from vllm import ModelRegistry
    if "Qwen2ForPrmModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen2ForPrmModel", "prm_model:Qwen2ForPrmModel")