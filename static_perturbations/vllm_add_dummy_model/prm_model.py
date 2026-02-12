import torch
from torch import nn

from vllm.model_executor.models.qwen2_rm import (
    Qwen2Model,
    AutoWeightsLoader,
    VllmConfig,
    maybe_prefix,
    IntermediateTensors,
    PoolingType, Pooler, PoolingMetadata, PoolerOutput,
    AttentionMetadata,
    Iterable,
    SupportsPP, SupportsLoRA,
    List, Union, Tuple, Optional, Set
)

class ValueHead(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output

class Qwen2ForPrmModel(nn.Module, SupportsLoRA, SupportsPP):
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

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

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
        self.v_head = ValueHead(self.config)
        self._pooler = Pooler.from_config_with_defaults(
            vllm_config.model_config.pooler_config,
            pooling_type=PoolingType.ALL,
            normalize=False,
            softmax=False
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        logits = self.v_head(hidden_states)
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
        ModelRegistry.register_model("Qwen2ForPrmModel", "vllm_add_dummy_model.prm_model:Qwen2ForPrmModel")
