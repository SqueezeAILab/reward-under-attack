import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.device import (
    get_device_id,
    get_device_name,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import DistProfiler

from verl.workers.fsdp_workers import (
    RewardModelWorker, get_sharding_strategy
)

from recipe.prm.modeling_utils import (
    MODEL_CLASS_MAP,
    PREPARE_INPUT_MAP,
    DERIVE_LAST_REWARDS_MAP,
    DERIVE_STEP_REWARDS_MAP,
)


device_name = get_device_name()

class PRMRewardModelWorker(RewardModelWorker):
    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)


        self._do_switch_chat_template = True
        input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
        self.input_tokenizer = hf_tokenizer(
            input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
        )
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        # Step-level rewards require mapping PRM step scores back to policy response
        # tokens, which is only well-defined when both share a vocabulary.
        self._prm_tokenizer_matches_policy = (
            self.input_tokenizer.get_vocab() == self.tokenizer.get_vocab()
        )

        trust_remote_code = config.model.get("trust_remote_code", False)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=False, mesh=self.device_mesh
        )
        assert config.model.path in MODEL_CLASS_MAP, f"Model {config.model.path} not found in MODEL_CLASS_MAP"
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = MODEL_CLASS_MAP[config.model.path].from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    def _expand_to_token_level(self, data: DataProto, rm_data: DataProto, scores: torch.Tensor):
        """Map each PRM per-step reward onto the policy's response tokens (step-level reward).

        Only supported when the policy and PRM share a tokenizer: the PRM re-tokenizes the
        response (decode -> split on "\\n\\n" -> re-encode), so aligning its per-step scores to
        the policy's response-token positions requires a common vocabulary.

        The rebuttal step-level runs use the "cheap path": the reward stays as GRPO
        (``compute_grpo_outcome_advantage``), which sums ``token_level_scores`` over the
        response into one scalar per trajectory. We place each step's reward on that step's
        boundary token, so the trajectory return equals the sum of per-step PRM rewards.
        Because the advantage only depends on that per-row sum, ``index_add_`` guarantees it
        is exact even if the (approximate) boundary alignment collides or drifts.
        """
        if not self._prm_tokenizer_matches_policy:
            raise NotImplementedError(
                "Step-level rewards (use_last_reward_only=False) are only implemented when the "
                "policy and PRM share a tokenizer; got mismatched vocabularies."
            )
        assert self.config.model.path in DERIVE_STEP_REWARDS_MAP, (
            f"Model {self.config.model.path} not found in DERIVE_STEP_REWARDS_MAP"
        )

        token_masks = rm_data.batch["token_masks"]
        # (B, T_prm): per-step reward at each step-separator position, zero elsewhere.
        step_rewards = DERIVE_STEP_REWARDS_MAP[self.config.model.path](scores, token_masks, self.tokenizer)

        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_mask = data.batch["response_mask"]
        response_length = responses.shape[-1]

        token_level_scores = torch.zeros_like(response_mask, dtype=step_rewards.dtype)

        # Response tokens live in the policy's space; _switch_chat_template built the PRM
        # steps with the policy tokenizer, so decode/re-encode with it here too.
        tokenizer = self.input_tokenizer
        eos_token = tokenizer.eos_token
        sep = "\n\n"
        for i in range(responses.shape[0]):
            # Per-step rewards for this row, in order (one per step separator).
            step_vals = step_rewards[i][token_masks[i].bool()]
            if step_vals.numel() == 0:
                continue

            valid_response_length = int(attention_mask[i][-response_length:].sum().item())
            if valid_response_length == 0:
                continue

            valid_response_ids = responses[i][:valid_response_length]
            # Reconstruct the same steps the PRM saw (see _switch_chat_template).
            response = tokenizer.decode(valid_response_ids.tolist())
            if eos_token:
                response = response.replace(eos_token, "")
            steps = [step for step in response.split(sep) if step != ""]

            # Approximate each step's boundary token index in the policy response by
            # re-encoding the step text (same tokenizer) and accumulating lengths.
            boundaries = []
            pos = 0
            for step in steps:
                pos += len(tokenizer.encode(step + sep, add_special_tokens=False))
                boundaries.append(min(pos - 1, valid_response_length - 1))

            boundary_idx = torch.tensor(boundaries, dtype=torch.long, device=token_level_scores.device)
            # Reconcile counts so every step reward is placed exactly once (sum stays exact).
            k = step_vals.numel()
            if boundary_idx.numel() < k:
                pad = boundary_idx.new_full((k - boundary_idx.numel(),), valid_response_length - 1)
                boundary_idx = torch.cat([boundary_idx, pad])
            elif boundary_idx.numel() > k:
                boundary_idx = boundary_idx[:k]
            boundary_idx = boundary_idx.clamp_(0, response_length - 1)

            token_level_scores[i].index_add_(0, boundary_idx, step_vals.to(token_level_scores.dtype))

        return token_level_scores

    def _get_last_reward(self, data: DataProto, scores: torch.Tensor):
        token_masks = data.batch["token_masks"]
        assert token_masks.dim() == 2, f"token_masks should be 2D, but got {token_masks.dim()}"
        assert self.config.model.path in DERIVE_LAST_REWARDS_MAP, f"Model {self.config.model.path} not found in DERIVE_LAST_REWARDS_MAP"
        last_reward = DERIVE_LAST_REWARDS_MAP[self.config.model.path](scores, token_masks, self.tokenizer)
        return last_reward

    def _forward_micro_batch(self, micro_batch):
        if is_cuda_available:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        elif is_npu_available:
            from transformers.integrations.npu_flash_attention import (
                index_first_axis,
                pad_input,
                rearrange,
                unpad_input,
            )

        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if self.use_remove_padding:
                raise NotImplementedError("use_remove_padding is not supported for PRM reward model")
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits

            return rm_score

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []
        rm_token_masks = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            problem = chat[0]["content"]

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            steps = response.split("\n\n")
            input_ids, token_masks = PREPARE_INPUT_MAP[self.config.model.path](problem, steps, target_tokenizer)

            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.get("truncation", "right"),
            )

            token_masks = verl_F.pad_sequence_to_length(
                token_masks, max_seq_len=max_length, pad_token_id=0, left_pad=True
            )

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)
            rm_token_masks.append(token_masks)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)
        rm_token_masks = torch.cat(rm_token_masks, dim=0)
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {
            "input_ids": rm_input_ids, 
            "attention_mask": rm_attention_mask, 
            "position_ids": rm_position_ids, 
            "token_masks": rm_token_masks
        }

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        rm_data = self._switch_chat_template(data)
        # Support all hardwares
        rm_data.batch = rm_data.batch.to(get_device_id())

        response_mask = data.batch["response_mask"]

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            if self.config.use_last_reward_only:
                outcome_rewards = self._get_last_reward(rm_data, scores)
                token_level_scores = torch.zeros_like(response_mask, dtype=outcome_rewards.dtype)
                token_level_scores[:, 0] = outcome_rewards
            else:
                token_level_scores = self._expand_to_token_level(data, rm_data, scores)
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        return output

