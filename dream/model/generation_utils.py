# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Modified from Dream repos: https://github.com/HKUNLP/Dream

import time
import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import numpy as np
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

from model.gdllm_utils import (
    _quota_for_target_nfe,
    detect_attn_sinks_,
    select_high_confidence_fill,
    select_parallel_tokens_conflict_mis,
    select_axon_adaptive_gate_anchors,
    select_axon_adaptive_minimal_anchors,
    select_axon_submodular_anchors,
)

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        threshold = kwargs.get("threshold", 0.9)
        tau_induce = kwargs.get("tau_induce", 0.9)
        tau_sink = kwargs.get("tau_sink", 0.01)
        tau_edge = kwargs.get("tau_edge", 0.07)
        conf_threshold = kwargs.get("conf_threshold", 0.7)
        kl_threshold = kwargs.get("kl_threshold", 0.015)
        factor = kwargs.get("factor", 1.0)
        block_length = kwargs.get("block_length", 32)
        relaxed_threshold = kwargs.get("relaxed_threshold", 0.8)
        radius = kwargs.get("radius", 4)
        candidate_topk = kwargs.get("candidate_topk", 32)
        candidate_min_topk = kwargs.get("candidate_min_topk", 32)
        candidate_ratio = kwargs.get("candidate_ratio", 1.0)
        conflict_threshold = kwargs.get("conflict_threshold", 0.08)
        min_select = kwargs.get("min_select", 1)
        unlock_lambda = kwargs.get("unlock_lambda", 1.0)
        unlock_gamma = kwargs.get("unlock_gamma", 1.0)
        unlock_exit_conf_threshold = kwargs.get("unlock_exit_conf_threshold", 0.85)
        unlock_target_nfe_per_block = kwargs.get("unlock_target_nfe_per_block", 16)
        unlock_max_anchor_steps = kwargs.get("unlock_max_anchor_steps", 1)
        axon_reveal_policy = kwargs.get("axon_reveal_policy", "fixed1")
        axon_step_combo_mode = kwargs.get("axon_step_combo_mode", "none")
        axon_need_gate_mode = kwargs.get("axon_need_gate_mode", "none")
        axon_need_min_reveal_frac = kwargs.get("axon_need_min_reveal_frac", 0.10)
        axon_need_residual_conf_threshold = kwargs.get("axon_need_residual_conf_threshold", 0.70)
        axon_need_graph_coverage_threshold = kwargs.get("axon_need_graph_coverage_threshold", 0.60)
        axon_plugin = kwargs.get("axon_plugin", "none")
        axon_cover_target = kwargs.get("axon_cover_target", 0.50)
        axon_max_reveal_frac = kwargs.get("axon_max_reveal_frac", 0.0625)
        axon_needed_rule = kwargs.get("axon_needed_rule", "none")
        axon_need_block_conf_threshold = kwargs.get("axon_need_block_conf_threshold", 0.80)
        axon_need_high_conf_frac_threshold = kwargs.get("axon_need_high_conf_frac_threshold", 0.50)
        axon_need_min_step = kwargs.get("axon_need_min_step", 0)
        axon_need_bad_streak = kwargs.get("axon_need_bad_streak", 1)
        axon_dawn_tau_sink = kwargs.get("axon_dawn_tau_sink", -1.0)
        axon_dawn_tau_edge = kwargs.get("axon_dawn_tau_edge", -1.0)
        axon_dawn_tau_induce = kwargs.get("axon_dawn_tau_induce", -1.0)
        axon_dawn_conf_threshold = kwargs.get("axon_dawn_conf_threshold", -1.0)
        axon_max_anchors_per_step = kwargs.get("axon_max_anchors_per_step", 1)
        axon_anchor_size_penalty = kwargs.get("axon_anchor_size_penalty", 0.0)
        axon_anchor_size_penalty_power = kwargs.get("axon_anchor_size_penalty_power", 1.0)
        axon_adaptive_gate = kwargs.get("axon_adaptive_gate", "progress_empty")
        axon_adaptive_selector = kwargs.get("axon_adaptive_selector", "fixed1")
        axon_stag_gate = kwargs.get("axon_stag_gate", "progress_stall")
        axon_stag_selector = kwargs.get("axon_stag_selector", "density")
        gate_alpha = float(kwargs.get("gate_alpha", 0.5))
        axon_submod_fn = kwargs.get("axon_submod_fn", "facility_location")
        axon_submod_monotone = kwargs.get("axon_submod_monotone", True)
        axon_submod_penalty = kwargs.get("axon_submod_penalty", "conflict")
        axon_submod_lambda = kwargs.get("axon_submod_lambda", 0.0)
        axon_step_mode = kwargs.get("axon_step_mode", "anchor_only")
        unlock_stage2_fill = kwargs.get("unlock_stage2_fill", False)
        unlock_fill_ratio = kwargs.get("unlock_fill_ratio", 0.5)
        unlock_high_conf_threshold = kwargs.get("unlock_high_conf_threshold", 0.0)
        unlock_anchor_min_conf = kwargs.get("unlock_anchor_min_conf", 0.0)
        union_dawn_keep_frac = kwargs.get("union_dawn_keep_frac", 1.0)
        adcov_threshold = kwargs.get("adcov_threshold", 0.5)
        adcov_warmup = kwargs.get("adcov_warmup", 256)
        axon_base_proposer = kwargs.get("axon_base_proposer", "dawn")

        result, nfe = self._sample_block(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            threshold=threshold,
            tau_induce=tau_induce,
            tau_sink=tau_sink,
            tau_edge=tau_edge,
            conf_threshold=conf_threshold,
            factor=factor,
            block_length=block_length,
            kl_threshold=kl_threshold,
            relaxed_threshold=relaxed_threshold,
            radius=radius,
            candidate_topk=candidate_topk,
            candidate_min_topk=candidate_min_topk,
            candidate_ratio=candidate_ratio,
            conflict_threshold=conflict_threshold,
            min_select=min_select,
            unlock_lambda=unlock_lambda,
            unlock_gamma=unlock_gamma,
            unlock_exit_conf_threshold=unlock_exit_conf_threshold,
            unlock_target_nfe_per_block=unlock_target_nfe_per_block,
            unlock_max_anchor_steps=unlock_max_anchor_steps,
            axon_reveal_policy=axon_reveal_policy,
            axon_step_combo_mode=axon_step_combo_mode,
            axon_need_gate_mode=axon_need_gate_mode,
            axon_need_min_reveal_frac=axon_need_min_reveal_frac,
            axon_need_residual_conf_threshold=axon_need_residual_conf_threshold,
            axon_need_graph_coverage_threshold=axon_need_graph_coverage_threshold,
            axon_plugin=axon_plugin,
            axon_cover_target=axon_cover_target,
            axon_max_reveal_frac=axon_max_reveal_frac,
            axon_needed_rule=axon_needed_rule,
            axon_need_block_conf_threshold=axon_need_block_conf_threshold,
            axon_need_high_conf_frac_threshold=axon_need_high_conf_frac_threshold,
            axon_need_min_step=axon_need_min_step,
            axon_need_bad_streak=axon_need_bad_streak,
            axon_dawn_tau_sink=axon_dawn_tau_sink,
            axon_dawn_tau_edge=axon_dawn_tau_edge,
            axon_dawn_tau_induce=axon_dawn_tau_induce,
            axon_dawn_conf_threshold=axon_dawn_conf_threshold,
            axon_max_anchors_per_step=axon_max_anchors_per_step,
            axon_anchor_size_penalty=axon_anchor_size_penalty,
            axon_anchor_size_penalty_power=axon_anchor_size_penalty_power,
            axon_adaptive_gate=axon_adaptive_gate,
            axon_adaptive_selector=axon_adaptive_selector,
            axon_stag_gate=axon_stag_gate,
            axon_stag_selector=axon_stag_selector,
            gate_alpha=gate_alpha,
            axon_submod_fn=axon_submod_fn,
            axon_submod_monotone=axon_submod_monotone,
            axon_submod_penalty=axon_submod_penalty,
            axon_submod_lambda=axon_submod_lambda,
            axon_step_mode=axon_step_mode,
            unlock_stage2_fill=unlock_stage2_fill,
            unlock_fill_ratio=unlock_fill_ratio,
            unlock_high_conf_threshold=unlock_high_conf_threshold,
            unlock_anchor_min_conf=unlock_anchor_min_conf,
            union_dawn_keep_frac=union_dawn_keep_frac,
            adcov_threshold=adcov_threshold,
            adcov_warmup=adcov_warmup,
            axon_base_proposer=axon_base_proposer,
        )
        return result, nfe

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        threshold: Optional[float] = 0.9,
        tau_induce: Optional[float] = 0.9,
        conf_threshold: Optional[float] = 0.7,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None
        start_time = time.time()
        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        i = 0
        if alg == 'confidence_threshold':
            mask_index = (x == mask_token_id)
            assert mask_index.sum() % steps == 0, "mask_index.sum() must be divisible by steps"
            assert x.shape[0] == 1, "batch size must be 1"

            number_transfer_tokens = mask_index.sum().item() // steps
            left_tokens_last_step = 0
        if alg == 'g-dllm':
            conf_arch = torch.full_like(x, 0.0, device=self.device, dtype=torch.bfloat16)
            conf_arch[:, :input_ids.shape[1]] = 1.0
        while i < steps:
            mask_index = (x == mask_token_id)
            if mask_index.sum() == 0:
                steps = i
                break
            output, avg_attn_scores = self(x, attention_mask, tok_idx, return_attn_scores= (alg == 'g-dllm'))
            logits = output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            if not alg == 'confidence_threshold':
                t = timesteps[i]
                s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            elif alg == 'confidence_threshold':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                current_transfer_tokens = number_transfer_tokens + left_tokens_last_step
                left_tokens_last_step = 0
                selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                select_index = select_index.to(x.device)
                transfer_index[0, select_index[0]] = True
                for k in range(1, current_transfer_tokens):
                    if selected_confidence[0, k] < threshold:
                        if i < steps - 1:
                            left_tokens_last_step += 1
                            transfer_index[0, select_index[0, k]] = False
                        else:
                            number_transfer_tokens = 0
                            steps += 1
                            left_tokens_last_step += 1
                            transfer_index[0, select_index[0, k]] = False

                x[transfer_index] = x_[transfer_index].clone()
            elif alg == 'g-dllm':
                assert avg_attn_scores is not None, 'avg_attn_scores is None'
                sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=0.02)
                key_sink_mask = sink_mask.unsqueeze(1)      # [B, 1, L]
                avg_attn_scores = avg_attn_scores.masked_fill(key_sink_mask, 0.0)  # [B, L, L]

                B, _, _ = avg_attn_scores.shape
                avg_attn_scores.diagonal(dim1=1, dim2=2).zero_()
                
                x0_p, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)

                quantile_mask = avg_attn_scores >= 0.07
                quantile_mask = quantile_mask.transpose(1, 2)

                # select the dependent nodes
                decoded_mask = (~mask_index & (conf_arch >= threshold_d)).unsqueeze(-1)
                decoded_edge = quantile_mask & decoded_mask # [B, ?, N]
                dependent_nodes = decoded_edge.any(dim=1) & mask_index
                dependent_conf = torch.where(
                                        decoded_edge, # [B, ?, N]                  
                                        conf_arch.unsqueeze(-1), # [B, ?, 1]        
                                        torch.zeros_like(conf_arch.unsqueeze(-1))  
                                    )
                dependent_conf, _ = dependent_conf.max(dim=1)
                conf_d = torch.where(dependent_nodes, x0_p, -np.inf)
                transfer_index = conf_d >= (threshold_d + threshold_c - dependent_conf)

                adj_ti_mask = quantile_mask & transfer_index.unsqueeze(-1)
                adj_ti_mask = adj_ti_mask.any(dim=1)
                node_mask = mask_index & (x0_p >= threshold_c) & ~transfer_index & ~adj_ti_mask

                if node_mask.sum(dim=-1).min().item() != 0:
                    _node_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1) # [B, N, N]
                    edge_mask = _node_mask & quantile_mask # [B, N, N]

                    confidence = torch.where(node_mask, x0_p, -np.inf)

                    transfer_index_c = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(B):
                        select_index = select_parallel_tokens_conflict_mis(edge_mask[j], node_mask[j], confidence[j])
                        transfer_index_c[j, select_index] = True

                    transfer_index = transfer_index | transfer_index_c
                
                if transfer_index.sum(dim=-1).min().item() == 0:
                    # x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)
                    
                    max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
                    force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
                    transfer_index = transfer_index | force_mask

                x[transfer_index] = x0[transfer_index]
                conf_arch[transfer_index] = x0_p[transfer_index]
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices,transfer_index] = x_[row_indices,transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
            i += 1
        
        print(f'used steps: {steps}')
        end_time = time.time()
        print(f'used time: {end_time - start_time}')
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

    def _sample_block(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = 32,
        tau_induce: Optional[float] = 0.9,
        conf_threshold: Optional[float] = 0.7,
        kl_threshold: Optional[float] = 0.015,
        relaxed_threshold: Optional[float] = 0.8,
        radius: Optional[int] = 4,
        tau_sink: Optional[float] = 0.01,
        tau_edge: Optional[float] = 0.07,
        factor: Optional[float] = 1.0,
        candidate_topk: Optional[int] = 32,
        candidate_min_topk: Optional[int] = 32,
        candidate_ratio: Optional[float] = 1.0,
        conflict_threshold: Optional[float] = 0.08,
        min_select: Optional[int] = 1,
        unlock_lambda: Optional[float] = 1.0,
        unlock_gamma: Optional[float] = 1.0,
        unlock_exit_conf_threshold: Optional[float] = 0.85,
        unlock_target_nfe_per_block: Optional[float] = 16,
        unlock_max_anchor_steps: Optional[int] = 1,
        axon_reveal_policy: Optional[str] = "fixed1",
        axon_step_combo_mode: Optional[str] = "none",
        axon_need_gate_mode: Optional[str] = "none",
        axon_need_min_reveal_frac: Optional[float] = 0.10,
        axon_need_residual_conf_threshold: Optional[float] = 0.70,
        axon_need_graph_coverage_threshold: Optional[float] = 0.60,
        axon_plugin: Optional[str] = "none",
        axon_cover_target: Optional[float] = 0.50,
        axon_max_reveal_frac: Optional[float] = 0.0625,
        axon_needed_rule: Optional[str] = "none",
        axon_need_block_conf_threshold: Optional[float] = 0.80,
        axon_need_high_conf_frac_threshold: Optional[float] = 0.50,
        axon_need_min_step: Optional[int] = 0,
        axon_need_bad_streak: Optional[int] = 1,
        axon_dawn_tau_sink: Optional[float] = -1.0,
        axon_dawn_tau_edge: Optional[float] = -1.0,
        axon_dawn_tau_induce: Optional[float] = -1.0,
        axon_dawn_conf_threshold: Optional[float] = -1.0,
        axon_max_anchors_per_step: Optional[int] = 1,
        axon_anchor_size_penalty: Optional[float] = 0.0,
        axon_anchor_size_penalty_power: Optional[float] = 1.0,
        axon_adaptive_gate: Optional[str] = "progress_empty",
        axon_adaptive_selector: Optional[str] = "fixed1",
        axon_stag_gate: Optional[str] = "progress_stall",
        axon_stag_selector: Optional[str] = "density",
        gate_alpha: Optional[float] = 0.5,
        axon_submod_fn: Optional[str] = "facility_location",
        axon_submod_monotone: Optional[bool] = True,
        axon_submod_penalty: Optional[str] = "conflict",
        axon_submod_lambda: Optional[float] = 0.0,
        axon_step_mode: Optional[str] = "anchor_only",
        unlock_stage2_fill: Optional[bool] = False,
        unlock_fill_ratio: Optional[float] = 0.5,
        unlock_high_conf_threshold: Optional[float] = 0.0,
        unlock_anchor_min_conf: Optional[float] = 0.0,
        union_dawn_keep_frac: Optional[float] = 1.0,
        adcov_threshold: Optional[float] = 0.5,
        adcov_warmup: Optional[int] = 256,
        axon_base_proposer: Optional[str] = "dawn",
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        kl_history_length = 2
        unmask_strategy = "all"

        histories = [] if (return_dict_in_generate and output_history) else None
        start_time = time.time()
        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        gen_length = max_length - input_ids.shape[1]
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)

        V = self.lm_head.out_features if hasattr(self, "lm_head") else self.config.vocab_size
        kl_history = torch.zeros((1, x.shape[1], kl_history_length), dtype=torch.float64, device=x.device)
        p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        nfe = 0
        self._last_axon_submodular_calls = 0
        self._last_axon_submodular_anchors = 0
        self._last_axon_step_records = []
        # Per-step profiler (opt-in). Stashed on the model instance by the eval
        # (dream monkey-patches diffusion_generate as a MethodType, so threading a
        # new arg through the signatures is fragile; self-attribute mirrors how the
        # axon records above are passed back). need_attn matches the forward flag.
        _prof = getattr(self, "_step_profiler", None)
        _prof_need_attn = (alg == 'dawn' or alg == 'axon')
        # adaptive_coverage selector state is global across the eval and
        # lazy-initialised on `self` (see the adaptive_coverage block below).

        for num_block in range(num_blocks):
            i = 0
            axon_anchor_steps_used = 0
            axon_bad_streak_used = 0
            prev_dawn_progress = None
            prev_graph_coverage = None
            prev_readiness = None
            prev_hybrid_bad = False
            while True:
                mask_index = (x == mask_token_id)
                mask_index[:, input_ids.shape[1] + (num_block + 1) * block_length:] = 0
                if mask_index.sum() == 0:
                    break
                _masked0 = mask_index.sum() if _prof is not None else None  # for base-arm reveal count
                if _prof is not None:
                    _prof.step_begin(_prof_need_attn)
                output, avg_attn_scores = self(x, attention_mask, tok_idx, return_attn_scores= (alg == 'dawn' or alg == 'axon'))
                if _prof is not None:
                    _prof.forward_end()
                logits = output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                # this allows user-defined logits control of the intermediate steps
                logits = generation_logits_hook_func(i, x, logits)

                mask_logits = logits[mask_index]
                nfe += 1
                if not alg == 'confidence_threshold' and not alg == 'dawn' and not alg == 'axon' and not alg == 'klass' and not alg == 'local_leap' and not alg == 'factor':
                    t = timesteps[i]
                    s = timesteps[i + 1]
            
                if alg == 'origin':
                    p_transfer = 1 - s / t if i < steps_per_block - 1 else 1
                    x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                    transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                    x[mask_index] = x0.clone()
                elif alg == 'confidence_threshold':
                    full_confidence, x_ = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    full_confidence = torch.where(mask_index, full_confidence, -np.inf)
                    current_transfer_tokens = mask_index.sum()
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_confidence[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = False
                    if transfer_index.sum() == 0:
                        _, force_index = torch.topk(full_confidence, 1)
                        transfer_index[0, force_index[0]] = True

                    x[transfer_index] = x_[transfer_index].clone()
                elif alg == 'factor':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence

                    transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
                    
                    for j in range(full_confidence.shape[0]):
                        num_tokens = int(num_transfer_tokens[j].item())
                        if num_tokens == 0:
                            continue
                        
                        ns=list(range(1,num_transfer_tokens[j]+1))
                        es=[factor/(n+1) for n in ns]
                        threshs=[1-e for e in es]

                        # at least one token is transferred
                        threshs[0]=-1
                        sorted_confidence=torch.sort(full_confidence[j][mask_index[j]],dim=-1,descending=True)[0]
                        assert len(sorted_confidence)==len(threshs)
                        for top_i in range(len(threshs)):
                            if sorted_confidence[top_i]<threshs[top_i]:
                                break

                        if top_i == 0 or top_i == len(threshs)-1:
                            top_i+=1

                        _, select_index = torch.topk(full_confidence[j], k=top_i)
                        transfer_index[j, select_index] = True
                    x[transfer_index] = x_[transfer_index].clone()
                elif alg == 'dawn':
                    assert avg_attn_scores is not None, 'avg_attn_scores is None'
                    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)
                    key_sink_mask = sink_mask.unsqueeze(1)      # [B, 1, L]
                    avg_attn_scores = avg_attn_scores.masked_fill(key_sink_mask, 0.0)  # [B, L, L]

                    B, _, _ = avg_attn_scores.shape
                    avg_attn_scores.diagonal(dim1=1, dim2=2).zero_()
                    
                    x0_p, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)

                    quantile_mask = avg_attn_scores >= tau_edge
                    quantile_mask = quantile_mask.transpose(1, 2)

                    confidence = torch.where(mask_index, x0_p, -np.inf)
                    transfer_index_conf = confidence >= 0.9

                    # select the dependent nodes
                    decoded_mask = (~mask_index & (x0_p >= 0.9)).unsqueeze(-1)
                    decoded_mask[:, input_ids.shape[1] + (num_block + 1) * block_length:] = False
                    decoded_edge = quantile_mask & decoded_mask # [B, ?, N]
                    dependent_nodes = decoded_edge.any(dim=1) & mask_index
                    conf_d = torch.where(dependent_nodes, x0_p, -np.inf)
                    transfer_index_a = conf_d >= tau_induce

                    adj_ti_mask = quantile_mask & transfer_index_a.unsqueeze(-1) & transfer_index_conf.unsqueeze(-1)
                    adj_ti_mask = adj_ti_mask.any(dim=1)
                    node_mask = mask_index & (x0_p >= conf_threshold) & (x0_p < 0.9) & ~transfer_index_a & ~adj_ti_mask

                    transfer_index_c = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

                    if node_mask.sum(dim=-1).min().item() != 0:
                        _node_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1) # [B, N, N]
                        edge_mask = _node_mask & quantile_mask # [B, N, N]

                        confidence = torch.where(node_mask, x0_p, -np.inf)

                        
                        for j in range(B):
                            select_index = select_parallel_tokens_conflict_mis(edge_mask[j], node_mask[j], confidence[j])
                            transfer_index_c[j, select_index] = True

                    transfer_index = transfer_index_a | transfer_index_conf | transfer_index_c
                    
                    if transfer_index.sum(dim=-1).min().item() == 0:
                        # x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, -np.inf)
                        
                        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
                        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
                        transfer_index = transfer_index | force_mask

                    x[transfer_index] = x0[transfer_index].clone()
                    # conf_arch[transfer_index] = x0_p[transfer_index]
                elif alg == 'axon':
                    assert avg_attn_scores is not None, 'avg_attn_scores is None'
                    x0_p, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)

                    axon_combo_mode = str(axon_step_combo_mode).strip().lower()
                    if axon_combo_mode in {"", "false", "0"}:
                        axon_combo_mode = "none"
                    if axon_combo_mode != "none":
                        raise ValueError(f"Dream AXON currently supports axon_step_combo_mode=none, got {axon_step_combo_mode}")
                    axon_need_mode = str(axon_need_gate_mode).strip().lower()
                    if axon_need_mode in {"", "false", "0"}:
                        axon_need_mode = "none"
                    if axon_need_mode not in {"none", "low_reveal", "residual_conf", "graph_gap", "low_reveal_or_graph_gap"}:
                        raise ValueError(f"Unsupported AXON need gate mode: {axon_need_gate_mode}")
                    axon_plugin_name = str(axon_plugin).strip().lower()
                    if axon_plugin_name in {"", "false", "0"}:
                        axon_plugin_name = "none"
                    if axon_plugin_name not in {"none", "adaptive_minimal", "needed_fixed1", "adaptive_gate", "stag_unified"}:
                        raise ValueError(f"Unsupported AXON plugin: {axon_plugin}")
                    axon_base_proposer_name = str(axon_base_proposer).strip().lower()
                    if axon_base_proposer_name in {"", "false", "0"}:
                        axon_base_proposer_name = "dawn"
                    if axon_base_proposer_name not in {"dawn", "local_leap", "confidence"}:
                        raise ValueError(f"Unsupported axon_base_proposer for Dream: {axon_base_proposer}")
                    axon_step_mode_name = str(axon_step_mode).strip().lower()
                    if axon_step_mode_name in {"", "false", "0"}:
                        axon_step_mode_name = "anchor_only"
                    if axon_step_mode_name not in {"anchor_only", "augmenting"}:
                        raise ValueError(f"Unsupported AXON step mode: {axon_step_mode}")
                    if axon_plugin_name == "stag_unified":
                        axon_adaptive_gate_name = str(axon_stag_gate).strip().lower()
                        axon_adaptive_selector_name = str(axon_stag_selector).strip().lower()
                    else:
                        axon_adaptive_gate_name = str(axon_adaptive_gate).strip().lower()
                        axon_adaptive_selector_name = str(axon_adaptive_selector).strip().lower()
                    if axon_adaptive_selector_name in {"", "false", "0"}:
                        axon_adaptive_selector_name = "fixed1"
                    if axon_adaptive_selector_name not in {"fixed1", "density", "cover", "nonmono"}:
                        raise ValueError(f"Unsupported AXON adaptive selector: {axon_adaptive_selector}")
                    supported_adaptive_gates = {
                        "progress_empty",
                        "progress_below_quota",
                        "progress_below_expected",
                        "progress_stalled",
                        "graph_below_uniform",
                        "graph_below_progress",
                        "graph_residual_gap",
                        "graph_candidate_advantage",
                        "readiness_lt_one",
                        "residual_below_block",
                        "residual_below_revealed",
                        "dispersion_dominates",
                        "progress_or_graph",
                        "progress_and_graph",
                        "readiness_or_graph",
                        "readiness_and_progress",
                        "dawn_weak_certificate",
                        "two_step_progress_stall",
                        "two_step_graph_gap",
                        "two_step_hybrid_stuck",
                        "progress_stall",
                        "coverage_gap",
                        "residual_risk",
                        "readiness_drop",
                        "candidate_advantage",
                        "adaptive_coverage",
                        "unified_axon",
                        "cp_alpha",
                    }
                    if axon_adaptive_gate_name not in supported_adaptive_gates:
                        raise ValueError(f"Unsupported AXON adaptive gate: {axon_adaptive_gate}")
                    axon_rule_name = str(axon_needed_rule).strip().lower()
                    if axon_rule_name in {"", "false", "0"}:
                        axon_rule_name = "none"
                    if axon_rule_name not in {
                        "none",
                        "avg_conf",
                        "high_conf_frac",
                        "dawn_reveal",
                        "graph",
                        "or_mild",
                        "and_strict",
                        "delayed_graph",
                        "stuck2",
                        "readiness",
                    }:
                        raise ValueError(f"Unsupported AXON needed rule: {axon_needed_rule}")

                    block_start = input_ids.shape[1] + num_block * block_length
                    block_end = input_ids.shape[1] + (num_block + 1) * block_length
                    dawn_transfer_index = None
                    dawn_tau_sink = tau_sink if float(axon_dawn_tau_sink) < 0.0 else float(axon_dawn_tau_sink)
                    dawn_tau_edge = tau_edge if float(axon_dawn_tau_edge) < 0.0 else float(axon_dawn_tau_edge)
                    dawn_tau_induce = tau_induce if float(axon_dawn_tau_induce) < 0.0 else float(axon_dawn_tau_induce)
                    dawn_conf_threshold = conf_threshold if float(axon_dawn_conf_threshold) < 0.0 else float(axon_dawn_conf_threshold)

                    def _dream_dawn_transfer_index():
                        sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=dawn_tau_sink)
                        key_sink_mask = sink_mask.unsqueeze(1)
                        avg_attn_scores_clean = avg_attn_scores.masked_fill(key_sink_mask, 0.0).clone()
                        avg_attn_scores_clean.diagonal(dim1=1, dim2=2).zero_()
                        quantile_mask = avg_attn_scores_clean >= dawn_tau_edge
                        quantile_mask = quantile_mask.transpose(1, 2)

                        confidence = torch.where(mask_index, x0_p, -np.inf)
                        transfer_index_conf = confidence >= 0.9
                        decoded_mask = (~mask_index & (x0_p >= 0.9)).unsqueeze(-1)
                        decoded_mask[:, input_ids.shape[1] + (num_block + 1) * block_length:] = False
                        decoded_edge = quantile_mask & decoded_mask
                        dependent_nodes = decoded_edge.any(dim=1) & mask_index
                        conf_d = torch.where(dependent_nodes, x0_p, -np.inf)
                        transfer_index_a = conf_d >= dawn_tau_induce
                        adj_ti_mask = quantile_mask & transfer_index_a.unsqueeze(-1) & transfer_index_conf.unsqueeze(-1)
                        adj_ti_mask = adj_ti_mask.any(dim=1)
                        node_mask = mask_index & (x0_p >= dawn_conf_threshold) & (x0_p < 0.9) & ~transfer_index_a & ~adj_ti_mask
                        transfer_index_c = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        if node_mask.sum(dim=-1).min().item() != 0:
                            _node_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
                            edge_mask = _node_mask & quantile_mask
                            confidence_c = torch.where(node_mask, x0_p, -np.inf)
                            for batch_idx in range(x.shape[0]):
                                select_index = select_parallel_tokens_conflict_mis(
                                    edge_mask[batch_idx],
                                    node_mask[batch_idx],
                                    confidence_c[batch_idx],
                                )
                                transfer_index_c[batch_idx, select_index] = True
                        transfer_index = transfer_index_a | transfer_index_conf | transfer_index_c
                        if transfer_index.sum(dim=-1).min().item() == 0:
                            confidence = torch.where(mask_index, x0_p, -np.inf)
                            max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
                            force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
                            transfer_index = transfer_index | force_mask
                        return transfer_index

                    def _dream_local_leap_transfer_index():
                        """Mirror of alg=='local_leap' step (around line 1451)."""
                        full_confidence_ll, _x_ll = sample_tokens(
                            logits, temperature=temperature, top_p=top_p, top_k=top_k
                        )
                        full_confidence_ll = torch.where(mask_index, full_confidence_ll, -np.inf)
                        current_transfer_tokens_ll = mask_index.sum()
                        _selected_conf_ll, select_index_ll = torch.topk(
                            full_confidence_ll, current_transfer_tokens_ll
                        )
                        ti_ll = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                        select_index_ll = select_index_ll.to(x.device)
                        ti_ll[0, select_index_ll[0]] = True
                        ti_ll = self.get_transfer_index_localleap(
                            ti_ll, full_confidence_ll, select_index_ll, mask_index, x, logits,
                            current_transfer_tokens_ll, conf_threshold, relaxed_threshold, radius,
                        )
                        return ti_ll

                    def _dream_confidence_transfer_index():
                        """Mirror of alg=='confidence_threshold' step (around line 824).
                        Uses `threshold` (default 0.9, the canonical 'parallel' baseline)."""
                        full_confidence_cf, _x_cf = sample_tokens(
                            logits, temperature=temperature, top_p=top_p, top_k=top_k
                        )
                        full_confidence_cf = torch.where(mask_index, full_confidence_cf, -np.inf)
                        current_transfer_tokens_cf = mask_index.sum()
                        selected_conf_cf, select_index_cf = torch.topk(
                            full_confidence_cf, current_transfer_tokens_cf
                        )
                        ti_cf = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                        select_index_cf = select_index_cf.to(x.device)
                        ti_cf[0, select_index_cf[0]] = True
                        for k in range(1, current_transfer_tokens_cf):
                            if selected_conf_cf[0, k] < threshold:
                                ti_cf[0, select_index_cf[0, k]] = False
                        if ti_cf.sum() == 0:
                            _, force_index = torch.topk(full_confidence_cf, 1)
                            ti_cf[0, force_index[0]] = True
                        return ti_cf

                    def _dream_get_base_proposal(name):
                        """Dispatch to a base decoder's per-step transfer_index."""
                        n = str(name).strip().lower()
                        if n in {"", "false", "0", "dawn"}:
                            return _dream_dawn_transfer_index()
                        if n == "local_leap":
                            return _dream_local_leap_transfer_index()
                        if n == "confidence":
                            return _dream_confidence_transfer_index()
                        raise ValueError(f"Unsupported axon_base_proposer for Dream: {name}")

                    can_anchor = axon_anchor_steps_used < int(unlock_max_anchor_steps)
                    use_dawn = not can_anchor
                    axon_step_record = None
                    if (
                        axon_need_mode == "none"
                        and axon_plugin_name != "needed_fixed1"
                        and not use_dawn
                        and float(unlock_exit_conf_threshold) > 0.0
                    ):
                        block_mask = mask_index[:, block_start:block_end]
                        if block_mask.sum() > 0:
                            avg_conf = x0_p[:, block_start:block_end][block_mask].float().mean().item()
                            if avg_conf > float(unlock_exit_conf_threshold):
                                use_dawn = True

                    if can_anchor and (
                        axon_need_mode != "none" or axon_plugin_name in {"adaptive_minimal", "needed_fixed1", "adaptive_gate", "stag_unified"}
                    ):
                        block_mask = mask_index[:, block_start:block_end]
                        masked_count = int(block_mask.sum().item())
                        block_conf = 1.0
                        high_conf_frac = 1.0
                        block_conf_std = 0.0
                        readiness = float("inf")
                        if masked_count > 0:
                            block_conf_values = x0_p[:, block_start:block_end][block_mask].float()
                            block_conf = float(block_conf_values.mean().item())
                            high_conf_frac = float((block_conf_values >= 0.90).float().mean().item())
                            block_conf_std = float(block_conf_values.std(unbiased=False).item())
                            readiness = (block_conf * high_conf_frac) / (block_conf_std + 1e-3)

                        reveal_frac = 1.0
                        residual_conf = 1.0
                        revealed_conf = 1.0
                        graph_coverage = 1.0
                        dawn_raw_coverage = 0.0
                        dawn_density = 0.0
                        residual_count = 0
                        remaining_target_steps = max(1, int(np.ceil(float(unlock_target_nfe_per_block))) - int(i))
                        expected_count = int(np.ceil(float(masked_count) / float(remaining_target_steps))) if masked_count > 0 else 0
                        expected_progress = float(expected_count) / float(max(1, masked_count))
                        low_reveal_need = False
                        residual_conf_need = False
                        graph_gap_need = False

                        block_conf_need = block_conf < float(axon_need_block_conf_threshold)
                        high_conf_frac_need = high_conf_frac < float(axon_need_high_conf_frac_threshold)
                        readiness_need = readiness < float(axon_need_block_conf_threshold)
                        needs_dawn_diagnostics = not (
                            axon_plugin_name == "needed_fixed1"
                            and axon_rule_name in {"avg_conf", "high_conf_frac", "readiness"}
                        )
                        if needs_dawn_diagnostics:
                            needs_graph_diagnostics = not (
                                axon_plugin_name == "needed_fixed1"
                                and axon_rule_name in {"dawn_reveal", "stuck2"}
                            )
                            if _prof is not None:
                                _prof.base_begin()
                            dawn_transfer_index = _dream_get_base_proposal(axon_base_proposer_name)
                            if _prof is not None:
                                _prof.base_end()
                            block_dawn = dawn_transfer_index[:, block_start:block_end] & block_mask
                            dawn_count = int(block_dawn.sum().item())
                            reveal_frac = float(dawn_count) / float(max(1, masked_count))
                            residual_mask = block_mask & ~block_dawn
                            residual_count = int(residual_mask.sum().item())
                            if dawn_count > 0:
                                revealed_conf = x0_p[:, block_start:block_end][block_dawn].float().mean().item()
                            if residual_count > 0:
                                residual_conf = x0_p[:, block_start:block_end][residual_mask].float().mean().item()

                            if needs_graph_diagnostics and residual_count > 0:
                                row_coverages = []
                                row_raw_coverages = []
                                for batch_idx in range(mask_index.shape[0]):
                                    residual_pos = torch.nonzero(residual_mask[batch_idx], as_tuple=False).flatten() + block_start
                                    anchor_pos = torch.nonzero(block_dawn[batch_idx], as_tuple=False).flatten() + block_start
                                    if residual_pos.numel() == 0:
                                        row_coverages.append(1.0)
                                        row_raw_coverages.append(0.0)
                                    elif anchor_pos.numel() == 0:
                                        row_coverages.append(0.0)
                                        row_raw_coverages.append(0.0)
                                    else:
                                        block_pos = torch.arange(block_start, block_end, device=mask_index.device)
                                        residual_attn = avg_attn_scores[batch_idx].index_select(0, residual_pos)
                                        anchor_mass = residual_attn.index_select(1, anchor_pos).sum(dim=1)
                                        block_mass = residual_attn.index_select(1, block_pos).sum(dim=1)
                                        row_coverages.append(
                                            float((anchor_mass / (block_mass + 1e-8)).clamp(0.0, 1.0).mean().item())
                                        )
                                        residual_conf_values = x0_p[batch_idx].index_select(0, residual_pos).float()
                                        anchor_conf_values = x0_p[batch_idx].index_select(0, anchor_pos).float()
                                        raw_weights = (
                                            residual_attn.index_select(1, anchor_pos).float()
                                            * (1.0 - residual_conf_values).clamp_min(0.0).unsqueeze(1)
                                            * anchor_conf_values.unsqueeze(0)
                                        )
                                        row_raw_coverages.append(float(raw_weights.max(dim=1).values.sum().item()))
                                graph_coverage = sum(row_coverages) / float(max(1, len(row_coverages)))
                                dawn_raw_coverage = sum(row_raw_coverages) / float(max(1, len(row_raw_coverages)))
                                dawn_density = dawn_raw_coverage / float(max(1, dawn_count))

                            low_reveal_need = reveal_frac < float(axon_need_min_reveal_frac)
                            residual_conf_need = residual_conf < float(axon_need_residual_conf_threshold)
                            graph_gap_need = graph_coverage < float(axon_need_graph_coverage_threshold)
                        if axon_plugin_name in {"adaptive_gate", "stag_unified"}:
                            progress_empty = dawn_count == 0
                            progress_below_expected = reveal_frac < expected_progress
                            progress_below_quota = dawn_count < expected_count
                            progress_stalled = progress_empty or progress_below_expected
                            graph_below_uniform = graph_coverage < reveal_frac
                            graph_below_progress = graph_coverage < reveal_frac
                            graph_residual_gap = residual_count > 0 and graph_coverage < (1.0 - residual_conf)
                            graph_candidate_advantage = residual_count > 0 and graph_coverage < block_conf
                            readiness_lt_one = readiness < 1.0
                            residual_below_block = residual_count > 0 and residual_conf < block_conf
                            residual_below_revealed = residual_count > 0 and residual_conf < revealed_conf
                            dispersion_dominates = block_conf_std > max(0.0, block_conf - residual_conf)
                            progress_or_graph = progress_below_expected or graph_below_uniform
                            progress_and_graph = progress_below_expected and graph_below_uniform
                            readiness_or_graph = readiness_lt_one or graph_below_uniform
                            readiness_and_progress = readiness_lt_one and progress_below_expected
                            dawn_weak_certificate = (
                                progress_below_expected
                                and residual_below_revealed
                                and (graph_below_uniform or graph_residual_gap)
                            )
                            two_step_progress_stall = (
                                prev_dawn_progress is not None
                                and reveal_frac <= prev_dawn_progress
                                and progress_below_expected
                            )
                            readiness_drop = (
                                prev_readiness is not None
                                and masked_count > 0
                                and readiness < prev_readiness
                            )
                            two_step_graph_gap = (
                                prev_graph_coverage is not None
                                and graph_coverage <= prev_graph_coverage
                                and graph_below_uniform
                            )
                            hybrid_bad = progress_or_graph or readiness_lt_one
                            two_step_hybrid_stuck = prev_hybrid_bad and hybrid_bad
                            coverage_gap = graph_below_uniform or graph_residual_gap
                            cp_alpha_thresh = (
                                float(gate_alpha) * reveal_frac
                                + (1.0 - float(gate_alpha)) * (1.0 - residual_conf)
                            )
                            cp_alpha = (residual_count > 0
                                        and graph_coverage < cp_alpha_thresh)
                            residual_risk = residual_below_block and progress_below_expected
                            candidate_advantage = graph_candidate_advantage
                            gate_values = {
                                "progress_empty": progress_empty,
                                "progress_below_quota": progress_below_quota,
                                "progress_below_expected": progress_below_expected,
                                "progress_stalled": progress_stalled,
                                "graph_below_uniform": graph_below_uniform,
                                "graph_below_progress": graph_below_progress,
                                "graph_residual_gap": graph_residual_gap,
                                "graph_candidate_advantage": graph_candidate_advantage,
                                "readiness_lt_one": readiness_lt_one,
                                "residual_below_block": residual_below_block,
                                "residual_below_revealed": residual_below_revealed,
                                "dispersion_dominates": dispersion_dominates,
                                "progress_or_graph": progress_or_graph,
                                "progress_and_graph": progress_and_graph,
                                "readiness_or_graph": readiness_or_graph,
                                "readiness_and_progress": readiness_and_progress,
                                "dawn_weak_certificate": dawn_weak_certificate,
                                "two_step_progress_stall": two_step_progress_stall,
                                "two_step_graph_gap": two_step_graph_gap,
                                "two_step_hybrid_stuck": two_step_hybrid_stuck,
                                "progress_stall": two_step_progress_stall,
                                "coverage_gap": coverage_gap,
                                "residual_risk": residual_risk,
                                "readiness_drop": readiness_drop,
                                "candidate_advantage": candidate_advantage,
                                "adaptive_coverage": coverage_gap,
                                "unified_axon": bool(coverage_gap and progress_stalled),
                                "cp_alpha": cp_alpha,
                            }
                            if axon_adaptive_gate_name == "adaptive_coverage":
                                # Label-free gate selector. coverage_gap's
                                # in-situ fire rate is accumulated GLOBALLY
                                # across the eval (per-generation windows are
                                # far too short to estimate it). For the first
                                # `adcov_warmup` gate-steps we run coverage_gap
                                # and measure; once enough steps are observed
                                # the choice locks for the rest of the eval:
                                # if the global fire rate >= adcov_threshold
                                # the base over-fires -> switch to
                                # progress_stall, else keep coverage_gap.
                                if not hasattr(self, "_adcov_steps"):
                                    self._adcov_steps = 0
                                    self._adcov_fire = 0
                                    self._adcov_choice = None
                                self._adcov_steps += 1
                                if bool(coverage_gap):
                                    self._adcov_fire += 1
                                if self._adcov_choice is None and self._adcov_steps >= int(adcov_warmup):
                                    self._adcov_choice = (
                                        "progress_stall"
                                        if (self._adcov_fire / max(1, self._adcov_steps)) >= float(adcov_threshold)
                                        else "coverage_gap"
                                    )
                                if self._adcov_choice == "progress_stall":
                                    need_axon = bool(two_step_progress_stall)
                                else:
                                    need_axon = bool(coverage_gap)
                            else:
                                need_axon = bool(gate_values[axon_adaptive_gate_name])
                            axon_step_record = {
                                "step_idx": int(num_block * steps_per_block + i),
                                "block_idx": int(num_block),
                                "step_in_block": int(i),
                                "gate": axon_adaptive_gate_name,
                                "selector": axon_adaptive_selector_name,
                                "need_axon": int(need_axon),
                                "can_anchor": int(can_anchor),
                                "masked_count": int(masked_count),
                                "dawn_count": int(dawn_count),
                                "dawn_reveal_frac": float(reveal_frac),
                                "expected_count": int(expected_count),
                                "expected_progress": float(expected_progress),
                                "graph_coverage": float(graph_coverage),
                                "block_conf": float(block_conf),
                                "residual_conf": float(residual_conf),
                                "revealed_conf": float(revealed_conf),
                                "selected_count": 0,
                            }
                            prev_dawn_progress = reveal_frac
                            prev_graph_coverage = graph_coverage
                            prev_readiness = readiness
                            prev_hybrid_bad = hybrid_bad
                        elif axon_plugin_name == "needed_fixed1":
                            if axon_rule_name == "avg_conf":
                                need_axon = block_conf_need
                            elif axon_rule_name == "high_conf_frac":
                                need_axon = high_conf_frac_need
                            elif axon_rule_name == "dawn_reveal":
                                need_axon = low_reveal_need and residual_conf_need
                            elif axon_rule_name == "graph":
                                need_axon = graph_gap_need and residual_conf_need
                            elif axon_rule_name == "or_mild":
                                need_axon = block_conf_need or (low_reveal_need and graph_gap_need)
                            elif axon_rule_name == "and_strict":
                                need_axon = block_conf_need and (low_reveal_need or graph_gap_need)
                            elif axon_rule_name == "delayed_graph":
                                need_axon = i >= int(axon_need_min_step) and (block_conf_need or graph_gap_need)
                            elif axon_rule_name == "stuck2":
                                bad_step = block_conf_need or low_reveal_need
                                axon_bad_streak_used = axon_bad_streak_used + 1 if bad_step else 0
                                need_axon = axon_bad_streak_used >= int(axon_need_bad_streak)
                            elif axon_rule_name == "readiness":
                                need_axon = readiness_need
                            else:
                                need_axon = False
                            if axon_rule_name != "stuck2":
                                axon_bad_streak_used = 0
                        elif axon_plugin_name == "adaptive_minimal":
                            need_axon = residual_conf_need and (low_reveal_need or graph_gap_need)
                        elif axon_need_mode == "low_reveal":
                            need_axon = low_reveal_need
                        elif axon_need_mode == "residual_conf":
                            need_axon = residual_conf_need
                        elif axon_need_mode == "graph_gap":
                            need_axon = graph_gap_need
                        else:
                            need_axon = low_reveal_need or graph_gap_need

                        if (not need_axon) or (not can_anchor):
                            use_dawn = True
                        else:
                            use_dawn = False

                    if use_dawn:
                        transfer_index = dawn_transfer_index if dawn_transfer_index is not None else _dream_get_base_proposal(axon_base_proposer_name)
                        if axon_step_record is not None:
                            self._last_axon_step_records.append(axon_step_record)
                    elif axon_plugin_name in {"adaptive_gate", "stag_unified"}:
                        if _prof is not None:
                            _prof.submod_begin()
                        transfer_index = select_axon_adaptive_gate_anchors(
                            mask_index=mask_index,
                            confidence=x0_p,
                            avg_attn_scores=avg_attn_scores,
                            tau_sink=tau_sink,
                            candidate_topk=candidate_topk,
                            candidate_min_topk=candidate_min_topk,
                            candidate_ratio=candidate_ratio,
                            min_select=min_select,
                            selector=axon_adaptive_selector_name,
                            dawn_density=dawn_density,
                            target_coverage=dawn_raw_coverage,
                            conflict_threshold=conflict_threshold,
                            axon_submod_fn=axon_submod_fn,
                            axon_submod_monotone=axon_submod_monotone,
                            axon_submod_penalty=axon_submod_penalty,
                            axon_submod_lambda=axon_submod_lambda,
                        )
                        if _prof is not None:
                            _prof.submod_end()
                        self._last_axon_submodular_calls += 1
                        # --- Experiment H: log anchor confidence diagnostics
                        #     (before any confidence guard is applied) ---
                        pre_guard_anchor_mask = transfer_index & mask_index
                        pre_guard_count = int(pre_guard_anchor_mask.sum().item())
                        if pre_guard_count > 0:
                            anchor_confs = x0_p[pre_guard_anchor_mask].float()
                            anchor_mean_conf = float(anchor_confs.mean().item())
                            anchor_min_conf = float(anchor_confs.min().item())
                            anchor_max_conf = float(anchor_confs.max().item())
                        else:
                            anchor_mean_conf = anchor_min_conf = anchor_max_conf = -1.0
                        # --- Experiment I: confidence-gated anchor commit ---
                        # Only keep anchor positions whose own confidence clears
                        # unlock_anchor_min_conf. If the guard removes everything,
                        # the step commits nothing extra (defers to base via the
                        # augmenting union / dawn_transfer_index downstream).
                        if float(unlock_anchor_min_conf) > 0.0:
                            conf_ok = x0_p >= float(unlock_anchor_min_conf)
                            transfer_index = transfer_index & conf_ok
                        selected_count = int((transfer_index & mask_index).sum().item())
                        self._last_axon_submodular_anchors += selected_count
                        if axon_step_record is not None:
                            axon_step_record["selected_count"] = selected_count
                            axon_step_record["anchor_count_pre_guard"] = pre_guard_count
                            axon_step_record["anchor_mean_conf"] = anchor_mean_conf
                            axon_step_record["anchor_min_conf"] = anchor_min_conf
                            axon_step_record["anchor_max_conf"] = anchor_max_conf
                            self._last_axon_step_records.append(axon_step_record)
                        axon_anchor_steps_used += 1
                    elif axon_plugin_name == "adaptive_minimal":
                        axon_mask_index = mask_index
                        if dawn_transfer_index is not None:
                            axon_mask_index = mask_index & ~dawn_transfer_index
                        transfer_index = select_axon_adaptive_minimal_anchors(
                            mask_index=axon_mask_index,
                            confidence=x0_p,
                            avg_attn_scores=avg_attn_scores,
                            tau_sink=tau_sink,
                            candidate_topk=candidate_topk,
                            candidate_min_topk=candidate_min_topk,
                            candidate_ratio=candidate_ratio,
                            min_select=min_select,
                            conflict_threshold=conflict_threshold,
                            axon_cover_target=axon_cover_target,
                            axon_max_reveal_frac=axon_max_reveal_frac,
                        )
                        if dawn_transfer_index is not None:
                            transfer_index = transfer_index | dawn_transfer_index
                        axon_anchor_steps_used += 1
                    else:
                        transfer_index = select_axon_submodular_anchors(
                            mask_index=mask_index,
                            confidence=x0_p,
                            avg_attn_scores=avg_attn_scores,
                            tau_sink=tau_sink,
                            candidate_topk=candidate_topk,
                            candidate_min_topk=candidate_min_topk,
                            candidate_ratio=candidate_ratio,
                            min_select=min_select,
                            step_in_block=axon_anchor_steps_used,
                            target_nfe_per_block=unlock_target_nfe_per_block,
                            block_length=block_length,
                            unlock_lambda=unlock_lambda,
                            unlock_gamma=unlock_gamma,
                            conflict_threshold=conflict_threshold,
                            axon_reveal_policy=axon_reveal_policy,
                            axon_max_anchors_per_step=axon_max_anchors_per_step,
                            axon_anchor_size_penalty=axon_anchor_size_penalty,
                            axon_anchor_size_penalty_power=axon_anchor_size_penalty_power,
                        )
                        self._last_axon_submodular_calls += 1
                        self._last_axon_submodular_anchors += int((transfer_index & mask_index).sum().item())
                        axon_anchor_steps_used += 1

                    # Augmenting step body: when AXON fires, optionally add a
                    # stage-2 fill of high-confidence masked tokens AND union
                    # with the DAWN proposal so the step body matches LLaDA's.
                    # Skipped when use_dawn (already a DAWN step) and gated by
                    # axon_step_mode == "augmenting" so legacy "anchor_only"
                    # behaviour is preserved.
                    if (not use_dawn) and axon_step_mode_name == "augmenting":
                        if bool(unlock_stage2_fill):
                            stage2_select_cap = _quota_for_target_nfe(
                                step_in_block=axon_anchor_steps_used,
                                num_masked=int(mask_index.sum().item()),
                                block_length=block_length,
                                target_nfe=unlock_target_nfe_per_block,
                                min_select=min_select,
                            )
                            fill_index = select_high_confidence_fill(
                                mask_index=mask_index,
                                confidence=x0_p,
                                already_selected=transfer_index,
                                candidate_topk=candidate_topk,
                                candidate_min_topk=candidate_min_topk,
                                candidate_ratio=candidate_ratio,
                                fill_ratio=float(unlock_fill_ratio),
                                select_cap=int(stage2_select_cap),
                                high_conf_threshold=float(unlock_high_conf_threshold),
                            )
                            transfer_index = transfer_index | fill_index
                        if axon_plugin_name != "adaptive_minimal":
                            if dawn_transfer_index is None:
                                dawn_transfer_index = _dream_get_base_proposal(axon_base_proposer_name)
                            # union_dawn_keep_frac: keep only the top
                            # fraction (by confidence) of the decoder's
                            # proposed tokens before unioning with the
                            # anchor. 1.0 = full union (default, no change).
                            _keep = float(union_dawn_keep_frac)
                            if 0.0 < _keep < 1.0:
                                _dk = torch.zeros_like(dawn_transfer_index)
                                for _b in range(dawn_transfer_index.shape[0]):
                                    _idx = dawn_transfer_index[_b].nonzero(as_tuple=True)[0]
                                    _c = int(_idx.numel())
                                    if _c == 0:
                                        continue
                                    _k = max(1, int(_c * _keep + 0.5))
                                    if _k >= _c:
                                        _dk[_b, _idx] = True
                                    else:
                                        _cv = x0_p[_b].index_select(0, _idx)
                                        _top = torch.topk(_cv, k=_k).indices
                                        _dk[_b, _idx[_top]] = True
                                dawn_transfer_index = _dk
                            transfer_index = transfer_index | dawn_transfer_index

                    if _prof is not None:
                        _prof.note_reveal(transfer_index, num_block, i)
                    x[transfer_index] = x0[transfer_index].clone()
                elif alg == 'klass':
                    p_curr_masked = torch.softmax(mask_logits, dim=-1).to(p_prev.dtype)  # [num_masked, V]
                    x0_masked = torch.argmax(p_curr_masked, dim=-1)  # [num_masked]
                    curr_conf_masked = torch.gather(p_curr_masked, -1, x0_masked.unsqueeze(-1)).squeeze(-1)  # [num_masked]
                    x0 = torch.zeros_like(x, dtype=torch.long)

                    x0[mask_index] = x0_masked

                    p_curr = F.softmax(logits.to(torch.float64), dim=-1)
                    curr_conf = torch.gather(p_curr, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    eps = 1e-12
                    kl_current_prev = (p_curr * (torch.log(p_curr + eps) - torch.log(p_prev + eps))).sum(dim=-1)
                    kl_history = torch.roll(kl_history, shifts=-1, dims=-1)
                    kl_history[..., -1] = kl_current_prev
                    p_prev.copy_(p_curr)
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                    conf_mask = (curr_conf >= conf_threshold)
                    ready_mask = stable_mask & conf_mask & mask_index

                    transfer_index = torch.zeros_like(mask_index)
                    ready_indices = torch.where(ready_mask)

                    # Handle cases where no ready indices are found
                    no_ready_indices = torch.where(~ready_mask.any(dim=1))[0]
                    if no_ready_indices.numel() > 0:
                        conf_fb, x0_fb = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                        chosen = torch.argmax(conf_fb, dim=-1)
                        global_idxs = torch.where(mask_index[no_ready_indices])
                        idx = global_idxs[1][chosen]
                        transfer_index[no_ready_indices, idx] = True
                    else:
                        # unmask strategy
                        batch_size = x.size(0)
                        for j in range(batch_size):
                            ready_j = torch.where(ready_mask[j])[0]
                            if len(ready_j) <= 1:
                                selected_indices = ready_j
                            elif unmask_strategy == "all":
                                selected_indices = ready_j
                            elif unmask_strategy == "random":
                                idx = torch.randint(0, len(ready_j), (1,))
                                selected_indices = ready_j[idx]
                            elif unmask_strategy == "max_conf":
                                conf_vals = curr_conf[j, ready_j]
                                max_idx = torch.argmax(conf_vals)
                                selected_indices = ready_j[max_idx:max_idx+1]
                            elif unmask_strategy == "min_kl":
                                kl_vals = kl_current_prev[j, ready_j]
                                min_idx = torch.argmin(kl_vals)
                                selected_indices = ready_j[min_idx:min_idx+1]
                            else:
                                selected_indices = ready_j
                            transfer_index[j, selected_indices] = True
                    x0_full = torch.zeros_like(x)
                    x0_full[mask_index] = x0_masked
                    x[transfer_index] = x0_full[transfer_index].clone()
                elif alg == 'local_leap':
                    full_confidence, x_ = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    full_confidence = torch.where(mask_index, full_confidence, -np.inf)
                    current_transfer_tokens = mask_index.sum()
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    transfer_index = self.get_transfer_index_localleap(
                                transfer_index, full_confidence, select_index, mask_index, x, logits, current_transfer_tokens, 
                                conf_threshold, relaxed_threshold, radius
                            )
                    x[transfer_index] = x_[transfer_index].clone()
                else:
                    if alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    elif alg == 'topk_margin':
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        x[row_indices,transfer_index] = x_[row_indices,transfer_index]

                # this allows user-defined token control of the intermediate steps
                x = generation_tokens_hook_func(i, x, logits)

                if histories is not None:
                    histories.append(x.clone())
                if _prof is not None:
                    # base arms (alg != 'axon') never hit the axon-commit note_reveal at :1590;
                    # derive their per-step reveal count from the drop in masked positions.
                    if alg != 'axon':
                        _ma = (x == mask_token_id)
                        _ma[:, input_ids.shape[1] + (num_block + 1) * block_length:] = 0
                        _prof.note_reveal(_masked0 - _ma.sum(), num_block, i)
                    _prof.step_end()
                i += 1
        
        print(f'nfe: {nfe}')
        end_time = time.time()
        print(f'used time: {end_time - start_time}')
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            ), nfe
        else:
            return x, nfe

    def get_transfer_index_localleap(self, transfer_index, full_confidence, select_index, mask_index_block, 
        block_x, block_logits, current_transfer_tokens, threshold=None, relaxed_threshold=None, radius=4):
        mask_positions = torch.where(mask_index_block[0])[0]
        
        if len(mask_positions) == 0:
            return transfer_index
        
        use_localleap = False
        neighbor_positions = set()

        mask_confidences = full_confidence[0][mask_positions]
        anchor_mask = mask_confidences >= threshold
        anchor_count = anchor_mask.sum().item()
        
        if relaxed_threshold is not None and anchor_count >= 1:
            use_localleap = True
            anchor_positions = mask_positions[anchor_mask]
            
            for pos in anchor_positions:
                pos_val = pos.item()
                for neignbor_pos in range(max(0, pos_val - radius), min(full_confidence.shape[1], pos_val + radius + 1)):
                    neighbor_positions.add(neignbor_pos)
             
        for k in range(1, current_transfer_tokens):
            if k >= len(select_index[0]):
                break

            pos = select_index[0, k].item()
            if use_localleap:
                effective_threshold = relaxed_threshold if pos in neighbor_positions else threshold
            else:
                effective_threshold = threshold
            
            if full_confidence[0, select_index[0, k]] < effective_threshold:
                transfer_index[0, select_index[0, k]] = False
        
        return transfer_index
