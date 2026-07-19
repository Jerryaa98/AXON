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
#
# SDAR port of the AXON/AXON submodular block-diffusion decode.
#
# This mirrors `llada/generate.py`:
#   * the DAWN / LocalLeap / confidence base-proposer kernels and the
#     `_get_base_proposal` dispatcher are ported verbatim from LLaDA;
#   * the `axon_plugin in {stag_unified, adaptive_gate}` gate (coverage_gap /
#     adaptive_coverage / cp_alpha, `_gate_values` / `_need_axon`) and the
#     `select_axon_adaptive_gate_anchors(...)` selector call are ported from
#     LLaDA's `elif unlock:` block (llada/generate.py:443-810);
#   * the outer loop follows SDAR's `block_diffusion_generate()` structure
#     (block loop -> per-block denoising steps -> `mask_index==mask_id` ->
#     logits -> confidence -> transfer), and the `axon_plugin=='none'` baseline
#     reproduces SDAR's native `low_confidence_dynamic` remasking.
#
# The one structural difference from SDAR's reference decode: we run the model
# on the FULL sequence each step (LLaDA-style, no KV cache), because the AXON
# machinery consumes an [B, L, L] attention graph over absolute positions
# (indexed with `prompt_length + num_block*block_length`). SDAR's shipped
# attention module returns None for attention weights, so `avg_attn_scores` is
# built by an opt-in eager-attention monkeypatch (`enable_attention_capture`).

import importlib

import numpy as np
import torch
import torch.nn.functional as F

from gdllm_utils import *  # noqa: F401,F403  (select_axon_adaptive_gate_anchors, detect_attn_sinks_, select_parallel_tokens_conflict_mis, ...)


# =============================================================================
# SDAR eager-attention capture
# =============================================================================
#
# SDAR (JetLM/SDAR-8B-Chat) ships a custom `SDARAttention.forward` that always
# routes through flash-attn / SDPA and returns `attn_output, None` — it ignores
# `output_attentions` entirely, so `outputs.attentions` is a tuple of `None`s
# and `attn_implementation="eager"` alone does NOT surface attention maps.
#
# `enable_attention_capture(model)` installs an eager attention path on each
# `self_attn` module (faithfully reproducing the module's own q_norm/k_norm +
# RoPE + GQA-repeat math). When `model._sdar_want_attn` is set, that path
# computes softmax(QK^T) weights, reduces them to the per-position head-mean
# [B, L, L] and accumulates them into `model._sdar_attn_accum` (summed over
# layers). We never materialise all layers/heads at once — each layer's
# [B, H, L, L] tensor is reduced to [B, L, L] immediately and dropped — so the
# memory cost is O(L^2), not O(layers * heads * L^2).
def enable_attention_capture(model):
    """Idempotently install the eager attention-weight capture hooks."""
    if getattr(model, "_sdar_attn_capture_installed", False):
        return

    layers = model.model.layers
    _mod = importlib.import_module(type(layers[0].self_attn).__module__)
    apply_rotary_pos_emb = _mod.apply_rotary_pos_emb
    repeat_kv = _mod.repeat_kv

    def _make_patched(attn, orig_forward):
        def patched(*args, **kwargs):
            # Fast path: model doesn't want attention this step -> original
            # (flash / SDPA) forward, which is faster and returns (out, None).
            if not getattr(model, "_sdar_want_attn", False):
                return orig_forward(*args, **kwargs)

            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            attention_mask = kwargs.get("attention_mask", None)
            position_embeddings = kwargs.get("position_embeddings", None)

            input_shape = hidden_states.shape[:-1]  # (B, L)
            hidden_shape = (*input_shape, -1, attn.head_dim)

            query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn.scaling
            if attention_mask is not None:
                mask = attention_mask[:, :, :, : key_states.shape[-2]]
                if mask.dtype == torch.bool:
                    neg = torch.finfo(attn_weights.dtype).min
                    attn_weights = attn_weights.masked_fill(~mask, neg)
                else:
                    attn_weights = attn_weights + mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

            # Reduce to per-position head-mean and accumulate across layers.
            head_mean = attn_weights.mean(dim=1).detach()  # [B, L, L] float32
            accum = getattr(model, "_sdar_attn_accum", None)
            if accum is None:
                model._sdar_attn_accum = head_mean.clone()
            else:
                accum.add_(head_mean)
            model._sdar_attn_layers = getattr(model, "_sdar_attn_layers", 0) + 1

            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_output = attn.o_proj(attn_output)
            # Return None for weights: the buffer already holds them, and we run
            # the model with output_attentions=False so HF won't try to collect.
            return attn_output, None

        return patched

    for layer in layers:
        attn = layer.self_attn
        attn.forward = _make_patched(attn, attn.forward)

    model._sdar_attn_capture_installed = True
    model._sdar_want_attn = False
    model._sdar_attn_accum = None
    model._sdar_attn_layers = 0


def _build_block_diffusion_mask(prompt_length, gen_length, block_length, device):
    """Semi-autoregressive (block-diffusion) attention mask, [1, 1, L, L] bool.

    Mirrors SDAR's block-causal structure but aligned to `prompt_length` (so it
    matches the AXON code's block indexing `prompt_length + b*block_length`):
      * the prompt is bidirectionally visible to itself and to every gen token;
      * gen block b attends to the prompt + gen blocks 0..b (causal across
        blocks, bidirectional within its own block).
    """
    L = prompt_length + gen_length
    m = torch.zeros((L, L), dtype=torch.bool, device=device)
    if prompt_length > 0:
        m[:prompt_length, :prompt_length] = True  # prompt <-> prompt
        m[prompt_length:, :prompt_length] = True  # gen -> prompt
    num_gen_blocks = gen_length // block_length
    for b in range(num_gen_blocks):
        bs = prompt_length + b * block_length
        be = bs + block_length
        m[bs:be, prompt_length:be] = True  # gen block b -> prompt..(<=block b)
    return m.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]


def _sdar_forward(model, x, attn_mask, position_ids, need_attn):
    """Full-sequence SDAR forward. Returns (logits, avg_attn_scores|None).

    avg_attn_scores is [B, L, L], the attention averaged over heads and layers.
    """
    if need_attn:
        model._sdar_want_attn = True
        model._sdar_attn_accum = None
        model._sdar_attn_layers = 0
        out = model(
            input_ids=x,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        num_layers = max(1, int(getattr(model, "_sdar_attn_layers", 0)))
        if getattr(model, "_sdar_attn_accum", None) is None:
            raise RuntimeError(
                "SDAR attention capture produced no weights. Call "
                "enable_attention_capture(model) before generate() when a "
                "DAWN proposer / graph-signal gate is active."
            )
        avg_attn_scores = model._sdar_attn_accum / float(num_layers)
        model._sdar_want_attn = False
        model._sdar_attn_accum = None
        model._sdar_attn_layers = 0
        return out.logits, avg_attn_scores

    out = model(
        input_ids=x,
        attention_mask=attn_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    return out.logits, None


# =============================================================================
# Ported LLaDA kernels (verbatim behaviour)
# =============================================================================
def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling helper (float64). No-op when temperature == 0."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """(B, L) bool mask -> (B, steps) int schedule of tokens to reveal per step."""
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps

    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)
    cols = torch.arange(steps, device=device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)
    return num_transfer_tokens


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    """LLaDA confidence base proposer. Returns (x0, transfer_index)."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | force_mask
        transfer_index = transfer_index & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    values, idx = torch.sort(confidence, dim=1, descending=True)
    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index


def get_transfer_index_dawn(logits, temperature, remasking, mask_index, x, num_transfer_tokens, avg_attn_scores,
                            tau_sink=0.01, tau_edge=0.07, tau_induce=0.7, tau_low=0.7,
                            num_block=0, block_length=32, prompt_length=None):
    """DAWN (attention-graph) base proposer. Needs avg_attn_scores [B, L, L]."""
    assert avg_attn_scores is not None, "avg_attn_scores is None"
    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)
    key_sink_mask = sink_mask.unsqueeze(1)
    avg_attn_scores = avg_attn_scores.masked_fill(key_sink_mask, 0.0)

    B, _, _ = avg_attn_scores.shape
    avg_attn_scores.diagonal(dim1=1, dim2=2).zero_()

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    quantile_mask = avg_attn_scores >= tau_edge
    quantile_mask = quantile_mask.transpose(1, 2)

    confidence = torch.where(mask_index, x0_p, -np.inf)
    transfer_index_conf = confidence >= 0.9

    decoded_mask = (~mask_index & (x0_p >= 0.9)).unsqueeze(-1)
    decoded_mask[:, prompt_length + (num_block + 1) * block_length:] = False
    decoded_edge = quantile_mask & decoded_mask
    dependent_nodes = decoded_edge.any(dim=1) & mask_index
    conf_d = torch.where(dependent_nodes, x0_p, -np.inf)
    transfer_index_a = conf_d >= tau_induce

    adj_ti_mask = quantile_mask & transfer_index_a.unsqueeze(-1) & transfer_index_conf.unsqueeze(-1)
    adj_ti_mask = adj_ti_mask.any(dim=1)
    node_mask = mask_index & (x0_p >= tau_low) & ~transfer_index_conf & ~transfer_index_a & ~adj_ti_mask
    transfer_index_c = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    if node_mask.sum(dim=-1).min().item() != 0:
        _node_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        edge_mask = _node_mask & quantile_mask

        confidence = torch.where(node_mask, x0_p, -np.inf)
        for j in range(B):
            select_index = select_parallel_tokens_conflict_mis(edge_mask[j], node_mask[j], confidence[j])
            transfer_index_c[j, select_index] = True

    transfer_index = transfer_index_a | transfer_index_conf | transfer_index_c

    if transfer_index.sum(dim=-1).min().item() == 0:
        confidence = torch.where(mask_index, x0_p, -np.inf)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | force_mask

    return x0, transfer_index


def get_transfer_index_localleap(logits, temperature, remasking, mask_index, x, num_transfer_tokens,
                                 threshold=0.9, relaxed_threshold=0.75, radius=4):
    """LocalLeap base proposer (KLASS-style local expansion)."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            mask_positions = torch.where(mask_index[j])[0]

            neighbor_positions = set()
            use_localleap = False

            if relaxed_threshold is not None:
                anchor_mask = confidence[j][mask_positions] >= threshold
                anchor_count = anchor_mask.sum().item()

                if anchor_count >= 1:
                    use_localleap = True
                    anchor_positions = mask_positions[anchor_mask]
                    for pos in anchor_positions:
                        pos_val = pos.item()
                        for neignbor_pos in range(max(0, pos_val - radius), min(confidence.shape[1], pos_val + radius + 1)):
                            neighbor_positions.add(neignbor_pos)

            for k in range(1, num_transfer_tokens[j]):
                pos = select_index[k].item()
                if use_localleap:
                    effective_threshold = relaxed_threshold if pos in neighbor_positions else threshold
                else:
                    effective_threshold = threshold

                if confidence[j, select_index[k]] < effective_threshold:
                    transfer_index[j, select_index[k]] = False

    return x0, transfer_index


# =============================================================================
# AXON base-proposer dispatch (ported from llada/generate.py; klass dropped)
# =============================================================================
def _get_base_proposal_local_leap(logits, temperature, remasking, mask_index, x,
                                   threshold, relaxed_threshold, radius):
    return get_transfer_index_localleap(
        logits, temperature, remasking, mask_index, x, None,
        threshold=threshold, relaxed_threshold=relaxed_threshold, radius=radius,
    )


def _get_base_proposal_confidence(logits, temperature, remasking, mask_index, x,
                                   num_transfer_tokens_step, threshold):
    return get_transfer_index(
        logits, temperature, remasking, mask_index, x,
        num_transfer_tokens_step if threshold is None else None,
        threshold,
    )


def _get_base_proposal(axon_base_proposer, *,
                       logits, temperature, remasking, mask_index, x, avg_attn_scores,
                       tau_sink, tau_edge, tau_induce, tau_low,
                       num_block, block_length, prompt_length,
                       ll_threshold=0.9, ll_relaxed_threshold=0.8, ll_radius=4,
                       conf_num_transfer_tokens=None, conf_step=None, conf_threshold_param=None):
    """Dispatch to the requested base decoder. Returns (x0, transfer_index)."""
    if axon_base_proposer == "dawn":
        return get_transfer_index_dawn(
            logits, temperature, remasking, mask_index, x, None, avg_attn_scores,
            tau_sink=tau_sink, tau_edge=tau_edge, tau_induce=tau_induce, tau_low=tau_low,
            num_block=num_block, block_length=block_length, prompt_length=prompt_length,
        )
    if axon_base_proposer == "local_leap":
        return _get_base_proposal_local_leap(
            logits, temperature, remasking, mask_index, x,
            ll_threshold, ll_relaxed_threshold, ll_radius,
        )
    if axon_base_proposer == "confidence":
        # Match LLaDA: default the reveal threshold to 0.9 (the canonical
        # "parallel" baseline) so the step-schedule fallback isn't triggered.
        cf_threshold = conf_threshold_param if conf_threshold_param is not None else 0.9
        return _get_base_proposal_confidence(
            logits, temperature, remasking, mask_index, x, None, cf_threshold,
        )
    raise ValueError(f"Unsupported axon_base_proposer: {axon_base_proposer}")


# =============================================================================
# SDAR AXON-aware block-diffusion decode
# =============================================================================
@torch.no_grad()
def generate(model, tokenizer, prompt_ids,
             gen_length=64, block_length=16, steps=64,
             mask_id=151669, temperature=0.0, remasking="low_confidence",
             confidence_threshold=0.85, threshold=None,
             # ---- AXON plugin / gate / selector / base proposer ----
             axon_plugin="none",
             axon_stag_gate="coverage_gap", axon_stag_selector="fixed1",
             axon_adaptive_gate="progress_empty", axon_adaptive_selector="fixed1",
             axon_base_proposer="dawn",
             gate_alpha=0.5, adcov_threshold=0.5, adcov_warmup=256,
             # ---- DAWN / LocalLeap graph-signal thresholds ----
             tau_sink=0.01, tau_edge=0.07, tau_induce=0.7, tau_low=0.7,
             relaxed_threshold=0.75, radius=4,
             # ---- submodular anchor selector knobs ----
             candidate_topk=64, candidate_min_topk=16, candidate_ratio=0.25,
             min_select=1, conflict_threshold=0.05,
             axon_beta_r=1.0, axon_beta_u=1.0, axon_fixed_k=0,
             axon_submod_fn="facility_location", axon_submod_monotone=True,
             axon_submod_penalty="conflict", axon_submod_lambda=0.0,
             # ---- anchor-step (submodular-exit) control ----
             unlock_target_nfe_per_block=8,
             unlock_min_anchor_steps=1, unlock_max_anchor_steps=1,
             step_profiler=None,
             **kwargs):
    """AXON/AXON block-diffusion decode for SDAR.

    Returns (output_ids [B, L], nfe, stats_dict) where stats_dict carries
    'axon_submod_calls' and 'axon_submod_anchors'.
    """
    device = model.device
    prompt_ids = prompt_ids.to(device)
    B = prompt_ids.shape[0]
    prompt_length = prompt_ids.shape[1]

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    L = prompt_length + gen_length
    x = torch.full((B, L), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt_ids.clone()

    attn_mask = _build_block_diffusion_mask(prompt_length, gen_length, block_length, device).expand(B, 1, L, L)
    position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    # ---- normalise / validate AXON knobs (mirror LLaDA) ----
    axon_plugin_name = str(axon_plugin).strip().lower()
    if axon_plugin_name in {"", "false", "0"}:
        axon_plugin_name = "none"
    if axon_plugin_name not in {"none", "adaptive_gate", "stag_unified"}:
        raise ValueError(f"Unsupported AXON plugin: {axon_plugin}")

    axon_base_proposer_name = str(axon_base_proposer).strip().lower()
    if axon_base_proposer_name in {"", "false", "0"}:
        axon_base_proposer_name = "dawn"
    if axon_base_proposer_name not in {"dawn", "local_leap", "confidence"}:
        raise ValueError(f"Unsupported axon_base_proposer: {axon_base_proposer}")

    if axon_plugin_name == "stag_unified":
        axon_gate_name = str(axon_stag_gate).strip().lower()
        axon_selector_name = str(axon_stag_selector).strip().lower()
    else:
        axon_gate_name = str(axon_adaptive_gate).strip().lower()
        axon_selector_name = str(axon_adaptive_selector).strip().lower()
    if axon_gate_name in {"", "false", "0"}:
        axon_gate_name = "progress_empty"
    if axon_selector_name in {"", "false", "0"}:
        axon_selector_name = "fixed1"

    _min_anchor_steps = max(0, int(unlock_min_anchor_steps))
    _max_anchor_steps = max(_min_anchor_steps, int(unlock_max_anchor_steps))

    nfe = 0
    stats = {"axon_submod_calls": 0, "axon_submod_anchors": 0}
    use_axon = axon_plugin_name in {"adaptive_gate", "stag_unified"}

    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # per-block AXON gate state (mirror llada/generate.py:259-266)
        prev_dawn_progress = None
        prev_graph_coverage = None
        prev_readiness = None
        prev_hybrid_bad = False
        axon_anchor_steps_used = 0

        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)

            gate_active = use_axon and (axon_anchor_steps_used < _max_anchor_steps)
            if not use_axon:
                need_attn = False
            else:
                need_attn = gate_active or (axon_base_proposer_name == "dawn")

            if step_profiler is not None:
                step_profiler.step_begin(need_attn)
            logits, avg_attn_scores = _sdar_forward(model, x, attn_mask, position_ids, need_attn)
            if step_profiler is not None:
                step_profiler.forward_end()
            mask_index[:, block_end:] = 0  # only current+past blocks are selectable

            # ---------------- Plain baseline: SDAR low_confidence_dynamic -----
            if not use_axon:
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                step = min(i, steps_per_block - 1)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[j, step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[j, step])
                        transfer_index[j, idx] = True
                transfer_index = transfer_index & mask_index
                if step_profiler is not None:
                    step_profiler.commit(transfer_index, num_block, i)
                x[transfer_index] = x0[transfer_index]
                i += 1
                if (x[:, block_start:block_end] == mask_id).sum() == 0:
                    break
                continue

            # ---------------- AXON gate (mirror llada/generate.py:443-685) ----
            _use_dawn = axon_anchor_steps_used >= _max_anchor_steps
            need_gate_dawn_x0 = None
            need_gate_dawn_transfer_index = None
            stag_dawn_density = 0.0
            stag_target_coverage = 0.0

            if gate_active and avg_attn_scores is not None:
                _can_anchor = axon_anchor_steps_used < _max_anchor_steps
                if step_profiler is not None:
                    step_profiler.base_begin()
                need_gate_dawn_x0, need_gate_dawn_transfer_index = _get_base_proposal(
                    axon_base_proposer_name,
                    logits=logits, temperature=temperature, remasking=remasking,
                    mask_index=mask_index, x=x, avg_attn_scores=avg_attn_scores,
                    tau_sink=tau_sink, tau_edge=tau_edge, tau_induce=tau_induce, tau_low=tau_low,
                    num_block=num_block, block_length=block_length, prompt_length=prompt_length,
                    ll_threshold=threshold if threshold is not None else 0.9,
                    ll_relaxed_threshold=relaxed_threshold, ll_radius=radius,
                    conf_num_transfer_tokens=num_transfer_tokens, conf_step=i,
                    conf_threshold_param=threshold,
                )
                if step_profiler is not None:
                    step_profiler.base_end()
                _bstart = block_start
                _bend = block_end
                _block_mask = mask_index[:, _bstart:_bend]
                _block_dawn = need_gate_dawn_transfer_index[:, _bstart:_bend] & _block_mask
                _masked_count = int(_block_mask.sum().item())
                _dawn_count = int(_block_dawn.sum().item())
                _reveal_frac = float(_dawn_count) / float(max(1, _masked_count))

                _bprobs = F.softmax(logits[:, _bstart:_bend].to(torch.float64), dim=-1)
                _block_conf = 1.0
                _block_conf_std = 0.0
                _high_conf_frac = 1.0
                _readiness = float("inf")
                if _masked_count > 0:
                    _block_conf_values = _bprobs.max(dim=-1).values[_block_mask]
                    _block_conf = float(_block_conf_values.mean().item())
                    _block_conf_std = float(_block_conf_values.std(unbiased=False).item())
                    _high_conf_frac = float((_block_conf_values >= 0.90).float().mean().item())
                    _readiness = (_block_conf * _high_conf_frac) / (_block_conf_std + 1e-3)

                _residual_mask = _block_mask & ~_block_dawn
                _residual_count = int(_residual_mask.sum().item())
                _residual_conf = 1.0
                _revealed_conf = 1.0
                if _dawn_count > 0:
                    _revealed_conf = float(_bprobs.max(dim=-1).values[_block_dawn].mean().item())
                if _residual_count > 0:
                    _residual_conf = _bprobs.max(dim=-1).values[_residual_mask].mean().item()

                _graph_coverage = 1.0
                _dawn_raw_coverage = 0.0
                if _residual_count > 0:
                    _row_coverages = []
                    _row_raw_coverages = []
                    for _row_idx in range(mask_index.shape[0]):
                        _residual_pos = torch.nonzero(_residual_mask[_row_idx], as_tuple=False).flatten() + _bstart
                        _anchor_pos = torch.nonzero(_block_dawn[_row_idx], as_tuple=False).flatten() + _bstart
                        if _residual_pos.numel() == 0:
                            _row_coverages.append(1.0)
                            _row_raw_coverages.append(0.0)
                        elif _anchor_pos.numel() == 0:
                            _row_coverages.append(0.0)
                            _row_raw_coverages.append(0.0)
                        else:
                            _block_pos = torch.arange(_bstart, _bend, device=mask_index.device)
                            _residual_attn = avg_attn_scores[_row_idx].index_select(0, _residual_pos)
                            _anchor_mass = _residual_attn.index_select(1, _anchor_pos).sum(dim=1)
                            _block_mass = _residual_attn.index_select(1, _block_pos).sum(dim=1)
                            _row_coverages.append(
                                float((_anchor_mass / (_block_mass + 1e-8)).clamp(0.0, 1.0).mean().item())
                            )
                            _residual_conf_values = _bprobs[_row_idx].max(dim=-1).values.index_select(0, _residual_pos - _bstart).float()
                            _anchor_conf_values = _bprobs[_row_idx].max(dim=-1).values.index_select(0, _anchor_pos - _bstart).float()
                            _raw_weights = (
                                _residual_attn.index_select(1, _anchor_pos).float()
                                * (1.0 - _residual_conf_values).clamp_min(0.0).unsqueeze(1)
                                * _anchor_conf_values.unsqueeze(0)
                            )
                            _row_raw_coverages.append(float(_raw_weights.max(dim=1).values.sum().item()))
                    _graph_coverage = sum(_row_coverages) / float(max(1, len(_row_coverages)))
                    _dawn_raw_coverage = sum(_row_raw_coverages) / float(max(1, len(_row_raw_coverages)))
                    stag_dawn_density = _dawn_raw_coverage / float(max(1, _dawn_count))
                    stag_target_coverage = _dawn_raw_coverage

                # ---- gate values (mirror llada/generate.py:537-649) ----
                _remaining_target_steps = max(1, int(np.ceil(float(unlock_target_nfe_per_block))) - int(i))
                _expected_count = int(np.ceil(float(_masked_count) / float(_remaining_target_steps))) if _masked_count > 0 else 0
                _expected_progress = float(_expected_count) / float(max(1, _masked_count))
                _progress_empty = _dawn_count == 0
                _progress_below_expected = _reveal_frac < _expected_progress
                _progress_below_quota = _dawn_count < _expected_count
                _progress_stalled = _progress_empty or _progress_below_expected
                _beta_r = float(axon_beta_r)
                _beta_u = float(axon_beta_u)
                _graph_below_uniform = _graph_coverage < _beta_r * _reveal_frac
                _graph_below_progress = _graph_coverage < _reveal_frac
                _graph_residual_gap = _residual_count > 0 and _graph_coverage < _beta_u * (1.0 - _residual_conf)
                _graph_candidate_advantage = _residual_count > 0 and _graph_coverage < _block_conf
                _readiness_lt_one = _readiness < 1.0
                _residual_below_block = _residual_count > 0 and _residual_conf < _block_conf
                _residual_below_revealed = _residual_count > 0 and _residual_conf < _revealed_conf
                _dispersion_dominates = _block_conf_std > max(0.0, _block_conf - _residual_conf)
                _progress_or_graph = _progress_below_expected or _graph_below_uniform
                _progress_and_graph = _progress_below_expected and _graph_below_uniform
                _readiness_or_graph = _readiness_lt_one or _graph_below_uniform
                _readiness_and_progress = _readiness_lt_one and _progress_below_expected
                _dawn_weak_certificate = (
                    _progress_below_expected
                    and _residual_below_revealed
                    and (_graph_below_uniform or _graph_residual_gap)
                )
                _two_step_progress_stall = (
                    prev_dawn_progress is not None
                    and _reveal_frac <= prev_dawn_progress
                    and _progress_below_expected
                )
                _readiness_drop = (
                    prev_readiness is not None
                    and _masked_count > 0
                    and _readiness < prev_readiness
                )
                _two_step_graph_gap = (
                    prev_graph_coverage is not None
                    and _graph_coverage <= prev_graph_coverage
                    and _graph_below_uniform
                )
                _hybrid_bad = _progress_or_graph or _readiness_lt_one
                _two_step_hybrid_stuck = prev_hybrid_bad and _hybrid_bad
                _coverage_gap = _graph_below_uniform or _graph_residual_gap
                _cp_alpha_thresh = (
                    float(gate_alpha) * _reveal_frac
                    + (1.0 - float(gate_alpha)) * (1.0 - _residual_conf)
                )
                _cp_alpha = (_residual_count > 0 and _graph_coverage < _cp_alpha_thresh)
                _residual_risk = _residual_below_block and _progress_below_expected
                _candidate_advantage = _graph_candidate_advantage
                _gate_values = {
                    "progress_empty": _progress_empty,
                    "progress_below_quota": _progress_below_quota,
                    "progress_below_expected": _progress_below_expected,
                    "progress_stalled": _progress_stalled,
                    "graph_below_uniform": _graph_below_uniform,
                    "graph_below_progress": _graph_below_progress,
                    "graph_residual_gap": _graph_residual_gap,
                    "graph_candidate_advantage": _graph_candidate_advantage,
                    "readiness_lt_one": _readiness_lt_one,
                    "residual_below_block": _residual_below_block,
                    "residual_below_revealed": _residual_below_revealed,
                    "dispersion_dominates": _dispersion_dominates,
                    "progress_or_graph": _progress_or_graph,
                    "progress_and_graph": _progress_and_graph,
                    "readiness_or_graph": _readiness_or_graph,
                    "readiness_and_progress": _readiness_and_progress,
                    "dawn_weak_certificate": _dawn_weak_certificate,
                    "two_step_progress_stall": _two_step_progress_stall,
                    "two_step_graph_gap": _two_step_graph_gap,
                    "two_step_hybrid_stuck": _two_step_hybrid_stuck,
                    "progress_stall": _two_step_progress_stall,
                    "coverage_gap": _coverage_gap,
                    "residual_risk": _residual_risk,
                    "readiness_drop": _readiness_drop,
                    "candidate_advantage": _candidate_advantage,
                    "adaptive_coverage": _coverage_gap,
                    "unified_axon": bool(_coverage_gap and _progress_stalled),
                    "cp_alpha": _cp_alpha,
                }
                if axon_gate_name == "adaptive_coverage":
                    # Label-free gate: accumulate coverage_gap's global fire rate
                    # over `adcov_warmup` gate-steps, then lock the choice.
                    if not hasattr(model, "_adcov_steps"):
                        model._adcov_steps = 0
                        model._adcov_fire = 0
                        model._adcov_choice = None
                    model._adcov_steps += 1
                    if bool(_coverage_gap):
                        model._adcov_fire += 1
                    if model._adcov_choice is None and model._adcov_steps >= int(adcov_warmup):
                        model._adcov_choice = (
                            "progress_stall"
                            if (model._adcov_fire / max(1, model._adcov_steps)) >= float(adcov_threshold)
                            else "coverage_gap"
                        )
                    if model._adcov_choice == "progress_stall":
                        _need_axon = bool(_two_step_progress_stall)
                    else:
                        _need_axon = bool(_coverage_gap)
                else:
                    if axon_gate_name not in _gate_values:
                        raise ValueError(f"Unsupported AXON adaptive gate: {axon_gate_name}")
                    _need_axon = bool(_gate_values[axon_gate_name])

                prev_dawn_progress = _reveal_frac
                prev_graph_coverage = _graph_coverage
                prev_readiness = _readiness
                prev_hybrid_bad = _hybrid_bad

                if (not _need_axon) or (not _can_anchor):
                    _use_dawn = True
                else:
                    _use_dawn = False

            # ---------------- reveal via base proposer (DAWN path) -----------
            if _use_dawn:
                if need_gate_dawn_transfer_index is None:
                    x0, transfer_index = _get_base_proposal(
                        axon_base_proposer_name,
                        logits=logits, temperature=temperature, remasking=remasking,
                        mask_index=mask_index, x=x, avg_attn_scores=avg_attn_scores,
                        tau_sink=tau_sink, tau_edge=tau_edge, tau_induce=tau_induce, tau_low=tau_low,
                        num_block=num_block, block_length=block_length, prompt_length=prompt_length,
                        ll_threshold=threshold if threshold is not None else 0.9,
                        ll_relaxed_threshold=relaxed_threshold, ll_radius=radius,
                        conf_num_transfer_tokens=num_transfer_tokens, conf_step=i,
                        conf_threshold_param=threshold,
                    )
                else:
                    x0 = need_gate_dawn_x0
                    transfer_index = need_gate_dawn_transfer_index
                if step_profiler is not None:
                    step_profiler.commit(transfer_index, num_block, i)
                x[transfer_index] = x0[transfer_index]
                i += 1
                if (x[:, block_start:block_end] == mask_id).sum() == 0:
                    break
                continue

            # ---------------- AXON fires: submodular anchor selection --------
            assert avg_attn_scores is not None, "avg_attn_scores is None"
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0 = torch.where(mask_index, x0, x)
            axon_mask_index = mask_index

            if step_profiler is not None:
                step_profiler.submod_begin()
            transfer_index = select_axon_adaptive_gate_anchors(
                mask_index=axon_mask_index,
                confidence=x0_p,
                avg_attn_scores=avg_attn_scores,
                tau_sink=tau_sink,
                candidate_topk=candidate_topk,
                candidate_min_topk=candidate_min_topk,
                candidate_ratio=candidate_ratio,
                min_select=min_select,
                selector=axon_selector_name,
                dawn_density=stag_dawn_density,
                target_coverage=stag_target_coverage,
                conflict_threshold=conflict_threshold,
                axon_fixed_k=axon_fixed_k,
                axon_submod_fn=axon_submod_fn,
                axon_submod_monotone=axon_submod_monotone,
                axon_submod_penalty=axon_submod_penalty,
                axon_submod_lambda=axon_submod_lambda,
            )
            if step_profiler is not None:
                step_profiler.submod_end()
            selected_count = int((transfer_index & axon_mask_index).sum().item())
            stats["axon_submod_calls"] += 1
            stats["axon_submod_anchors"] += selected_count

            if step_profiler is not None:
                step_profiler.commit(transfer_index, num_block, i)
            x[transfer_index] = x0[transfer_index]
            axon_anchor_steps_used += 1
            i += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

    return x, nfe, stats
