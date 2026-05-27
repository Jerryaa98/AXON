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
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx

from gdllm_utils import *


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
             dawn=False, unlock=False, local_leap=False,
             tau_sink=0.01, tau_edge=0.07, tau_induce=0.7, tau_low=0.7,
             candidate_topk=64, candidate_min_topk=16, candidate_ratio=0.25,
             warmup_steps=1, conflict_threshold=0.05, multiplier_after_warmup=2.0, min_select=1,
             unlock_weight_func='base', unlock_alpha=0.0, unlock_beta=0.0,
             unlock_score_mode='current', unlock_lambda=0.0, unlock_gamma=0.0,
             unlock_hazard_weight=1.0,
             unlock_budget_mode='none', unlock_budget_eta=0.5,
             unlock_budget_alpha=0.0, unlock_budget_threshold=0.0, unlock_budget_gamma=0.0,
             unlock_tail_merge_steps=0,
             unlock_first_reveal_then_dawn=False,
             unlock_steps_per_block=1,
             unlock_num_blocks=-1,
             unlock_conf_threshold=0.0,
             unlock_exit_mode='none',
             unlock_min_anchor_steps=1,
             unlock_max_anchor_steps=1,
             unlock_exit_conf_threshold=0.0,
             unlock_exit_gain_eta=0.0,
             unlock_exit_coverage_threshold=1.0,
             unlock_use_clean=False,
             unlock_stage2_fill=False, unlock_fill_ratio=0.5,
             unlock_easy_then_unlock=False,
             unlock_easy_score_mode='conf_only',
             unlock_easy_prefix_mode='avg_threshold',
             unlock_easy_threshold=0.85,
             unlock_easy_eta=0.8,
             unlock_technique='none',
             unlock_target_nfe_per_block=8,
             unlock_block_length=None,
             unlock_backload_power=1.25,
             unlock_optional_multiplier=1.5,
             unlock_optional_conf_threshold=0.5,
             unlock_stability_conf_threshold=0.70,
             unlock_high_conf_threshold=0.75,
             unlock_anchor_min_conf=0.0,
             union_dawn_keep_frac=1.0,
             unlock_easy_stable_frac=0.60,
             unlock_easy_conf_frac=0.65,
             unlock_hard_stable_frac=0.30,
             unlock_gate_max_nfe_multiplier=2.0,
             axon_stag_gate='progress_stall',
             axon_stag_selector='fixed1',
             axon_base_proposer='dawn',
             relaxed_threshold=0.75, radius=4, unlock_profile=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    # conf_arch = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), 0, dtype=torch.float64).to(model.device)
    # conf_arch[:, :prompt.shape[1]] = 1

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    if unlock_profile is not None:
        unlock_profile.setdefault('scheduler_steps', 0)
        unlock_profile.setdefault('candidate_pool_total', 0)
        unlock_profile.setdefault('selected_total', 0)
        unlock_profile.setdefault('active_rows', 0)
        unlock_profile.setdefault('step_records', [])
        unlock_profile.setdefault('stag_step_records', [])

    nfe = 0
    for num_block in range(num_blocks):
        prev_prediction_ids = None
        prev_confidence = None
        prev_unlock_step_stats = None
        prev_dawn_progress = None
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            output, avg_attn_scores = model(x, return_attn_scores=(dawn or unlock))
            logits = output.logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is not None:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            elif unlock:
                technique_name = str(unlock_technique).strip().lower()
                axon_base_proposer_name = str(axon_base_proposer).strip().lower()
                if axon_base_proposer_name in {"", "false", "0"}:
                    axon_base_proposer_name = "dawn"
                if axon_base_proposer_name not in {"dawn", "local_leap", "confidence"}:
                    raise ValueError(f"Unsupported axon_base_proposer: {axon_base_proposer}")
                axon_adaptive_gate_name = str(axon_stag_gate).strip().lower()
                if axon_adaptive_gate_name not in {"progress_stall", "coverage_gap"}:
                    raise ValueError(f"Unsupported AXON gate: {axon_stag_gate}")
                axon_adaptive_selector_name = str(axon_stag_selector).strip().lower()
                if axon_adaptive_selector_name not in {"fixed1", "cover"}:
                    raise ValueError(f"Unsupported AXON selector: {axon_stag_selector}")
                need_gate_dawn_x0 = None
                need_gate_dawn_transfer_index = None
                _use_dawn = False
                if unlock_first_reveal_then_dawn or technique_name == 'submodular-exit':
                    _min_anchor_steps = int(unlock_min_anchor_steps) if technique_name == 'submodular-exit' else int(unlock_steps_per_block)
                    _max_anchor_steps = int(unlock_max_anchor_steps) if technique_name == 'submodular-exit' else int(unlock_steps_per_block)
                    _min_anchor_steps = max(0, _min_anchor_steps)
                    _max_anchor_steps = max(_min_anchor_steps, _max_anchor_steps)
                    _use_dawn = i >= _max_anchor_steps
                    if not _use_dawn and int(unlock_num_blocks) >= 0 and num_block >= int(unlock_num_blocks):
                        _use_dawn = True
                    _conf_threshold = float(unlock_exit_conf_threshold) if technique_name == 'submodular-exit' else float(unlock_conf_threshold)
                    _anchor_gate_count = i
                    if not _use_dawn and _anchor_gate_count >= _min_anchor_steps and _conf_threshold > 0.0:
                        _bstart = prompt.shape[1] + num_block * block_length
                        _bend = prompt.shape[1] + (num_block + 1) * block_length
                        _bmask = mask_index[:, _bstart:_bend]
                        if _bmask.sum() > 0:
                            _bprobs = F.softmax(logits[:, _bstart:_bend].to(torch.float64), dim=-1)
                            _avg_conf = _bprobs.max(dim=-1).values[_bmask].mean().item()
                            if _avg_conf > _conf_threshold:
                                _use_dawn = True
                    if not _use_dawn and technique_name == 'submodular-exit' and _anchor_gate_count >= _min_anchor_steps:
                        _exit_mode = str(unlock_exit_mode).strip().lower()
                        _bstart = prompt.shape[1] + num_block * block_length
                        _bend = prompt.shape[1] + (num_block + 1) * block_length
                        _remaining = int(mask_index[:, _bstart:_bend].sum().item())
                        _total = max(1, int(mask_index.shape[0]) * int(block_length))
                        _coverage = 1.0 - (float(_remaining) / float(_total))
                        _graph_coverage = 0.0
                        if avg_attn_scores is not None:
                            _block_mask = mask_index[:, _bstart:_bend]
                            _revealed = ~_block_mask
                            _row_coverages = []
                            for _row_idx in range(mask_index.shape[0]):
                                _residual_pos = torch.nonzero(_block_mask[_row_idx], as_tuple=False).flatten() + _bstart
                                _anchor_pos = torch.nonzero(_revealed[_row_idx], as_tuple=False).flatten() + _bstart
                                if _residual_pos.numel() == 0:
                                    _row_coverages.append(1.0)
                                elif _anchor_pos.numel() == 0:
                                    _row_coverages.append(0.0)
                                else:
                                    _block_pos = torch.arange(_bstart, _bend, device=mask_index.device)
                                    _residual_attn = avg_attn_scores[_row_idx].index_select(0, _residual_pos)
                                    _anchor_mass = _residual_attn.index_select(1, _anchor_pos).sum(dim=1)
                                    _block_mass = _residual_attn.index_select(1, _block_pos).sum(dim=1)
                                    _row_coverage = (_anchor_mass / (_block_mass + 1e-8)).clamp(0.0, 1.0).mean().item()
                                    _row_coverages.append(float(_row_coverage))
                            _graph_coverage = sum(_row_coverages) / float(max(1, len(_row_coverages)))
                        _last_gain = None
                        if prev_unlock_step_stats is not None:
                            _last_gain = prev_unlock_step_stats.get('avg_raw_gain_last')
                            if _last_gain is None:
                                _last_gain = prev_unlock_step_stats.get('avg_best_unlock_term')
                        _gain_exit = (
                            _last_gain is not None
                            and float(unlock_exit_gain_eta) > 0.0
                            and float(_last_gain) <= float(unlock_exit_gain_eta)
                        )
                        _coverage_exit = (
                            float(unlock_exit_coverage_threshold) < 1.0
                            and _coverage >= float(unlock_exit_coverage_threshold)
                        )
                        _graph_coverage_exit = (
                            float(unlock_exit_coverage_threshold) < 1.0
                            and _graph_coverage >= float(unlock_exit_coverage_threshold)
                        )
                        if _exit_mode == 'coverage_or_confidence':
                            _use_dawn = _use_dawn or _coverage_exit
                        elif _exit_mode == 'graph_coverage_or_confidence':
                            _use_dawn = _use_dawn or _graph_coverage_exit
                        elif _exit_mode == 'gain_or_confidence':
                            _use_dawn = _use_dawn or _gain_exit
                        elif _exit_mode == 'coverage_or_gain_or_confidence':
                            _use_dawn = _use_dawn or _coverage_exit or _gain_exit
                        elif _exit_mode == 'graph_coverage_or_gain_or_confidence':
                            _use_dawn = _use_dawn or _graph_coverage_exit or _gain_exit
                        elif _exit_mode in {'none', 'confidence'}:
                            pass
                        else:
                            raise ValueError(f"Unsupported UNLOCK exit mode: {unlock_exit_mode}")
                stag_dawn_density = 0.0
                stag_target_coverage = 0.0
                stag_step_record = None
                if (
                    technique_name == 'submodular-exit'
                    and i < _max_anchor_steps
                    and avg_attn_scores is not None
                ):
                    _can_anchor = i < _max_anchor_steps
                    need_gate_dawn_x0, need_gate_dawn_transfer_index = _get_base_proposal(
                        axon_base_proposer_name,
                        logits=logits, temperature=temperature, remasking=remasking,
                        mask_index=mask_index, x=x, avg_attn_scores=avg_attn_scores,
                        tau_sink=tau_sink, tau_edge=tau_edge, tau_induce=tau_induce, tau_low=tau_low,
                        num_block=num_block, block_length=block_length, prompt_length=prompt.shape[1],
                        ll_threshold=threshold if threshold is not None else 0.9,
                        ll_relaxed_threshold=relaxed_threshold,
                        ll_radius=radius,
                        conf_num_transfer_tokens=num_transfer_tokens,
                        conf_step=i,
                        conf_threshold_param=threshold,
                    )
                    _bstart = prompt.shape[1] + num_block * block_length
                    _bend = prompt.shape[1] + (num_block + 1) * block_length
                    _block_mask = mask_index[:, _bstart:_bend]
                    _block_dawn = need_gate_dawn_transfer_index[:, _bstart:_bend] & _block_mask
                    _masked_count = int(_block_mask.sum().item())
                    _dawn_count = int(_block_dawn.sum().item())
                    _reveal_frac = float(_dawn_count) / float(max(1, _masked_count))

                    _bprobs = F.softmax(logits[:, _bstart:_bend].to(torch.float64), dim=-1)

                    _residual_mask = _block_mask & ~_block_dawn
                    _residual_count = int(_residual_mask.sum().item())
                    _residual_conf = 1.0
                    if _residual_count > 0:
                        _residual_conf = _bprobs.max(dim=-1).values[_residual_mask].mean().item()

                    # g^(t): structural support — fraction of in-block attention
                    # mass that the residual pays to the base-decoder commits.
                    # Also accumulates the per-row anchor weight w_ij = A * (1-c_i) * c_j
                    # which gives the coverage threshold target tau = C(S_base).
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

                    # Paper Eq. (d_pace): pace deficit. Required pace
                    # rho^(t) = ceil(|M|/(T-t)) / |M| ~ 1/(T-t).
                    # Gate fires when the base pace is below required AND
                    # is not increasing relative to the previous step.
                    _remaining_target_steps = max(1, int(np.ceil(float(unlock_target_nfe_per_block))) - int(i))
                    _expected_count = int(np.ceil(float(_masked_count) / float(_remaining_target_steps))) if _masked_count > 0 else 0
                    _expected_progress = float(_expected_count) / float(max(1, _masked_count))
                    _progress_below_expected = _reveal_frac < _expected_progress
                    _two_step_progress_stall = (
                        prev_dawn_progress is not None
                        and _reveal_frac <= prev_dawn_progress
                        and _progress_below_expected
                    )
                    # Paper Eq. (d_cov): coverage deficit. Equivalent to
                    # g^(t) < max(r^(t), 1 - c_bar_M^(t)).
                    _graph_below_uniform = _graph_coverage < _reveal_frac
                    _graph_residual_gap = _residual_count > 0 and _graph_coverage < (1.0 - _residual_conf)
                    _coverage_gap = _graph_below_uniform or _graph_residual_gap

                    _gate_values = {
                        "progress_stall": _two_step_progress_stall,
                        "coverage_gap":   _coverage_gap,
                    }
                    _need_axon = bool(_gate_values[axon_adaptive_gate_name])
                    prev_dawn_progress = _reveal_frac
                    stag_step_record = {
                        "step_idx": int(num_block * steps + i),
                        "block_idx": int(num_block),
                        "step_in_block": int(i),
                        "gate": axon_adaptive_gate_name,
                        "selector": axon_adaptive_selector_name,
                        "need_axon": int(_need_axon),
                        "can_anchor": int(_can_anchor),
                        "masked_count": int(_masked_count),
                        "dawn_count": int(_dawn_count),
                        "dawn_reveal_frac": float(_reveal_frac),
                        "expected_count": int(_expected_count),
                        "expected_progress": float(_expected_progress),
                        "graph_coverage": float(_graph_coverage),
                        "residual_conf": float(_residual_conf),
                        "selected_count": 0,
                    }

                    if (not _need_axon) or (not _can_anchor):
                        _use_dawn = True
                    else:
                        _use_dawn = False
                if _use_dawn:
                    if need_gate_dawn_transfer_index is None:
                        x0, transfer_index = _get_base_proposal(
                            axon_base_proposer_name,
                            logits=logits, temperature=temperature, remasking=remasking,
                            mask_index=mask_index, x=x, avg_attn_scores=avg_attn_scores,
                            tau_sink=tau_sink, tau_edge=tau_edge, tau_induce=tau_induce, tau_low=tau_low,
                            num_block=num_block, block_length=block_length, prompt_length=prompt.shape[1],
                            ll_threshold=threshold if threshold is not None else 0.9,
                            ll_relaxed_threshold=relaxed_threshold,
                            ll_radius=radius,
                            conf_num_transfer_tokens=num_transfer_tokens,
                            conf_step=i,
                            conf_threshold_param=threshold,
                        )
                    else:
                        x0 = need_gate_dawn_x0
                        transfer_index = need_gate_dawn_transfer_index
                    if unlock_profile is not None and stag_step_record is not None:
                        unlock_profile.setdefault('stag_step_records', []).append(stag_step_record)
                    x[transfer_index] = x0[transfer_index]
                    i += 1
                    if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                        break
                    continue

                assert avg_attn_scores is not None, 'avg_attn_scores is None'

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0 = torch.where(mask_index, x0, x)
                axon_mask_index = mask_index
                transfer_index = select_axon_adaptive_gate_anchors(
                    mask_index=axon_mask_index,
                    confidence=x0_p,
                    avg_attn_scores=avg_attn_scores,
                    tau_sink=tau_sink,
                    candidate_topk=candidate_topk,
                    candidate_min_topk=candidate_min_topk,
                    candidate_ratio=candidate_ratio,
                    min_select=min_select,
                    selector=axon_adaptive_selector_name,
                    dawn_density=stag_dawn_density,
                    target_coverage=stag_target_coverage,
                    conflict_threshold=conflict_threshold,
                )
                if float(unlock_anchor_min_conf) > 0.0:
                    conf_ok = x0_p >= float(unlock_anchor_min_conf)
                    transfer_index = transfer_index & conf_ok
                selected_count = int((transfer_index & axon_mask_index).sum().item())
                unlock_step_stats = {
                    "candidate_pool_total": 0,
                    "selected_total": selected_count,
                    "active_rows": int(mask_index.shape[0]),
                    "step_idx": int(num_block * steps + i),
                    "masked_before_total": int(axon_mask_index.sum().item()),
                    "selected_scored_count": selected_count,
                    "technique_used": "axon",
                }
                if stag_step_record is not None:
                    stag_step_record["selected_count"] = selected_count

                if unlock_profile is not None:
                    unlock_profile['scheduler_steps'] += 1
                    unlock_profile['candidate_pool_total'] += unlock_step_stats['candidate_pool_total']
                    unlock_profile['selected_total'] += unlock_step_stats['selected_total']
                    unlock_profile['active_rows'] += unlock_step_stats['active_rows']
                    unlock_profile['step_records'].append(unlock_step_stats)
                    if stag_step_record is not None:
                        unlock_profile.setdefault('stag_step_records', []).append(stag_step_record)
                prev_prediction_ids = x0.detach().clone()
                prev_confidence = x0_p.detach().clone()
                prev_unlock_step_stats = unlock_step_stats
            elif dawn:
                x0, transfer_index = get_transfer_index_dawn(logits, temperature, remasking, mask_index, x, None, avg_attn_scores, tau_sink=tau_sink, tau_edge=tau_edge, tau_induce=tau_induce, tau_low=tau_low, num_block = num_block, block_length = block_length, prompt_length = prompt.shape[1])
            elif local_leap:
                x0, transfer_index = get_transfer_index_localleap(logits, temperature, remasking, mask_index, x, None, threshold = threshold, relaxed_threshold = relaxed_threshold, radius = radius)
            else:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe

@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    return x, nfe



def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def get_transfer_index_dawn(logits, temperature, remasking, mask_index, x, num_transfer_tokens, avg_attn_scores, tau_sink=0.01, tau_edge=0.07, tau_induce=0.7, tau_low=0.7, num_block=0, block_length=32, prompt_length=None):
    # attn sink removal
    assert avg_attn_scores is not None, 'avg_attn_scores is None'
    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)
    key_sink_mask = sink_mask.unsqueeze(1)      # [B, 1, L]
    avg_attn_scores = avg_attn_scores.masked_fill(key_sink_mask, 0.0)  # [B, L, L]

    B, _, _ = avg_attn_scores.shape
    avg_attn_scores.diagonal(dim1=1, dim2=2).zero_()

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    quantile_mask = avg_attn_scores >= tau_edge
    quantile_mask = quantile_mask.transpose(1, 2)

    # confidence aware 
    confidence = torch.where(mask_index, x0_p, -np.inf)
    transfer_index_conf = confidence >= 0.9

    # select the dependent nodes
    decoded_mask = (~mask_index & (x0_p >= 0.9)).unsqueeze(-1)
    decoded_mask[:, prompt_length + (num_block + 1) * block_length:] = False
    decoded_edge = quantile_mask & decoded_mask # [B, ?, N]
    dependent_nodes = decoded_edge.any(dim=1) & mask_index
    conf_d = torch.where(dependent_nodes, x0_p, -np.inf)
    transfer_index_a = conf_d >= tau_induce

    # select the conflicting nodes 
    adj_ti_mask = quantile_mask & transfer_index_a.unsqueeze(-1) & transfer_index_conf.unsqueeze(-1)
    adj_ti_mask = adj_ti_mask.any(dim=1)
    node_mask = mask_index & (x0_p >= tau_low) & ~transfer_index_conf & ~transfer_index_a & ~adj_ti_mask
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
        confidence = torch.where(mask_index, x0_p, -np.inf)
        
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | force_mask
        
    return x0, transfer_index

@torch.no_grad()
def generate_klass(
    model, input_ids_original, gen_length, steps, block_length, temperature=0., mask_id=126336,
    conf_threshold=0.6, kl_threshold=0.015, kl_history_length=2, 
    alg="klass",
    unmask_strategy="all"
):
    """
    reference: https://github.com/shkim0116/KLASS
    used for baseline method in the paper, remove analysis of the model's output

    Decoding strategy: Unmask tokens that are both high-confidence and have stable (low KL-divergence) softmax distributions over H steps.
    Implements alg options: default, random, topk_margin, entropy.
    """
    mask_id = 126336
    x = torch.full((1, input_ids_original.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :input_ids_original.shape[1]] = input_ids_original.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    used_steps = 0

    # History buffers
    V = model.lm_head.out_features if hasattr(model, "lm_head") else model.config.vocab_size
    kl_history = torch.zeros((1, x.shape[1], kl_history_length), dtype=torch.float64, device=x.device)
    p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64, device=x.device)

    all_step_outputs = []

    for num_block in range(num_blocks):
        block_start = input_ids_original.shape[1] + num_block * block_length
        block_end = input_ids_original.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            # --- Restrict to current block ---
            block_mask = torch.zeros_like(mask_index)
            block_mask[:, block_start:block_end] = True
            mask_index = mask_index & block_mask

            # --- Break if all tokens in current block are unmasked ---
            if not mask_index[:, block_start:block_end].any():
                break

            output, _ = model(x)
            logits = output.logits
            if temperature > 0:
                logits = add_gumbel_noise(logits, temperature)
            p_curr = F.softmax(logits.to(torch.float64), dim=-1)
            x0 = torch.argmax(p_curr, dim=-1)

            # --- Compute confidence according to alg ---
            if alg == "random":
                curr_conf = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif alg == "topk_margin":
                sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
                top1 = sorted_probs[..., 0]
                top2 = sorted_probs[..., 1]
                curr_conf = top1 - top2
            elif alg == "entropy":
                eps_ent = 1e-10
                log_p = torch.log(p_curr + eps_ent)
                curr_conf = -torch.sum(p_curr * log_p, dim=-1)  # negative entropy (lower entropy = higher confidence)
            else:  # default (top confidence)
                curr_conf = torch.squeeze(torch.gather(p_curr, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            # KL divergence between current and previous step
            eps = 1e-12
            kl_current_prev = (p_curr * (torch.log(p_curr + eps)
                            - torch.log(p_prev + eps))
                 ).sum(dim=-1)
            # Shift kl_history and insert new KL at the end
            kl_history = torch.roll(kl_history, shifts=-1, dims=-1)
            kl_history[..., -1] = kl_current_prev

            p_prev = p_curr.clone()

            if alg == "klass":
                # --- KL threshold logic ---
                if step >= kl_history_length - 1:
                    stable_mask = torch.all(kl_history < kl_threshold, dim=-1)
                else:
                    stable_mask = torch.zeros_like(curr_conf, dtype=torch.bool)
                # --- Confidence threshold logic ---
                conf_mask = curr_conf > conf_threshold

                ready_mask = stable_mask & conf_mask & mask_index
            else:
                ready_mask = torch.zeros_like(curr_conf, dtype=torch.bool)

            # Select top-k tokens to unmask
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
            decoded_token_info = [] 

            for j in range(ready_mask.shape[0]):
                ready_indices = torch.where(ready_mask[j])[0]
                if len(ready_indices) > 0:
                    if len(ready_indices) > 1 and unmask_strategy != "all":
                        if unmask_strategy == "max_conf":
                            # Pick the one with highest confidence
                            conf_vals = curr_conf[j, ready_indices]
                            max_idx = torch.argmax(conf_vals)
                            selected_indices = ready_indices[max_idx:max_idx+1]
                        elif unmask_strategy == "min_kl":
                            # Pick the one with lowest KL divergence
                            kl_vals = kl_current_prev[j, ready_indices]
                            min_idx = torch.argmin(kl_vals)
                            selected_indices = ready_indices[min_idx:min_idx+1]
                        elif unmask_strategy == "random":
                            selected_indices = ready_indices[torch.randint(0, len(ready_indices), (1,))]
                        else:
                            selected_indices = ready_indices
                    else:
                        selected_indices = ready_indices
                    transfer_index[j, selected_indices] = True
                # If no tokens meet both criteria, select top-k by confidence
                else:
                    curr_conf[:, input_ids_original.shape[1] + (num_block + 1) * block_length:] = -np.inf
                    confidence = torch.where(mask_index, curr_conf, -np.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    _, selected_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, step].item())
                    transfer_index[j, selected_indices] = True

            x[transfer_index] = x0[transfer_index]
            used_steps += 1

    return x, used_steps

def get_transfer_index_localleap(logits, temperature, remasking, mask_index, x, num_transfer_tokens, 
    threshold=0.9, relaxed_threshold=0.75, radius=4):
    '''
    reference: https://github.com/shkim0116/KLASS
    used for local leap method in the paper
    '''
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
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
                        # Add all positions of the anchor's neighbors.
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
# AXON base-proposer dispatch
# =============================================================================
#
# Each helper computes (x0, transfer_index) for a single decoding step using
# a different base decoder. The dispatcher `_get_base_proposal` picks one based
# on `axon_base_proposer`. The AXON diagnostic / gate / selector code remains
# unchanged — only the proposal that feeds them changes.

def _get_base_proposal_local_leap(logits, temperature, remasking, mask_index, x,
                                   threshold, relaxed_threshold, radius):
    """LocalLeap base proposer."""
    return get_transfer_index_localleap(
        logits, temperature, remasking, mask_index, x, None,
        threshold=threshold, relaxed_threshold=relaxed_threshold, radius=radius,
    )


def _get_base_proposal_confidence(logits, temperature, remasking, mask_index, x,
                                   num_transfer_tokens_step, threshold):
    """Low-confidence-first base proposer (LLaDA's default decoder)."""
    return get_transfer_index(
        logits, temperature, remasking, mask_index, x,
        num_transfer_tokens_step if threshold is None else None,
        threshold,
    )


def _get_base_proposal(axon_base_proposer, *,
                        logits, temperature, remasking, mask_index, x, avg_attn_scores,
                        # DAWN args
                        tau_sink, tau_edge, tau_induce, tau_low,
                        num_block, block_length, prompt_length,
                        # LocalLeap args
                        ll_threshold=0.9, ll_relaxed_threshold=0.8, ll_radius=4,
                        # Confidence args
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
        cf_threshold = conf_threshold_param if conf_threshold_param is not None else 0.9
        return _get_base_proposal_confidence(
            logits, temperature, remasking, mask_index, x, None, cf_threshold,
        )
    raise ValueError(f"Unsupported axon_base_proposer: {axon_base_proposer}")


def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
