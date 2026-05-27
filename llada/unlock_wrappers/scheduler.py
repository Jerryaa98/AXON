"""Wrapped scheduler: adaptive-stop greedy with pluggable schedules.

Correctness design:
  - When the active Config is neutral for scheduler knobs, the wrappers MUST
    return byte-for-byte the same tensor as the original functions. This is
    enforced by a short-circuit that delegates directly to the originals
    before any tensor manipulation happens.
  - When knobs are active, we implement a parallel greedy that mirrors the
    exact structure of the originals in gdllm_utils.py:
      * Sink-corrected attention.
      * Candidate pool by top-confidence among masked positions.
      * Influence = target_uncertainty[i] * source_conf[j] * attn[i, j].
      * Symmetrized conflict weighted by pairwise uncertainty.
      * Greedy with conflict feasibility and risk_reward scoring.
    The only changes:
      (a) effective (lambda, gamma) are computed via the schedules module.
      (b) greedy stops early when best_score < gain_stop_ratio * best_ever.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

import gdllm_utils as _gd
from gdllm_utils import _normalize_vec, _to_bool, detect_attn_sinks_

from . import config as _cfg
from . import preprocess as _pre
from . import schedules as _sched


# Module-level container for cross-step state (currently reserved for future
# warm-start implementations — see plan v2). Reset at every new block boundary.
_WARM_STATE: Dict[str, Any] = {}


def _reset_warm_state_if_new_block(step_idx: int, step_in_block: int) -> None:
    # step_in_block == 0 AND step_idx > 0 means we just moved into a new block.
    # step_idx == 0 is the very first call, which also needs a clean state.
    if int(step_in_block) == 0:
        _WARM_STATE.clear()


def _effective_lambda_gamma(
    unlock_lambda: float,
    unlock_gamma: float,
    step_in_block: int,
    steps_in_block: int,
) -> Tuple[float, float]:
    c = _cfg.get()
    lam_fn = _sched.resolve(c.lambda_schedule)
    gam_fn = _sched.resolve(c.gamma_schedule)
    lam = lam_fn(float(unlock_lambda), int(step_in_block), int(steps_in_block))
    gam = gam_fn(float(unlock_gamma), int(step_in_block), int(steps_in_block))
    return float(lam), float(gam)


def _progress_t(
    step_in_block: int,
    steps_in_block: int,
    num_masked: int,
    num_masked_start: int,
) -> float:
    """Progress signal for the budget schedule.

    "step"   : t = step_in_block / (steps_in_block - 1). When blocks exit
               early this never approaches 1.0, so the ramp never fires.
    "masked" : t = 1 - num_masked / num_masked_start. Reaches 1.0 exactly
               as the block empties, regardless of how many steps it took.
    """
    c = _cfg.get()
    mode = str(c.budget_progress).strip().lower()
    if mode == "masked":
        start = max(1, int(num_masked_start))
        t = 1.0 - float(num_masked) / float(start)
    else:
        denom = max(1, int(steps_in_block) - 1)
        t = float(step_in_block) / float(denom)
    return max(0.0, min(1.0, t))


def _effective_multiplier_from_t(base_multiplier: float, t: float) -> float:
    """Return `base_multiplier * factor(t)` using the configured budget schedule."""
    c = _cfg.get()
    factor = _sched.budget_factor_by_t(
        c.budget_schedule, float(c.budget_floor), float(c.budget_ceil), float(t),
    )
    return float(base_multiplier) * float(factor)


def _should_short_circuit_to_original() -> bool:
    """When every scheduler-affecting knob is neutral, use the originals.

    Must cover every knob that our parallel greedy would change relative to
    the original implementation. If any knob is active, we route through the
    wrapped greedy so the knob's effect is realized."""
    c = _cfg.get()
    budget_neutral = (
        c.budget_schedule == "constant"
        and float(c.budget_floor) == 1.0
        and float(c.budget_ceil) == 1.0
    )
    return (
        c.gain_stop_ratio >= 1.0
        and c.lambda_schedule == "constant"
        and c.gamma_schedule == "constant"
        and float(c.sparse_conflict_threshold) <= 0.0
        and budget_neutral
        and not c.batch_greedy
        and not c.greedy_warm_start
        and str(c.gain_fn).strip().lower() == "linear"
    )


def _gain_from_influence(
    influence: torch.Tensor, cover: torch.Tensor, gain_fn: str,
) -> torch.Tensor:
    """Marginal submodular gain of each candidate column j given current cover.

    influence: [num_masked, num_candidates]
    cover:     [num_masked]  (per-target saturation already accumulated)
    Returns:   [num_candidates]
    """
    cover_col = cover.unsqueeze(1)  # [num_masked, 1]
    new_cover = torch.maximum(cover_col, influence)
    if str(gain_fn).strip().lower() == "log":
        per_target = torch.log1p(new_cover) - torch.log1p(cover_col)
    else:
        per_target = new_cover - cover_col
    return per_target.sum(dim=0)


def _get_warm_cover_row(
    b: int, masked_positions: torch.Tensor, seq_len: int, device, dtype,
) -> torch.Tensor:
    """Read per-position warm cover for batch b at masked_positions.

    Returns a [num_masked] tensor. If no warm state yet, returns zeros."""
    state = _WARM_STATE.get("cover")
    if state is None:
        return torch.zeros(masked_positions.numel(), dtype=dtype, device=device)
    return state[b].index_select(0, masked_positions).to(dtype=dtype)


def _write_warm_cover_row(
    b: int, masked_positions: torch.Tensor, cover: torch.Tensor,
    batch_size: int, seq_len: int, device,
) -> None:
    """Persist per-position cover values after a step. Lazily allocate."""
    state = _WARM_STATE.get("cover")
    if state is None:
        state = torch.zeros(batch_size, seq_len, dtype=cover.dtype, device=device)
        _WARM_STATE["cover"] = state
    state[b, masked_positions] = cover.to(dtype=state.dtype)


@torch.no_grad()
def get_transfer_index_unlock_clean_wrapped(
    mask_index,
    confidence,
    avg_attn_scores,
    tau_sink=0.01,
    candidate_topk=64,
    candidate_min_topk=16,
    candidate_ratio=0.25,
    warmup_steps=1,
    conflict_threshold=0.05,
    multiplier_after_warmup=2.0,
    min_select=1,
    step_idx=0,
    step_in_block=0,
    steps_in_block=1,
    unlock_lambda=0.0,
    unlock_gamma=0.0,
    unlock_technique="none",
    unlock_target_nfe_per_block=8,
    unlock_backload_power=1.25,
    unlock_optional_multiplier=1.5,
    unlock_optional_conf_threshold=0.5,
):
    assert avg_attn_scores is not None, "avg_attn_scores is None"

    if _should_short_circuit_to_original():
        return _gd.get_transfer_index_unlock_clean(
            mask_index=mask_index,
            confidence=confidence,
            avg_attn_scores=avg_attn_scores,
            tau_sink=tau_sink,
            candidate_topk=candidate_topk,
            candidate_min_topk=candidate_min_topk,
            candidate_ratio=candidate_ratio,
            warmup_steps=warmup_steps,
            conflict_threshold=conflict_threshold,
            multiplier_after_warmup=multiplier_after_warmup,
            min_select=min_select,
            step_idx=step_idx,
            step_in_block=step_in_block,
            steps_in_block=steps_in_block,
            unlock_lambda=unlock_lambda,
            unlock_gamma=unlock_gamma,
            unlock_technique=unlock_technique,
            unlock_target_nfe_per_block=unlock_target_nfe_per_block,
            unlock_backload_power=unlock_backload_power,
            unlock_optional_multiplier=unlock_optional_multiplier,
            unlock_optional_conf_threshold=unlock_optional_conf_threshold,
        )

    c = _cfg.get()
    _reset_warm_state_if_new_block(step_idx, step_in_block)

    eff_lambda, eff_gamma = _effective_lambda_gamma(
        unlock_lambda, unlock_gamma, step_in_block, steps_in_block
    )

    # Step 6: optionally sparsify the attention kernel before building conflict.
    avg_attn_scores = _pre.sparsify_attention(
        avg_attn_scores, c.sparse_conflict_threshold,
    )

    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)
    a_clean = avg_attn_scores.masked_fill(sink_mask.unsqueeze(1), 0.0).clone()
    a_clean.diagonal(dim1=1, dim2=2).zero_()

    device = mask_index.device
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=device)

    candidate_pool_total = 0
    selected_total = 0
    active_rows = 0
    masked_before_total = 0

    first_block = int(step_idx) < max(1, int(steps_in_block))
    warmup_active = first_block and (int(step_in_block) < int(warmup_steps))

    batch_size = mask_index.shape[0]
    for b in range(batch_size):
        masked_positions = torch.nonzero(mask_index[b], as_tuple=True)[0]
        num_masked = masked_positions.numel()
        if num_masked == 0:
            continue

        active_rows += 1
        masked_before_total += int(num_masked)
        conf_masked = confidence[b].index_select(0, masked_positions)

        k_t = int(candidate_ratio * int(num_masked))
        k_t = max(int(candidate_min_topk), k_t)
        k_t = min(int(candidate_topk), k_t, int(num_masked))
        k_t = max(k_t, 1)

        remaining_steps = max(1, int(steps_in_block) - int(step_in_block))
        base_quota = max(int(min_select), (int(num_masked) + remaining_steps - 1) // remaining_steps)

        # Capture num_masked at block start per batch row so masked-fraction
        # progress is well-defined for subsequent steps. _WARM_STATE is
        # already cleared on step_in_block == 0 by the reset above.
        masked_start_key = f"num_masked_start_{int(b)}"
        if int(step_in_block) == 0 or masked_start_key not in _WARM_STATE:
            _WARM_STATE[masked_start_key] = int(num_masked)
        num_masked_start = int(_WARM_STATE.get(masked_start_key, num_masked))

        if warmup_active:
            select_cap = min(k_t, base_quota)
        else:
            t_progress = _progress_t(
                int(step_in_block), int(steps_in_block),
                int(num_masked), num_masked_start,
            )
            eff_multiplier = _effective_multiplier_from_t(
                float(multiplier_after_warmup), t_progress,
            )
            scaled_quota = int(np.ceil(float(eff_multiplier) * float(base_quota)))
            select_cap = min(k_t, max(int(min_select), scaled_quota))
        select_cap = max(select_cap, 1)

        if warmup_active:
            warmup_k = min(select_cap, int(num_masked))
            _, warmup_rel = torch.topk(conf_masked, k=warmup_k, dim=0)
            selected_positions = masked_positions.index_select(0, warmup_rel)
            transfer_index[b, selected_positions] = True
            candidate_pool_total += int(warmup_k)
            selected_total += int(selected_positions.numel())
            continue

        _, candidate_rel = torch.topk(conf_masked, k=k_t, dim=0)
        candidate_positions = masked_positions.index_select(0, candidate_rel)
        if candidate_positions.numel() == 0:
            top_rel = torch.argmax(conf_masked, dim=0, keepdim=True)
            candidate_positions = masked_positions.index_select(0, top_rel)

        candidate_pool_total += int(candidate_positions.numel())

        source_conf = confidence[b].index_select(0, candidate_positions)
        target_uncertainty = 1.0 - conf_masked

        a_targets_sources = a_clean[b].index_select(0, masked_positions).index_select(1, candidate_positions)
        influence = target_uncertainty.unsqueeze(1) * source_conf.unsqueeze(0) * a_targets_sources

        a_candidate = a_clean[b].index_select(0, candidate_positions).index_select(1, candidate_positions)
        candidate_uncertainty = 1.0 - source_conf
        conflict = 0.5 * (a_candidate + a_candidate.transpose(0, 1))
        conflict = conflict * (candidate_uncertainty.unsqueeze(1) * candidate_uncertainty.unsqueeze(0))
        conflict.diagonal().zero_()

        num_candidates = candidate_positions.numel()
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)

        if bool(c.greedy_warm_start):
            current_cover = _get_warm_cover_row(
                b, masked_positions, int(mask_index.shape[1]),
                device, influence.dtype,
            )
        else:
            current_cover = torch.zeros(num_masked, dtype=influence.dtype, device=device)

        selected_count = 0
        best_score_seen = 0.0
        stop_ratio = float(c.gain_stop_ratio)
        gain_fn = str(c.gain_fn).strip().lower()

        if bool(c.batch_greedy):
            # One-shot scoring (no iterative cover/risk update). Risk term is
            # zero at the start, so final_score = unlock_term + lambda*conf_term.
            gain_vec = _gain_from_influence(influence, current_cover, gain_fn)
            unlock_term = gain_vec / float(max(1, int(num_masked)))
            unlock_term_n = _normalize_vec(unlock_term)
            conf_term_n = _normalize_vec(source_conf)
            final_score_all = unlock_term_n + float(eff_lambda) * conf_term_n

            order = torch.argsort(final_score_all, descending=True)
            for idx_t in order:
                if selected_count >= int(select_cap):
                    break
                idx = int(idx_t.item())
                if final_score_all[idx] <= 0.0:
                    break
                selected_indices = torch.nonzero(selected_mask, as_tuple=True)[0]
                if selected_indices.numel() > 0:
                    pair_conflict = conflict[idx].index_select(0, selected_indices)
                    if not torch.all(pair_conflict <= float(conflict_threshold)):
                        continue
                selected_mask[idx] = True
                selected_count += 1
                current_cover = torch.maximum(current_cover, influence[:, idx])
        else:
            while True:
                if selected_count >= int(select_cap):
                    break

                remaining = torch.nonzero(~selected_mask, as_tuple=True)[0]
                if remaining.numel() == 0:
                    break

                feasible = remaining
                selected = torch.nonzero(selected_mask, as_tuple=True)[0]
                if selected.numel() > 0:
                    pair_conflict = conflict.index_select(0, remaining).index_select(1, selected)
                    feasible = remaining[(pair_conflict <= float(conflict_threshold)).all(dim=1)]

                if feasible.numel() == 0:
                    break

                influence_feasible = influence.index_select(1, feasible)
                gain = _gain_from_influence(influence_feasible, current_cover, gain_fn)

                selected_count_now = int(selected.numel())
                if selected_count_now > 0:
                    risk_pair = conflict.index_select(0, feasible).index_select(1, selected)
                    risk_term = risk_pair.sum(dim=1) / float(max(1, selected_count_now))
                else:
                    risk_term = torch.zeros(feasible.numel(), dtype=gain.dtype, device=device)

                unlock_term = gain / float(max(1, int(num_masked)))
                conf_term_base = source_conf.index_select(0, feasible)

                unlock_term_n = _normalize_vec(unlock_term)
                conf_term_n = _normalize_vec(conf_term_base)
                risk_term_n = _normalize_vec(risk_term)

                final_score = (
                    unlock_term_n
                    + float(eff_lambda) * conf_term_n
                    - float(eff_gamma) * risk_term_n
                )

                best_pos = torch.argmax(final_score)
                best_score = float(final_score[best_pos].item())
                if best_score <= 0.0:
                    break

                # Adaptive gain-driven stop.
                if best_score > best_score_seen:
                    best_score_seen = best_score
                if stop_ratio < 1.0 and selected_count >= max(1, int(min_select)):
                    if best_score < stop_ratio * best_score_seen:
                        break

                best_local = feasible[best_pos]
                selected_mask[best_local] = True
                selected_count += 1
                current_cover = torch.maximum(current_cover, influence[:, best_local])

        if bool(c.greedy_warm_start):
            _write_warm_cover_row(
                b, masked_positions, current_cover,
                int(mask_index.shape[0]), int(mask_index.shape[1]), device,
            )

        selected = torch.nonzero(selected_mask, as_tuple=True)[0]
        selected_positions = candidate_positions.index_select(0, selected)

        if selected_positions.numel() == 0:
            fallback_k = max(1, int(min_select))
            fallback_k = min(fallback_k, int(num_masked))
            _, fallback_rel = torch.topk(conf_masked, k=fallback_k, dim=0)
            selected_positions = masked_positions.index_select(0, fallback_rel)

        transfer_index[b, selected_positions] = True
        selected_total += int(selected_positions.numel())

    stats = {
        "step_idx": int(step_idx),
        "step_in_block": int(step_in_block),
        "candidate_pool_total": candidate_pool_total,
        "selected_total": selected_total,
        "active_rows": active_rows,
        "masked_before_total": masked_before_total,
    }
    return transfer_index, stats


@torch.no_grad()
def get_transfer_index_unlock_wrapped(
    mask_index,
    confidence,
    avg_attn_scores,
    tau_sink=0.01,
    candidate_topk=64,
    candidate_min_topk=16,
    candidate_ratio=0.25,
    warmup_steps=1,
    conflict_threshold=0.05,
    multiplier_after_warmup=2.0,
    min_select=1,
    step_idx=0,
    step_in_block=0,
    steps_in_block=1,
    weight_func="base",
    alpha=0.0,
    beta=0.0,
    unlock_score_mode="current",
    unlock_lambda=0.0,
    unlock_gamma=0.0,
    unlock_stage2_fill=False,
    unlock_fill_ratio=0.5,
    unlock_budget_mode="none",
    unlock_budget_eta=0.5,
):
    """Wrapper for the full get_transfer_index_unlock.

    The full function supports multiple score modes, weight_funcs, and
    stage-2 fill. To avoid duplicating that entire state machine, the
    wrapper only intercepts the risk_reward / default-weight / no-stage-2
    cases — which match what the Pareto sweeps exercise. Any other combo
    falls through to the original implementation verbatim."""
    c = _cfg.get()

    score_mode = str(unlock_score_mode).strip().lower()
    stage2 = _to_bool(unlock_stage2_fill)
    if _should_short_circuit_to_original() or score_mode != "risk_reward" or stage2 or weight_func != "base":
        return _gd.get_transfer_index_unlock(
            mask_index=mask_index,
            confidence=confidence,
            avg_attn_scores=avg_attn_scores,
            tau_sink=tau_sink,
            candidate_topk=candidate_topk,
            candidate_min_topk=candidate_min_topk,
            candidate_ratio=candidate_ratio,
            warmup_steps=warmup_steps,
            conflict_threshold=conflict_threshold,
            multiplier_after_warmup=multiplier_after_warmup,
            min_select=min_select,
            step_idx=step_idx,
            step_in_block=step_in_block,
            steps_in_block=steps_in_block,
            weight_func=weight_func,
            alpha=alpha,
            beta=beta,
            unlock_score_mode=unlock_score_mode,
            unlock_lambda=unlock_lambda,
            unlock_gamma=unlock_gamma,
            unlock_stage2_fill=unlock_stage2_fill,
            unlock_fill_ratio=unlock_fill_ratio,
            unlock_budget_mode=unlock_budget_mode,
            unlock_budget_eta=unlock_budget_eta,
        )

    # For the wrappable subset, share the clean-path implementation — behaviorally
    # equivalent in risk_reward/base/no-stage-2 mode and much cleaner to maintain.
    return get_transfer_index_unlock_clean_wrapped(
        mask_index=mask_index,
        confidence=confidence,
        avg_attn_scores=avg_attn_scores,
        tau_sink=tau_sink,
        candidate_topk=candidate_topk,
        candidate_min_topk=candidate_min_topk,
        candidate_ratio=candidate_ratio,
        warmup_steps=warmup_steps,
        conflict_threshold=conflict_threshold,
        multiplier_after_warmup=multiplier_after_warmup,
        min_select=min_select,
        step_idx=step_idx,
        step_in_block=step_in_block,
        steps_in_block=steps_in_block,
        unlock_lambda=unlock_lambda,
        unlock_gamma=unlock_gamma,
    )
