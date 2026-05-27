import torch
import numpy as np
import torch.nn.functional as F


ADAPTIVE_UNLOCK_TECHNIQUES = {
    "confidence-adaptive",
    "safe-leaf",
    "stable-anchor",
    "leaf-anchor-hybrid",
    "hazard-clipped",
    "gain-elbow",
    "confidence-mass",
    "risk-adaptive",
    "deadline-backload",
    "coverage-efficiency",
    "self-governed",
}


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        item = value.strip().lower()
        if item in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if item in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return bool(value)


def _normalize_vec(x, eps=1e-8):
    if x.numel() == 0:
        return x
    x_min = torch.min(x)
    x_max = torch.max(x)
    return (x - x_min) / (x_max - x_min + float(eps))


def _resolve_budget_args(
    unlock_budget_mode,
    unlock_budget_eta,
    unlock_budget_alpha,
    unlock_budget_threshold,
    unlock_budget_gamma,
):
    budget_mode = str(unlock_budget_mode).strip().lower()
    if not budget_mode:
        budget_mode = "none"

    # Keep legacy modes for backward compatibility while adding new raw-gain modes.
    allowed_budget_modes = {
        "none",
        "first_gain_ratio",
        "running_avg_ratio",
        "prev_gain_ratio",
        "raw_gain_first_ratio",
        "raw_gain_running_avg",
        "raw_gain_plus_conf",
        "raw_gain_soft_penalty",
    }
    if budget_mode not in allowed_budget_modes:
        raise ValueError(f"Unsupported UNLOCK budget mode: {unlock_budget_mode}")

    budget_eta = float(unlock_budget_eta)
    budget_alpha = float(unlock_budget_alpha)
    budget_threshold = float(unlock_budget_threshold)
    budget_gamma = float(unlock_budget_gamma)
    return budget_mode, budget_eta, budget_alpha, budget_threshold, budget_gamma


def _should_stop_by_budget_mode(
    budget_mode,
    selected_count,
    base_quota,
    budget_eta,
    budget_alpha,
    budget_threshold,
    budget_gamma,
    raw_gain,
    raw_conf,
    raw_pairwise_conflict,
    selected_raw_gain_sequence,
    selected_raw_gain_sum,
    legacy_gain,
    selected_legacy_gain_sequence,
    selected_legacy_gain_sum,
):
    if int(selected_count) < int(base_quota):
        return False

    if budget_mode == "none":
        return False

    # Legacy (kept unchanged for compatibility).
    if budget_mode in {"first_gain_ratio", "running_avg_ratio", "prev_gain_ratio"}:
        if len(selected_legacy_gain_sequence) == 0:
            return False
        if budget_mode == "first_gain_ratio":
            reference_gain = selected_legacy_gain_sequence[0]
        elif budget_mode == "running_avg_ratio":
            reference_gain = selected_legacy_gain_sum / float(max(1, len(selected_legacy_gain_sequence)))
        else:
            reference_gain = selected_legacy_gain_sequence[-1]
        return float(legacy_gain) < float(budget_eta) * float(reference_gain)

    # New raw-gain based adaptive stopping.
    if budget_mode == "raw_gain_first_ratio":
        if len(selected_raw_gain_sequence) == 0:
            return False
        return float(raw_gain) < float(budget_eta) * float(selected_raw_gain_sequence[0])

    if budget_mode == "raw_gain_running_avg":
        if len(selected_raw_gain_sequence) == 0:
            return False
        running_avg = selected_raw_gain_sum / float(max(1, len(selected_raw_gain_sequence)))
        return float(raw_gain) < float(budget_eta) * float(running_avg)

    if budget_mode == "raw_gain_plus_conf":
        return (float(raw_gain) + float(budget_alpha) * float(raw_conf)) < float(budget_threshold)

    if budget_mode == "raw_gain_soft_penalty":
        effective_gain = float(raw_gain) - float(budget_gamma) * float(raw_pairwise_conflict)
        return effective_gain <= 0.0

    return False


def _resolve_easy_then_unlock_args(
    unlock_easy_then_unlock,
    unlock_easy_score_mode,
    unlock_easy_prefix_mode,
    unlock_easy_threshold,
    unlock_easy_eta,
):
    easy_then_unlock = _to_bool(unlock_easy_then_unlock)
    easy_score_mode = str(unlock_easy_score_mode).strip().lower()
    easy_prefix_mode = str(unlock_easy_prefix_mode).strip().lower()
    easy_threshold = float(unlock_easy_threshold)
    easy_eta = float(unlock_easy_eta)

    if easy_then_unlock:
        if easy_score_mode not in {"conf_only", "conf_maskdep"}:
            raise ValueError(f"Unsupported UNLOCK easy score mode: {unlock_easy_score_mode}")
        if easy_prefix_mode not in {"avg_threshold", "relative_drop"}:
            raise ValueError(f"Unsupported UNLOCK easy prefix mode: {unlock_easy_prefix_mode}")

    return easy_then_unlock, easy_score_mode, easy_prefix_mode, easy_threshold, easy_eta


def _build_monotonic_anchor_quota_schedule(block_length, target_nfe, min_select, backload_power):
    target_steps = max(1, int(np.ceil(float(target_nfe))))
    total_budget = max(1, int(block_length))
    min_quota = max(1, int(min_select))

    if target_steps == 1:
        return [total_budget]

    avg_quota = float(total_budget) / float(target_steps)
    ramp_strength = min(0.9, max(0.0, float(backload_power) - 0.75))
    half_span = avg_quota * ramp_strength
    raw = np.linspace(avg_quota - half_span, avg_quota + half_span, target_steps)

    quotas = [max(min_quota, int(np.rint(x))) for x in raw]
    for i in range(1, len(quotas)):
        quotas[i] = max(quotas[i], quotas[i - 1])

    def can_decrement(idx):
        if quotas[idx] <= min_quota:
            return False
        if idx > 0 and quotas[idx] - 1 < quotas[idx - 1]:
            return False
        return True

    def can_increment(idx):
        if idx + 1 < len(quotas) and quotas[idx] + 1 > quotas[idx + 1]:
            return False
        return True

    diff = int(sum(quotas) - total_budget)
    while diff > 0:
        changed = False
        for idx in range(len(quotas) - 1, -1, -1):
            if diff <= 0:
                break
            if can_decrement(idx):
                quotas[idx] -= 1
                diff -= 1
                changed = True
        if not changed:
            break

    diff = int(total_budget - sum(quotas))
    while diff > 0:
        changed = False
        for idx in range(len(quotas) - 1, -1, -1):
            if diff <= 0:
                break
            if can_increment(idx):
                quotas[idx] += 1
                diff -= 1
                changed = True
        if not changed:
            quotas[-1] += diff
            diff = 0

    return quotas


def _anchor_confidence_quota(
    step_in_block,
    num_masked,
    block_length,
    target_nfe,
    min_select,
    backload_power,
):
    schedule = _build_monotonic_anchor_quota_schedule(
        block_length=block_length,
        target_nfe=target_nfe,
        min_select=min_select,
        backload_power=backload_power,
    )
    schedule_idx = min(max(0, int(step_in_block)), len(schedule) - 1)
    if int(step_in_block) >= len(schedule) - 1:
        return int(num_masked), schedule
    return min(int(num_masked), int(schedule[schedule_idx])), schedule


def _mean_bool(mask):
    if mask.numel() == 0:
        return 0.0
    return float(mask.to(torch.float32).mean().item())


def _select_easy_prefix(
    source_conf,
    candidate_positions,
    masked_positions,
    a_clean_row,
    easy_score_mode,
    easy_prefix_mode,
    easy_threshold,
    easy_eta,
):
    if source_conf.numel() == 0:
        empty = source_conf.new_empty((0,))
        return empty.to(torch.long), empty

    if easy_score_mode == "conf_only":
        easy_scores = source_conf
    elif easy_score_mode == "conf_maskdep":
        attn_from_candidate = a_clean_row.index_select(0, candidate_positions).index_select(1, masked_positions)
        if attn_from_candidate.numel() == 0:
            maskdep = torch.zeros_like(source_conf)
        else:
            maskdep = attn_from_candidate.mean(dim=1)
        easy_scores = source_conf * (1.0 - maskdep)
    else:
        raise ValueError(f"Unsupported UNLOCK easy score mode: {easy_score_mode}")

    sorted_scores, sorted_rel = torch.sort(easy_scores, descending=True)
    if sorted_scores.numel() == 0:
        empty = source_conf.new_empty((0,))
        return empty.to(torch.long), empty

    m = 0
    if easy_prefix_mode == "avg_threshold":
        running_sum = torch.cumsum(sorted_scores, dim=0)
        denom = torch.arange(1, sorted_scores.numel() + 1, device=sorted_scores.device, dtype=sorted_scores.dtype)
        running_avg = running_sum / denom
        valid = torch.nonzero(running_avg >= float(easy_threshold), as_tuple=True)[0]
        if valid.numel() > 0:
            m = int(valid[-1].item()) + 1
    elif easy_prefix_mode == "relative_drop":
        best = sorted_scores[0]
        valid = torch.nonzero(sorted_scores >= float(easy_eta) * best, as_tuple=True)[0]
        if valid.numel() > 0:
            m = int(valid[-1].item()) + 1
    else:
        raise ValueError(f"Unsupported UNLOCK easy prefix mode: {easy_prefix_mode}")

    if m <= 0:
        empty = source_conf.new_empty((0,))
        return empty.to(torch.long), empty

    selected_rel = sorted_rel[:m]
    selected_scores = sorted_scores[:m]
    return selected_rel, selected_scores

def select_parallel_tokens_conflict_mis(edge_mask, node_mask, confidence, max_parallel=None):
    K = node_mask.sum().item()
    if K == 0:
        return []

    conflict = edge_mask | edge_mask.T

    select_index = []

    if max_parallel is None:
        max_parallel = K

    while len(select_index) < max_parallel:
        best_node_idx = torch.argmax(confidence).item()
        select_index.append(best_node_idx)
        confidence[best_node_idx] = -np.inf

        neigh_bool = conflict[best_node_idx]
        neigh_idx = torch.nonzero(neigh_bool, as_tuple=True)[0].tolist()
        node_mask[neigh_idx] = False
        confidence[neigh_idx] = -np.inf

    return select_index

def detect_attn_sinks(attn_scores, ratio=None, topk=None):
    B, L, _ = attn_scores.shape
    barA = attn_scores.mean(dim=1)  # [B, L]

    k1 = 0
    if ratio is not None and ratio > 0:
        k1 = max(1, int(L * ratio))
    k2 = topk if (topk is not None and topk > 0) else 0
    k = min(L, max(k1, k2))

    global_mask = torch.zeros((B, L), device=attn_scores.device, dtype=torch.bool)
    if k > 0:
        _, idx = torch.topk(barA, k=k, dim=-1)
        global_mask.scatter_(1, idx, True)
    return global_mask

def detect_attn_sinks_(attn_scores, threshold=0.05):
    """
    attn_scores: [B, S, S] softmax attention
    """
    B, S, _ = attn_scores.shape

    # column mean
    barA = attn_scores.mean(dim=1)          # [B, S]
    return barA > threshold


def select_axon_adaptive_gate_anchors(
    mask_index,
    confidence,
    avg_attn_scores,
    tau_sink=0.01,
    candidate_topk=32,
    candidate_min_topk=32,
    candidate_ratio=1.0,
    min_select=1,
    selector="fixed1",
    dawn_density=0.0,
    target_coverage=0.0,
    conflict_threshold=0.08,
):
    """Select AXON anchors by greedy maximisation of the facility-location
    coverage objective C(S) = sum_i max_{j in S} w_ij with
    w_ij = A_ij * (1 - c_i) * c_j (Eq. score_def).

    Two selectors:
      - "fixed1": commit a single top-marginal-gain anchor (k = 1).
      - "cover":  add anchors greedily until C(S) >= C(S_base) = target_coverage
                  (paper Alg. 1 stop rule).
    """
    assert avg_attn_scores is not None, "avg_attn_scores is None"

    selector = str(selector).strip().lower()
    if selector not in {"fixed1", "cover"}:
        raise ValueError(f"Unsupported AXON selector: {selector}")

    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)
    device = mask_index.device
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=device)
    target_coverage = max(0.0, float(target_coverage))

    for b in range(mask_index.shape[0]):
        masked_positions = torch.nonzero(mask_index[b], as_tuple=True)[0]
        num_masked = int(masked_positions.numel())
        if num_masked == 0:
            continue

        conf_masked = confidence[b].index_select(0, masked_positions)
        k_t = int(float(candidate_ratio) * num_masked)
        k_t = max(int(candidate_min_topk), k_t)
        k_t = min(int(candidate_topk), k_t, num_masked)
        k_t = max(k_t, 1)
        _, candidate_rel = torch.topk(conf_masked, k=k_t, dim=0)
        candidate_positions = masked_positions.index_select(0, candidate_rel)
        candidate_conf = confidence[b].index_select(0, candidate_positions).float()

        max_select = 1 if selector == "fixed1" else int(np.ceil(np.log2(float(num_masked) + 1.0)))
        max_select = min(num_masked, int(candidate_positions.numel()), max(int(min_select), max_select))

        residual_conf = confidence[b].index_select(0, masked_positions).float()
        uncertainty = (1.0 - residual_conf).clamp_min(0.0)
        cand_sink_mask = sink_mask[b].index_select(0, candidate_positions)
        attn_res_to_cand = (
            avg_attn_scores[b]
            .index_select(0, masked_positions)
            .index_select(1, candidate_positions)
            .float()
            .clone()
        )
        if cand_sink_mask.any():
            attn_res_to_cand[:, cand_sink_mask] = 0.0
        diagonal_mask = masked_positions.unsqueeze(1) == candidate_positions.unsqueeze(0)
        if diagonal_mask.any():
            attn_res_to_cand.masked_fill_(diagonal_mask, 0.0)

        # w_ij = A_ij * (1 - c_i) * c_j  (paper Eq. score_def)
        weights = attn_res_to_cand * uncertainty.unsqueeze(1) * candidate_conf.unsqueeze(0)
        current_coverage = torch.zeros((num_masked,), device=device, dtype=weights.dtype)
        selected_rel = []
        available = torch.ones((candidate_positions.numel(),), device=device, dtype=torch.bool)

        while len(selected_rel) < max_select:
            # cover-stop rule: stop once C(S) >= C(S_base)
            if selector == "cover" and len(selected_rel) >= int(min_select):
                if float(current_coverage.sum().item()) >= target_coverage:
                    break

            remaining = torch.nonzero(available, as_tuple=True)[0]
            if remaining.numel() == 0:
                break

            feasible_weights = weights.index_select(1, remaining)
            # facility-location marginal: sum_i [ max(C_i(S), w_ij) - C_i(S) ]
            coverage_gain = (
                torch.maximum(current_coverage.unsqueeze(1), feasible_weights)
                - current_coverage.unsqueeze(1)
            ).sum(dim=0)
            score = coverage_gain + 1e-6 * candidate_conf.index_select(0, remaining)

            best_pos = torch.argmax(score)
            best_idx = int(remaining[best_pos].item())
            best_gain = float(coverage_gain[best_pos].item())

            if best_gain <= 0.0 and len(selected_rel) >= int(min_select):
                break

            selected_rel.append(best_idx)
            available[best_idx] = False
            current_coverage = torch.maximum(current_coverage, weights[:, best_idx])

        if not selected_rel:
            selected_rel = [int(torch.argmax(candidate_conf).item())]

        selected_tensor = torch.tensor(selected_rel, device=device, dtype=torch.long)
        selected_positions = candidate_positions.index_select(0, selected_tensor)
        transfer_index[b, selected_positions] = True

    return transfer_index


def get_transfer_index_unlock(
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
    unlock_budget_alpha=0.0,
    unlock_budget_threshold=0.0,
    unlock_budget_gamma=0.0,
    unlock_tail_merge_steps=0,
    unlock_easy_then_unlock=False,
    unlock_easy_score_mode="conf_only",
    unlock_easy_prefix_mode="avg_threshold",
    unlock_easy_threshold=0.85,
    unlock_easy_eta=0.8,
):
    """
    UNLOCK is a training-free decoding scheduler for diffusion LLMs.
    At each denoising step, it selects a small set of masked positions that
    maximize coverage of the remaining sequence uncertainty while avoiding
    strongly coupled uncertain pairs.
    It uses only:
    - current token confidence
    - DAWN-style sink-corrected attention
    - a small candidate-restricted greedy selection
    It is designed to improve wall-clock inference by reducing decoding steps
    without introducing expensive scheduler overhead.

        Optional two-stage extension:
        - stage 1 runs the regular UNLOCK greedy selection on a reduced quota
        - stage 2 (if enabled) fills remaining slots with high-confidence masked
            positions from the remaining candidate pool
    """
    assert avg_attn_scores is not None, "avg_attn_scores is None"

    # Reuse DAWN sink correction and keep tensors on device.
    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)
    a_clean = avg_attn_scores.masked_fill(sink_mask.unsqueeze(1), 0.0).clone()
    a_clean.diagonal(dim1=1, dim2=2).zero_()

    device = mask_index.device
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=device)

    score_mode = str(unlock_score_mode).strip().lower()
    if score_mode not in {"current", "risk_reward"}:
        raise ValueError(f"Unsupported UNLOCK score mode: {unlock_score_mode}")

    budget_mode, budget_eta, budget_alpha, budget_threshold, budget_gamma = _resolve_budget_args(
        unlock_budget_mode=unlock_budget_mode,
        unlock_budget_eta=unlock_budget_eta,
        unlock_budget_alpha=unlock_budget_alpha,
        unlock_budget_threshold=unlock_budget_threshold,
        unlock_budget_gamma=unlock_budget_gamma,
    )

    easy_then_unlock, easy_score_mode, easy_prefix_mode, easy_threshold, easy_eta = _resolve_easy_then_unlock_args(
        unlock_easy_then_unlock=unlock_easy_then_unlock,
        unlock_easy_score_mode=unlock_easy_score_mode,
        unlock_easy_prefix_mode=unlock_easy_prefix_mode,
        unlock_easy_threshold=unlock_easy_threshold,
        unlock_easy_eta=unlock_easy_eta,
    )

    stage2_enabled = _to_bool(unlock_stage2_fill)
    fill_ratio = float(unlock_fill_ratio)
    tail_merge_steps = max(0, int(unlock_tail_merge_steps))

    candidate_pool_total = 0
    selected_total = 0
    active_rows = 0
    masked_before_total = 0

    selected_scored_count = 0
    unlock_term_selected_sum = 0.0
    conf_term_selected_sum = 0.0
    risk_term_selected_sum = 0.0
    risk_penalty_selected_sum = 0.0
    final_score_selected_sum = 0.0

    unlock_dominant_count = 0
    conf_dominant_count = 0
    risk_dominant_count = 0
    mixed_count = 0
    selected_classified_count = 0

    best_count = 0
    best_score_sum = 0.0
    best_unlock_term_sum = 0.0
    best_conf_term_sum = 0.0
    best_risk_term_sum = 0.0
    best_risk_penalty_sum = 0.0

    selected_gain_sequence_length_total = 0
    first_selected_gain_sum = 0.0
    last_selected_gain_sum = 0.0
    running_avg_selected_gain_sum = 0.0
    raw_gain_first_sum = 0.0
    raw_gain_last_sum = 0.0
    final_selected_count_total = 0
    base_quota_total = 0
    stopping_trigger_count = 0
    stop_cap_count = 0
    stop_no_remaining_count = 0
    stop_conflict_count = 0
    stop_nonpositive_count = 0
    stop_budget_count = 0
    stop_other_count = 0
    fallback_row_count = 0

    easy_selected_count_total = 0
    residual_selected_count_total = 0
    easy_score_selected_sum = 0.0
    easy_score_selected_count = 0
    mode_used = (
        f"easy_then_unlock:{easy_score_mode}:{easy_prefix_mode}"
        if easy_then_unlock else "default"
    )

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
        base_quota_total += int(base_quota)
        if int(step_in_block) < int(warmup_steps):
            select_cap = min(k_t, base_quota)
        else:
            # Permit moderate acceleration after warmup while preventing one-shot collapse.
            scaled_quota = int(np.ceil(float(multiplier_after_warmup) * float(base_quota)))
            select_cap = min(k_t, max(int(min_select), scaled_quota))
        select_cap = max(select_cap, 1)

        budget_base_quota = int(base_quota)
        if tail_merge_steps > 1 and int(num_masked) <= int(2 * tail_merge_steps):
            # Optional tail compaction: merge several tiny end-of-row reveals.
            merged_quota = int(np.ceil(float(base_quota) * float(tail_merge_steps)))
            budget_base_quota = min(int(select_cap), max(int(base_quota), merged_quota))

        # Keep original UNLOCK behavior unless two-stage fill is explicitly enabled.
        # In risk_reward mode, let stage-1 explore the entire candidate pool and
        # stop only on conflicts or non-positive score.
        if score_mode == "risk_reward":
            stage1_cap = int(select_cap)
        elif stage2_enabled:
            stage1_cap = int(np.floor(float(fill_ratio) * float(select_cap)))
            stage1_cap = max(1, min(int(select_cap), int(stage1_cap)))
        else:
            stage1_cap = int(select_cap)

        # Warmup should be conservative and confidence-driven to stabilize early denoising.
        if int(step_in_block) < int(warmup_steps) and not easy_then_unlock:
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

        easy_rel = source_conf.new_empty((0,), dtype=torch.long)
        easy_scores_selected = source_conf.new_empty((0,))
        if easy_then_unlock:
            easy_rel_full, easy_scores_full = _select_easy_prefix(
                source_conf=source_conf,
                candidate_positions=candidate_positions,
                masked_positions=masked_positions,
                a_clean_row=a_clean[b],
                easy_score_mode=easy_score_mode,
                easy_prefix_mode=easy_prefix_mode,
                easy_threshold=easy_threshold,
                easy_eta=easy_eta,
            )
            if easy_rel_full.numel() > 0:
                easy_count = min(int(select_cap), int(easy_rel_full.numel()))
                easy_rel = easy_rel_full[:easy_count]
                easy_scores_selected = easy_scores_full[:easy_count]

        target_uncertainty = 1.0 - conf_masked

        # Compute influence only on masked targets x candidate sources.
        a_targets_sources = a_clean[b].index_select(0, masked_positions).index_select(1, candidate_positions)
        influence = target_uncertainty.unsqueeze(1) * source_conf.unsqueeze(0) * a_targets_sources
        if weight_func == "diag_bonus":
            # Bonus for directly revealing its own position (i == j).
            same_position = (masked_positions.unsqueeze(1) == candidate_positions.unsqueeze(0)).to(influence.dtype)
            influence = influence + float(alpha) * same_position * source_conf.unsqueeze(0)
        elif weight_func not in {"base", "gain_conf_bonus"}:
            raise ValueError(f"Unsupported UNLOCK weight_func: {weight_func}")

        # Build conflict only on the candidate pool.
        a_candidate = a_clean[b].index_select(0, candidate_positions).index_select(1, candidate_positions)
        candidate_uncertainty = 1.0 - source_conf
        conflict = 0.5 * (a_candidate + a_candidate.transpose(0, 1))
        conflict = conflict * (candidate_uncertainty.unsqueeze(1) * candidate_uncertainty.unsqueeze(0))
        conflict.diagonal().zero_()

        num_candidates = candidate_positions.numel()
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
        current_cover = torch.zeros(num_masked, dtype=influence.dtype, device=device)
        selected_count = 0
        selected_raw_gain_sequence = []
        selected_raw_gain_sum = 0.0
        selected_legacy_gain_sequence = []
        selected_legacy_gain_sum = 0.0

        if easy_rel.numel() > 0:
            selected_mask[easy_rel] = True
            selected_count = int(easy_rel.numel())
            easy_selected_count_total += int(easy_rel.numel())
            easy_score_selected_sum += float(easy_scores_selected.sum().item())
            easy_score_selected_count += int(easy_scores_selected.numel())
            for idx in easy_rel:
                current_cover = torch.maximum(current_cover, influence[:, idx])

        # Bug fix: previously greedy could over-select and then prune by confidence, which broke
        # the greedy objective. We now stop inside greedy as soon as select_cap is reached.
        loop_stop_reason = "other"
        while True:
            if selected_count >= stage1_cap:
                loop_stop_reason = "cap"
                break

            remaining = torch.nonzero(~selected_mask, as_tuple=True)[0]
            if remaining.numel() == 0:
                loop_stop_reason = "no_remaining"
                break

            feasible = remaining
            selected = torch.nonzero(selected_mask, as_tuple=True)[0]
            if selected.numel() > 0:
                pair_conflict = conflict.index_select(0, remaining).index_select(1, selected)
                feasible = remaining[(pair_conflict <= conflict_threshold).all(dim=1)]

            if feasible.numel() == 0:
                loop_stop_reason = "conflict"
                break

            influence_feasible = influence.index_select(1, feasible)
            gain = torch.maximum(current_cover.unsqueeze(1), influence_feasible) - current_cover.unsqueeze(1)
            gain = gain.sum(dim=0)
            if weight_func == "gain_conf_bonus":
                gain = gain + float(beta) * source_conf.index_select(0, feasible)

            if score_mode == "risk_reward":
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

                conf_term = float(unlock_lambda) * conf_term_n
                risk_penalty = float(unlock_gamma) * risk_term_n
                final_score = unlock_term_n + conf_term - risk_penalty

                best_pos = torch.argmax(final_score)
                best_score = final_score[best_pos]

                best_count += 1
                best_score_sum += float(best_score.item())
                best_unlock_term_sum += float(unlock_term_n[best_pos].item())
                best_conf_term_sum += float(conf_term[best_pos].item())
                best_risk_term_sum += float(risk_term_n[best_pos].item())
                best_risk_penalty_sum += float(risk_penalty[best_pos].item())

                if float(best_score.item()) <= 0.0:
                    loop_stop_reason = "nonpositive_score"
                    break

                raw_gain_value = float(gain[best_pos].item())
                raw_conf_value = float(conf_term_base[best_pos].item())
                if selected_count_now > 0:
                    raw_pairwise_conflict = float(risk_pair[best_pos].sum().item())
                else:
                    raw_pairwise_conflict = 0.0
                legacy_gain_value = float(best_score.item())

                if _should_stop_by_budget_mode(
                    budget_mode=budget_mode,
                    selected_count=selected_count,
                    base_quota=budget_base_quota,
                    budget_eta=budget_eta,
                    budget_alpha=budget_alpha,
                    budget_threshold=budget_threshold,
                    budget_gamma=budget_gamma,
                    raw_gain=raw_gain_value,
                    raw_conf=raw_conf_value,
                    raw_pairwise_conflict=raw_pairwise_conflict,
                    selected_raw_gain_sequence=selected_raw_gain_sequence,
                    selected_raw_gain_sum=selected_raw_gain_sum,
                    legacy_gain=legacy_gain_value,
                    selected_legacy_gain_sequence=selected_legacy_gain_sequence,
                    selected_legacy_gain_sum=selected_legacy_gain_sum,
                ):
                    stopping_trigger_count += 1
                    loop_stop_reason = "budget"
                    break

                best_local = feasible[best_pos]

                selected_scored_count += 1
                unlock_term_value = float(unlock_term_n[best_pos].item())
                conf_term_value = float(conf_term[best_pos].item())
                risk_term_value = float(risk_term_n[best_pos].item())
                risk_penalty_value = float(risk_penalty[best_pos].item())
                final_score_value = float(best_score.item())

                unlock_term_selected_sum += unlock_term_value
                conf_term_selected_sum += conf_term_value
                risk_term_selected_sum += risk_term_value
                risk_penalty_selected_sum += risk_penalty_value
                final_score_selected_sum += final_score_value

                selected_classified_count += 1
                dominance_terms = [
                    ("unlock", unlock_term_value),
                    ("conf", conf_term_value),
                    ("risk", risk_penalty_value),
                ]
                dominance_terms.sort(key=lambda item: item[1], reverse=True)
                top_name, top_value = dominance_terms[0]
                second_value = dominance_terms[1][1]

                if top_value > 0.0 and (second_value / top_value) > 0.8:
                    mixed_count += 1
                elif top_name == "unlock":
                    unlock_dominant_count += 1
                elif top_name == "conf":
                    conf_dominant_count += 1
                elif top_name == "risk":
                    risk_dominant_count += 1
                else:
                    mixed_count += 1

                selected_raw_gain_sequence.append(raw_gain_value)
                selected_raw_gain_sum += raw_gain_value
                selected_legacy_gain_sequence.append(legacy_gain_value)
                selected_legacy_gain_sum += legacy_gain_value
            else:
                best_pos = torch.argmax(gain)
                best_gain = float(gain[best_pos].item())
                best_local = feasible[best_pos]
                selected_count_now = int(selected.numel())
                raw_conf_value = float(source_conf[best_local].item())
                if selected_count_now > 0:
                    raw_pairwise_conflict = float(conflict[best_local].index_select(0, selected).sum().item())
                else:
                    raw_pairwise_conflict = 0.0
                legacy_gain_value = best_gain

                if _should_stop_by_budget_mode(
                    budget_mode=budget_mode,
                    selected_count=selected_count,
                    base_quota=budget_base_quota,
                    budget_eta=budget_eta,
                    budget_alpha=budget_alpha,
                    budget_threshold=budget_threshold,
                    budget_gamma=budget_gamma,
                    raw_gain=best_gain,
                    raw_conf=raw_conf_value,
                    raw_pairwise_conflict=raw_pairwise_conflict,
                    selected_raw_gain_sequence=selected_raw_gain_sequence,
                    selected_raw_gain_sum=selected_raw_gain_sum,
                    legacy_gain=legacy_gain_value,
                    selected_legacy_gain_sequence=selected_legacy_gain_sequence,
                    selected_legacy_gain_sum=selected_legacy_gain_sum,
                ):
                    stopping_trigger_count += 1
                    loop_stop_reason = "budget"
                    break

                selected_raw_gain_sequence.append(best_gain)
                selected_raw_gain_sum += best_gain
                selected_legacy_gain_sequence.append(legacy_gain_value)
                selected_legacy_gain_sum += legacy_gain_value

            selected_mask[best_local] = True
            selected_count += 1
            current_cover = torch.maximum(current_cover, influence[:, best_local])

        if loop_stop_reason == "cap":
            stop_cap_count += 1
        elif loop_stop_reason == "no_remaining":
            stop_no_remaining_count += 1
        elif loop_stop_reason == "conflict":
            stop_conflict_count += 1
        elif loop_stop_reason == "nonpositive_score":
            stop_nonpositive_count += 1
        elif loop_stop_reason == "budget":
            stop_budget_count += 1
        else:
            stop_other_count += 1

        selected = torch.nonzero(selected_mask, as_tuple=True)[0]

        # Stage 2: fill remaining budget using highest-confidence remaining candidates.
        if score_mode == "current" and stage2_enabled and selected.numel() < int(select_cap):
            remaining = torch.nonzero(~selected_mask, as_tuple=True)[0]
            if remaining.numel() > 0:
                remaining_conf = source_conf.index_select(0, remaining)
                remaining_order = torch.argsort(remaining_conf, descending=True)
                fill_candidates = remaining.index_select(0, remaining_order)

                selected_dyn = selected
                for idx in fill_candidates:
                    if selected_dyn.numel() >= int(select_cap):
                        break
                    if selected_dyn.numel() > 0:
                        pair_conflict = conflict[idx].index_select(0, selected_dyn)
                        if not torch.all(pair_conflict <= float(conflict_threshold)):
                            continue
                    selected_mask[idx] = True
                    selected_dyn = torch.cat((selected_dyn, idx.view(1)))

                selected = selected_dyn

        selected_positions = candidate_positions.index_select(0, selected)

        if selected_positions.numel() == 0:
            fallback_row_count += 1
            fallback_k = max(1, int(min_select))
            fallback_k = min(fallback_k, int(num_masked))
            _, fallback_rel = torch.topk(conf_masked, k=fallback_k, dim=0)
            selected_positions = masked_positions.index_select(0, fallback_rel)

        residual_selected_count_total += max(0, int(selected_positions.numel()) - int(easy_rel.numel()))

        selected_gain_sequence_length_total += int(len(selected_legacy_gain_sequence))
        if len(selected_legacy_gain_sequence) > 0:
            first_selected_gain_sum += float(selected_legacy_gain_sequence[0])
            last_selected_gain_sum += float(selected_legacy_gain_sequence[-1])
            running_avg_selected_gain_sum += float(selected_legacy_gain_sum / float(len(selected_legacy_gain_sequence)))
        if len(selected_raw_gain_sequence) > 0:
            raw_gain_first_sum += float(selected_raw_gain_sequence[0])
            raw_gain_last_sum += float(selected_raw_gain_sequence[-1])
        final_selected_count_total += int(selected_positions.numel())

        transfer_index[b, selected_positions] = True
        selected_total += int(selected_positions.numel())

    diag_denom = max(1, active_rows)
    gate_denom = max(1, gate_metric_count)

    stats = {
        "step_idx": int(step_idx),
        "step_in_block": int(step_in_block),
        "candidate_pool_total": candidate_pool_total,
        "selected_total": selected_total,
        "active_rows": active_rows,
        "masked_before_total": masked_before_total,
        "selected_scored_count": selected_scored_count,
        "unlock_term_selected_sum": unlock_term_selected_sum,
        "conf_term_selected_sum": conf_term_selected_sum,
        "risk_term_selected_sum": risk_term_selected_sum,
        "risk_penalty_selected_sum": risk_penalty_selected_sum,
        "final_score_selected_sum": final_score_selected_sum,
        "unlock_dominant_count": unlock_dominant_count,
        "conf_dominant_count": conf_dominant_count,
        "risk_dominant_count": risk_dominant_count,
        "mixed_count": mixed_count,
        "selected_classified_count": selected_classified_count,
        "best_count": best_count,
        "best_score_sum": best_score_sum,
        "best_unlock_term_sum": best_unlock_term_sum,
        "best_conf_term_sum": best_conf_term_sum,
        "best_risk_term_sum": best_risk_term_sum,
        "best_risk_penalty_sum": best_risk_penalty_sum,
        "selected_gain_sequence_length": selected_gain_sequence_length_total / float(diag_denom),
        "first_selected_gain": first_selected_gain_sum / float(diag_denom),
        "last_selected_gain": last_selected_gain_sum / float(diag_denom),
        "running_avg_selected_gain": running_avg_selected_gain_sum / float(diag_denom),
        "avg_raw_gain_first": raw_gain_first_sum / float(diag_denom),
        "avg_raw_gain_last": raw_gain_last_sum / float(diag_denom),
        "final_selected_count": final_selected_count_total / float(diag_denom),
        "avg_selected_per_step": selected_total / float(diag_denom),
        "base_quota": base_quota_total / float(diag_denom),
        "stopping_trigger_count": int(stopping_trigger_count),
        "stop_cap_count": int(stop_cap_count),
        "stop_no_remaining_count": int(stop_no_remaining_count),
        "stop_conflict_count": int(stop_conflict_count),
        "stop_nonpositive_count": int(stop_nonpositive_count),
        "stop_budget_count": int(stop_budget_count),
        "stop_other_count": int(stop_other_count),
        "fallback_row_count": int(fallback_row_count),
        "budget_mode_used": budget_mode,
        "easy_selected_count": easy_selected_count_total,
        "residual_selected_count": residual_selected_count_total,
        "avg_easy_score": easy_score_selected_sum / float(max(1, easy_score_selected_count)),
        "mode_used": mode_used,
    }
    return transfer_index, stats


def get_transfer_index_unlock_clean(
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
    unlock_hazard_weight=1.0,
    unlock_budget_mode="none",
    unlock_budget_eta=0.5,
    unlock_budget_alpha=0.0,
    unlock_budget_threshold=0.0,
    unlock_budget_gamma=0.0,
    unlock_tail_merge_steps=0,
    unlock_easy_then_unlock=False,
    unlock_easy_score_mode="conf_only",
    unlock_easy_prefix_mode="avg_threshold",
    unlock_easy_threshold=0.85,
    unlock_easy_eta=0.8,
    unlock_technique="none",
    unlock_target_nfe_per_block=8,
    unlock_block_length=None,
    unlock_backload_power=1.25,
    unlock_optional_multiplier=1.5,
    unlock_optional_conf_threshold=0.5,
    prediction_ids=None,
    prev_prediction_ids=None,
    prev_confidence=None,
    unlock_stability_conf_threshold=0.70,
    unlock_high_conf_threshold=0.75,
    unlock_easy_stable_frac=0.60,
    unlock_easy_conf_frac=0.65,
    unlock_hard_stable_frac=0.30,
    unlock_gate_max_nfe_multiplier=2.0,
    axon_reveal_policy="fixed1",
    axon_gain_eta=0.0,
    axon_mass_ratio=0.20,
):
    """
    Clean one-stage UNLOCK scheduler for the current normalized risk_reward method.
    - No score-mode branching
    - No two-stage fill
    - No optional weighting variants
    """
    assert avg_attn_scores is not None, "avg_attn_scores is None"

    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)
    a_clean = avg_attn_scores.masked_fill(sink_mask.unsqueeze(1), 0.0).clone()
    a_clean.diagonal(dim1=1, dim2=2).zero_()

    device = mask_index.device
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=device)

    budget_mode, budget_eta, budget_alpha, budget_threshold, budget_gamma = _resolve_budget_args(
        unlock_budget_mode=unlock_budget_mode,
        unlock_budget_eta=unlock_budget_eta,
        unlock_budget_alpha=unlock_budget_alpha,
        unlock_budget_threshold=unlock_budget_threshold,
        unlock_budget_gamma=unlock_budget_gamma,
    )

    technique = str(unlock_technique).strip().lower()
    if not technique:
        technique = "none"
    allowed_techniques = {
        "none",
        "anchor-confidence",
        "stability-gated-anchor",
        "certified-influence",
    } | ADAPTIVE_UNLOCK_TECHNIQUES
    if technique not in allowed_techniques:
        raise ValueError(f"Unsupported UNLOCK technique: {unlock_technique}")
    use_anchor_confidence = technique == "anchor-confidence"
    use_stability_gate = technique == "stability-gated-anchor"
    use_certified_influence = technique == "certified-influence"
    use_adaptive_technique = technique in ADAPTIVE_UNLOCK_TECHNIQUES
    use_certified_like = use_certified_influence or use_adaptive_technique
    use_anchor_style = use_anchor_confidence or use_stability_gate or use_certified_like
    target_nfe = max(1.0, float(unlock_target_nfe_per_block))
    anchor_block_length = int(unlock_block_length) if unlock_block_length is not None else int(steps_in_block)
    anchor_block_length = max(1, anchor_block_length)
    backload_power = max(1e-6, float(unlock_backload_power))
    optional_multiplier = max(1.0, float(unlock_optional_multiplier))
    optional_conf_threshold = float(unlock_optional_conf_threshold)
    hazard_weight = float(unlock_hazard_weight)
    reveal_policy = str(axon_reveal_policy).strip().lower()
    if not reveal_policy:
        reveal_policy = "fixed1"
    if reveal_policy not in {"fixed1", "quota", "marginal_gain", "confidence_mass"}:
        raise ValueError(f"Unsupported AXON reveal policy: {axon_reveal_policy}")
    gain_eta = float(axon_gain_eta)
    mass_ratio = float(axon_mass_ratio)
    stability_conf_threshold = float(unlock_stability_conf_threshold)
    high_conf_threshold = float(unlock_high_conf_threshold)
    easy_stable_frac = float(unlock_easy_stable_frac)
    easy_conf_frac = float(unlock_easy_conf_frac)
    hard_stable_frac = float(unlock_hard_stable_frac)
    gate_max_steps = max(1, int(np.ceil(float(target_nfe) * max(1.0, float(unlock_gate_max_nfe_multiplier)))))

    easy_then_unlock, easy_score_mode, easy_prefix_mode, easy_threshold, easy_eta = _resolve_easy_then_unlock_args(
        unlock_easy_then_unlock=unlock_easy_then_unlock,
        unlock_easy_score_mode=unlock_easy_score_mode,
        unlock_easy_prefix_mode=unlock_easy_prefix_mode,
        unlock_easy_threshold=unlock_easy_threshold,
        unlock_easy_eta=unlock_easy_eta,
    )

    candidate_pool_total = 0
    selected_total = 0
    active_rows = 0
    masked_before_total = 0

    selected_scored_count = 0
    unlock_term_selected_sum = 0.0
    conf_term_selected_sum = 0.0
    risk_term_selected_sum = 0.0
    risk_penalty_selected_sum = 0.0
    final_score_selected_sum = 0.0

    unlock_dominant_count = 0
    conf_dominant_count = 0
    risk_dominant_count = 0
    mixed_count = 0
    selected_classified_count = 0

    best_count = 0
    best_score_sum = 0.0
    best_unlock_term_sum = 0.0
    best_conf_term_sum = 0.0
    best_risk_term_sum = 0.0
    best_risk_penalty_sum = 0.0

    easy_selected_count_total = 0
    residual_selected_count_total = 0
    easy_score_selected_sum = 0.0
    easy_score_selected_count = 0
    raw_gain_first_sum = 0.0
    raw_gain_last_sum = 0.0
    quota_floor_total = 0.0
    quota_cap_total = 0.0
    anchor_weight_total = 0.0
    conf_weight_total = 0.0
    risk_weight_total = 0.0
    gate_anchor_mode_count = 0
    gate_mixed_mode_count = 0
    gate_bulk_mode_count = 0
    gate_metric_count = 0
    stable_frac_sum = 0.0
    stable_high_conf_frac_sum = 0.0
    high_conf_frac_sum = 0.0
    gate_conf_sum = 0.0
    cert_selected_sum = 0.0
    hazard_selected_sum = 0.0
    out_influence_selected_sum = 0.0
    certified_gain_selected_sum = 0.0
    hazard_anchor_selected_count = 0
    certified_selected_count = 0
    hazard_block_count = 0
    stopping_trigger_count = 0
    stop_cap_count = 0
    stop_no_remaining_count = 0
    stop_conflict_count = 0
    stop_nonpositive_count = 0
    stop_budget_count = 0
    stop_other_count = 0
    fallback_row_count = 0
    tail_merge_steps = max(0, int(unlock_tail_merge_steps))
    mode_used = (
        f"easy_then_unlock:{easy_score_mode}:{easy_prefix_mode}"
        if easy_then_unlock else "default"
    )

    first_block = int(step_idx) < max(1, int(steps_in_block))
    warmup_active = first_block and (int(step_in_block) < int(warmup_steps)) and not easy_then_unlock

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

        quota_floor = int(base_quota)
        quota_cap = int(base_quota)
        anchor_weight = 1.0
        conf_weight = float(unlock_lambda)
        risk_weight = float(unlock_gamma)
        gate_mode = "default"
        has_prediction_history = (
            prediction_ids is not None
            and prev_prediction_ids is not None
            and int(step_in_block) > 0
        )
        stable_frac = 0.0
        stable_high_conf_frac = 0.0
        high_conf_frac = _mean_bool(conf_masked >= float(high_conf_threshold))
        avg_gate_conf = float(conf_masked.mean().item()) if conf_masked.numel() > 0 else 0.0
        if has_prediction_history:
            current_pred = prediction_ids[b].index_select(0, masked_positions)
            prev_pred = prev_prediction_ids[b].index_select(0, masked_positions)
            stable_mask = current_pred == prev_pred
            high_conf_mask = conf_masked >= float(high_conf_threshold)
            if prev_confidence is not None:
                prev_conf_masked = prev_confidence[b].index_select(0, masked_positions)
                stable_conf_mask = (
                    stable_mask
                    & (conf_masked >= float(stability_conf_threshold))
                    & (prev_conf_masked >= float(stability_conf_threshold))
                )
            else:
                stable_conf_mask = stable_mask & (conf_masked >= float(stability_conf_threshold))
            stable_frac = _mean_bool(stable_mask)
            stable_high_conf_frac = _mean_bool(stable_conf_mask)
            high_conf_frac = _mean_bool(high_conf_mask)

        if warmup_active:
            select_cap = min(k_t, base_quota)
        elif use_anchor_confidence or use_certified_like:
            quota_floor, _ = _anchor_confidence_quota(
                step_in_block=step_in_block,
                num_masked=num_masked,
                block_length=anchor_block_length,
                target_nfe=target_nfe,
                min_select=min_select,
                backload_power=backload_power,
            )
            quota_floor = max(int(min_select), quota_floor)
            quota_floor = min(int(num_masked), quota_floor)
            quota_cap = int(np.ceil(float(quota_floor) * optional_multiplier))
            quota_cap = max(int(quota_floor), quota_cap)
            quota_cap = min(int(num_masked), quota_cap)

            k_t = min(int(num_masked), max(int(k_t), int(quota_floor)))
            select_cap = min(k_t, max(int(quota_floor), int(quota_cap)))

            p = min(float(step_in_block) / float(max(1.0, target_nfe - 1.0)), 1.0)
            if use_certified_like:
                anchor_weight = 1.0
                conf_weight = float(unlock_lambda)
                risk_weight = float(unlock_gamma)
                if technique == "confidence-adaptive":
                    confidence_scale = 1.0 + (optional_multiplier - 1.0) * (
                        0.35 * avg_gate_conf + 0.65 * high_conf_frac
                    )
                    quota_cap = min(
                        int(num_masked),
                        max(int(quota_floor), int(np.ceil(float(quota_floor) * confidence_scale))),
                    )
                elif technique == "safe-leaf":
                    leaf_multiplier = min(optional_multiplier, 1.5)
                    quota_cap = min(
                        int(num_masked),
                        max(int(quota_floor), int(np.ceil(float(quota_floor) * leaf_multiplier))),
                    )
                elif technique == "deadline-backload":
                    deadline_progress = min(
                        float(step_in_block + 1) / float(max(1.0, target_nfe)),
                        1.0,
                    )
                    quota_floor = min(
                        int(num_masked),
                        max(
                            int(min_select),
                            int(np.ceil(float(base_quota) * max(0.5, deadline_progress))),
                        ),
                    )
                    backload_scale = 0.75 + (optional_multiplier + 0.75) * (
                        deadline_progress ** backload_power
                    )
                    quota_cap = min(
                        int(num_masked),
                        max(int(quota_floor), int(np.ceil(float(quota_floor) * backload_scale))),
                    )
                elif technique == "risk-adaptive":
                    risk_weight = float(unlock_gamma) * (1.45 - 0.65 * avg_gate_conf)
                elif technique == "leaf-anchor-hybrid":
                    if has_prediction_history and stable_high_conf_frac >= max(0.25, hard_stable_frac):
                        anchor_weight = 1.15
                        conf_weight = float(unlock_lambda)
                        risk_weight = float(unlock_gamma)
                    else:
                        anchor_weight = 0.65
                        conf_weight = float(unlock_lambda) * 0.75
                        risk_weight = float(unlock_gamma) * 1.20
                elif technique == "self-governed":
                    conf_scale = max(0.0, (avg_gate_conf - 0.5) / 0.4)
                    adaptive_multiplier = 1.0 + (optional_multiplier - 1.0) * conf_scale
                    quota_cap = min(
                        int(num_masked),
                        max(int(quota_floor), int(np.ceil(float(quota_floor) * adaptive_multiplier))),
                    )
                    
                    if avg_gate_conf > 0.8:
                        anchor_weight = 0.7
                        conf_weight = float(unlock_lambda) * 1.2
                        risk_weight = float(unlock_gamma) * 0.8
                    elif avg_gate_conf < 0.6:
                        anchor_weight = 1.3
                        conf_weight = float(unlock_lambda) * 0.7
                        risk_weight = float(unlock_gamma) * 1.4
                    else:
                        anchor_weight = 1.0
                        conf_weight = float(unlock_lambda)
                        risk_weight = float(unlock_gamma)

                k_t = min(int(num_masked), max(int(k_t), int(quota_floor), int(quota_cap)))
                select_cap = min(k_t, max(int(quota_floor), int(quota_cap)))
            else:
                anchor_weight = 1.0 - 0.6 * p
                conf_weight = float(unlock_lambda) * (0.6 + 0.7 * p)
                risk_weight = float(unlock_gamma) * (1.2 - 0.4 * p)

            if use_anchor_confidence:
                if reveal_policy == "fixed1":
                    quota_floor = min(int(num_masked), max(int(min_select), 1))
                    quota_cap = quota_floor
                    select_cap = min(k_t, quota_cap)
                elif reveal_policy == "quota":
                    pass
                elif reveal_policy in {"marginal_gain", "confidence_mass"}:
                    quota_floor = min(int(num_masked), max(int(min_select), 1))
                    quota_cap = min(int(num_masked), int(k_t))
                    select_cap = min(k_t, max(int(quota_floor), int(quota_cap)))
        elif use_stability_gate:
            has_prediction_history = (
                prediction_ids is not None
                and prev_prediction_ids is not None
                and int(step_in_block) > 0
            )
            if has_prediction_history:
                current_pred = prediction_ids[b].index_select(0, masked_positions)
                prev_pred = prev_prediction_ids[b].index_select(0, masked_positions)
                stable_mask = current_pred == prev_pred
                high_conf_mask = conf_masked >= float(high_conf_threshold)
                if prev_confidence is not None:
                    prev_conf_masked = prev_confidence[b].index_select(0, masked_positions)
                    stable_conf_mask = (
                        stable_mask
                        & (conf_masked >= float(stability_conf_threshold))
                        & (prev_conf_masked >= float(stability_conf_threshold))
                    )
                else:
                    stable_conf_mask = stable_mask & (conf_masked >= float(stability_conf_threshold))
                stable_frac = _mean_bool(stable_mask)
                stable_high_conf_frac = _mean_bool(stable_conf_mask)
                high_conf_frac = _mean_bool(high_conf_mask)

            final_deadline_step = int(step_in_block) >= int(gate_max_steps - 1)
            deadline_progress_step = int(step_in_block) >= int(np.ceil(float(target_nfe)))
            if final_deadline_step:
                gate_mode = "bulk"
                quota_floor = int(num_masked)
                quota_cap = int(num_masked)
            elif deadline_progress_step:
                remaining_deadline_steps = max(1, int(gate_max_steps) - int(step_in_block))
                gate_mode = "bulk" if high_conf_frac >= float(hard_stable_frac) else "mixed"
                quota_floor = int(np.ceil(float(num_masked) / float(remaining_deadline_steps)))
                quota_floor = max(int(min_select), quota_floor)
                quota_cap = min(int(num_masked), max(int(quota_floor), int(quota_floor + 2)))
            elif (
                has_prediction_history
                and stable_frac >= float(easy_stable_frac)
                and high_conf_frac >= float(easy_conf_frac)
                and stable_high_conf_frac >= (0.75 * float(easy_conf_frac))
                and avg_gate_conf >= float(stability_conf_threshold)
            ):
                gate_mode = "bulk"
                quota_floor = min(int(num_masked), max(int(min_select), 1))
                quota_cap = min(int(num_masked), max(int(quota_floor), 10))
            elif (
                not has_prediction_history
                or stable_frac < float(hard_stable_frac)
                or avg_gate_conf < float(stability_conf_threshold)
            ):
                gate_mode = "anchor"
                quota_floor = min(int(num_masked), max(int(min_select), 1))
                quota_cap = min(int(num_masked), max(int(quota_floor), 2))
            else:
                gate_mode = "mixed"
                quota_floor = min(int(num_masked), max(int(min_select), 2))
                quota_cap = min(int(num_masked), max(int(quota_floor), 5))

            k_t = min(int(num_masked), max(int(k_t), int(quota_cap)))
            select_cap = min(k_t, max(int(quota_floor), int(quota_cap)))
            if gate_mode == "anchor":
                anchor_weight = 1.25
                conf_weight = float(unlock_lambda) * 0.35
                risk_weight = float(unlock_gamma) * 1.25
            elif gate_mode == "mixed":
                anchor_weight = 0.90
                conf_weight = float(unlock_lambda) * 0.90
                risk_weight = float(unlock_gamma)
            else:
                anchor_weight = 0.25
                conf_weight = float(unlock_lambda) * 1.40
                risk_weight = float(unlock_gamma) * 0.70
        else:
            scaled_quota = int(np.ceil(float(multiplier_after_warmup) * float(base_quota)))
            select_cap = min(k_t, max(int(min_select), scaled_quota))
        select_cap = max(select_cap, 1)
        quota_floor = min(int(select_cap), max(1, int(quota_floor)))
        quota_cap = min(int(select_cap), max(int(quota_floor), int(quota_cap)))

        quota_floor_total += float(quota_floor)
        quota_cap_total += float(quota_cap)
        anchor_weight_total += float(anchor_weight)
        conf_weight_total += float(conf_weight)
        risk_weight_total += float(risk_weight)
        if use_stability_gate or use_certified_like:
            gate_metric_count += 1
            stable_frac_sum += float(stable_frac)
            stable_high_conf_frac_sum += float(stable_high_conf_frac)
            high_conf_frac_sum += float(high_conf_frac)
            gate_conf_sum += float(avg_gate_conf)
            if gate_mode == "anchor":
                gate_anchor_mode_count += 1
            elif gate_mode == "mixed":
                gate_mixed_mode_count += 1
            elif gate_mode == "bulk":
                gate_bulk_mode_count += 1

        budget_base_quota = int(base_quota)
        if tail_merge_steps > 1 and int(num_masked) <= int(2 * tail_merge_steps):
            # Optional tail compaction: merge several tiny end-of-row reveals.
            merged_quota = int(np.ceil(float(base_quota) * float(tail_merge_steps)))
            budget_base_quota = min(int(select_cap), max(int(base_quota), merged_quota))

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

        easy_rel = source_conf.new_empty((0,), dtype=torch.long)
        easy_scores_selected = source_conf.new_empty((0,))
        if easy_then_unlock:
            easy_rel_full, easy_scores_full = _select_easy_prefix(
                source_conf=source_conf,
                candidate_positions=candidate_positions,
                masked_positions=masked_positions,
                a_clean_row=a_clean[b],
                easy_score_mode=easy_score_mode,
                easy_prefix_mode=easy_prefix_mode,
                easy_threshold=easy_threshold,
                easy_eta=easy_eta,
            )
            if easy_rel_full.numel() > 0:
                easy_count = min(int(select_cap), int(easy_rel_full.numel()))
                easy_rel = easy_rel_full[:easy_count]
                easy_scores_selected = easy_scores_full[:easy_count]

        target_uncertainty = 1.0 - conf_masked

        a_targets_sources = a_clean[b].index_select(0, masked_positions).index_select(1, candidate_positions)
        influence = target_uncertainty.unsqueeze(1) * source_conf.unsqueeze(0) * a_targets_sources
        cert = source_conf * source_conf
        outgoing_influence = (target_uncertainty.unsqueeze(1) * a_targets_sources).mean(dim=0)
        hazard = (1.0 - cert) * outgoing_influence
        certified_mask = cert >= float(stability_conf_threshold)
        hazard_anchor_mask = (outgoing_influence > outgoing_influence.mean()) & ~certified_mask
        if use_certified_like:
            if has_prediction_history:
                current_candidate_pred = prediction_ids[b].index_select(0, candidate_positions)
                prev_candidate_pred = prev_prediction_ids[b].index_select(0, candidate_positions)
                stable_candidate = current_candidate_pred == prev_candidate_pred
                if prev_confidence is not None:
                    prev_source_conf = prev_confidence[b].index_select(0, candidate_positions)
                    cert = torch.where(
                        stable_candidate,
                        torch.minimum(source_conf, prev_source_conf),
                        torch.zeros_like(source_conf),
                    )
                else:
                    cert = torch.where(stable_candidate, source_conf, torch.zeros_like(source_conf))
            else:
                cert = source_conf * source_conf
            certified_mask = cert >= float(stability_conf_threshold)
            hazard = (1.0 - cert) * outgoing_influence
            hazard_anchor_mask = (outgoing_influence > outgoing_influence.mean()) & ~certified_mask
            cert_based_influence = (
                use_certified_influence
                or technique in {"stable-anchor", "hazard-clipped", "coverage-efficiency"}
                or (
                    technique == "leaf-anchor-hybrid"
                    and has_prediction_history
                    and stable_high_conf_frac >= max(0.25, float(hard_stable_frac))
                )
            )
            if cert_based_influence:
                influence = target_uncertainty.unsqueeze(1) * cert.unsqueeze(0) * a_targets_sources
            hazard_block_count += int(hazard_anchor_mask.any().item())

        if technique == "confidence-mass" and source_conf.numel() > 0:
            sorted_conf = torch.sort(source_conf, descending=True).values
            total_conf_mass = float(sorted_conf.sum().item())
            if total_conf_mass > 0.0:
                mass_eta = min(0.95, max(0.05, float(budget_eta)))
                cumulative_mass = torch.cumsum(sorted_conf, dim=0)
                mass_count = int((cumulative_mass < (mass_eta * total_conf_mass)).sum().item()) + 1
                quota_cap = min(int(select_cap), max(int(quota_floor), mass_count))
                select_cap = max(1, min(int(select_cap), int(quota_cap)))

        a_candidate = a_clean[b].index_select(0, candidate_positions).index_select(1, candidate_positions)
        candidate_uncertainty = 1.0 - source_conf
        conflict = 0.5 * (a_candidate + a_candidate.transpose(0, 1))
        conflict = conflict * (candidate_uncertainty.unsqueeze(1) * candidate_uncertainty.unsqueeze(0))
        conflict.diagonal().zero_()

        num_candidates = candidate_positions.numel()
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
        current_cover = torch.zeros(num_masked, dtype=influence.dtype, device=device)
        selected_count = 0
        selected_conf_mass = 0.0
        confidence_mass_target = 0.0
        if reveal_policy == "confidence_mass":
            confidence_mass_target = max(
                0.0,
                float(mass_ratio) * float(target_uncertainty.clamp_min(0.0).sum().item()),
            )
        selected_raw_gain_sequence = []
        selected_raw_gain_sum = 0.0
        selected_legacy_gain_sequence = []
        selected_legacy_gain_sum = 0.0

        if easy_rel.numel() > 0:
            selected_mask[easy_rel] = True
            selected_count = int(easy_rel.numel())
            selected_conf_mass += float(source_conf.index_select(0, easy_rel).sum().item())
            easy_selected_count_total += int(easy_rel.numel())
            easy_score_selected_sum += float(easy_scores_selected.sum().item())
            easy_score_selected_count += int(easy_scores_selected.numel())
            for idx in easy_rel:
                current_cover = torch.maximum(current_cover, influence[:, idx])

        loop_stop_reason = "other"
        while True:
            if selected_count >= int(select_cap):
                loop_stop_reason = "cap"
                break

            remaining = torch.nonzero(~selected_mask, as_tuple=True)[0]
            if remaining.numel() == 0:
                loop_stop_reason = "no_remaining"
                break

            feasible = remaining
            selected = torch.nonzero(selected_mask, as_tuple=True)[0]
            if selected.numel() > 0:
                pair_conflict = conflict.index_select(0, remaining).index_select(1, selected)
                feasible = remaining[(pair_conflict <= float(conflict_threshold)).all(dim=1)]
                if technique == "risk-adaptive" and feasible.numel() > 0:
                    adaptive_threshold = float(conflict_threshold) * (0.55 + 0.75 * avg_gate_conf)
                    adaptive_threshold = min(float(conflict_threshold), max(0.0, adaptive_threshold))
                    adaptive_pair_conflict = conflict.index_select(0, feasible).index_select(1, selected)
                    adaptive_feasible = feasible[(adaptive_pair_conflict <= adaptive_threshold).all(dim=1)]
                    if adaptive_feasible.numel() > 0 or selected_count >= int(quota_floor):
                        feasible = adaptive_feasible

            hazard_block_floor = int(min_select) if technique == "hazard-clipped" else int(quota_floor)
            if (
                (use_certified_influence or technique == "hazard-clipped")
                and selected_count >= hazard_block_floor
                and feasible.numel() > 0
            ):
                safe_feasible = feasible[~hazard_anchor_mask.index_select(0, feasible)]
                if safe_feasible.numel() == 0:
                    loop_stop_reason = "budget"
                    break
                feasible = safe_feasible

            if feasible.numel() == 0:
                if use_anchor_style and selected_count < int(min_select):
                    if use_certified_like:
                        remaining_score = source_conf.index_select(0, remaining) - hazard.index_select(0, remaining)
                        best_remaining_pos = torch.argmax(remaining_score)
                    else:
                        remaining_conf = source_conf.index_select(0, remaining)
                        best_remaining_pos = torch.argmax(remaining_conf)
                    best_local = remaining[best_remaining_pos]
                    selected_mask[best_local] = True
                    selected_count += 1
                    current_cover = torch.maximum(current_cover, influence[:, best_local])
                    continue
                else:
                    loop_stop_reason = "conflict"
                    break

            influence_feasible = influence.index_select(1, feasible)
            gain = torch.maximum(current_cover.unsqueeze(1), influence_feasible) - current_cover.unsqueeze(1)
            gain = gain.sum(dim=0)

            selected_count_now = int(selected.numel())
            if selected_count_now > 0:
                risk_pair = conflict.index_select(0, feasible).index_select(1, selected)
                risk_term = risk_pair.sum(dim=1) / float(max(1, selected_count_now))
            else:
                risk_term = torch.zeros(feasible.numel(), dtype=gain.dtype, device=device)

            conf_term_base = source_conf.index_select(0, feasible)

            if use_adaptive_technique:
                unlock_term = gain / float(max(1, int(num_masked)))
                unlock_term_n = _normalize_vec(unlock_term)
                conf_term_n = _normalize_vec(conf_term_base)
                risk_term_n = _normalize_vec(risk_term)
                cert_term = cert.index_select(0, feasible)
                cert_term_n = _normalize_vec(cert_term)
                hazard_term = hazard.index_select(0, feasible)
                hazard_term_n = _normalize_vec(hazard_term)
                out_term = outgoing_influence.index_select(0, feasible)
                out_term_n = _normalize_vec(out_term)
                leaf_term = conf_term_base * (1.0 - out_term_n)
                leaf_term_n = _normalize_vec(leaf_term)

                score_conf_n = conf_term_n
                score_hazard_n = hazard_term_n
                risk_scale = 1.0
                if technique == "confidence-adaptive":
                    reveal_scale = 0.75 + 0.75 * avg_gate_conf + 0.50 * high_conf_frac
                    final_score = reveal_scale * unlock_term_n + float(conf_weight) * conf_term_n
                elif technique == "safe-leaf":
                    final_score = leaf_term_n + 0.35 * unlock_term_n + 0.25 * conf_term_n
                    score_hazard_n = torch.maximum(hazard_term_n, out_term_n)
                elif technique == "stable-anchor":
                    score_conf_n = cert_term_n
                    final_score = unlock_term_n + float(conf_weight) * cert_term_n + cert_term_n * out_term_n
                elif technique == "leaf-anchor-hybrid":
                    if has_prediction_history and stable_high_conf_frac >= max(0.25, float(hard_stable_frac)):
                        score_conf_n = cert_term_n
                        final_score = unlock_term_n + float(conf_weight) * cert_term_n + 0.5 * cert_term_n * out_term_n
                    else:
                        final_score = leaf_term_n + 0.45 * unlock_term_n + 0.25 * conf_term_n
                        score_hazard_n = torch.maximum(hazard_term_n, out_term_n)
                elif technique == "hazard-clipped":
                    score_conf_n = cert_term_n
                    final_score = unlock_term_n + float(conf_weight) * cert_term_n
                elif technique == "gain-elbow":
                    final_score = unlock_term_n + float(conf_weight) * conf_term_n
                elif technique == "confidence-mass":
                    final_score = conf_term_n + 0.45 * unlock_term_n
                    score_hazard_n = 0.5 * hazard_term_n
                elif technique == "risk-adaptive":
                    risk_scale = 1.35 - 0.65 * avg_gate_conf
                    final_score = unlock_term_n + float(conf_weight) * conf_term_n
                elif technique == "deadline-backload":
                    deadline_progress = min(float(step_in_block + 1) / float(max(1.0, target_nfe)), 1.0)
                    final_score = (0.65 + deadline_progress) * unlock_term_n + float(conf_weight) * conf_term_n
                    score_hazard_n = (1.0 - deadline_progress) * hazard_term_n
                elif technique == "coverage-efficiency":
                    final_score = cover_term_n + float(conf_weight) * conf_term_n
                elif technique == "self-governed":
                    final_score = unlock_term_n + float(conf_weight) * conf_term_n
                else:
                    final_score = unlock_term_n + float(conf_weight) * conf_term_n

                conf_term = float(conf_weight) * score_conf_n
                risk_penalty = float(risk_weight) * risk_scale * risk_term_n + float(hazard_weight) * score_hazard_n
                final_score = final_score - risk_penalty
            elif use_certified_influence:
                unlock_term_n = gain
                conf_term_n = cert.index_select(0, feasible)
                risk_term_n = risk_term
                hazard_term = hazard.index_select(0, feasible)
                conf_term = float(conf_weight) * conf_term_n
                risk_penalty = float(risk_weight) * risk_term_n + float(hazard_weight) * hazard_term
                final_score = unlock_term_n + conf_term - risk_penalty
            else:
                unlock_term = gain / float(max(1, int(num_masked)))
                unlock_term_n = _normalize_vec(unlock_term)
                conf_term_n = _normalize_vec(conf_term_base)
                risk_term_n = _normalize_vec(risk_term)
                conf_term = float(conf_weight) * conf_term_n
                risk_penalty = float(risk_weight) * risk_term_n

                if use_anchor_style:
                    final_score = float(anchor_weight) * unlock_term_n + conf_term - risk_penalty
                else:
                    final_score = unlock_term_n + conf_term - risk_penalty

            best_pos = torch.argmax(final_score)
            best_score = final_score[best_pos]

            best_count += 1
            best_score_sum += float(best_score.item())
            best_unlock_term_sum += float(unlock_term_n[best_pos].item())
            best_conf_term_sum += float(conf_term[best_pos].item())
            best_risk_term_sum += float(risk_term_n[best_pos].item())
            best_risk_penalty_sum += float(risk_penalty[best_pos].item())

            before_quota_floor = use_anchor_style and selected_count < int(min_select)

            if float(best_score.item()) <= 0.0 and not before_quota_floor:
                loop_stop_reason = "nonpositive_score"
                break
            if (
                use_anchor_confidence
                and reveal_policy == "marginal_gain"
                and not before_quota_floor
                and float(best_score.item()) <= float(gain_eta)
            ):
                stopping_trigger_count += 1
                loop_stop_reason = "budget"
                break
            if (
                use_anchor_confidence
                and reveal_policy == "confidence_mass"
                and not before_quota_floor
                and selected_conf_mass >= confidence_mass_target
            ):
                stopping_trigger_count += 1
                loop_stop_reason = "budget"
                break

            best_local = feasible[best_pos]

            raw_gain_value = float(gain[best_pos].item())
            if use_certified_like:
                raw_conf_value = float(cert.index_select(0, feasible)[best_pos].item())
            else:
                raw_conf_value = float(conf_term_base[best_pos].item())
            if selected_count_now > 0:
                raw_pairwise_conflict = float(risk_pair[best_pos].sum().item())
            else:
                raw_pairwise_conflict = 0.0
            legacy_gain_value = float(best_score.item())

            should_stop_by_budget = _should_stop_by_budget_mode(
                budget_mode=budget_mode,
                selected_count=selected_count,
                base_quota=budget_base_quota,
                budget_eta=budget_eta,
                budget_alpha=budget_alpha,
                budget_threshold=budget_threshold,
                budget_gamma=budget_gamma,
                raw_gain=raw_gain_value,
                raw_conf=raw_conf_value,
                raw_pairwise_conflict=raw_pairwise_conflict,
                selected_raw_gain_sequence=selected_raw_gain_sequence,
                selected_raw_gain_sum=selected_raw_gain_sum,
                legacy_gain=legacy_gain_value,
                selected_legacy_gain_sequence=selected_legacy_gain_sequence,
                selected_legacy_gain_sum=selected_legacy_gain_sum,
            )
            if should_stop_by_budget and not before_quota_floor:
                stopping_trigger_count += 1
                loop_stop_reason = "budget"
                break

            if (
                technique == "gain-elbow"
                and not before_quota_floor
                and len(selected_raw_gain_sequence) > 0
                and selected_raw_gain_sequence[0] > 0.0
                and raw_gain_value < float(budget_eta) * float(selected_raw_gain_sequence[0])
            ):
                stopping_trigger_count += 1
                loop_stop_reason = "budget"
                break

            if (
                use_anchor_style
                and selected_count >= int(min_select)
                and (
                    (
                        use_anchor_confidence
                        and float(conf_term_n[best_pos].item()) < float(optional_conf_threshold)
                    )
                    or (
                        use_stability_gate
                        and gate_mode == "bulk"
                        and float(conf_term_base[best_pos].item()) < float(high_conf_threshold)
                    )
                )
            ):
                loop_stop_reason = "budget"
                break

            selected_scored_count += 1
            unlock_term_value = float(unlock_term_n[best_pos].item())
            conf_term_value = float(conf_term[best_pos].item())
            risk_term_value = float(risk_term_n[best_pos].item())
            risk_penalty_value = float(risk_penalty[best_pos].item())
            final_score_value = float(best_score.item())
            if use_certified_like:
                best_local_for_diag = feasible[best_pos]
                cert_value = float(cert[best_local_for_diag].item())
                hazard_value = float(hazard[best_local_for_diag].item())
                out_influence_value = float(outgoing_influence[best_local_for_diag].item())
                hazard_anchor_value = bool(hazard_anchor_mask[best_local_for_diag].item())
                certified_value = bool(certified_mask[best_local_for_diag].item())
                cert_selected_sum += cert_value
                hazard_selected_sum += hazard_value
                out_influence_selected_sum += out_influence_value
                certified_gain_selected_sum += raw_gain_value
                hazard_anchor_selected_count += int(hazard_anchor_value)
                certified_selected_count += int(certified_value)

            unlock_term_selected_sum += unlock_term_value
            conf_term_selected_sum += conf_term_value
            risk_term_selected_sum += risk_term_value
            risk_penalty_selected_sum += risk_penalty_value
            final_score_selected_sum += final_score_value

            selected_classified_count += 1
            dominance_terms = [
                ("unlock", unlock_term_value),
                ("conf", conf_term_value),
                ("risk", risk_penalty_value),
            ]
            dominance_terms.sort(key=lambda item: item[1], reverse=True)
            top_name, top_value = dominance_terms[0]
            second_value = dominance_terms[1][1]

            if top_value > 0.0 and (second_value / top_value) > 0.8:
                mixed_count += 1
            elif top_name == "unlock":
                unlock_dominant_count += 1
            elif top_name == "conf":
                conf_dominant_count += 1
            elif top_name == "risk":
                risk_dominant_count += 1
            else:
                mixed_count += 1

            selected_raw_gain_sequence.append(raw_gain_value)
            selected_raw_gain_sum += raw_gain_value
            selected_legacy_gain_sequence.append(legacy_gain_value)
            selected_legacy_gain_sum += legacy_gain_value

            selected_mask[best_local] = True
            selected_count += 1
            selected_conf_mass += float(source_conf[best_local].item())
            current_cover = torch.maximum(current_cover, influence[:, best_local])

        if loop_stop_reason == "cap":
            stop_cap_count += 1
        elif loop_stop_reason == "no_remaining":
            stop_no_remaining_count += 1
        elif loop_stop_reason == "conflict":
            stop_conflict_count += 1
        elif loop_stop_reason == "nonpositive_score":
            stop_nonpositive_count += 1
        elif loop_stop_reason == "budget":
            stop_budget_count += 1
        else:
            stop_other_count += 1

        selected = torch.nonzero(selected_mask, as_tuple=True)[0]
        selected_positions = candidate_positions.index_select(0, selected)

        if selected_positions.numel() == 0:
            fallback_row_count += 1
            if use_certified_like and candidate_positions.numel() > 0:
                fallback_score = source_conf - hazard
                fallback_local = torch.argmax(fallback_score, dim=0, keepdim=True)
                selected_positions = candidate_positions.index_select(0, fallback_local)
            else:
                fallback_k = max(1, int(min_select))
                fallback_k = min(fallback_k, int(num_masked))
                _, fallback_rel = torch.topk(conf_masked, k=fallback_k, dim=0)
                selected_positions = masked_positions.index_select(0, fallback_rel)

        residual_selected_count_total += max(0, int(selected_positions.numel()) - int(easy_rel.numel()))

        if len(selected_raw_gain_sequence) > 0:
            raw_gain_first_sum += float(selected_raw_gain_sequence[0])
            raw_gain_last_sum += float(selected_raw_gain_sequence[-1])

        transfer_index[b, selected_positions] = True
        selected_total += int(selected_positions.numel())

    diag_denom = max(1, active_rows)
    gate_denom = max(1, gate_metric_count)

    stats = {
        "step_idx": int(step_idx),
        "step_in_block": int(step_in_block),
        "candidate_pool_total": candidate_pool_total,
        "selected_total": selected_total,
        "active_rows": active_rows,
        "masked_before_total": masked_before_total,
        "selected_scored_count": selected_scored_count,
        "unlock_term_selected_sum": unlock_term_selected_sum,
        "conf_term_selected_sum": conf_term_selected_sum,
        "risk_term_selected_sum": risk_term_selected_sum,
        "risk_penalty_selected_sum": risk_penalty_selected_sum,
        "final_score_selected_sum": final_score_selected_sum,
        "unlock_dominant_count": unlock_dominant_count,
        "conf_dominant_count": conf_dominant_count,
        "risk_dominant_count": risk_dominant_count,
        "mixed_count": mixed_count,
        "selected_classified_count": selected_classified_count,
        "best_count": best_count,
        "best_score_sum": best_score_sum,
        "best_unlock_term_sum": best_unlock_term_sum,
        "best_conf_term_sum": best_conf_term_sum,
        "best_risk_term_sum": best_risk_term_sum,
        "best_risk_penalty_sum": best_risk_penalty_sum,
        "avg_raw_gain_first": raw_gain_first_sum / float(diag_denom),
        "avg_raw_gain_last": raw_gain_last_sum / float(diag_denom),
        "avg_selected_per_step": selected_total / float(diag_denom),
        "technique_used": technique,
        "avg_quota_floor": quota_floor_total / float(diag_denom),
        "avg_quota_cap": quota_cap_total / float(diag_denom),
        "avg_anchor_weight": anchor_weight_total / float(diag_denom),
        "avg_conf_weight": conf_weight_total / float(diag_denom),
        "avg_risk_weight": risk_weight_total / float(diag_denom),
        "gate_anchor_mode_count": int(gate_anchor_mode_count),
        "gate_mixed_mode_count": int(gate_mixed_mode_count),
        "gate_bulk_mode_count": int(gate_bulk_mode_count),
        "gate_metric_count": int(gate_metric_count),
        "avg_stable_frac": stable_frac_sum / float(gate_denom),
        "avg_stable_high_conf_frac": stable_high_conf_frac_sum / float(gate_denom),
        "avg_high_conf_frac": high_conf_frac_sum / float(gate_denom),
        "avg_gate_conf": gate_conf_sum / float(gate_denom),
        "cert_selected_sum": cert_selected_sum,
        "hazard_selected_sum": hazard_selected_sum,
        "out_influence_selected_sum": out_influence_selected_sum,
        "certified_gain_selected_sum": certified_gain_selected_sum,
        "hazard_anchor_selected_count": int(hazard_anchor_selected_count),
        "certified_selected_count": int(certified_selected_count),
        "hazard_block_count": int(hazard_block_count),
        "stopping_trigger_count": int(stopping_trigger_count),
        "stop_cap_count": int(stop_cap_count),
        "stop_no_remaining_count": int(stop_no_remaining_count),
        "stop_conflict_count": int(stop_conflict_count),
        "stop_nonpositive_count": int(stop_nonpositive_count),
        "stop_budget_count": int(stop_budget_count),
        "stop_other_count": int(stop_other_count),
        "fallback_row_count": int(fallback_row_count),
        "budget_mode_used": budget_mode,
        "axon_reveal_policy": reveal_policy,
        "axon_gain_eta": float(gain_eta),
        "axon_mass_ratio": float(mass_ratio),
        "easy_selected_count": easy_selected_count_total,
        "residual_selected_count": residual_selected_count_total,
        "avg_easy_score": easy_score_selected_sum / float(max(1, easy_score_selected_count)),
        "mode_used": mode_used,
    }
    return transfer_index, stats
