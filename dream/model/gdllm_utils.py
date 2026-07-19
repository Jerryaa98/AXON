import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

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

def detect_attn_sinks_(attn_scores, threshold=0.02):
    """
    attn_scores: [B, S, S] softmax attention
    """
    B, S, _ = attn_scores.shape

    # column mean
    barA = attn_scores.mean(dim=1)          # [B, S]
    return barA > threshold


def _quota_for_target_nfe(step_in_block, num_masked, block_length, target_nfe, min_select=1):
    target_steps = max(1, int(np.ceil(float(target_nfe))))
    if int(step_in_block) >= target_steps - 1:
        return int(num_masked)
    avg_quota = max(1, int(np.ceil(float(block_length) / float(target_steps))))
    return min(int(num_masked), max(int(min_select), avg_quota))


def _normalize_vec(values):
    if values.numel() == 0:
        return values
    v_min = values.min()
    v_max = values.max()
    span = v_max - v_min
    if float(span.item()) <= 1e-12:
        return torch.zeros_like(values)
    return (values - v_min) / (span + 1e-8)


def select_high_confidence_fill(
    mask_index,
    confidence,
    already_selected,
    candidate_topk=32,
    candidate_min_topk=32,
    candidate_ratio=1.0,
    fill_ratio=0.5,
    select_cap=None,
    high_conf_threshold=0.0,
):
    """Stage-2 fill: pick top-confidence masked positions excluding
    `already_selected`, capped at ceil(fill_ratio * select_cap) per row.

    Mirrors the stage-2 logic from llada/gdllm_utils.py:get_transfer_index_unlock
    so a Dream AXON step can match LLaDA's per-step token throughput.
    """
    B = mask_index.shape[0]
    fill_index = torch.zeros_like(mask_index, dtype=torch.bool)
    if select_cap is None:
        return fill_index
    fill_count = max(0, int(np.ceil(float(fill_ratio) * float(select_cap))))
    if fill_count <= 0:
        return fill_index

    for b in range(B):
        available_mask = mask_index[b] & ~already_selected[b]
        available_pos = torch.nonzero(available_mask, as_tuple=True)[0]
        if available_pos.numel() == 0:
            continue
        avail_conf = confidence[b].index_select(0, available_pos).float()
        if float(high_conf_threshold) > 0.0:
            keep = avail_conf >= float(high_conf_threshold)
            if not keep.any():
                continue
            available_pos = available_pos[keep]
            avail_conf = avail_conf[keep]

        n_avail = int(available_pos.numel())
        k_t = max(int(candidate_min_topk), int(float(candidate_ratio) * n_avail))
        k_t = min(int(candidate_topk), k_t, n_avail)
        k_t = max(k_t, 1)
        n_take = min(int(fill_count), int(k_t), n_avail)
        if n_take <= 0:
            continue
        _, top_rel = torch.topk(avail_conf, k=n_take)
        fill_index[b, available_pos.index_select(0, top_rel)] = True

    return fill_index


def select_axon_submodular_anchors(
    mask_index,
    confidence,
    avg_attn_scores,
    tau_sink=0.01,
    candidate_topk=32,
    candidate_min_topk=32,
    candidate_ratio=1.0,
    min_select=1,
    step_in_block=0,
    target_nfe_per_block=16,
    block_length=32,
    unlock_lambda=1.0,
    unlock_gamma=1.0,
    conflict_threshold=0.08,
    axon_reveal_policy="fixed1",
    axon_max_anchors_per_step=1,
    axon_anchor_size_penalty=0.0,
    axon_anchor_size_penalty_power=1.0,
):
    """Select a small AXON anchor set with coverage + confidence - redundancy.

    The set function is:
        C(S) + lambda M(S) - gamma R(S)
    where C is attention coverage over residual masked tokens, M is confidence
    mass, and R is pairwise conflict/redundancy among selected anchors.
    """
    assert avg_attn_scores is not None, "avg_attn_scores is None"

    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)

    device = mask_index.device
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=device)
    eps = 1e-8

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

        reveal_policy = str(axon_reveal_policy).strip().lower()
        if reveal_policy in {"", "false", "0"}:
            reveal_policy = "fixed1"
        if reveal_policy == "fixed1":
            quota = min(num_masked, max(int(min_select), 1))
        elif reveal_policy in {"capped", "fixedk"}:
            quota = min(num_masked, max(int(min_select), int(axon_max_anchors_per_step)))
        else:
            quota = _quota_for_target_nfe(
                step_in_block=step_in_block,
                num_masked=num_masked,
                block_length=block_length,
                target_nfe=target_nfe_per_block,
                min_select=min_select,
            )
        quota = min(quota, int(candidate_positions.numel()))

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
        weights = attn_res_to_cand * uncertainty.unsqueeze(1) * candidate_conf.unsqueeze(0)
        current_coverage = torch.zeros((num_masked,), device=device, dtype=weights.dtype)
        selected_rel: list[int] = []
        available = torch.ones((candidate_positions.numel(),), device=device, dtype=torch.bool)
        a_candidate = (
            avg_attn_scores[b]
            .index_select(0, candidate_positions)
            .index_select(1, candidate_positions)
            .float()
            .clone()
        )
        if cand_sink_mask.any():
            a_candidate[:, cand_sink_mask] = 0.0
        a_candidate.diagonal().zero_()
        candidate_uncertainty = (1.0 - candidate_conf).clamp_min(0.0)
        conflict = 0.5 * (a_candidate + a_candidate.transpose(0, 1))
        conflict = conflict * (candidate_uncertainty.unsqueeze(1) * candidate_uncertainty.unsqueeze(0))
        conflict.diagonal().zero_()

        for _ in range(quota):
            remaining = torch.nonzero(available, as_tuple=True)[0]
            if remaining.numel() == 0:
                break

            feasible = remaining
            if selected_rel:
                selected_tensor = torch.tensor(selected_rel, device=device, dtype=torch.long)
                pair_conflict = conflict.index_select(0, remaining).index_select(1, selected_tensor)
                feasible = remaining[(pair_conflict <= float(conflict_threshold)).all(dim=1)]
            if feasible.numel() == 0:
                if len(selected_rel) < int(min_select):
                    feasible = remaining
                else:
                    break

            feasible_weights = weights.index_select(1, feasible)
            coverage_gain = (torch.maximum(current_coverage.unsqueeze(1), feasible_weights) - current_coverage.unsqueeze(1)).sum(dim=0)
            coverage_term = _normalize_vec(coverage_gain / float(max(1, num_masked)))
            conf_term = _normalize_vec(candidate_conf.index_select(0, feasible))
            if selected_rel:
                selected_tensor = torch.tensor(selected_rel, device=device, dtype=torch.long)
                risk_term = conflict.index_select(0, feasible).index_select(1, selected_tensor).sum(dim=1) / float(max(1, len(selected_rel)))
            else:
                risk_term = torch.zeros(feasible.numel(), dtype=weights.dtype, device=device)
            risk_term = _normalize_vec(risk_term)

            size_penalty = float(axon_anchor_size_penalty) * float(len(selected_rel) + 1) ** float(axon_anchor_size_penalty_power)
            score = coverage_term + float(unlock_lambda) * conf_term - float(unlock_gamma) * risk_term - size_penalty
            best_pos = torch.argmax(score)
            best_gain = score[best_pos]
            if float(best_gain.item()) <= 0.0 and len(selected_rel) >= int(min_select):
                break
            best_idx = int(feasible[best_pos].item())
            selected_rel.append(best_idx)
            available[best_idx] = False
            current_coverage = torch.maximum(current_coverage, weights[:, best_idx])

        if not selected_rel:
            selected_rel = [int(torch.argmax(candidate_conf).item())]

        selected_tensor = torch.tensor(selected_rel, device=device, dtype=torch.long)
        selected_positions = candidate_positions.index_select(0, selected_tensor)
        transfer_index[b, selected_positions] = True

    return transfer_index


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
    axon_submod_fn="facility_location",
    axon_submod_monotone=True,
    axon_submod_penalty="conflict",
    axon_submod_lambda=0.0,
):
    """Select AXON anchors with an adaptive reveal budget.

    The selector decides how many anchors to reveal from the current residual
    state, using DAWN's own coverage certificate as the comparison scale.

    axon_submod_fn:       "facility_location" (sum_i max_{j in S} w_ij) or
                          "graph_cut" (sum_i sum_{j in S} w_ij, modular marginal).
    axon_submod_monotone: when False, subtract lambda * sum_{j,k in S} penalty_jk
                          from the greedy score (non-monotone objective).
    axon_submod_penalty:  "conflict" (existing uncertainty-weighted matrix) or
                          "attn" (raw symmetrized candidate attention).
    axon_submod_lambda:   penalty weight (only used when monotone is False).
    Defaults reproduce the original facility-location monotone behavior exactly.
    """
    assert avg_attn_scores is not None, "avg_attn_scores is None"

    selector = str(selector).strip().lower()
    if selector in {"", "false", "0"}:
        selector = "fixed1"
    if selector not in {"fixed1", "density", "cover", "nonmono"}:
        raise ValueError(f"Unsupported adaptive AXON selector: {selector}")

    submod_fn = str(axon_submod_fn).strip().lower()
    if submod_fn in {"", "fl", "facility", "facility-location"}:
        submod_fn = "facility_location"
    if submod_fn in {"gc", "graphcut", "graph-cut"}:
        submod_fn = "graph_cut"
    if submod_fn not in {"facility_location", "graph_cut"}:
        raise ValueError(f"Unsupported axon_submod_fn: {axon_submod_fn}")
    if isinstance(axon_submod_monotone, str):
        submod_monotone = axon_submod_monotone.strip().lower() not in {"false", "0", "no", ""}
    else:
        submod_monotone = bool(axon_submod_monotone)
    submod_penalty = str(axon_submod_penalty).strip().lower()
    if submod_penalty in {"", "conflict"}:
        submod_penalty = "conflict"
    elif submod_penalty in {"attn", "attention", "raw"}:
        submod_penalty = "attn"
    else:
        raise ValueError(f"Unsupported axon_submod_penalty: {axon_submod_penalty}")
    submod_lambda = float(axon_submod_lambda)

    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)
    device = mask_index.device
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=device)
    dawn_density = max(0.0, float(dawn_density))
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

        weights = attn_res_to_cand * uncertainty.unsqueeze(1) * candidate_conf.unsqueeze(0)
        current_coverage = torch.zeros((num_masked,), device=device, dtype=weights.dtype)
        selected_rel: list[int] = []
        available = torch.ones((candidate_positions.numel(),), device=device, dtype=torch.bool)

        a_candidate = (
            avg_attn_scores[b]
            .index_select(0, candidate_positions)
            .index_select(1, candidate_positions)
            .float()
            .clone()
        )
        if cand_sink_mask.any():
            a_candidate[:, cand_sink_mask] = 0.0
        a_candidate.diagonal().zero_()
        candidate_uncertainty = (1.0 - candidate_conf).clamp_min(0.0)
        conflict = 0.5 * (a_candidate + a_candidate.transpose(0, 1))
        conflict = conflict * (candidate_uncertainty.unsqueeze(1) * candidate_uncertainty.unsqueeze(0))
        conflict.diagonal().zero_()
        if submod_penalty == "attn":
            pen_mat = 0.5 * (a_candidate + a_candidate.transpose(0, 1))
            pen_mat.diagonal().zero_()
        else:
            pen_mat = conflict

        while len(selected_rel) < max_select:
            if selector == "cover" and len(selected_rel) >= int(min_select):
                if float(current_coverage.sum().item()) >= target_coverage:
                    break

            remaining = torch.nonzero(available, as_tuple=True)[0]
            if remaining.numel() == 0:
                break

            feasible = remaining
            if selected_rel and selector == "nonmono":
                selected_tensor = torch.tensor(selected_rel, device=device, dtype=torch.long)
                pair_conflict = conflict.index_select(0, remaining).index_select(1, selected_tensor)
                feasible = remaining[(pair_conflict <= float(conflict_threshold)).all(dim=1)]
                if feasible.numel() == 0:
                    if len(selected_rel) < int(min_select):
                        feasible = remaining
                    else:
                        break

            feasible_weights = weights.index_select(1, feasible)
            if submod_fn == "graph_cut":
                # Modular per-target sum: marginal gain is independent of the
                # current selection (greedy reduces to top-k by column mass).
                coverage_gain = feasible_weights.sum(dim=0)
            else:
                coverage_gain = (
                    torch.maximum(current_coverage.unsqueeze(1), feasible_weights)
                    - current_coverage.unsqueeze(1)
                ).sum(dim=0)

            if (not submod_monotone) and submod_lambda != 0.0 and selected_rel:
                sel_idx_t = torch.tensor(selected_rel, device=device, dtype=torch.long)
                submod_pen = submod_lambda * (
                    pen_mat.index_select(0, feasible).index_select(1, sel_idx_t).sum(dim=1)
                )
            else:
                submod_pen = 0.0

            if selector == "nonmono":
                if selected_rel:
                    selected_tensor = torch.tensor(selected_rel, device=device, dtype=torch.long)
                    risk = conflict.index_select(0, feasible).index_select(1, selected_tensor).sum(dim=1)
                else:
                    risk = torch.zeros(feasible.numel(), dtype=weights.dtype, device=device)
                positive = coverage_gain[coverage_gain > 0]
                size_cost = positive.median() / float(max(1, max_select)) if positive.numel() else torch.tensor(0.0, device=device)
                score = coverage_gain + 1e-3 * candidate_conf.index_select(0, feasible) - risk - size_cost * float(len(selected_rel) + 1)
                score = score - submod_pen
            else:
                score = coverage_gain + 1e-6 * candidate_conf.index_select(0, feasible) - submod_pen

            best_pos = torch.argmax(score)
            best_idx = int(feasible[best_pos].item())
            best_gain = float(coverage_gain[best_pos].item())
            best_score = float(score[best_pos].item())

            if selector == "density" and len(selected_rel) >= int(min_select) and best_gain < dawn_density:
                break
            if selector == "nonmono" and len(selected_rel) >= int(min_select) and best_score <= 0.0:
                break
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


def select_axon_adaptive_minimal_anchors(
    mask_index,
    confidence,
    avg_attn_scores,
    tau_sink=0.01,
    candidate_topk=32,
    candidate_min_topk=32,
    candidate_ratio=1.0,
    min_select=1,
    conflict_threshold=0.08,
    axon_cover_target=0.50,
    axon_max_reveal_frac=0.0625,
):
    """Select the smallest feasible AXON anchor set for partial coverage.

    This is the monotone submodular partial-cover plugin:
        C(S) = sum_i max_{j in S} uncertainty_i * confidence_j * attention_ij
    Greedy adds anchors by marginal coverage gain and stops as soon as the
    requested fraction of candidate-pool coverage is reached.
    """
    assert avg_attn_scores is not None, "avg_attn_scores is None"

    sink_mask = detect_attn_sinks_(avg_attn_scores, threshold=tau_sink)

    device = mask_index.device
    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool, device=device)
    cover_target = min(1.0, max(0.0, float(axon_cover_target)))
    max_reveal_frac = max(0.0, float(axon_max_reveal_frac))

    for b in range(mask_index.shape[0]):
        masked_positions = torch.nonzero(mask_index[b], as_tuple=True)[0]
        num_masked = int(masked_positions.numel())
        if num_masked == 0:
            continue

        conf_masked = confidence[b].index_select(0, masked_positions).float()
        k_t = int(float(candidate_ratio) * num_masked)
        k_t = max(int(candidate_min_topk), k_t)
        k_t = min(int(candidate_topk), k_t, num_masked)
        k_t = max(k_t, 1)
        _, candidate_rel = torch.topk(conf_masked, k=k_t, dim=0)
        candidate_positions = masked_positions.index_select(0, candidate_rel)
        candidate_conf = confidence[b].index_select(0, candidate_positions).float()

        quota = int(np.ceil(max_reveal_frac * float(num_masked)))
        quota = min(num_masked, max(int(min_select), quota))
        quota = min(quota, int(candidate_positions.numel()))

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
        weights = attn_res_to_cand * uncertainty.unsqueeze(1) * candidate_conf.unsqueeze(0)
        target_coverage = cover_target * float(weights.max(dim=1).values.sum().item())

        current_coverage = torch.zeros((num_masked,), device=device, dtype=weights.dtype)
        selected_rel: list[int] = []
        available = torch.ones((candidate_positions.numel(),), device=device, dtype=torch.bool)

        a_candidate = (
            avg_attn_scores[b]
            .index_select(0, candidate_positions)
            .index_select(1, candidate_positions)
            .float()
            .clone()
        )
        if cand_sink_mask.any():
            a_candidate[:, cand_sink_mask] = 0.0
        a_candidate.diagonal().zero_()
        candidate_uncertainty = (1.0 - candidate_conf).clamp_min(0.0)
        conflict = 0.5 * (a_candidate + a_candidate.transpose(0, 1))
        conflict = conflict * (candidate_uncertainty.unsqueeze(1) * candidate_uncertainty.unsqueeze(0))
        conflict.diagonal().zero_()

        while len(selected_rel) < quota:
            covered = float(current_coverage.sum().item())
            if len(selected_rel) >= int(min_select) and covered >= target_coverage:
                break

            remaining = torch.nonzero(available, as_tuple=True)[0]
            if remaining.numel() == 0:
                break

            feasible = remaining
            if selected_rel:
                selected_tensor = torch.tensor(selected_rel, device=device, dtype=torch.long)
                pair_conflict = conflict.index_select(0, remaining).index_select(1, selected_tensor)
                feasible = remaining[(pair_conflict <= float(conflict_threshold)).all(dim=1)]
            if feasible.numel() == 0:
                if len(selected_rel) < int(min_select):
                    feasible = remaining
                else:
                    break

            feasible_weights = weights.index_select(1, feasible)
            coverage_gain = (
                torch.maximum(current_coverage.unsqueeze(1), feasible_weights)
                - current_coverage.unsqueeze(1)
            ).sum(dim=0)
            tie_break = 1e-6 * candidate_conf.index_select(0, feasible)
            best_pos = torch.argmax(coverage_gain + tie_break)
            best_gain = coverage_gain[best_pos]
            if float(best_gain.item()) <= 0.0 and len(selected_rel) >= int(min_select):
                break

            best_idx = int(feasible[best_pos].item())
            selected_rel.append(best_idx)
            available[best_idx] = False
            current_coverage = torch.maximum(current_coverage, weights[:, best_idx])

        if not selected_rel:
            selected_rel = [int(torch.argmax(candidate_conf).item())]

        selected_tensor = torch.tensor(selected_rel, device=device, dtype=torch.long)
        selected_positions = candidate_positions.index_select(0, selected_tensor)
        transfer_index[b, selected_positions] = True

    return transfer_index
