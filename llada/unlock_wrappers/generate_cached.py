"""KV-cached unlock generation path.

Composes the block-prefix-cache pattern from generate.generate_with_prefix_cache
with the unlock scheduler. The first forward of each block is full-sequence
(and produces both the attention matrix for unlock and the prefix K/V cache);
subsequent within-block steps run only over `x[:, current_block_start:]` with
the prefix cached.

CORRECTNESS CAVEAT (mirrors the existing cached path in generate.py):
  LLaDA attention is bidirectional. The prefix K/V cache freezes the prefix
  representation as of the first forward pass, before any tokens in the
  current block are revealed. This is an approximation; outputs may diverge
  slightly from the uncached path. The existing `generate_with_prefix_cache`
  in generate.py accepts the same tradeoff, and the KV-cache equivalence
  launcher reports drift quantitatively.

Only the unlock-use path is supported (the flag that activates the scheduler).
All non-unlock branches fall through to the original `generate()`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import generate as _gen_mod
from generate import (
    add_gumbel_noise,
    get_num_transfer_tokens,
)


@torch.no_grad()
def generate_with_cache_and_unlock(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
    dawn=False,
    unlock=False,
    local_leap=False,
    tau_sink=0.01,
    tau_edge=0.07,
    tau_induce=0.7,
    tau_low=0.7,
    candidate_topk=64,
    candidate_min_topk=16,
    candidate_ratio=0.25,
    warmup_steps=1,
    conflict_threshold=0.05,
    multiplier_after_warmup=2.0,
    min_select=1,
    unlock_weight_func="base",
    unlock_alpha=0.0,
    unlock_beta=0.0,
    unlock_score_mode="current",
    unlock_lambda=0.0,
    unlock_gamma=0.0,
    unlock_use_clean=False,
    unlock_stage2_fill=False,
    unlock_fill_ratio=0.5,
    relaxed_threshold=0.75,
    radius=4,
    unlock_profile=None,
):
    # Only the unlock path is accelerated; everything else delegates.
    if not unlock or factor is not None or dawn or local_leap:
        return _gen_mod.generate(
            model, prompt,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=temperature, remasking=remasking, mask_id=mask_id,
            threshold=threshold, factor=factor,
            dawn=dawn, unlock=unlock, local_leap=local_leap,
            tau_sink=tau_sink, tau_edge=tau_edge, tau_induce=tau_induce, tau_low=tau_low,
            candidate_topk=candidate_topk, candidate_min_topk=candidate_min_topk,
            candidate_ratio=candidate_ratio, warmup_steps=warmup_steps,
            conflict_threshold=conflict_threshold,
            multiplier_after_warmup=multiplier_after_warmup, min_select=min_select,
            unlock_weight_func=unlock_weight_func, unlock_alpha=unlock_alpha, unlock_beta=unlock_beta,
            unlock_score_mode=unlock_score_mode, unlock_lambda=unlock_lambda, unlock_gamma=unlock_gamma,
            unlock_use_clean=unlock_use_clean,
            unlock_stage2_fill=unlock_stage2_fill, unlock_fill_ratio=unlock_fill_ratio,
            relaxed_threshold=relaxed_threshold, radius=radius, unlock_profile=unlock_profile,
        )

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id, dtype=torch.long,
    ).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    if unlock_profile is not None:
        unlock_profile.setdefault("scheduler_steps", 0)
        unlock_profile.setdefault("candidate_pool_total", 0)
        unlock_profile.setdefault("selected_total", 0)
        unlock_profile.setdefault("active_rows", 0)
        unlock_profile.setdefault("step_records", [])

    full_len = x.shape[1]
    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        i = 0

        # First forward of the block: full sequence, with cache and attention.
        output, avg_attn_scores = model(
            x, use_cache=True, return_attn_scores=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits
        nfe += 1

        # Truncate cache to prefix only (positions strictly before current block).
        past_key_values = tuple(
            tuple(layer[j][:, :, :current_block_start] for j in range(len(layer)))
            for layer in past_key_values
        )

        while True:
            mask_index = (x == mask_id)
            mask_index[:, current_block_end:] = 0

            transfer_index, x0 = _unlock_step(
                logits=logits,
                avg_attn_scores=avg_attn_scores,
                mask_index=mask_index,
                x=x,
                temperature=temperature,
                remasking=remasking,
                num_block=num_block,
                step_in_block=i,
                steps_in_block=steps_per_block,
                unlock_use_clean=unlock_use_clean,
                tau_sink=tau_sink,
                candidate_topk=candidate_topk,
                candidate_min_topk=candidate_min_topk,
                candidate_ratio=candidate_ratio,
                warmup_steps=warmup_steps,
                conflict_threshold=conflict_threshold,
                multiplier_after_warmup=multiplier_after_warmup,
                min_select=min_select,
                unlock_weight_func=unlock_weight_func,
                unlock_alpha=unlock_alpha,
                unlock_beta=unlock_beta,
                unlock_score_mode=unlock_score_mode,
                unlock_lambda=unlock_lambda,
                unlock_gamma=unlock_gamma,
                unlock_stage2_fill=unlock_stage2_fill,
                unlock_fill_ratio=unlock_fill_ratio,
                unlock_profile=unlock_profile,
            )
            x[transfer_index] = x0[transfer_index]
            i += 1

            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break

            # Subsequent within-block step: cached prefix + window forward.
            nfe += 1
            out_window, attn_window = model(
                x[:, current_block_start:],
                past_key_values=past_key_values,
                use_cache=True,
                return_attn_scores=True,
            )
            logits_window = out_window.logits

            # Expand window logits back to full-sequence shape expected downstream.
            # The unlock scheduler indexes only masked positions (always in the
            # current-block window), so leaving prefix slots zero is safe here.
            logits = torch.zeros(
                x.shape[0], full_len, logits_window.shape[-1],
                dtype=logits_window.dtype, device=logits_window.device,
            )
            logits[:, current_block_start:] = logits_window

            # Attention from the window forward has rows only for window queries
            # but columns over the full sequence (since prefix K/V is cached).
            # Synthesize (B, full_len, full_len) with zero prefix rows — the
            # scheduler never indexes those rows (masked positions are all within
            # the current block onwards).
            if attn_window is None:
                avg_attn_scores = torch.zeros(
                    x.shape[0], full_len, full_len,
                    dtype=logits.dtype, device=logits.device,
                )
            else:
                avg_attn_scores = torch.zeros(
                    x.shape[0], full_len, full_len,
                    dtype=attn_window.dtype, device=attn_window.device,
                )
                avg_attn_scores[:, current_block_start:, :] = attn_window

    return x, nfe


def _unlock_step(
    *,
    logits,
    avg_attn_scores,
    mask_index,
    x,
    temperature,
    remasking,
    num_block,
    step_in_block,
    steps_in_block,
    unlock_use_clean,
    tau_sink,
    candidate_topk,
    candidate_min_topk,
    candidate_ratio,
    warmup_steps,
    conflict_threshold,
    multiplier_after_warmup,
    min_select,
    unlock_weight_func,
    unlock_alpha,
    unlock_beta,
    unlock_score_mode,
    unlock_lambda,
    unlock_gamma,
    unlock_stage2_fill,
    unlock_fill_ratio,
    unlock_profile,
):
    """One unlock denoising step. Produces (transfer_index, x0) for merging."""
    # The scheduler names are looked up on the generate module so monkey-patches
    # by dispatch.py transparently activate here too.
    get_unlock = _gen_mod.get_transfer_index_unlock
    get_unlock_clean = _gen_mod.get_transfer_index_unlock_clean

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
    step_idx = num_block * steps_in_block + step_in_block

    if unlock_use_clean:
        transfer_index, stats = get_unlock_clean(
            mask_index=mask_index,
            confidence=x0_p,
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
    else:
        transfer_index, stats = get_unlock(
            mask_index=mask_index,
            confidence=x0_p,
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
            weight_func=unlock_weight_func,
            alpha=unlock_alpha,
            beta=unlock_beta,
            unlock_score_mode=unlock_score_mode,
            unlock_lambda=unlock_lambda,
            unlock_gamma=unlock_gamma,
            unlock_stage2_fill=unlock_stage2_fill,
            unlock_fill_ratio=unlock_fill_ratio,
        )

    if unlock_profile is not None:
        unlock_profile["scheduler_steps"] += 1
        unlock_profile["candidate_pool_total"] += stats["candidate_pool_total"]
        unlock_profile["selected_total"] += stats["selected_total"]
        unlock_profile["active_rows"] += stats["active_rows"]
        unlock_profile["step_records"].append(stats)

    return transfer_index, x0
