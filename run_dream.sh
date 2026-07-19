#!/usr/bin/env bash
# run_dream.sh
# Run Dream on a benchmark with one of the four supported AXON variants.
#
#   MODEL    = dream_base | dream_instruct                (default: dream_base)
#   TASK     = humaneval | mbpp | gsm8k | minerva_math
#            | truthfulqa_gen                             (default: humaneval)
#   NUM_FEWSHOT  = integer   (default: per-task -- gsm8k 5, minerva_math 4,
#                             mbpp 3, humaneval 0, truthfulqa_gen 0)
#   VARIANT  = coverage_gap_fixed1 | coverage_gap_cover
#            | progress_stall_fixed1 | progress_stall_cover | all
#                                                         (default: progress_stall_fixed1)
#   LENGTH       = max generation length                 (default: 256)
#   BLOCK_LENGTH = denoising block length                (default: 32)
#   LIMIT        = optional --limit N for lm-eval        (default: unset)
#   RESULTS_DIR  = output root                           (default: ./results)
#
# Notes
#   * Dream's per-family gate is `progress_stall` (the "pace-deficit" gate:
#     reveal fraction non-increasing and behind schedule). The Dream base
#     proposer is strong/well-calibrated, so this rarely-firing gate is the
#     default here; `coverage_gap` (the coverage-deficit gate, and the default
#     used for LLaDA) is exposed via VARIANT for parity with run_llada.sh.
#   * TruthfulQA (truthfulqa_gen) is open-ended 0-shot QA: on the instruct model
#     raw completion collapses to empty, so the chat template is applied
#     automatically for that task. HumanEval likewise force-applies the chat
#     template internally on the instruct model (see dream/eval.py).
#
# Requires an active Python env with torch, transformers==4.49.0, accelerate,
# lm-eval-harness, etc. installed (see requirements-lock.txt).
set -euo pipefail

# HumanEval / MBPP run the `code_eval` metric which executes generated Python
# code; the metric refuses to run unless this is set. lm-eval's
# --confirm_run_unsafe_code flag is *separate* from this env var.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

MODEL=${MODEL:-dream_base}
TASK=${TASK:-humaneval}
VARIANT=${VARIANT:-progress_stall_fixed1}
LENGTH=${LENGTH:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
LIMIT=${LIMIT:-}
RESULTS_DIR=${RESULTS_DIR:-./results}

case "$MODEL" in
  dream_base)     MODEL_PATH=Dream-org/Dream-v0-Base-7B;     MODEL_NAME=Dream-v0-Base-7B;     TAU_EDGE=0.05 ;;
  dream_instruct) MODEL_PATH=Dream-org/Dream-v0-Instruct-7B; MODEL_NAME=Dream-v0-Instruct-7B; TAU_EDGE=0.10 ;;
  *) echo "Unknown MODEL=$MODEL"; exit 1 ;;
esac

# Per-task few-shot default (overridable via NUM_FEWSHOT env var).
default_fewshot() {
  case "$1" in
    gsm8k)        echo 5 ;;
    minerva_math) echo 4 ;;
    mbpp)         echo 3 ;;
    *)            echo 0 ;;   # humaneval, truthfulqa_gen
  esac
}
NUM_FEWSHOT=${NUM_FEWSHOT:-$(default_fewshot "$TASK")}

run_one() {
  local variant="$1"
  local gate selector
  case "$variant" in
    coverage_gap_fixed1)   gate=coverage_gap;   selector=fixed1 ;;
    coverage_gap_cover)    gate=coverage_gap;   selector=cover  ;;
    progress_stall_fixed1) gate=progress_stall; selector=fixed1 ;;
    progress_stall_cover)  gate=progress_stall; selector=cover  ;;
    *) echo "Unknown VARIANT=$variant"; exit 1 ;;
  esac

  local out_dir="${RESULTS_DIR}/${MODEL_NAME}/${variant}/${TASK}-ns${NUM_FEWSHOT}-${LENGTH}"
  mkdir -p "${out_dir}"

  local n_proc
  n_proc=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  [ "${n_proc}" -lt 1 ] && n_proc=1
  local port=$((20000 + RANDOM % 40000))

  # gsm8k on the base model uses a slightly lower confidence threshold.
  local conf_threshold=0.8
  [ "${MODEL}" = "dream_base" ] && [ "${TASK}" = "gsm8k" ] && conf_threshold=0.75

  # Dream-Instruct produces empty raw completions on 0-shot open-ended QA;
  # chat-template the TruthfulQA prompts (control and AXON both inherit it).
  local chat_arg=""
  [ "${TASK}" = "truthfulqa_gen" ] && chat_arg=",apply_chat_template=true"

  local model_args="pretrained=${MODEL_PATH},max_new_tokens=${LENGTH},\
outp_path=${out_dir}/results.jsonl,\
add_bos_token=true,diffusion_steps=${LENGTH},block_length=${BLOCK_LENGTH},show_speed=True,\
tau_sink=0.03,tau_edge=${TAU_EDGE},tau_induce=0.75,conf_threshold=${conf_threshold}${chat_arg},\
alg=axon,candidate_topk=32,candidate_min_topk=32,candidate_ratio=1.0,\
conflict_threshold=0.08,min_select=1,unlock_lambda=1.0,unlock_gamma=1.0,\
unlock_exit_conf_threshold=0.85,unlock_target_nfe_per_block=16,unlock_max_anchor_steps=1,\
axon_step_combo_mode=none,axon_reveal_policy=fixed1,\
axon_plugin=stag_unified,axon_stag_gate=${gate},axon_stag_selector=${selector},\
axon_base_proposer=dawn"

  local limit_args=()
  [ -n "${LIMIT}" ] && limit_args=(--limit "${LIMIT}")

  echo "=============================================================="
  echo "  MODEL=${MODEL_NAME}  TASK=${TASK}  VARIANT=${variant}"
  echo "  GPUs=${n_proc}  OUT=${out_dir}"
  echo "=============================================================="

  ( cd dream && \
    accelerate launch --num_processes "${n_proc}" --main_process_port "${port}" \
      eval.py --model dream \
        --model_args "${model_args}" \
        --tasks "${TASK}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --batch_size 1 \
        --confirm_run_unsafe_code \
        --output_path "${out_dir}" \
        --log_samples "${limit_args[@]}" )
}

if [ "${VARIANT}" = "all" ]; then
  for v in coverage_gap_fixed1 coverage_gap_cover \
           progress_stall_fixed1 progress_stall_cover; do
    run_one "$v"
  done
else
  run_one "${VARIANT}"
fi
