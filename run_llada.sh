#!/usr/bin/env bash
# run_llada.sh
# Run LLaDA on a benchmark with one of the four supported AXON variants.
#
#   MODEL    = llada15 | llada8b_instruct              (default: llada15)
#   TASK     = humaneval | mbpp | gsm8k | minerva_math (default: humaneval)
#   NUM_FEWSHOT  = integer                             (default: 0)
#   VARIANT  = coverage_gap_fixed1 | coverage_gap_cover
#            | progress_stall_fixed1 | progress_stall_cover | all
#                                                      (default: coverage_gap_fixed1)
#   LENGTH       = max generation length               (default: 256)
#   BLOCK_LENGTH = denoising block length              (default: 32)
#   LIMIT        = optional --limit N for lm-eval      (default: unset)
#   RESULTS_DIR  = output root                         (default: ./results)
#
# Requires an active Python env with torch, transformers, accelerate,
# lm-eval-harness, etc. installed (see requirements-lock.txt).
set -euo pipefail

# HumanEval / MBPP run the `code_eval` metric which executes generated
# Python code; the metric refuses to run unless this is set. lm-eval's
# --confirm_run_unsafe_code flag is *separate* from this env var.
export HF_ALLOW_CODE_EVAL=1

MODEL=${MODEL:-llada15}
TASK=${TASK:-humaneval}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
VARIANT=${VARIANT:-coverage_gap_fixed1}
LENGTH=${LENGTH:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
LIMIT=${LIMIT:-}
RESULTS_DIR=${RESULTS_DIR:-./results}

case "$MODEL" in
  llada15)          MODEL_PATH=GSAI-ML/LLaDA-1.5;          MODEL_NAME=LLaDA-1.5 ;;
  llada8b_instruct) MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct;  MODEL_NAME=LLaDA-8B-Instruct ;;
  *) echo "Unknown MODEL=$MODEL"; exit 1 ;;
esac

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

  local model_args="model_path=${MODEL_PATH},gen_length=${LENGTH},steps=${LENGTH},\
block_length=${BLOCK_LENGTH},show_speed=True,outp_path=${out_dir}/results.jsonl,\
tau_sink=0.01,tau_edge=0.07,tau_induce=0.7,tau_low=0.7,\
unlock=True,candidate_topk=32,candidate_min_topk=32,candidate_ratio=1.0,\
warmup_steps=0,conflict_threshold=0.08,min_select=1,\
unlock_weight_func=base,unlock_alpha=0.0,unlock_beta=0.0,unlock_score_mode=risk_reward,\
unlock_lambda=1.0,unlock_gamma=1.0,unlock_use_clean=True,unlock_stage2_fill=True,\
unlock_fill_ratio=0.5,unlock_exit_mode=confidence,unlock_max_anchor_steps=1,\
unlock_exit_conf_threshold=0.85,unlock_target_nfe_per_block=16,\
unlock_technique=submodular-exit,\
axon_stag_gate=${gate},axon_stag_selector=${selector},\
axon_base_proposer=dawn"

  local limit_args=()
  [ -n "${LIMIT}" ] && limit_args=(--limit "${LIMIT}")

  echo "=============================================================="
  echo "  MODEL=${MODEL_NAME}  TASK=${TASK}  VARIANT=${variant}"
  echo "  GPUs=${n_proc}  OUT=${out_dir}"
  echo "=============================================================="

  ( cd llada && \
    accelerate launch --multi_gpu --num_processes "${n_proc}" --main_process_port "${port}" \
      eval_llada.py --model llada_dist \
        --model_args "${model_args}" \
        --tasks "${TASK}" \
        --num_fewshot "${NUM_FEWSHOT}" \
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
