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
# lm-eval model wrapper for JetLM/SDAR-8B-Chat running the AXON/AXON
# submodular block-diffusion decode. Mirrors `llada/eval_llada.py`
# (@register_model("llada_dist") / LLaDAEvalHarness).

import os
import json
import re
import time
import random

import numpy as np
import torch

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate import generate, enable_attention_capture
from step_profiler import StepProfiler


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("sdar_dist")
class SDAREvalHarness(LM):
    def __init__(
        self,
        model_path="JetLM/SDAR-8B-Chat",
        mask_id=151669,
        max_length=4096,
        batch_size=1,
        device="cuda",
        # ---- block-diffusion decode shape ----
        gen_length=64,
        steps=64,
        block_length=16,
        remasking="low_confidence",
        temperature=0.0,
        confidence_threshold=0.85,
        threshold=None,
        # ---- prompting ----
        # HumanEval ALWAYS uses the chat turn (it carries the implement-the-function
        # instruction). This flag controls every OTHER task. Default True preserves the
        # previous behaviour; set false for completion-style prompts such as
        # truthfulqa_gen, whose doc_to_text is a few-shot "Q: ...\nA: ..." primer that a
        # chat model echoes instead of answering.
        use_chat_template=True,
        # MBPP prompt/decode recipe. "legacy" is the completion-style Branch B (no EOS cut):
        # SDAR-Chat ends its turn immediately on ~20% of problems -> "" , and when it never
        # emits the `[DONE]` stop sequence the raw arm runs to the full gen_length canvas and
        # degenerates. The other modes share an EOS-cut + [BEGIN]/[DONE] extraction decode and
        # differ only in the prompt, so a sweep isolates chat-template vs instruction effects.
        mbpp_mode="legacy",
        # TruthfulQA prompt/decode recipe. "legacy" is Branch B (completion-style until-split):
        # raw -> 10/10 empty (SDAR won't continue the 6-shot Q:/A: primer), chat -> echoes the
        # primer back (the whole primer becomes the chat turn). The non-legacy modes strip the
        # few-shot primer to the FINAL question and use an EOS-cut decode, then cut at the first
        # '\nQ:'/'\n\n' before extracting -- terminator FIRST, then extract (the MBPP bug lesson).
        tqa_mode="legacy",
        # ---- reporting ----
        show_speed=False,
        profile_steps=False,
        profile_tag="",
        outp_path=None,
        save_dir=None,
        # ---- AXON plugin / gate / selector / base proposer ----
        axon_plugin="none",
        axon_stag_gate="coverage_gap",
        axon_stag_selector="fixed1",
        axon_adaptive_gate="progress_empty",
        axon_adaptive_selector="fixed1",
        axon_base_proposer="dawn",
        gate_alpha=0.5,
        adcov_threshold=0.5,
        adcov_warmup=256,
        # ---- DAWN / LocalLeap graph-signal thresholds ----
        tau_sink=0.01,
        tau_edge=0.07,
        tau_induce=0.7,
        tau_low=0.7,
        relaxed_threshold=0.75,
        radius=4,
        # ---- submodular anchor selector knobs ----
        candidate_topk=64,
        candidate_min_topk=16,
        candidate_ratio=0.25,
        min_select=1,
        conflict_threshold=0.05,
        axon_beta_r=1.0,
        axon_beta_u=1.0,
        axon_fixed_k=0,
        axon_submod_fn="facility_location",
        axon_submod_monotone=True,
        axon_submod_penalty="conflict",
        axon_submod_lambda=0.0,
        # ---- anchor-step (submodular-exit) control ----
        unlock_target_nfe_per_block=8,
        unlock_min_anchor_steps=1,
        unlock_max_anchor_steps=1,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.model_path = model_path
        self.mask_id = int(mask_id)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

        # decode shape
        self.gen_length = int(gen_length)
        self.steps = int(steps)
        self.block_length = int(block_length)
        self.remasking = str(remasking)
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.threshold = float(threshold) if threshold is not None else None

        # prompting
        if isinstance(use_chat_template, str):
            self.use_chat_template = use_chat_template.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.use_chat_template = bool(use_chat_template)

        self.mbpp_mode = str(mbpp_mode).strip().lower()
        _MBPP_MODES = {"legacy", "chat_extract", "chat_noinstr", "raw_extract", "official"}
        if self.mbpp_mode not in _MBPP_MODES:
            raise ValueError(f"mbpp_mode must be one of {sorted(_MBPP_MODES)}, got {self.mbpp_mode!r}")

        self.tqa_mode = str(tqa_mode).strip().lower()
        _TQA_MODES = {"legacy", "raw_extract", "chat_extract"}
        if self.tqa_mode not in _TQA_MODES:
            raise ValueError(f"tqa_mode must be one of {sorted(_TQA_MODES)}, got {self.tqa_mode!r}")

        # reporting
        if isinstance(show_speed, str):
            self.show_speed = show_speed.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.show_speed = bool(show_speed)
        if isinstance(profile_steps, str):
            self.profile_steps = profile_steps.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.profile_steps = bool(profile_steps)
        self.profile_tag = str(profile_tag)
        self.outp_path = outp_path
        self.save_dir = save_dir

        # AXON knobs
        self.axon_plugin = str(axon_plugin)
        self.axon_stag_gate = str(axon_stag_gate)
        self.axon_stag_selector = str(axon_stag_selector)
        self.axon_adaptive_gate = str(axon_adaptive_gate)
        self.axon_adaptive_selector = str(axon_adaptive_selector)
        self.axon_base_proposer = str(axon_base_proposer)
        self.gate_alpha = float(gate_alpha)
        self.adcov_threshold = float(adcov_threshold)
        self.adcov_warmup = int(adcov_warmup)

        self.tau_sink = float(tau_sink)
        self.tau_edge = float(tau_edge)
        self.tau_induce = float(tau_induce)
        self.tau_low = float(tau_low)
        self.relaxed_threshold = float(relaxed_threshold) if relaxed_threshold is not None else None
        self.radius = int(radius)

        self.candidate_topk = int(candidate_topk)
        self.candidate_min_topk = int(candidate_min_topk)
        self.candidate_ratio = float(candidate_ratio)
        self.min_select = int(min_select)
        self.conflict_threshold = float(conflict_threshold)
        self.axon_beta_r = float(axon_beta_r)
        self.axon_beta_u = float(axon_beta_u)
        self.axon_fixed_k = int(axon_fixed_k)
        self.axon_submod_fn = str(axon_submod_fn)
        self.axon_submod_monotone = (
            axon_submod_monotone if isinstance(axon_submod_monotone, bool)
            else str(axon_submod_monotone).strip().lower() not in {"false", "0", "no", ""}
        )
        self.axon_submod_penalty = str(axon_submod_penalty)
        self.axon_submod_lambda = float(axon_submod_lambda)

        self.unlock_target_nfe_per_block = float(unlock_target_nfe_per_block)
        self.unlock_min_anchor_steps = int(unlock_min_anchor_steps)
        self.unlock_max_anchor_steps = int(unlock_max_anchor_steps)

        self._rank = 0
        self._world_size = 1

        # ---- load SDAR (Qwen3-derived) via the VENDORED + patched modeling ----
        # We use sdar/model/modeling_sdar.py (flash-attn removed: pure-torch RMSNorm +
        # SDPA attention) instead of trust_remote_code, which would re-require flash_attn.
        # SDAR's own attention returns None for weights, so `enable_attention_capture`
        # is what surfaces the attention maps for the DAWN proposer / graph-signal gates.
        from model.modeling_sdar import SDARForCausalLM
        from model.configuration_sdar import SDARConfig
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        cfg = SDARConfig.from_pretrained(model_path)
        self.model = SDARForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.device).eval()
        enable_attention_capture(self.model)

        self.is_instruct = True

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # ---- lm-eval requires these; not used for generative AXON evals ----
    def loglikelihood(self, requests):
        raise NotImplementedError("SDAREvalHarness only supports generate_until.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("SDAREvalHarness only supports generate_until.")

    def _axon_kwargs(self):
        return dict(
            mask_id=self.mask_id,
            temperature=self.temperature,
            remasking=self.remasking,
            confidence_threshold=self.confidence_threshold,
            threshold=self.threshold,
            axon_plugin=self.axon_plugin,
            axon_stag_gate=self.axon_stag_gate,
            axon_stag_selector=self.axon_stag_selector,
            axon_adaptive_gate=self.axon_adaptive_gate,
            axon_adaptive_selector=self.axon_adaptive_selector,
            axon_base_proposer=self.axon_base_proposer,
            gate_alpha=self.gate_alpha,
            adcov_threshold=self.adcov_threshold,
            adcov_warmup=self.adcov_warmup,
            tau_sink=self.tau_sink,
            tau_edge=self.tau_edge,
            tau_induce=self.tau_induce,
            tau_low=self.tau_low,
            relaxed_threshold=self.relaxed_threshold,
            radius=self.radius,
            candidate_topk=self.candidate_topk,
            candidate_min_topk=self.candidate_min_topk,
            candidate_ratio=self.candidate_ratio,
            min_select=self.min_select,
            conflict_threshold=self.conflict_threshold,
            axon_beta_r=self.axon_beta_r,
            axon_beta_u=self.axon_beta_u,
            axon_fixed_k=self.axon_fixed_k,
            axon_submod_fn=self.axon_submod_fn,
            axon_submod_monotone=self.axon_submod_monotone,
            axon_submod_penalty=self.axon_submod_penalty,
            axon_submod_lambda=self.axon_submod_lambda,
            unlock_target_nfe_per_block=self.unlock_target_nfe_per_block,
            unlock_min_anchor_steps=self.unlock_min_anchor_steps,
            unlock_max_anchor_steps=self.unlock_max_anchor_steps,
        )

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        axon_calls = 0
        axon_anchors = 0

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.mask_id

        axon_kwargs = self._axon_kwargs()
        prof = StepProfiler(enabled=self.profile_steps)
        prof_problem_idx = 0

        start_time = time.time()
        for req in tqdm(requests, desc="Generating..."):
            question = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            stop_tokens = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []

            # Detect HumanEval up front so the prompt AND the decode branch below use
            # the same flag.
            doc = getattr(req, "doc", None)
            task_id = str(doc.get("task_id", "")) if isinstance(doc, dict) else ""
            is_humaneval = task_id.lower().startswith("humaneval")
            # MBPP's task_id is an int, so key off the schema instead.
            is_mbpp = (not is_humaneval) and isinstance(doc, dict) and "test_list" in doc
            mbpp_fix = is_mbpp and self.mbpp_mode != "legacy"
            # TruthfulQA-gen has a distinctive doc schema.
            is_tqa = isinstance(doc, dict) and "best_answer" in doc
            tqa_fix = is_tqa and self.tqa_mode != "legacy"

            # SDAR OFFICIAL OpenCompass HumanEval recipe
            # (humaneval_openai_sample_evals_gen_dcae0e.py, role=HUMAN): the user turn is
            # an explicit implement-the-function instruction + the raw stub, NOT a bare
            # LLaDA-style stub. Without it the instruction-tuned chat model ends its turn
            # immediately (all-EOS -> ""), i.e. the ~37% empties. Placed in this shared
            # path so the plain and axon (axon_plugin) variants get a byte-identical prompt.
            if is_humaneval:
                question = (
                    "Read the following function signature and docstring, and fully "
                    "implement the function described. Your response should only contain "
                    "the code for this function.\n" + question
                )

            # HumanEval always takes the chat turn (it carries the instruction above). Other
            # tasks honour `use_chat_template`. truthfulqa_gen's doc_to_text is a completion-
            # style few-shot "Q: ...\nA: ..." primer; chat-templating it makes SDAR echo the
            # primer back ("Q: What is the square root of banana?") instead of answering, and
            # leaks an "A: " prefix into the scored string. Run those tasks raw, like LLaDA-1.5.
            if self.mbpp_mode == "chat_extract" and is_mbpp:
                question = (
                    "Complete the final Python task below. Follow the format of the worked "
                    "examples: write only the function implementation, then output [DONE].\n\n"
                    + question
                )

            # SDAR OFFICIAL OpenCompass MBPP recipe (sanitized_mbpp_mdblock_0shot_nocot_gen):
            # 0-shot, chat template, "expert Python programmer" prompt rebuilt from the doc's
            # own fields (IGNORE lm-eval's 3-shot doc_to_text), ```python fence extraction.
            # This is the MBPP analog of the HumanEval fix (SDAR-8B-Chat -> 72.0 vs our 45%).
            if self.mbpp_mode == "official" and is_mbpp:
                _tl = doc.get("test_list", [])
                if isinstance(_tl, (list, tuple)):
                    _tl = "\n".join(str(t) for t in _tl)
                # full MBPP calls the problem field 'text'; sanitized MBPP calls it 'prompt'.
                _desc = doc.get("text") or doc.get("prompt") or ""
                question = (
                    "You are an expert Python programmer, and here is your task:\n"
                    f"{_desc}\n"
                    "Your code should pass these tests:\n\n"
                    f"{_tl}\n"
                    " You should submit your final solution in the following format: "
                    "```python\n\n```"
                )

            # TruthfulQA chat_extract: drop the 6-shot Q:/A: primer entirely and ask the
            # single question directly (doc['question']); wrapping the whole primer in a chat
            # turn is what makes SDAR echo it back. raw_extract keeps the standard few-shot
            # primer as a completion.
            if self.tqa_mode == "chat_extract" and is_tqa:
                question = str(doc.get("question", question)).strip()

            if mbpp_fix:
                use_chat = self.mbpp_mode in ("chat_extract", "chat_noinstr", "official")
            elif tqa_fix:
                use_chat = (self.tqa_mode == "chat_extract")
            else:
                use_chat = is_humaneval or self.use_chat_template

            if use_chat:
                m = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                input_ids = self.tokenizer(user_input, add_special_tokens=False)["input_ids"]
            else:
                input_ids = self.tokenizer(question)["input_ids"]
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            out_ids, nfe, stats = generate(
                self.model,
                self.tokenizer,
                input_ids,
                gen_length=self.gen_length,
                block_length=self.block_length,
                steps=self.steps,
                step_profiler=(prof if self.profile_steps else None),
                **axon_kwargs,
            )
            if self.profile_steps:
                prof.finish_problem(prof_problem_idx, batch_size=1)
                prof_problem_idx += 1

            generated_ids = out_ids[0][input_ids.shape[1]:]
            # HumanEval decode mirrors SDAR's official OpenCompass path. There is no
            # cross-block EOS early-stop, so trailing blocks after the first end-of-turn
            # can carry stray tokens: cut generated_ids at the first <|im_end|>(151645) /
            # <|endoftext|>(151643), drop residual <|MASK|>, decode without specials, then
            # extract the first ```...``` code fence the implement-the-function instruction
            # induces (humaneval_postprocess_v2). Non-HumanEval tasks (mbpp few-shot,
            # truthfulqa_gen, ...) keep Branch B (completion-style `until` split).
            if is_humaneval:
                ids = generated_ids.tolist()
                for eos_id in (151645, 151643):
                    if eos_id in ids:
                        ids = ids[:ids.index(eos_id)]
                ids = [t for t in ids if t != self.mask_id]
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                fences = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
                generated_answer = fences[0] if fences else text
                answer_ids = generated_ids
            elif mbpp_fix:
                # Same EOS-cut + mask-strip as HumanEval, then recover the function body.
                # Preference order matches how the model actually answers: the few-shot primer
                # teaches [BEGIN]...[DONE]; the chat turn sometimes wraps it in a code fence;
                # otherwise take everything before the first [DONE].
                ids = generated_ids.tolist()
                for eos_id in (151645, 151643):
                    if eos_id in ids:
                        ids = ids[:ids.index(eos_id)]
                ids = [t for t in ids if t != self.mask_id]
                text = self.tokenizer.decode(ids, skip_special_tokens=True)

                if self.mbpp_mode == "official":
                    # 0-shot recipe: extract the first ```python fence (no BEGIN/DONE markers
                    # exist in this prompt), exactly like the HumanEval branch.
                    fences = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
                    code = fences[0] if fences else text
                else:
                    # Order matters. SDAR has no early stop, so in completion mode it runs past
                    # its answer and hallucinates a fresh "You are an expert ... [BEGIN] <other
                    # code>". Searching for [BEGIN] first therefore returns ANOTHER PROBLEM'S
                    # solution (observed: doc 7 answered longest_increasing_subsequence instead
                    # of remove_dirty_chars). Cut at the first [DONE] before looking for [BEGIN].
                    code = text.split("[DONE]")[0] if "[DONE]" in text else text
                    if "[BEGIN]" in code:
                        # a chat turn sometimes re-emits the marker ahead of the code
                        code = code.split("[BEGIN]", 1)[1]
                    else:
                        fences = re.findall(r"```[^\n]*\n(.*?)```", code, re.DOTALL)
                        if fences:
                            code = fences[0]
                generated_answer = code.strip("\n")
                answer_ids = (
                    torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                    if generated_answer
                    else torch.tensor([], dtype=torch.long)
                )
            elif tqa_fix:
                # EOS-cut + mask-strip (as HumanEval), then take the answer text before the
                # first '\nQ:' / '\n\n' (SDAR has no early stop and continues into a fresh
                # Q/A pair). Terminator FIRST, then extract -- the MBPP-bug lesson. Strip a
                # leading 'A:' the primer format induces.
                ids = generated_ids.tolist()
                for eos_id in (151645, 151643):
                    if eos_id in ids:
                        ids = ids[:ids.index(eos_id)]
                ids = [t for t in ids if t != self.mask_id]
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                cut = len(text)
                for term in ("\nQ:", "\n\n"):
                    j = text.find(term)
                    if j != -1:
                        cut = min(cut, j)
                ans = text[:cut].strip()
                if ans[:2].lower() == "a:":
                    ans = ans[2:].strip()
                generated_answer = ans
                answer_ids = (
                    torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                    if generated_answer
                    else torch.tensor([], dtype=torch.long)
                )
            else:
                generated_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                # A generation that opens with a newline would otherwise be split at offset 0
                # by a completion-style stop sequence (truthfulqa's until is '\n\n'), yielding
                # "". Observed as 2/5 empty generations on truthfulqa_gen before this lstrip.
                generated_answer = generated_answer.lstrip()
                for stop_seq in stop_tokens:
                    if stop_seq and stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]
                answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                generated_answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

            if self.show_speed:
                num_tokens += (answer_ids != pad_id).sum()
                num_nfe += nfe
            axon_calls += int(stats.get("axon_submod_calls", 0))
            axon_anchors += int(stats.get("axon_submod_anchors", 0))

            output.append(generated_answer)

            print("=" * 20)
            print("answer: ", generated_answer)
            print("nfe: ", nfe)
            print("=" * 20, end="\n\n")

        end_time = time.time()

        if self.show_speed:
            elapsed = end_time - start_time
            total_tokens = int(num_tokens)
            denom_out = max(1, len(output))
            print(f"Total number of tokens generated: {total_tokens}")
            print(f"Total time taken: {elapsed} seconds")
            print(f"Tokens per second: {total_tokens / elapsed if elapsed > 0 else 0.0}")
            print(f"Total NFE is {num_nfe}")
            print(f"Average NFE is {num_nfe / denom_out}")
            print(f"AXON submodular calls: {axon_calls}  anchors: {axon_anchors}")

            if self.outp_path is not None:
                dirpath = os.path.dirname(self.outp_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                output_metrics = {
                    "Total Number of Tokens": total_tokens,
                    "Total Time Taken": elapsed,
                    "Tokens per Second": (total_tokens / elapsed) if elapsed > 0 else 0.0,
                    "Total NFE": num_nfe,
                    "Average NFE": num_nfe / denom_out,
                    "AXON Submodular Calls": axon_calls,
                    "AXON Submodular Anchors": axon_anchors,
                }
                if self.profile_steps:
                    output_metrics.update(prof.summary())
                with open(self.outp_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_metrics, ensure_ascii=False) + "\n")

        if self.profile_steps and self.outp_path is not None:
            _tag = self.profile_tag.split("/") if self.profile_tag else []
            prof.write_csv(
                os.path.join(os.path.dirname(self.outp_path) or ".", "step_profile.csv"),
                family="sdar",
                model=(_tag[0] if len(_tag) > 0 else os.path.basename(self.model_path)),
                task=(_tag[1] if len(_tag) > 1 else ""),
                arm=(_tag[2] if len(_tag) > 2 else ""),
                decoder=str(self.axon_base_proposer),
            )

        return output


if __name__ == "__main__":
    cli_evaluate()
