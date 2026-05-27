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

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import os
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache, generate_klass
from model.modeling_llada import LLaDAModelLM
import json
import time
import csv
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        use_cache=False,
        threshold=None,
        factor=None,
        save_dir=None,
        show_speed=False,
        dual_cache=False,
        dawn=False,
        unlock=False,
        klass=False,
        local_leap=False,
        outp_path=None,
        threshold_klass=0.6,
        kl_threshold=0.015,
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
        unlock_weight_func='base',
        unlock_alpha=0.0,
        unlock_beta=0.0,
        unlock_score_mode='current',
        unlock_lambda=0.0,
        unlock_gamma=0.0,
        unlock_hazard_weight=1.0,
        unlock_budget_mode='none',
        unlock_budget_eta=0.5,
        unlock_budget_alpha=0.0,
        unlock_budget_threshold=0.0,
        unlock_budget_gamma=0.0,
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
        unlock_stage2_fill=False,
        unlock_fill_ratio=0.5,
        unlock_easy_then_unlock=False,
        unlock_easy_score_mode='conf_only',
        unlock_easy_prefix_mode='avg_threshold',
        unlock_easy_threshold=0.85,
        unlock_easy_eta=0.8,
        unlock_technique='none',
        unlock_target_nfe_per_block=8,
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
        unlock_low_reveal_threshold=3,
        unlock_low_reveal_mode='total',
        unlock_nonfinal_mode='remaining_mask',
        relaxed_threshold=0.75,
        radius=4,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.is_instruct = True if 'instruct' in model_path.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        self.dawn = dawn
        self.unlock = unlock
        self.outp_path = outp_path
        self.klass = klass
        self.local_leap = local_leap
        self.threshold_klass = threshold_klass
        self.kl_threshold = kl_threshold
        self.tau_sink = tau_sink
        self.tau_edge = tau_edge
        self.tau_induce = tau_induce
        self.tau_low = tau_low
        self.candidate_topk = candidate_topk
        self.candidate_min_topk = candidate_min_topk
        self.candidate_ratio = candidate_ratio
        self.warmup_steps = warmup_steps
        self.conflict_threshold = conflict_threshold
        self.multiplier_after_warmup = multiplier_after_warmup
        self.min_select = min_select
        self.unlock_weight_func = unlock_weight_func
        self.unlock_alpha = unlock_alpha
        self.unlock_beta = unlock_beta
        self.unlock_score_mode = str(unlock_score_mode)
        self.unlock_lambda = float(unlock_lambda)
        self.unlock_gamma = float(unlock_gamma)
        self.unlock_hazard_weight = float(unlock_hazard_weight)
        self.unlock_budget_mode = str(unlock_budget_mode)
        self.unlock_budget_eta = float(unlock_budget_eta)
        self.unlock_budget_alpha = float(unlock_budget_alpha)
        self.unlock_budget_threshold = float(unlock_budget_threshold)
        self.unlock_budget_gamma = float(unlock_budget_gamma)
        self.unlock_tail_merge_steps = int(unlock_tail_merge_steps)
        if isinstance(unlock_first_reveal_then_dawn, str):
            self.unlock_first_reveal_then_dawn = unlock_first_reveal_then_dawn.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.unlock_first_reveal_then_dawn = bool(unlock_first_reveal_then_dawn)
        self.unlock_steps_per_block = int(unlock_steps_per_block)
        self.unlock_num_blocks = int(unlock_num_blocks)
        self.unlock_conf_threshold = float(unlock_conf_threshold)
        self.unlock_exit_mode = str(unlock_exit_mode)
        self.unlock_min_anchor_steps = int(unlock_min_anchor_steps)
        self.unlock_max_anchor_steps = int(unlock_max_anchor_steps)
        self.unlock_exit_conf_threshold = float(unlock_exit_conf_threshold)
        self.unlock_exit_gain_eta = float(unlock_exit_gain_eta)
        self.unlock_exit_coverage_threshold = float(unlock_exit_coverage_threshold)
        if isinstance(unlock_use_clean, str):
            self.unlock_use_clean = unlock_use_clean.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.unlock_use_clean = bool(unlock_use_clean)
        if isinstance(unlock_stage2_fill, str):
            self.unlock_stage2_fill = unlock_stage2_fill.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.unlock_stage2_fill = bool(unlock_stage2_fill)
        self.unlock_fill_ratio = float(unlock_fill_ratio)
        if isinstance(unlock_easy_then_unlock, str):
            self.unlock_easy_then_unlock = unlock_easy_then_unlock.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.unlock_easy_then_unlock = bool(unlock_easy_then_unlock)
        self.unlock_easy_score_mode = str(unlock_easy_score_mode)
        self.unlock_easy_prefix_mode = str(unlock_easy_prefix_mode)
        self.unlock_easy_threshold = float(unlock_easy_threshold)
        self.unlock_easy_eta = float(unlock_easy_eta)
        self.unlock_technique = str(unlock_technique)
        self.unlock_target_nfe_per_block = float(unlock_target_nfe_per_block)
        self.unlock_backload_power = float(unlock_backload_power)
        self.unlock_optional_multiplier = float(unlock_optional_multiplier)
        self.unlock_optional_conf_threshold = float(unlock_optional_conf_threshold)
        self.unlock_stability_conf_threshold = float(unlock_stability_conf_threshold)
        self.unlock_high_conf_threshold = float(unlock_high_conf_threshold)
        self.unlock_anchor_min_conf = float(unlock_anchor_min_conf)
        self.union_dawn_keep_frac = float(union_dawn_keep_frac)
        self.unlock_easy_stable_frac = float(unlock_easy_stable_frac)
        self.unlock_easy_conf_frac = float(unlock_easy_conf_frac)
        self.unlock_hard_stable_frac = float(unlock_hard_stable_frac)
        self.unlock_gate_max_nfe_multiplier = float(unlock_gate_max_nfe_multiplier)
        self.axon_stag_gate = str(axon_stag_gate)
        self.axon_stag_selector = str(axon_stag_selector)
        self.axon_base_proposer = str(axon_base_proposer)
        self.unlock_low_reveal_threshold = int(unlock_low_reveal_threshold)
        self.unlock_low_reveal_mode = str(unlock_low_reveal_mode)
        self.unlock_nonfinal_mode = str(unlock_nonfinal_mode)
        self.relaxed_threshold = relaxed_threshold
        self.radius = radius
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        unlock_stats = None
        if self.show_speed and self.unlock:
            unlock_stats = {
                'scheduler_steps': 0,
                'candidate_pool_total': 0,
                'selected_total': 0,
                'active_rows': 0,
                'step_records': [],
                'stag_step_records': [],
            }
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        
        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])
        
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                if self.is_instruct:
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = self.tokenizer(user_input)['input_ids']
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)['input_ids']
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))
            
            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device), torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)
            
            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros((batched_input_ids.shape[0], 1, max_len+self.gen_length, max_len+self.gen_length), device=self.device, dtype=torch.bool)
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True


            stop_tokens = req.args[1]['until']
            input_ids = batched_input_ids
            if self.use_cache:
                if self.dual_cache:
                    generated_answer, nfe = generate_with_dual_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
                else:
                    generated_answer, nfe = generate_with_prefix_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
            elif self.klass:
                generated_answer, nfe = generate_klass(self.model, input_ids, gen_length=self.gen_length, steps=self.steps, block_length=self.block_length, 
                                        temperature=0, mask_id=self.mask_id, conf_threshold=self.threshold_klass, kl_threshold=self.kl_threshold,)
            else:
                unlock_profile = {} if unlock_stats is not None else None
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor,
                                        dawn=self.dawn, unlock=self.unlock, local_leap=self.local_leap,
                                        tau_sink=self.tau_sink, tau_edge=self.tau_edge, tau_induce=self.tau_induce, tau_low=self.tau_low,
                                        candidate_topk=self.candidate_topk, candidate_min_topk=self.candidate_min_topk,
                                        candidate_ratio=self.candidate_ratio, warmup_steps=self.warmup_steps,
                                        conflict_threshold=self.conflict_threshold,
                                        multiplier_after_warmup=self.multiplier_after_warmup,
                                        min_select=self.min_select,
                                        unlock_weight_func=self.unlock_weight_func,
                                        unlock_alpha=self.unlock_alpha,
                                        unlock_beta=self.unlock_beta,
                                        unlock_score_mode=self.unlock_score_mode,
                                        unlock_lambda=self.unlock_lambda,
                                        unlock_gamma=self.unlock_gamma,
                                        unlock_hazard_weight=self.unlock_hazard_weight,
                                        unlock_budget_mode=self.unlock_budget_mode,
                                        unlock_budget_eta=self.unlock_budget_eta,
                                        unlock_budget_alpha=self.unlock_budget_alpha,
                                        unlock_budget_threshold=self.unlock_budget_threshold,
                                        unlock_budget_gamma=self.unlock_budget_gamma,
                                        unlock_tail_merge_steps=self.unlock_tail_merge_steps,
                                        unlock_first_reveal_then_dawn=self.unlock_first_reveal_then_dawn,
                                        unlock_steps_per_block=self.unlock_steps_per_block,
                                        unlock_num_blocks=self.unlock_num_blocks,
                                        unlock_conf_threshold=self.unlock_conf_threshold,
                                        unlock_exit_mode=self.unlock_exit_mode,
                                        unlock_min_anchor_steps=self.unlock_min_anchor_steps,
                                        unlock_max_anchor_steps=self.unlock_max_anchor_steps,
                                        unlock_exit_conf_threshold=self.unlock_exit_conf_threshold,
                                        unlock_exit_gain_eta=self.unlock_exit_gain_eta,
                                        unlock_exit_coverage_threshold=self.unlock_exit_coverage_threshold,
                                        unlock_use_clean=self.unlock_use_clean,
                                        unlock_stage2_fill=self.unlock_stage2_fill,
                                        unlock_fill_ratio=self.unlock_fill_ratio,
                                        unlock_easy_then_unlock=self.unlock_easy_then_unlock,
                                        unlock_easy_score_mode=self.unlock_easy_score_mode,
                                        unlock_easy_prefix_mode=self.unlock_easy_prefix_mode,
                                        unlock_easy_threshold=self.unlock_easy_threshold,
                                        unlock_easy_eta=self.unlock_easy_eta,
                                        unlock_technique=self.unlock_technique,
                                        unlock_target_nfe_per_block=self.unlock_target_nfe_per_block,
                                        unlock_backload_power=self.unlock_backload_power,
                                        unlock_optional_multiplier=self.unlock_optional_multiplier,
                                        unlock_optional_conf_threshold=self.unlock_optional_conf_threshold,
                                        unlock_stability_conf_threshold=self.unlock_stability_conf_threshold,
                                        unlock_high_conf_threshold=self.unlock_high_conf_threshold,
                                        unlock_anchor_min_conf=self.unlock_anchor_min_conf,
                                        union_dawn_keep_frac=self.union_dawn_keep_frac,
                                        unlock_easy_stable_frac=self.unlock_easy_stable_frac,
                                        unlock_easy_conf_frac=self.unlock_easy_conf_frac,
                                        unlock_hard_stable_frac=self.unlock_hard_stable_frac,
                                        unlock_gate_max_nfe_multiplier=self.unlock_gate_max_nfe_multiplier,
                                        axon_stag_gate=self.axon_stag_gate,
                                        axon_stag_selector=self.axon_stag_selector,
                                        axon_base_proposer=self.axon_base_proposer,
                                        relaxed_threshold=self.relaxed_threshold, radius=self.radius,
                                        unlock_profile=unlock_profile)

                if unlock_stats is not None:
                    unlock_stats['scheduler_steps'] += unlock_profile.get('scheduler_steps', 0)
                    unlock_stats['candidate_pool_total'] += unlock_profile.get('candidate_pool_total', 0)
                    unlock_stats['selected_total'] += unlock_profile.get('selected_total', 0)
                    unlock_stats['active_rows'] += unlock_profile.get('active_rows', 0)
                    unlock_stats['step_records'].extend(unlock_profile.get('step_records', []))
                    unlock_stats['stag_step_records'].extend(unlock_profile.get('stag_step_records', []))

            # torch.cuda.empty_cache()
            
            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                batched_generated_answer = [self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) for i in range(len(generated_answer_ids))]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    generated_answer_i = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                # print('question: ', question)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')
            # self.accelerator.wait_for_everyone()
        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")
            print(f"Tokens per NFE is {num_tokens / num_nfe}")
            print(f"Average NFE is {num_nfe / len(output)}")


            dirpath = os.path.dirname(self.outp_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            with open(self.outp_path, "a", encoding="utf-8") as f:
                output_metrics = {
                    'Total Number of Tokens': num_tokens.item(),
                    'Total Time Taken': end_time - start_time,
                    'Tokens per Second': num_tokens.item() / (end_time - start_time),
                    'Total NFE': num_nfe,
                    'Tokens per NFE': num_tokens.item() / num_nfe,
                    'Average NFE': num_nfe / len(output),
                }
                if unlock_stats is not None and unlock_stats['scheduler_steps'] > 0:
                    denom = max(1, unlock_stats['active_rows'])
                    output_metrics.update({
                        'UNLOCK Avg Candidate Pool Size': unlock_stats['candidate_pool_total'] / denom,
                        'UNLOCK Avg Selected Count': unlock_stats['selected_total'] / denom,
                        'UNLOCK Scheduler Steps': unlock_stats['scheduler_steps'],
                        'AXON Submodular Calls': unlock_stats['scheduler_steps'],
                        'AXON Submodular Anchors': unlock_stats['selected_total'],
                    })
                f.write(json.dumps(
                    output_metrics,
                    ensure_ascii=False
                ) + "\n")

            if unlock_stats is not None and unlock_stats['step_records']:
                step_metrics_path = os.path.join(dirpath if dirpath else '.', 'step_metrics.csv')
                grouped = {}
                for rec in unlock_stats['step_records']:
                    step_key = int(rec.get('step_idx', 0))
                    rec_selected_total = int(rec.get('selected_total', 0))
                    rec_active_rows = int(rec.get('active_rows', 0))
                    rec_masked_before_total = int(rec.get('masked_before_total', 0))
                    if self.unlock_low_reveal_mode == 'per_row_avg':
                        reveal_measure = float(rec_selected_total) / float(max(1, rec_active_rows))
                    else:
                        reveal_measure = float(rec_selected_total)

                    if self.unlock_nonfinal_mode == 'stop_no_remaining':
                        rec_nonfinal = int(rec.get('stop_no_remaining_count', 0)) == 0
                    else:
                        rec_nonfinal = rec_selected_total < rec_masked_before_total

                    rec_low_reveal_nonfinal = (
                        reveal_measure < float(self.unlock_low_reveal_threshold) and rec_nonfinal
                    )

                    bucket = grouped.setdefault(step_key, {
                        'step_idx': step_key,
                        'calls': 0,
                        'active_rows': 0,
                        'selected_total': 0,
                        'masked_before_total': 0,
                        'candidate_pool_total': 0,
                        'selected_scored_count': 0,
                        'unlock_term_selected_sum': 0.0,
                        'conf_term_selected_sum': 0.0,
                        'risk_term_selected_sum': 0.0,
                        'risk_penalty_selected_sum': 0.0,
                        'final_score_selected_sum': 0.0,
                        'unlock_dominant_count': 0,
                        'conf_dominant_count': 0,
                        'risk_dominant_count': 0,
                        'mixed_count': 0,
                        'selected_classified_count': 0,
                        'best_count': 0,
                        'best_score_sum': 0.0,
                        'best_unlock_term_sum': 0.0,
                        'best_conf_term_sum': 0.0,
                        'best_risk_term_sum': 0.0,
                        'best_risk_penalty_sum': 0.0,
                        'avg_raw_gain_first_sum': 0.0,
                        'avg_raw_gain_last_sum': 0.0,
                        'avg_selected_per_step_sum': 0.0,
                        'technique_used': 'none',
                        'avg_quota_floor_sum': 0.0,
                        'avg_quota_cap_sum': 0.0,
                        'avg_anchor_weight_sum': 0.0,
                        'avg_conf_weight_sum': 0.0,
                        'avg_risk_weight_sum': 0.0,
                        'gate_anchor_mode_count_total': 0,
                        'gate_mixed_mode_count_total': 0,
                        'gate_bulk_mode_count_total': 0,
                        'gate_metric_count_total': 0,
                        'avg_stable_frac_sum': 0.0,
                        'avg_stable_high_conf_frac_sum': 0.0,
                        'avg_high_conf_frac_sum': 0.0,
                        'avg_gate_conf_sum': 0.0,
                        'cert_selected_sum': 0.0,
                        'hazard_selected_sum': 0.0,
                        'out_influence_selected_sum': 0.0,
                        'certified_gain_selected_sum': 0.0,
                        'hazard_anchor_selected_count': 0,
                        'certified_selected_count': 0,
                        'hazard_block_count': 0,
                        'stopping_trigger_count_total': 0,
                        'fallback_row_count_total': 0,
                        'stop_cap_count_total': 0,
                        'stop_no_remaining_count_total': 0,
                        'stop_conflict_count_total': 0,
                        'stop_nonpositive_count_total': 0,
                        'stop_budget_count_total': 0,
                        'stop_other_count_total': 0,
                        'budget_mode_used': 'none',
                        'low_reveal_nonfinal_count': 0,
                    })
                    bucket['calls'] += 1
                    bucket['active_rows'] += rec_active_rows
                    bucket['selected_total'] += rec_selected_total
                    bucket['masked_before_total'] += rec_masked_before_total
                    bucket['candidate_pool_total'] += int(rec.get('candidate_pool_total', 0))
                    bucket['selected_scored_count'] += int(rec.get('selected_scored_count', 0))
                    bucket['unlock_term_selected_sum'] += float(rec.get('unlock_term_selected_sum', 0.0))
                    bucket['conf_term_selected_sum'] += float(rec.get('conf_term_selected_sum', 0.0))
                    bucket['risk_term_selected_sum'] += float(rec.get('risk_term_selected_sum', 0.0))
                    bucket['risk_penalty_selected_sum'] += float(rec.get('risk_penalty_selected_sum', 0.0))
                    bucket['final_score_selected_sum'] += float(rec.get('final_score_selected_sum', 0.0))
                    bucket['unlock_dominant_count'] += int(rec.get('unlock_dominant_count', 0))
                    bucket['conf_dominant_count'] += int(rec.get('conf_dominant_count', 0))
                    bucket['risk_dominant_count'] += int(rec.get('risk_dominant_count', 0))
                    bucket['mixed_count'] += int(rec.get('mixed_count', 0))
                    bucket['selected_classified_count'] += int(rec.get('selected_classified_count', 0))
                    bucket['best_count'] += int(rec.get('best_count', 0))
                    bucket['best_score_sum'] += float(rec.get('best_score_sum', 0.0))
                    bucket['best_unlock_term_sum'] += float(rec.get('best_unlock_term_sum', 0.0))
                    bucket['best_conf_term_sum'] += float(rec.get('best_conf_term_sum', 0.0))
                    bucket['best_risk_term_sum'] += float(rec.get('best_risk_term_sum', 0.0))
                    bucket['best_risk_penalty_sum'] += float(rec.get('best_risk_penalty_sum', 0.0))
                    bucket['avg_raw_gain_first_sum'] += float(rec.get('avg_raw_gain_first', 0.0))
                    bucket['avg_raw_gain_last_sum'] += float(rec.get('avg_raw_gain_last', 0.0))
                    bucket['avg_selected_per_step_sum'] += float(rec.get('avg_selected_per_step', 0.0))
                    bucket['technique_used'] = str(rec.get('technique_used', bucket['technique_used']))
                    bucket['avg_quota_floor_sum'] += float(rec.get('avg_quota_floor', 0.0))
                    bucket['avg_quota_cap_sum'] += float(rec.get('avg_quota_cap', 0.0))
                    bucket['avg_anchor_weight_sum'] += float(rec.get('avg_anchor_weight', 0.0))
                    bucket['avg_conf_weight_sum'] += float(rec.get('avg_conf_weight', 0.0))
                    bucket['avg_risk_weight_sum'] += float(rec.get('avg_risk_weight', 0.0))
                    bucket['gate_anchor_mode_count_total'] += int(rec.get('gate_anchor_mode_count', 0))
                    bucket['gate_mixed_mode_count_total'] += int(rec.get('gate_mixed_mode_count', 0))
                    bucket['gate_bulk_mode_count_total'] += int(rec.get('gate_bulk_mode_count', 0))
                    bucket['gate_metric_count_total'] += int(rec.get('gate_metric_count', 0))
                    bucket['avg_stable_frac_sum'] += float(rec.get('avg_stable_frac', 0.0))
                    bucket['avg_stable_high_conf_frac_sum'] += float(rec.get('avg_stable_high_conf_frac', 0.0))
                    bucket['avg_high_conf_frac_sum'] += float(rec.get('avg_high_conf_frac', 0.0))
                    bucket['avg_gate_conf_sum'] += float(rec.get('avg_gate_conf', 0.0))
                    bucket['cert_selected_sum'] += float(rec.get('cert_selected_sum', 0.0))
                    bucket['hazard_selected_sum'] += float(rec.get('hazard_selected_sum', 0.0))
                    bucket['out_influence_selected_sum'] += float(rec.get('out_influence_selected_sum', 0.0))
                    bucket['certified_gain_selected_sum'] += float(rec.get('certified_gain_selected_sum', 0.0))
                    bucket['hazard_anchor_selected_count'] += int(rec.get('hazard_anchor_selected_count', 0))
                    bucket['certified_selected_count'] += int(rec.get('certified_selected_count', 0))
                    bucket['hazard_block_count'] += int(rec.get('hazard_block_count', 0))
                    bucket['stopping_trigger_count_total'] += int(rec.get('stopping_trigger_count', 0))
                    bucket['fallback_row_count_total'] += int(rec.get('fallback_row_count', 0))
                    bucket['stop_cap_count_total'] += int(rec.get('stop_cap_count', 0))
                    bucket['stop_no_remaining_count_total'] += int(rec.get('stop_no_remaining_count', 0))
                    bucket['stop_conflict_count_total'] += int(rec.get('stop_conflict_count', 0))
                    bucket['stop_nonpositive_count_total'] += int(rec.get('stop_nonpositive_count', 0))
                    bucket['stop_budget_count_total'] += int(rec.get('stop_budget_count', 0))
                    bucket['stop_other_count_total'] += int(rec.get('stop_other_count', 0))
                    bucket['budget_mode_used'] = str(rec.get('budget_mode_used', bucket['budget_mode_used']))
                    if rec_low_reveal_nonfinal:
                        bucket['low_reveal_nonfinal_count'] += 1

                with open(step_metrics_path, 'w', encoding='utf-8', newline='') as sf:
                    fieldnames = [
                        'step_idx',
                        'total_selected_tokens',
                        'avg_selected_tokens',
                        'avg_masked_before_step',
                        'avg_candidate_pool_size',
                        'avg_unlock_term_selected',
                        'avg_conf_term_selected',
                        'avg_risk_term_selected',
                        'avg_risk_penalty_selected',
                        'avg_final_score_selected',
                        'pct_unlock_dominant',
                        'pct_conf_dominant',
                        'pct_risk_dominant',
                        'pct_mixed',
                        'avg_best_score',
                        'avg_best_unlock_term',
                        'avg_best_conf_term',
                        'avg_best_risk_term',
                        'avg_best_risk_penalty',
                        'avg_raw_gain_first',
                        'avg_raw_gain_last',
                        'avg_selected_scored_per_row',
                        'avg_selected_per_step',
                        'technique_used',
                        'avg_quota_floor',
                        'avg_quota_cap',
                        'avg_anchor_weight',
                        'avg_conf_weight',
                        'avg_risk_weight',
                        'gate_anchor_mode_count',
                        'gate_mixed_mode_count',
                        'gate_bulk_mode_count',
                        'pct_gate_anchor_mode',
                        'pct_gate_mixed_mode',
                        'pct_gate_bulk_mode',
                        'avg_stable_frac',
                        'avg_stable_high_conf_frac',
                        'avg_high_conf_frac',
                        'avg_gate_conf',
                        'avg_cert_selected',
                        'avg_hazard_selected',
                        'avg_out_influence_selected',
                        'avg_certified_gain_selected',
                        'pct_hazard_anchor_selected',
                        'pct_certified_selected',
                        'hazard_block_count',
                        'selected_scored_count',
                        'selected_classified_count',
                        'best_count',
                        'fallback_row_count',
                        'pct_fallback_rows',
                        'stop_cap_count',
                        'stop_no_remaining_count',
                        'stop_conflict_count',
                        'stop_nonpositive_count',
                        'stop_budget_count',
                        'stop_other_count',
                        'pct_stop_cap',
                        'pct_stop_no_remaining',
                        'pct_stop_conflict',
                        'pct_stop_nonpositive',
                        'pct_stop_budget',
                        'pct_stop_other',
                        'stopping_trigger_count',
                        'budget_mode_used',
                        'low_reveal_nonfinal_count',
                        'low_reveal_nonfinal_ratio',
                        'low_reveal_nonfinal_count_over_calls',
                        'low_reveal_threshold_used',
                        'low_reveal_mode_used',
                        'nonfinal_mode_used',
                    ]
                    writer = csv.DictWriter(sf, fieldnames=fieldnames)
                    writer.writeheader()
                    for step_key in sorted(grouped.keys()):
                        row = grouped[step_key]
                        rows_denom = max(1, int(row['active_rows']))
                        selected_denom = max(1, int(row['selected_scored_count']))
                        class_denom = max(1, int(row['selected_classified_count']))
                        best_denom = max(1, int(row['best_count']))
                        calls_denom = max(1, int(row['calls']))
                        gate_denom = max(1, int(row['gate_metric_count_total']))
                        writer.writerow({
                            'step_idx': step_key,
                            'total_selected_tokens': int(row['selected_total']),
                            'avg_selected_tokens': float(row['selected_total']) / rows_denom,
                            'avg_masked_before_step': float(row['masked_before_total']) / rows_denom,
                            'avg_candidate_pool_size': float(row['candidate_pool_total']) / rows_denom,
                            'avg_unlock_term_selected': float(row['unlock_term_selected_sum']) / selected_denom,
                            'avg_conf_term_selected': float(row['conf_term_selected_sum']) / selected_denom,
                            'avg_risk_term_selected': float(row['risk_term_selected_sum']) / selected_denom,
                            'avg_risk_penalty_selected': float(row['risk_penalty_selected_sum']) / selected_denom,
                            'avg_final_score_selected': float(row['final_score_selected_sum']) / selected_denom,
                            'pct_unlock_dominant': 100.0 * float(row['unlock_dominant_count']) / class_denom,
                            'pct_conf_dominant': 100.0 * float(row['conf_dominant_count']) / class_denom,
                            'pct_risk_dominant': 100.0 * float(row['risk_dominant_count']) / class_denom,
                            'pct_mixed': 100.0 * float(row['mixed_count']) / class_denom,
                            'avg_best_score': float(row['best_score_sum']) / best_denom,
                            'avg_best_unlock_term': float(row['best_unlock_term_sum']) / best_denom,
                            'avg_best_conf_term': float(row['best_conf_term_sum']) / best_denom,
                            'avg_best_risk_term': float(row['best_risk_term_sum']) / best_denom,
                            'avg_best_risk_penalty': float(row['best_risk_penalty_sum']) / best_denom,
                            'avg_raw_gain_first': float(row['avg_raw_gain_first_sum']) / calls_denom,
                            'avg_raw_gain_last': float(row['avg_raw_gain_last_sum']) / calls_denom,
                            'avg_selected_scored_per_row': float(row['selected_scored_count']) / rows_denom,
                            'avg_selected_per_step': float(row['avg_selected_per_step_sum']) / calls_denom,
                            'technique_used': row['technique_used'],
                            'avg_quota_floor': float(row['avg_quota_floor_sum']) / calls_denom,
                            'avg_quota_cap': float(row['avg_quota_cap_sum']) / calls_denom,
                            'avg_anchor_weight': float(row['avg_anchor_weight_sum']) / calls_denom,
                            'avg_conf_weight': float(row['avg_conf_weight_sum']) / calls_denom,
                            'avg_risk_weight': float(row['avg_risk_weight_sum']) / calls_denom,
                            'gate_anchor_mode_count': int(row['gate_anchor_mode_count_total']),
                            'gate_mixed_mode_count': int(row['gate_mixed_mode_count_total']),
                            'gate_bulk_mode_count': int(row['gate_bulk_mode_count_total']),
                            'pct_gate_anchor_mode': 100.0 * float(row['gate_anchor_mode_count_total']) / gate_denom,
                            'pct_gate_mixed_mode': 100.0 * float(row['gate_mixed_mode_count_total']) / gate_denom,
                            'pct_gate_bulk_mode': 100.0 * float(row['gate_bulk_mode_count_total']) / gate_denom,
                            'avg_stable_frac': float(row['avg_stable_frac_sum']) / calls_denom,
                            'avg_stable_high_conf_frac': float(row['avg_stable_high_conf_frac_sum']) / calls_denom,
                            'avg_high_conf_frac': float(row['avg_high_conf_frac_sum']) / calls_denom,
                            'avg_gate_conf': float(row['avg_gate_conf_sum']) / calls_denom,
                            'avg_cert_selected': float(row['cert_selected_sum']) / selected_denom,
                            'avg_hazard_selected': float(row['hazard_selected_sum']) / selected_denom,
                            'avg_out_influence_selected': float(row['out_influence_selected_sum']) / selected_denom,
                            'avg_certified_gain_selected': float(row['certified_gain_selected_sum']) / selected_denom,
                            'pct_hazard_anchor_selected': 100.0 * float(row['hazard_anchor_selected_count']) / selected_denom,
                            'pct_certified_selected': 100.0 * float(row['certified_selected_count']) / selected_denom,
                            'hazard_block_count': int(row['hazard_block_count']),
                            'selected_scored_count': int(row['selected_scored_count']),
                            'selected_classified_count': int(row['selected_classified_count']),
                            'best_count': int(row['best_count']),
                            'fallback_row_count': int(row['fallback_row_count_total']),
                            'pct_fallback_rows': 100.0 * float(row['fallback_row_count_total']) / rows_denom,
                            'stop_cap_count': int(row['stop_cap_count_total']),
                            'stop_no_remaining_count': int(row['stop_no_remaining_count_total']),
                            'stop_conflict_count': int(row['stop_conflict_count_total']),
                            'stop_nonpositive_count': int(row['stop_nonpositive_count_total']),
                            'stop_budget_count': int(row['stop_budget_count_total']),
                            'stop_other_count': int(row['stop_other_count_total']),
                            'pct_stop_cap': 100.0 * float(row['stop_cap_count_total']) / rows_denom,
                            'pct_stop_no_remaining': 100.0 * float(row['stop_no_remaining_count_total']) / rows_denom,
                            'pct_stop_conflict': 100.0 * float(row['stop_conflict_count_total']) / rows_denom,
                            'pct_stop_nonpositive': 100.0 * float(row['stop_nonpositive_count_total']) / rows_denom,
                            'pct_stop_budget': 100.0 * float(row['stop_budget_count_total']) / rows_denom,
                            'pct_stop_other': 100.0 * float(row['stop_other_count_total']) / rows_denom,
                            'stopping_trigger_count': int(row['stopping_trigger_count_total']),
                            'budget_mode_used': row['budget_mode_used'],
                            'low_reveal_nonfinal_count': int(row['low_reveal_nonfinal_count']),
                            'low_reveal_nonfinal_ratio': float(row['low_reveal_nonfinal_count']) / calls_denom,
                            'low_reveal_nonfinal_count_over_calls': f"{int(row['low_reveal_nonfinal_count'])}/{int(row['calls'])}",
                            'low_reveal_threshold_used': int(self.unlock_low_reveal_threshold),
                            'low_reveal_mode_used': self.unlock_low_reveal_mode,
                            'nonfinal_mode_used': self.unlock_nonfinal_mode,
                        })

            if unlock_stats is not None and unlock_stats.get('stag_step_records'):
                stag_step_metrics_path = os.path.join(dirpath if dirpath else '.', 'stag_step_metrics.csv')
                stag_fieldnames = [
                    'step_idx',
                    'block_idx',
                    'step_in_block',
                    'gate',
                    'selector',
                    'need_axon',
                    'can_anchor',
                    'masked_count',
                    'dawn_count',
                    'dawn_reveal_frac',
                    'expected_count',
                    'expected_progress',
                    'graph_coverage',
                    'block_conf',
                    'residual_conf',
                    'revealed_conf',
                    'selected_count',
                    'anchor_count_pre_guard',
                    'anchor_mean_conf',
                    'anchor_min_conf',
                    'anchor_max_conf',
                ]
                with open(stag_step_metrics_path, 'w', encoding='utf-8', newline='') as sf:
                    writer = csv.DictWriter(sf, fieldnames=stag_fieldnames)
                    writer.writeheader()
                    for rec in unlock_stats['stag_step_records']:
                        writer.writerow({key: rec.get(key) for key in stag_fieldnames})
            
        return output


if __name__ == "__main__":
    cli_evaluate()
    
