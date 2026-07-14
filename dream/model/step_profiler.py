"""Opt-in per-step CUDA timing + reveal counting for diffusion decode loops.

Design (see plan): torch.cuda.Event pairs recorded with non-blocking .record();
NOTHING is resolved until finish_problem() issues a single torch.cuda.synchronize().
Reveal counts are kept as GPU scalars and .item()-ed only at problem end, so no
sync is injected mid-loop (a mid-loop .item() would drain the pipeline and corrupt
the next step's forward timing).

This exact file is vendored verbatim into llada/, dream/model/, sdar/ so each
package stays self-contained (same pattern as gdllm_utils.py). Keep the three copies
identical.

All hooks are no-ops unless `enabled=True`, so the default decode path is byte-for-byte
unchanged (event creation and deferred .item() touch neither RNG nor dtypes).
"""
from time import perf_counter

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class StepProfiler:
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.cuda = self.enabled and _HAS_TORCH and torch.cuda.is_available()
        self.rows = []          # resolved per-step rows (plain dicts, floats/ints)
        self._pending = []      # steps awaiting the problem-end sync
        self._cur = None        # current step bag

    # ------------------------------------------------------------------ per step
    def step_begin(self, need_attn):
        if not self.enabled:
            return
        self._cur = {"need_attn": int(bool(need_attn)), "axon_fired": 0,
                     "ev": {}, "reveal": None,
                     "block_idx": -1, "step_in_block": -1}
        self._mark("step0")

    def forward_end(self):
        self._mark("fwd")

    def base_begin(self):
        self._mark("base0")

    def base_end(self):
        self._mark("base1")

    def submod_begin(self):
        self._mark("sel0")

    def submod_end(self):
        self._mark("sel1")
        if self._cur is not None:
            self._cur["axon_fired"] = 1

    def note_reveal(self, transfer_index, block_idx, step_in_block):
        """Record reveal count (as a GPU scalar) WITHOUT closing the step.

        Use at a commit site when the step is closed later at a common tail
        (Dream: base algs commit in many places but share one `i += 1`).
        """
        if not self.enabled or self._cur is None:
            return
        try:
            self._cur["reveal"] = transfer_index.sum()  # 0-dim tensor, NOT .item()
        except Exception:
            self._cur["reveal"] = None
        self._cur["block_idx"] = int(block_idx)
        self._cur["step_in_block"] = int(step_in_block)

    def step_end(self):
        """Close the current step (record end event, push to pending)."""
        if not self.enabled or self._cur is None:
            return
        self._mark("end")
        self._pending.append(self._cur)
        self._cur = None

    def commit(self, transfer_index, block_idx, step_in_block):
        """Convenience: record reveal AND close the step in one call.

        Use where the commit site IS the natural end of the step body
        (LLaDA, SDAR).
        """
        self.note_reveal(transfer_index, block_idx, step_in_block)
        self.step_end()

    def _mark(self, name):
        if not self.enabled or self._cur is None:
            return
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._cur["ev"][name] = ev
        else:
            self._cur["ev"][name] = perf_counter()

    # --------------------------------------------------------------- per problem
    def finish_problem(self, problem_idx, batch_size=1, **meta):
        """Sync once, resolve every pending step's events into self.rows."""
        if not self.enabled:
            return
        if self.cuda:
            torch.cuda.synchronize()
        for p in self._pending:
            ev = p["ev"]

            def dt(a, b):
                if a in ev and b in ev:
                    if self.cuda:
                        return float(ev[a].elapsed_time(ev[b]))   # milliseconds
                    return float((ev[b] - ev[a]) * 1e3)
                return float("nan")

            forward_ms = dt("step0", "fwd")
            base_ms = dt("base0", "base1") if ("base0" in ev and "base1" in ev) else 0.0
            submod_ms = dt("sel0", "sel1") if ("sel0" in ev and "sel1" in ev) else 0.0
            step_total_ms = dt("step0", "end")
            reveal = p["reveal"]
            reveal_count = int(reveal.item()) if reveal is not None else -1
            row = dict(meta)
            row.update(problem_idx=int(problem_idx), batch_size=int(batch_size),
                       block_idx=p["block_idx"], step_in_block=p["step_in_block"],
                       need_attn=p["need_attn"], axon_fired=p["axon_fired"],
                       reveal_count=reveal_count,
                       forward_ms=forward_ms, base_ms=base_ms, submod_ms=submod_ms,
                       step_total_ms=step_total_ms)
            self.rows.append(row)
        self._pending = []

    # ------------------------------------------------------------------- outputs
    FIELDNAMES = ["family", "model", "task", "arm", "decoder",
                  "problem_idx", "batch_size", "block_idx", "step_in_block",
                  "need_attn", "axon_fired", "reveal_count",
                  "forward_ms", "base_ms", "submod_ms", "step_total_ms"]

    def write_csv(self, path, **run_meta):
        """Write self.rows to `path` with run-level metadata merged into each row."""
        if not self.enabled or not self.rows:
            return
        import csv
        import os
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            for r in self.rows:
                out = dict(run_meta)
                out.update(r)
                w.writerow(out)

    def summary(self):
        """Raw component means for results.jsonl quick-look. The arm-aware
        model/decoder/axon %-decomposition is done in aggregate_step_profile.py
        (only it knows the arm). Empty dict if disabled / no rows."""
        if not self.enabled or not self.rows:
            return {}
        ok = lambda v: v == v  # not NaN
        fwd = [r["forward_ms"] for r in self.rows if ok(r["forward_ms"])]
        base = [r["base_ms"] for r in self.rows]
        submod = [r["submod_ms"] for r in self.rows]
        tot = [r["step_total_ms"] for r in self.rows if ok(r["step_total_ms"])]
        reveals = [r["reveal_count"] for r in self.rows if r["reveal_count"] >= 0]
        byprob = {}
        for r in self.rows:
            byprob.setdefault(r["problem_idx"], []).append(r["step_total_ms"])
        prob_means = [sum(v) / len(v) for v in byprob.values() if v]

        def mean(x):
            return (sum(x) / len(x)) if x else float("nan")

        return {
            "Profile Steps": len(self.rows),
            "Avg Forward ms/step": mean(fwd),
            "Avg Base ms/step": mean(base),
            "Avg Submod ms/step": mean(submod),
            "Avg Step ms": mean(tot),
            "Avg Problem ms": mean(prob_means),
            "Pct Steps Reveal==1": (100.0 * sum(1 for x in reveals if x == 1) / len(reveals)) if reveals else float("nan"),
            "Pct Steps Reveal<=2": (100.0 * sum(1 for x in reveals if x <= 2) / len(reveals)) if reveals else float("nan"),
        }
