<div align="center">


<h1>AXON: Supportive Token Revealing for Fast Diffusion Language Model Decoding</h1>

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/pdf/2606.04236)
[![license](https://img.shields.io/badge/License-MIT%202.0-blue)](./LICENSE)

</div>

AXON is a **training-free**, plug-in **scheduler** for masked diffusion language models.

![method](figure/method.png)

## 🚀 Features

- Training-free, plug-and-play on top of standard masked-diffusion decoders.
- Improves quality–speed: gates avoid revealing tokens that would conflict, and submodular selection picks anchors with the highest coverage of residual uncertainty.

## 🔍 Key Details

AXON augments a base diffusion-LLM decoder with three plug-in components:

1. **Adaptive gate.** A lightweight diagnostic that fires when the decoder needs help.

2. **Submodular anchor selection.** When the gate fires, AXON builds a weight matrix $w_{ij}$ over the candidate masked positions and picks an anchor set $S$ by maximising a submodular objective.

    $$f(S) \;=\; \sum_i \max_{j \in S} w_{ij}\,,$$

## 🔧 Installation
### Option A: Quick start (recommended)
```bash
pip install -r requirements.txt
```

### Option B: Reproducible install
```bash
pip install -r requirements-lock.txt
```

## ✨ Eval

Run `run_llada.sh` script at the repo root. It auto-detects the visible GPUs and launches with `accelerate`.

```bash
# Run AXON on HumanEval (LLaDA-1.5, default):
VARIANT=all TASK=humaneval bash run_llada.sh
```

## 🙏 Acknowledgements

We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada), [DAWN](https://github.com/lizhuo-luo/DAWN) and [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) for their open-source contributions.
