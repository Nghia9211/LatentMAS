# Adaptive latent handoff controller

Research extension of [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)
using frozen Qwen3-4B sender A and receiver B. The controller experiments choose
HANDOFF/CONTINUE at K10–80. Current comparisons also examine latent transforms
and selective KV communication.

**Execution default for new runs: HF native with direct Full KV A→B handoff.**
Realignment transforms A's latent input; inter-agent communication passes
past_key_values. Historical HF+vLLM embedding replay results have a different
protocol. Read [the native-KV contract](docs/README.md#native-kv-contract) and
[controller design and evidence](docs/README.md) before launching.

## Verified results — 27 October 2026

MedQA test500, strict boxed-answer scoring. This cohort has been used repeatedly;
it is not an untouched final test. A single run does not establish a robust gain.

### Matched native-HF communication comparison

A K10, batch4, seed42, same cohort and decoder. The replay comparator also uses HF.

| Handoff | Correct /500 | Accuracy | Measured total time (s) |
|---|---:|---:|---:|
| Embedding replay | 349 | 69.8% | 18931 |
| Direct Full KV | 341 | 68.2% | 17099 |
| Selective KVComm | 355 | 71.0% | 19622 |

Selective KVComm saves 69.28% of prefix-cache bytes, but measured runtime is 14.76%
longer than Full KV. Accuracy delta is +2.8 percentage points, with paired
question-bootstrap 95% CI [-0.4, +6.0]. This does not yet establish a repeatable
accuracy gain or an end-to-end speedup. Full-KV B decoding accounts for 99.35%
of measured time; A accounts for 0.49%. Calibration overhead is reported separately
in [the full results](docs/README.md#verified-results).

### Historical controller and EMA results

These runs used **hybrid HF+vLLM embedding replay**, not native Full KV.

| Method | Test500 accuracy | Mean K |
|---|---:|---:|
| Fixed K10 reference | 69.6% | 10.00 |
| PPO cosine64, validation-best iteration 32 | 69.6% | 23.22 |
| GRPO C1 late fusion | 70.0% | 10.38 |
| Supervised first-success C1 | 70.4% | 10.12 |

The supervised controller still chooses K10 on 497/500 questions; the three changed
budgets explain none of its net gain over the historical reference. The current
evidence does not establish a controller that reliably outperforms fixed K10.

EMA beta=0.05 has completed K10–80. In the fresh matched K40 comparison,
Realignment scored 65.4% and EMA 67.8%; loops fell 20→1, while the accuracy
interval still crosses zero. [Full K10–80, SRR, debate, stopping and Claude
diagnostics](docs/README.md#verified-results) are separated by protocol.

### Current job status at the snapshot

- Selective A K10→B latent K0/K10 test500: 180 paired questions preserved; the
  multi-GPU batch32/16 continuation is waiting for VRAM. No final 500-question result.
- Native seven-K adaptive campaign: canceled by the user after 6 diagnostic
  training questions; the dependent stages never ran.
- A K2 shortcut distillation: not trained; still blocked by the canceled
  native-profile prerequisite. Reusing historical labels has not been implemented.

The train profile's adaptive-help group is 99/1024 questions under a ≥2/3-seed
criterion. This is neither “K10 is correct on 90%” nor a promised 9.67-point gain.
See [the data priority](docs/README.md#train-data-priority), observability diagnostics
and [canonical research/operation guide](docs/README.md).

## Published repository and server-only workflows

scripts/ and tests/ are excluded from GitHub and ignored; their server copies,
frozen job snapshots, raw results and checkpoints are preserved. Several controller
modules import helpers from scripts/, so a GitHub clone alone cannot run the
private controller training/evaluation workflow or internal regression suite.
Module documentation is in [controller/README.md](controller/README.md);
configuration templates are in [configs/README.md](configs/README.md).

The latest server-only cleanup verification passed 306 tests with 1 historical skip.
It is CPU regression evidence, not real-model GPU parity or experimental accuracy.
Generated datasets, profiles, checkpoints, evaluation outputs and local credentials
are also excluded. New runs require their own frozen configuration and runtime.

The material below is the upstream LatentMAS README. Its paper results and
performance claims concern the upstream framework and are separate from the
local results above.

---

<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets_latentmas/logo.png">
    <img alt="LatentMAS" src="assets_latentmas/logo.png" width=500>
  </picture>
</p>

<h3 align="center">
Latent Collaboration in Multi-Agent Systems
</h3>



<p align="center">
    <a href="https://arxiv.org/abs/2511.20639"><img src="https://img.shields.io/badge/arXiv-2511.20639-B31B1B.svg?logo=arxiv" alt="Arxiv"></a>
    <a href="https://github.com/Gen-Verse/LatentMAS/blob/main/assets/LatentMAS_slides.pdf"><img src="https://img.shields.io/badge/Slides-LatentMAS-FF6F00.svg?logo=googleslides" alt="Paper Slides"></a>
    <a href="https://huggingface.co/papers/2511.20639"><img src="https://img.shields.io/badge/Huggingface-DailyPaper-FFD21E.svg?logo=huggingface" alt="Huggingface Paper"></a>
    <a href="https://x.com/Jiaru_Zou/status/1994724438135169196"><img src="https://img.shields.io/badge/Coverage-LatentMAS-2176BC.svg?logo=x" alt="X"></a>
    <a href="https://github.com/Gen-Verse/LatentMAS/tree/Science-LatentMAS"><img src="https://img.shields.io/badge/Science--LatentMAS-Branch-2D8CFF.svg?logo=github" alt="Science-LatentMAS Branch"></a>
</p>

---

<p align="center">
  <img src="assets_latentmas/main_res.png" width="1000">
</p>

## 💡 Introduction

Local adaptive-handoff extension: core code is in [controller/](controller/README.md)
and verified research results are in [docs/](docs/README.md#verified-results).
Experiment scripts and tests are maintained privately on the server.


**LatentMAS** is a multi-agent reasoning framework that **moves agent collaboration from token space into the model’s latent space**.  
Instead of producing long textual reasoning traces, agents communicate by **passing latent thoughts** through their own **working memory**. LatentMAS has the following key features:

- **Efficient** multi-step reasoning with drastically fewer tokens  
- **Training-free** latent-space alignment for stable generation  
- **A general technique** compatible with **any HF model** and optionally **vLLM** backends.

Overall, LatentMAS achieves **superior performance**, **lower token usage**, and **major wall-clock speedups** of the multi-agent system.

<p align="center">
  <img src="assets_latentmas/main.png" width="1000">
</p>


## 🔔 News
- **[2026-05-01]** LatentMAS has been accepted into ICML 2026 as a **spotlight** ! 
- **[2026-02-26]** 🦞 Check out [**OpenClaw-RL**](https://github.com/Gen-Verse/OpenClaw-RL) from our Gen-Verse group! OpenClaw-RL is a fully asynchronous RL framework that trains personalized AI agents directly from natural conversation feedback — no manual labels, no API keys. It introduces two learning paradigms (Binary RL via GRPO and On-Policy Distillation) and runs the entire stack on your own infrastructure. A great complement to LatentMAS's efficient multi-agent reasoning! 
- **[2025-12-20]** Check [**Science-LatentMAS**](https://github.com/Gen-Verse/LatentMAS/tree/Science-LatentMAS), an excellent extension of LatentMAS developed by Prof. Markus J. Buehler and the [LAMM Lab](https://github.com/lamm-mit) at MIT. Science-LatentMAS is specifically designed for the scientific discovery downstream applications! For more details and instructions, please check our README section "Science-LatentMAS" below and the new `Science-LatentMAS` branch.
- **[2025-12-15]** Check out these amazing community-driven extensions of LatentMAS!
  - **[KNN-LatentMAS](https://github.com/Bookmaster9/kNN-latentMAS)** — Enables more efficient KV utilization for latent memory.
  - **[Hybrid-LatentMAS](https://github.com/nhminle/LatentMAS-Hybrid)** — Extends LatentMAS to support hybrid, heterogeneous multi-agent systems.

- **[2025-11-25]** We have released our paper and code implementations for LatentMAS! Stay tuned for more model-backbone supports and advanced features!
- **[2025-11-25]** We are featured as 🤗 [**HuggingFace 1st Paper of the Day**](https://huggingface.co/papers/2511.20639)!


## 🌐 Awesome Works Built on Top of LatentMAS

Explore community-driven extensions that expand LatentMAS into new domains, architectures, and collaboration patterns:


### 🔬 1. **Science-LatentMAS**
**By Prof. Markus J. Buehler & MIT LAMM Group**  
- **New Branch:** https://github.com/Gen-Verse/LatentMAS/tree/Science-LatentMAS  
- **Original Code:** https://github.com/lamm-mit/LatentMAS/tree/flexible_agents  
**New Features:** Extends LatentMAS for scientific modeling and material-system collaboration, enabling flexible agent types and specialized latent communication for science domains.


### 🧠 2. **KNN-LatentMAS**
**By Bookmaster9**
- **Blog (Overview):** https://bookmaster9.github.io/kNN-latentMAS/  
- **Code:** https://github.com/Bookmaster9/kNN-latentMAS  
- **New Features:** Introduce kNN-based latent retrieval to improve KV-cache usage, boosting memory efficiency and multi-step reasoning stability across agents.

### 🤖 3. **Hybrid-LatentMAS**
**By nhminle**
- **Code:** https://github.com/nhminle/LatentMAS-Hybrid  
- **New Features:** Support heterogeneous/hybrid agent collaboration (LLM + non-LLM agents), enabling modular multi-agent pipelines that mix models, tools, and reasoning strategies.


### 🌍 4. **Awareness Network**
**By Everest-AN**
- **Website:** https://awareness.market/
- **Code:** https://github.com/everest-an/Awareness-Market
- **New Features:** A decentralized AI awareness market product built on LatentMAS research, enabling autonomous agent collaboration and memory sharing.

### 🧩 5. LatentMAS-SLoRA
**By Arifuzzaman Joy**
- **Demo:** https://www.youtube.com/watch?v=g7sxYjwgRRk
- **Code:** https://github.com/Arifuzzamanjoy/latent_mas_slora
- **New Features:** Augment LatentMAS with role-specialized, dynamically switchable LoRA adapters for better specialization and adaptability.

### 🛰️ 6. AVP (Agent Vector Protocol)
**By VectorArc**
- **Blog:** https://blog.avprotocol.ai/avp-binary-protocol-latent-agent-communication/
- **Code:** https://github.com/VectorArc/avp-python
- **New Features:** Enables agents to share KV-cache and hidden states instead of text, supporting zero-training latent handoff, cross-model transfer, and faster multi-agent collaboration.

**If your work extends LatentMAS, feel free to open a PR and we’ll feature it here! 🚀**


## 📊 Experiments Overview

### ⭐ Main Results  
Three main tables from our paper spanning 9 tasks across math & science reasoning, commensonse reasoning, and code generation:

- **Table 1 — LatentMAS under the Sequantial MAS setting**  
  <p align="center"><img src="assets_latentmas/main_table1.png" width="1000"></p>

- **Table 2 — LatentMAS under the Hierarchical MAS setting**  
  <p align="center"><img src="assets_latentmas/main_table2.png" width="1000"></p>

- **Table 3 — Main Results on Reasoning Intensive Tasks**
  <p align="center"><img src="assets_latentmas/main_table3.png" width="1000"></p>


### ⚡ Superior Efficiency on **Time and Tokens**

Overall, LatentMAS reduces:
- **~50–80% tokens**
- **~3×–7× wall-clock time**
compared to standard Text-MAS or chain-of-thought baselines.


## 🛠️ Getting Started

This repository provides all code for reproducing LatentMAS, TextMAS, and baseline single-agent experiments across GSM8K, AIME24/25, GPQA, ARC-Easy/Challenge, MBPP+, HumanEval+, and MedQA.

### ⚙️ Setup Environment Variables

We recommend setting your HF cache directory to avoid repeated downloads:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
````

Models and datasets will automatically be downloaded into `$HF_HOME`.


### 📦 Install Packages

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas

pip install -r requirements.txt
```

If you want **vLLM support**, also install:

```bash
pip install vllm
```

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Gen-Verse/LatentMAS.git
cd LatentMAS
```

### 2. Repository Structure

```
LatentMAS/
│── run.py                 # Main entry for experiments
│── models.py              # Wrapper for HF + vLLM + latent realignment
│── methods/
│   ├── baseline.py        # Single-agent baseline
│   ├── text_mas.py        # Token-space multi-agent method
│   └── latent_mas.py      # Latent-space multi-agent (our method)
│── prompts.py             # Prompt constructors
│── data.py                # Dataset loaders
│── data/                  # Provided data + figures (We give medqa.json as an example here)
│── utils.py               # Answer parsing / timeout / helpers
│── example_logs/          # Example logs from LatentMAS
│── requirements.txt
```


## 🧪 Running Experiments (standard HF backend)

### 🔹 **Baseline (single model)**

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --max_new_tokens 2048
```


### 🔹 **TextMAS (text based multi-agent system)**

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```


### 🔹 **LatentMAS (our latent mas method)**

```bash
# 4B example command
python run.py --method latent_mas --model_name Qwen/Qwen3-4B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048

# 8B example command
python run.py --method latent_mas --model_name Qwen/Qwen3-8B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048

# 14B example command
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```

#### Notes:

* **`--latent_steps`** ∈ [0, 80]
  Tune for best performance.
* **`--latent_space_realign`**
  Enables latent→embedding alignment
  We treat this as a **hyperparameter** — enable/disable depending on task/model:

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --latent_space_realign --max_new_tokens 2048
```


## 📘 Example Logs

Two example LatentMAS logs are provided for reference purposes:

* `example_logs/qwen3_14b_mbppplus_sequential.txt`
* `example_logs/qwen3_14b_humanevalplus_hierarchical.txt`


Please refer to additional experiment logs [here](https://drive.google.com/drive/folders/1evGv5YAmLb4YM_D9Yu0ABa1nfqHC5N-l?usp=drive_link).
You can open them to view the full agent interaction traces and outputs.


## ⚡ vLLM Integration

LatentMAS supports vLLM for faster inference.

### 🔹 Baseline with vLLM

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --use_vllm --max_new_tokens 2048
```

### 🔹 TextMAS with vLLM

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --use_vllm --max_new_tokens 2048
```

### 🔹 LatentMAS with vLLM

LatentMAS supports a **hybrid HF + vLLM pipeline** for fast inference:
- vLLM handles **final text generation** (with prefix caching, tensor parallelism, etc.)
- A HuggingFace model handles **latent-space rollout** and hidden-state alignment

For this setup, we recommend using two GPUs:
- One GPU for vLLM (`--device`, e.g., `cuda:0`)
- One GPU for the auxiliary HF model (`--device2`, e.g., `cuda:1`)

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048 \
  --use_vllm \
  --use_second_HF_model \
  --enable_prefix_caching \
  --device2 cuda:1
```

**📍Important Note:**

> vLLM does **not** officially support modifying KV-cache or prompting via latent embeddings.
> We modify the partial inner package inside vLLM backend for our method implementation.
> Note minor numeric differences may arise compared to offical HF backend due to different decoding (generation) strategies. Please Use the HF backend to reproduce the official published results.

## 📚 Citation

💫 If you find **LatentMAS** helpful, please kindly give us a star ⭐️ and cite below. Thanks!

```
@inproceedings{
zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Jiaru Zou and Ruizhong Qiu and Gaotang Li and Xiyuan Yang and Katherine Tieu and Pan Lu and Ke Shen and Hanghang Tong and Yejin Choi and Jingrui He and James Zou and Mengdi Wang and Ling Yang},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

## 🤝 Ackowledgement 

This code is partially based on the amazing work of [vLLM](https://github.com/vllm-project/vllm).
