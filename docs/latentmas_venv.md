# Isolated LatentMAS + vLLM environment

Environment: `/home/research/nghialt/latentmas` (Python 3.11).
The separate KVComm `.venv` is unchanged.

Activate before running: vLLM JIT compilation invokes tools such as `ninja`
from the environment's `bin` directory.

```bash
source /home/research/nghialt/latentmas/bin/activate
cd /home/research/nghialt/LatentMAS
```

Installed versions are recorded in `requirements-vllm.lock.txt` for this
Linux aarch64 / NVIDIA GB10 environment. Key versions: vLLM 0.25.1,
Transformers 5.17.0, PyTorch 2.11.0 (CUDA 13.0), datasets 5.0.0.

Small offline smoke test using cached Qwen3-4B and local MedQA data:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_NO_USAGE_STATS=1 python run.py \
  --method latent_mas --model_name Qwen/Qwen3-4B --task medqa \
  --num_agents 2 --prompt sequential --max_samples 1 --generate_bs 1 \
  --latent_steps 2 --max_new_tokens 64 --use_vllm --latent_space_realign \
  --device cuda:0 --device2 cuda:0 --gpu_memory_utilization 0.2
```

This is a runtime smoke test, not an accuracy benchmark. Both the HF model
and vLLM engine use GPU 0; two agent roles do not mean four model instances.

Verified on 2026-09-16: the command above completed with exit code 0.
Output: `output/latent_mas_Qwen_Qwen3-4B_medqa_20260916_014600.json`.
Both Planner and Judger ran; the 64-token response stopped during thinking
and had no final answer. The recorded 3.4932-second evaluation time excludes
model/engine initialization and is not a benchmark timing result.

`uv pip check` reports a platform-tag warning for `nvidia-cusparselt-cu13`
0.8.0 (`manylinux2014_sbsa`). Do not interpret dependency installation alone
as confirmation that every CUDA kernel or workload is supported.
