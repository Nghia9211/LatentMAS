import argparse
import json
import os
import sys
import math
from itertools import islice
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def save_results(preds: List[Dict], summary: Dict, args: argparse.Namespace) -> str:
    os.makedirs("output", exist_ok=True)

    model_short = args.model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"{args.method}_{model_short}_{args.task}_{timestamp}"

    json_path = os.path.join("output", f"{base_name}.json")
    with open(json_path, "x", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": preds}, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join("output", f"{base_name}_summary.txt")
    with open(txt_path, "x", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("           EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Method         : {summary['method']}\n")
        if summary.get("latent_steps") is not None:
            f.write(f"Latent steps   : {summary['latent_steps']} (per latent agent)\n")
        f.write(f"Model          : {summary['model']}\n")
        f.write(f"Task           : {summary['task']}\n")
        f.write(f"Split          : {summary['split']}\n")
        f.write(f"Seed           : {summary['seed']}\n")
        f.write(f"Max samples    : {summary['max_samples']}\n")
        f.write(f"Correct        : {summary['correct']} / {summary['max_samples']}\n")
        f.write(f"Accuracy       : {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)\n")
        if 'medqa_boxed_v1_accuracy' in summary:
            f.write(f"Strict boxed   : {summary['medqa_boxed_v1_accuracy']:.4f}\n")
            f.write(f"Parse failures : {summary['medqa_boxed_v1_parse_failures']}\n")
        f.write(f"Total time     : {summary['total_time_sec']} sec\n")
        f.write(f"Time/sample    : {summary['time_per_sample_sec']} sec\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("Per-sample results:\n")
        f.write("-" * 60 + "\n")
        for i, pred in enumerate(preds, 1):
            status = "✓" if pred.get("correct") else "✗"
            f.write(f"[{i:4d}] {status}  Pred: {pred.get('prediction', '')}  |  Gold: {pred.get('gold', '')}\n")

    print(f"\n[Output] Predictions saved  -> {json_path}")
    print(f"[Output] Summary saved      -> {txt_path}")
    return base_name


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) != len(current_batch):
        raise RuntimeError(f'Expected {len(current_batch)} results, got {len(results)}; refusing incomplete evaluation')
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def _load_yaml_config(path: str) -> dict:
    """Đọc file YAML và trả về dict config."""
    if not _HAS_YAML:
        print("[WARNING] pyyaml chưa được cài. Chạy: pip install pyyaml")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError('YAML config must be a mapping of argument names to values')
    print(f"[Config] Đã đọc config từ: {path}")
    return cfg


def parse_args(argv=None):
    # --- Bước 1: parse --config trước để lấy YAML defaults ---
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="Đường dẫn tới file YAML config (vd: configs/baseline.yaml)")
    pre_args, _ = pre_parser.parse_known_args(argv)

    yaml_defaults: dict = {}
    if pre_args.config:
        yaml_defaults = _load_yaml_config(pre_args.config)

    # --- Bước 2: build parser chính với defaults từ YAML ---
    parser = argparse.ArgumentParser(
        description="LatentMAS experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Đường dẫn tới file YAML config")

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"],
                        default=yaml_defaults.get("method", None), required=("method" not in yaml_defaults),
                        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
    parser.add_argument("--model_name", type=str,
                        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"],
                        default=yaml_defaults.get("model_name", None), required=("model_name" not in yaml_defaults),
                        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').")
    parser.add_argument("--max_samples", type=int,
                        default=yaml_defaults.get("max_samples", -1),
                        help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task",
                        choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", "humanevalplus", "medqa"],
                        default=yaml_defaults.get("task", "gsm8k"),
                        help="Dataset/task to evaluate.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"],
                        default=yaml_defaults.get("prompt", "sequential"),
                        help="Multi-agent system architecture.")

    parser.add_argument("--num_agents", type=int, choices=[2, 4],
                        default=yaml_defaults.get("num_agents", 2),
                        help="LatentMAS/TextMAS agents: 2 = Planner/Judger; 4 = original pipeline")

    # other args
    parser.add_argument("--device", type=str, default=yaml_defaults.get("device", "cuda"))
    parser.add_argument("--split", type=str, default=yaml_defaults.get("split", None),
                        help="Dataset split; default train for AIME and test for other tasks")
    parser.add_argument("--max_new_tokens", type=int, default=yaml_defaults.get("max_new_tokens", 4096))
    parser.add_argument("--latent_steps", type=int, default=yaml_defaults.get("latent_steps", 0),
                        help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=yaml_defaults.get("temperature", 0.6))
    parser.add_argument("--top_p", type=float, default=yaml_defaults.get("top_p", 0.95))
    parser.add_argument("--generate_bs", type=int, default=yaml_defaults.get("generate_bs", 20),
                        help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int,
                        default=yaml_defaults.get("text_mas_context_length", -1),
                        help="TextMAS context length limit")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=yaml_defaults.get("think", False),
                        help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action=argparse.BooleanOptionalAction,
                        default=yaml_defaults.get("latent_space_realign", False))
    parser.add_argument("--seed", type=int, default=yaml_defaults.get("seed", 42))

    # vLLM support
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction,
                        default=yaml_defaults.get("use_vllm", False),
                        help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action=argparse.BooleanOptionalAction,
                        default=yaml_defaults.get("enable_prefix_caching", False),
                        help="Enable prefix caching in vLLM")
    parser.add_argument("--use_second_HF_model", action=argparse.BooleanOptionalAction,
                        default=yaml_defaults.get("use_second_HF_model", False),
                        help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default=yaml_defaults.get("device2", "cuda:1"))
    parser.add_argument("--tensor_parallel_size", type=int,
                        default=yaml_defaults.get("tensor_parallel_size", 1),
                        help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float,
                        default=yaml_defaults.get("gpu_memory_utilization", 0.3),
                        help="Target GPU memory utilization for vLLM")

    known = {a.dest: a for a in parser._actions}
    unknown = set(yaml_defaults) - set(known)
    if unknown:
        parser.error(f'Unknown YAML keys: {sorted(unknown)}')
    # argparse does not validate numeric/boolean YAML defaults or their choices.
    for name, value in yaml_defaults.items():
        action = known[name]
        expected = bool if isinstance(action, argparse.BooleanOptionalAction) else action.type
        if expected in (bool, int, str) and type(value) is not expected:
            parser.error(f'YAML {name} must have type {expected.__name__}')
        if expected is float and (type(value) not in (float, int) or not math.isfinite(value)):
            parser.error(f'YAML {name} must be a finite number')
        if action.choices is not None and value not in action.choices:
            parser.error(f'Invalid YAML {name}: {value!r}; choices: {action.choices}')
    args = parser.parse_args(argv)
    for name in ('generate_bs', 'max_new_tokens', 'tensor_parallel_size'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.max_samples != -1 and args.max_samples < 1:
        parser.error('max_samples must be -1 or positive')
    if args.latent_steps < 0:
        parser.error('latent_steps must be nonnegative')
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error('temperature must be finite and nonnegative')
    if not 0 < args.top_p <= 1 or not 0 < args.gpu_memory_utilization < 1:
        parser.error('Require 0 < top_p <= 1 and 0 < gpu_memory_utilization < 1')
    if not 0 <= args.seed < 2**32:
        parser.error('seed must be between 0 and 2**32-1')
    if args.split is None:
        args.split = 'train' if args.task in ('aime2024', 'aime2025') else 'test'
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True
    return args


def load_items(args):
    loaders = dict(gsm8k=load_gsm8k, aime2024=load_aime2024, aime2025=load_aime2025,
                   gpqa=load_gpqa_diamond, arc_easy=load_arc_easy, arc_challenge=load_arc_challenge,
                   mbppplus=load_mbppplus, humanevalplus=load_humanevalplus, medqa=load_medqa)
    dataset = loaders[args.task](split=args.split)
    items = list(dataset if args.max_samples == -1 else islice(dataset, args.max_samples))
    if not items:
        raise ValueError('Dataset selection is empty; no model was loaded')
    return items


def main():
    args = parse_args()
    set_seed(args.seed)
    # Validate data before allocating VRAM; do not silently relabel train as test.
    data_start = time.perf_counter()
    dataset_iter = load_items(args)
    data_loading_seconds = time.perf_counter() - data_start
    requested_max_samples = args.max_samples
    sample_limit = len(dataset_iter)
    device = auto_device(args.device)
    init_start = time.perf_counter()
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    initialization_seconds = time.perf_counter() - init_start
    
    start_time = time.perf_counter()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    progress = tqdm(total=sample_limit)

    for item in dataset_iter:
        if processed >= sample_limit:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == sample_limit:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                sample_limit,
                args,
            )
            batch = []
            if processed >= sample_limit:
                break

    if batch and processed < sample_limit:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=sample_limit,
            args=args,
        )
    progress.close()
    
    total_time = time.perf_counter() - start_time

    acc, correct = evaluate(preds)

    summary = {
        "run_config": vars(args).copy(),
        "initialization_seconds": initialization_seconds,
        "timing_note": "Evaluation excludes initialization/data loading but includes logging; time/sample is amortized throughput, not request latency.",
        "method": args.method,
        "num_agents": len(method.agents) if hasattr(method, "agents") else 1,
        "latent_steps": args.latent_steps if args.method == "latent_mas" else None,
        "model": args.model_name,
        "task": args.task,
        "split": args.split,
        "seed": args.seed,
        "max_samples": len(preds),
        "num_samples": len(preds),
        "requested_max_samples": requested_max_samples,
        "data_loading_seconds": data_loading_seconds,
        "accuracy": acc,
        "correct": correct,
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": round(total_time / len(preds), 4),
    }

    # In kết quả ra terminal
    from importlib.metadata import version, PackageNotFoundError
    summary["package_versions"] = {}
    for name in ("torch", "transformers", "vllm"):
        try:
            summary["package_versions"][name] = version(name)
        except PackageNotFoundError:
            summary["package_versions"][name] = None
    summary["backend"] = "hf+vllm" if args.use_vllm and args.method == "latent_mas" else ("vllm" if args.use_vllm else "hf")
    if args.task == 'medqa':
        summary['dataset_source'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'medqa.json')
        summary['dataset_note'] = 'Packaged local evaluation subset; official split provenance not verified.'
    if args.task == "medqa" and preds and all("medqa_boxed_v1" in p for p in preds):
        summary["metric_version"] = "legacy_with_medqa_boxed_v1"
        summary["medqa_boxed_v1_accuracy"] = sum(p["medqa_boxed_v1"]["correct"] for p in preds) / len(preds)
        summary["medqa_boxed_v1_parse_failures"] = sum(p["medqa_boxed_v1"]["parse_failed"] for p in preds)
    print(json.dumps(summary, ensure_ascii=False))

    # Lưu kết quả vào folder output/
    save_results(preds, summary, args)




if __name__ == "__main__":
    main()
