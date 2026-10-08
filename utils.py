import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed host/torch RNGs; vLLM engine and requests must also receive seed.

    PYTHONHASHSEED set here affects future child interpreters, not hash
    randomization in this already-running interpreter. For that, launch with
    PYTHONHASHSEED=42 (or the chosen seed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def medqa_scoring_metadata(text, gold, legacy_prediction, legacy_correct):
    """Strict boxed-choice score alongside unchanged historical scoring."""
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    final = boxes[-1].strip().lower() if boxes else None
    prediction = final if final in ("a", "b", "c", "d") else None
    return {
        "metric_version": "legacy_with_medqa_boxed_v1",
        "legacy_prediction": legacy_prediction,
        "legacy_correct": bool(legacy_correct),
        "medqa_boxed_v1": {
            "prediction": prediction,
            "correct": prediction is not None and prediction == normalize_answer(gold),
            "parse_failed": prediction is None,
        },
    }


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# Keep the public import used by all three methods.
from code_execution import run_with_timeout
