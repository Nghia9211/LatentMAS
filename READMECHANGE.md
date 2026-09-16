# READMECHANGE — Changelog so với LatentMAS gốc

> Tài liệu này ghi lại toàn bộ thay đổi so với upstream repository  
> (`upstream/main` — commit `9a9e4d3`).

---

## 1. Bug Fix: Tương thích `transformers >= 4.38` / 5.x

**File:** `models.py`

### Vấn đề
Transformers phiên bản mới (>= 4.38, hiện tại 5.14.1) thay đổi kiểu trả về của `past_key_values` từ `tuple[tuple[Tensor]]` sang `DynamicCache` object. Hàm `_past_length` cố index vào tuple → crash:

```
TypeError: 'DynamicCache' object is not subscriptable
k = past_key_values[0][0]  # ← lỗi ở đây
```

### Fix
```diff
-def _past_length(past_key_values: Optional[Tuple]) -> int:
+def _past_length(past_key_values) -> int:
+    if past_key_values is None:
+        return 0
+    # Transformers >= 4.38: DynamicCache / Cache object
+    if hasattr(past_key_values, "get_seq_length"):
+        return past_key_values.get_seq_length()
+    # Fallback: legacy tuple-of-tuples (transformers < 4.38)
     if not past_key_values:
         return 0
     k = past_key_values[0][0]
     return k.shape[-2]
```

**Ảnh hưởng đến kết quả:** Không. Chỉ thay đổi cách *đọc* độ dài cache, không ảnh hưởng tính toán.

---

### Fix thêm: xoá `cache_position` khỏi `generate_text_batch`

```diff
-        cache_position = None
         if past_key_values is not None:
             past_len = _past_length(past_key_values)
-            cache_position = torch.arange(
-                past_len, past_len + input_ids.shape[-1],
-                dtype=torch.long, device=self.device,
-            )
             ...
         outputs = self.model.generate(
             ...
-            cache_position=cache_position,
         )
```

`cache_position` không cần truyền tường minh khi dùng `DynamicCache` — transformers tự quản lý nội bộ.

---

## 2. Bug Fix: `enable_prefix_caching` chỉ hoạt động với `latent_mas`

**File:** `models.py`

### Vấn đề
`--enable_prefix_caching` chỉ được truyền vào vLLM khi `method == "latent_mas"`. Với `baseline` và `text_mas`, flag bị bỏ qua hoàn toàn.

### Fix
```diff
-if args.enable_prefix_caching and args.method == "latent_mas":
-    self.vllm_engine = LLM(..., enable_prefix_caching=True, enable_prompt_embeds=True)
-else:
-    self.vllm_engine = LLM(...)

+use_prefix_caching = bool(getattr(args, "enable_prefix_caching", False))
+is_latent_mas = getattr(args, "method", "") == "latent_mas"
+if is_latent_mas and use_prefix_caching:
+    # latent_mas cần cả enable_prompt_embeds
+    self.vllm_engine = LLM(..., enable_prefix_caching=True, enable_prompt_embeds=True)
+elif use_prefix_caching:
+    # baseline / text_mas: prefix caching only
+    self.vllm_engine = LLM(..., enable_prefix_caching=True)
+else:
+    self.vllm_engine = LLM(...)
```

---

## 3. Tính năng mới: Lưu kết quả vào `output/`

**File:** `run.py`

### Mô tả
Sau mỗi lần chạy, tự động tạo 2 file trong folder `output/`:

| File | Nội dung |
|------|----------|
| `output/<method>_<model>_<task>_<timestamp>.json` | Toàn bộ predictions + summary |
| `output/<method>_<model>_<task>_<timestamp>_summary.txt` | Bảng tóm tắt dễ đọc |

### Code thêm
```python
def save_results(preds: List[Dict], summary: Dict, args) -> str:
    os.makedirs("output", exist_ok=True)
    model_short = args.model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.method}_{model_short}_{args.task}_{timestamp}"
    # ... ghi JSON và TXT
```

### Bổ sung vào summary dict
Thêm trường `"task"` vào summary JSON (trường này bị thiếu trong bản gốc):
```diff
 summary = {
     "method": args.method,
     "model": args.model_name,
+    "task": args.task,
     "split": args.split,
     ...
 }
```

---

## 4. Tính năng mới: YAML Config System

**Files mới:**
- `configs/baseline.yaml`
- `configs/text_mas.yaml`
- `configs/latent_mas.yaml`

**File sửa:** `run.py`, `requirements.txt`

### Mô tả
Thêm hỗ trợ đọc config từ file YAML qua `--config`. Giá trị YAML là default; CLI args sẽ **override** nếu được truyền vào.

```bash
# Chạy bằng config YAML
python3 run.py --config configs/baseline.yaml

# Override một tham số cụ thể
python3 run.py --config configs/baseline.yaml --task aime2024 --max_samples 30
```

### Cơ chế hoạt động
```python
# Bước 1: đọc --config trước (pre-parser)
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--config", ...)
pre_args, _ = pre_parser.parse_known_args()
yaml_defaults = _load_yaml_config(pre_args.config) if pre_args.config else {}

# Bước 2: dùng YAML làm defaults cho argparse chính
parser.add_argument("--method", default=yaml_defaults.get("method", None), ...)
```

### Dependency thêm
```diff
# requirements.txt
+pyyaml
```

---

## 5. Cập nhật: `run_baseline.sh`

**Trước:**
```bash
python3 run.py --method baseline --model_name Qwen/Qwen3-4B --task gsm8k \
    --max_samples -1 --max_new_tokens 2048 --use_vllm \
    --gpu_memory_utilization 0.3
```

**Sau:**
```bash
CONFIG="configs/text_mas.yaml"   # đổi dòng này để chuyển method
python3 run.py --config "$CONFIG"
```

---

## Tóm tắt files thay đổi

| File | Loại thay đổi |
|------|--------------|
| `models.py` | Bug fix (DynamicCache, prefix_caching) |
| `run.py` | Bug fix + tính năng mới (output saving, YAML config) |
| `run_baseline.sh` | Cập nhật để dùng YAML config |
| `requirements.txt` | Thêm `pyyaml` |
| `configs/baseline.yaml` | **[MỚI]** Config cho baseline |
| `configs/text_mas.yaml` | **[MỚI]** Config cho text_mas |
| `configs/latent_mas.yaml` | **[MỚI]** Config cho latent_mas |
| `output/` | **[MỚI]** Folder tự tạo khi chạy |
