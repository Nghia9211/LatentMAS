import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    # Transformers >= 4.38: DynamicCache / Cache object
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    # Fallback: legacy tuple-of-tuples format  (transformers < 4.38)
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]



class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        if use_vllm and not _HAS_VLLM:
            raise RuntimeError('use_vllm=True requires vLLM; install it or explicitly use the HF backend.')
        self.use_vllm = use_vllm
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args
        self.seed = int(getattr(args, "seed", 42))

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            use_prefix_caching = bool(getattr(args, "enable_prefix_caching", False))
            is_latent_mas = getattr(args, "method", "") == "latent_mas"
            if is_latent_mas:
                # Prompt embeddings are required even when prefix caching is off.
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=use_prefix_caching, enable_prompt_embeds=True, seed=self.seed)
            elif use_prefix_caching:
                # baseline / text_mas: prefix caching only (no prompt embeds needed)
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, seed=self.seed)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=False, seed=self.seed)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.tokenizer.padding_side = 'left'
            
            use_second_hf = is_latent_mas or bool(getattr(args, "use_second_HF_model", False))
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = 'left'
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=self.seed,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if getattr(self.args, "latent_space_realign", False):
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        attention_mask = attention_mask.to(input_ids.device)
        if (attention_mask.shape != input_ids.shape
                or not ((attention_mask == 0) | (attention_mask == 1)).all()
                or not attention_mask.any(dim=1).all()):
            raise ValueError('Each prompt needs a matching binary mask and at least one real token')
        if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError('Require max_new_tokens > 0, temperature >= 0 and 0 < top_p <= 1')
        # Decoder-only generation samples logits at the last physical position.
        # Normalize incoming left/right-padded prompts to left padding.
        padded = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        prompt_mask = torch.zeros_like(attention_mask)
        for i in range(input_ids.shape[0]):
            real = input_ids[i, attention_mask[i].bool()]
            padded[i, -len(real):] = real
            prompt_mask[i, -len(real):] = 1
        input_ids, attention_mask = padded, prompt_mask
        generation_kwargs = {}
        offset = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = getattr(past_key_values, '_latent_attention_mask', None)
                offsets = getattr(past_key_values, '_latent_next_position', None)
                if past_mask is None or offsets is None:
                    raise ValueError('Cached decoding requires saved mask/position metadata')
                past_mask = past_mask.to(attention_mask.device)
                if past_mask.shape != (input_ids.shape[0], past_len):
                    raise ValueError('Cache batch/length does not match the saved padding mask')
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
                # HF expects input_ids to include positions already in the cache.
                # Their IDs are placeholders only; generate must process the new suffix.
                prefix_ids = input_ids.new_full((input_ids.shape[0], past_len), self.tokenizer.pad_token_id)
                input_ids = torch.cat([prefix_ids, input_ids], dim=-1)
                offset = offsets.to(input_ids.device) - past_mask.sum(-1)
                if (offset != 0).any():
                    if not hasattr(self.model, '_prepare_position_ids_for_generation'):
                        raise ValueError('Decoding a truncated latent cache requires Transformers 5.x')
                    positions = attention_mask.long().cumsum(-1) - 1 + offset[:, None]
                    generation_kwargs['position_ids'] = positions.masked_fill(attention_mask == 0, 0)
        if temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=top_p)
        else:
            generation_kwargs.update(temperature=None, top_p=None, top_k=None)
        prompt_width = input_ids.shape[1]
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            **generation_kwargs,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx in range(sequences.shape[0]):
            generated_ids = sequences[idx, prompt_width:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        cache = outputs.past_key_values
        if cache is not None and hasattr(cache, 'get_seq_length'):
            added = cache.get_seq_length() - attention_mask.shape[1]
            cache._latent_attention_mask = torch.cat([
                attention_mask, attention_mask.new_ones((input_ids.shape[0], max(0, added)))], dim=-1).detach()
            cache._latent_next_position = (cache._latent_attention_mask.sum(-1) + offset).detach()
        return generations, cache

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        from latent_batch import rollout
        cache, _ = rollout(self, input_ids, attention_mask, latent_steps=latent_steps,
                           past_key_values=past_key_values, return_embeddings=False)
        return cache

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self, input_ids, attention_mask=None, *, latent_steps, past_key_values=None,
    ):
        """Return HF cache and a ragged list of unpadded input/latent embeddings."""
        from latent_batch import rollout
        return rollout(self, input_ids, attention_mask, latent_steps=latent_steps,
                       past_key_values=past_key_values)
