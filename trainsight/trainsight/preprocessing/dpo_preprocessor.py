"""
Shared DPO Preprocessor & Collator Contract.

Guarantees identical tokenization, truncation rules, and prompt sharing
between pre-flight TrainSight inspection and TRL DPOTrainer.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class DPOExample:
    prompt_input_ids: List[int]
    chosen_input_ids: List[int]
    rejected_input_ids: List[int]
    raw_prompt_len: int
    raw_chosen_len: int
    raw_rejected_len: int
    truncated_prompt: bool
    truncated_chosen: bool
    truncated_rejected: bool
    score_chosen: Optional[float] = None
    score_rejected: Optional[float] = None


class DPOPreprocessor:
    """
    Standardized Preprocessor for Direct Preference Optimization (DPO).

    Contract:
    1. Prompt is formatted with Qwen ChatML template:
       <|im_start|>system...<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    2. Completions (chosen and rejected) are tokenized as assistant responses, terminated with <|im_end|>.
    3. Truncation policy:
       - Prompt: left-truncated ([-max_prompt_length:]) if raw prompt > max_prompt_length.
       - Completion: right-truncated ([:max_completion_length]) if raw completion > max_completion_length.
    4. Exact prompt sharing invariant:
       Both chosen and rejected branches must condition on the identical prompt tokens.
    """

    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_prompt_length: int = 1024,
        max_completion_length: int = 1024,
        max_length: int = 2048,
        add_eos_to_completion: bool = True,
    ):
        self.tokenizer_name = tokenizer_name
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.max_length = max_length
        self.add_eos_to_completion = add_eos_to_completion

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def extract_texts(self, sample: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extracts prompt, chosen assistant text, and rejected assistant text."""
        prompt = sample.get("prompt") or ""
        prompt_str = prompt.strip() if isinstance(prompt, str) else str(prompt).strip()

        raw_chosen = sample.get("chosen")
        raw_rejected = sample.get("rejected")

        asst_chosen = self._extract_assistant(raw_chosen)
        asst_rejected = self._extract_assistant(raw_rejected)

        return prompt_str, asst_chosen, asst_rejected

    def _extract_assistant(self, field: Any) -> str:
        if not field:
            return ""
        if isinstance(field, str):
            return field.strip()
        if isinstance(field, list):
            for msg in field:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "").strip()
            if len(field) >= 2 and isinstance(field[1], dict):
                return field[1].get("content", "").strip()
        return str(field).strip()

    def process_sample(self, sample: Dict[str, Any]) -> DPOExample:
        prompt_str, asst_chosen, asst_rejected = self.extract_texts(sample)

        # Format prompt with ChatML template
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_str}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = self.tokenizer.encode(chat_prompt, add_special_tokens=False)

        # Tokenize completions
        chosen_tokens = self.tokenizer.encode(asst_chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer.encode(asst_rejected, add_special_tokens=False)

        if self.add_eos_to_completion and self.tokenizer.eos_token_id is not None:
            chosen_tokens = chosen_tokens + [self.tokenizer.eos_token_id]
            rejected_tokens = rejected_tokens + [self.tokenizer.eos_token_id]

        raw_p_len = len(prompt_tokens)
        raw_c_len = len(chosen_tokens)
        raw_r_len = len(rejected_tokens)

        # Left-truncate prompt if over budget
        truncated_prompt = raw_p_len > self.max_prompt_length
        retained_prompt_tokens = (
            prompt_tokens[-self.max_prompt_length:] if truncated_prompt else prompt_tokens
        )

        # Right-truncate completions if over budget
        truncated_chosen = raw_c_len > self.max_completion_length
        retained_chosen_tokens = (
            chosen_tokens[:self.max_completion_length] if truncated_chosen else chosen_tokens
        )

        truncated_rejected = raw_r_len > self.max_completion_length
        retained_rejected_tokens = (
            rejected_tokens[:self.max_completion_length] if truncated_rejected else rejected_tokens
        )

        sc = sample.get("score_chosen")
        sr = sample.get("score_rejected")

        return DPOExample(
            prompt_input_ids=retained_prompt_tokens,
            chosen_input_ids=retained_chosen_tokens,
            rejected_input_ids=retained_rejected_tokens,
            raw_prompt_len=raw_p_len,
            raw_chosen_len=raw_c_len,
            raw_rejected_len=raw_r_len,
            truncated_prompt=truncated_prompt,
            truncated_chosen=truncated_chosen,
            truncated_rejected=truncated_rejected,
            score_chosen=float(sc) if sc is not None else None,
            score_rejected=float(sr) if sr is not None else None,
        )

    def collate_batch(self, examples: List[DPOExample]) -> Dict[str, torch.Tensor]:
        """
        Pads examples into standard TRL preference format:
        - prompt_input_ids / prompt_attention_mask: left-padded
        - chosen_input_ids / chosen_attention_mask: right-padded
        - rejected_input_ids / rejected_attention_mask: right-padded
        """
        pad_id = self.tokenizer.pad_token_id or 0

        p_tensors = [torch.tensor(e.prompt_input_ids, dtype=torch.long) for e in examples]
        c_tensors = [torch.tensor(e.chosen_input_ids, dtype=torch.long) for e in examples]
        r_tensors = [torch.tensor(e.rejected_input_ids, dtype=torch.long) for e in examples]

        max_p = max(len(t) for t in p_tensors)
        max_c = max(len(t) for t in c_tensors)
        max_r = max(len(t) for t in r_tensors)

        batch_size = len(examples)

        # Left-pad prompts
        prompt_input_ids = torch.full((batch_size, max_p), pad_id, dtype=torch.long)
        prompt_attention_mask = torch.zeros((batch_size, max_p), dtype=torch.long)
        for i, t in enumerate(p_tensors):
            prompt_input_ids[i, max_p - len(t):] = t
            prompt_attention_mask[i, max_p - len(t):] = 1

        # Right-pad completions
        chosen_input_ids = torch.full((batch_size, max_c), pad_id, dtype=torch.long)
        chosen_attention_mask = torch.zeros((batch_size, max_c), dtype=torch.long)
        for i, t in enumerate(c_tensors):
            chosen_input_ids[i, :len(t)] = t
            chosen_attention_mask[i, :len(t)] = 1

        rejected_input_ids = torch.full((batch_size, max_r), pad_id, dtype=torch.long)
        rejected_attention_mask = torch.zeros((batch_size, max_r), dtype=torch.long)
        for i, t in enumerate(r_tensors):
            rejected_input_ids[i, :len(t)] = t
            rejected_attention_mask[i, :len(t)] = 1

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

    def build_concatenated_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Builds the exact input tensors consumed by the policy forward pass (concatenated forward):
        - Concatenates prompt and chosen, and prompt and rejected
        - Computes loss_mask (0 for prompt and pad positions, 1 strictly for valid completion tokens)
        """
        pad_id = self.tokenizer.pad_token_id or 0
        p_ids = batch["prompt_input_ids"]
        p_mask = batch["prompt_attention_mask"]
        c_ids = batch["chosen_input_ids"]
        c_mask = batch["chosen_attention_mask"]
        r_ids = batch["rejected_input_ids"]
        r_mask = batch["rejected_attention_mask"]

        # Duplicate prompts across branches
        full_p_ids = torch.cat([p_ids, p_ids], dim=0)
        full_p_mask = torch.cat([p_mask, p_mask], dim=0)

        # Pad completions to shared maximum width
        max_comp = max(c_ids.shape[1], r_ids.shape[1])
        c_ids_padded = torch.full((c_ids.shape[0], max_comp), pad_id, dtype=torch.long)
        c_mask_padded = torch.zeros((c_mask.shape[0], max_comp), dtype=torch.long)
        c_ids_padded[:, :c_ids.shape[1]] = c_ids
        c_mask_padded[:, :c_mask.shape[1]] = c_mask

        r_ids_padded = torch.full((r_ids.shape[0], max_comp), pad_id, dtype=torch.long)
        r_mask_padded = torch.zeros((r_mask.shape[0], max_comp), dtype=torch.long)
        r_ids_padded[:, :r_ids.shape[1]] = r_ids
        r_mask_padded[:, :r_mask.shape[1]] = r_mask

        full_comp_ids = torch.cat([c_ids_padded, r_ids_padded], dim=0)
        full_comp_mask = torch.cat([c_mask_padded, r_mask_padded], dim=0)

        input_ids = torch.cat([full_p_ids, full_comp_ids], dim=1)
        attention_mask = torch.cat([full_p_mask, full_comp_mask], dim=1)

        # loss_mask: 0 for prompt, 0 for padding, 1 strictly for completion tokens
        loss_mask = torch.cat([torch.zeros_like(full_p_mask), full_comp_mask], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }
