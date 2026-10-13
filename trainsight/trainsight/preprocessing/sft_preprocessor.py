"""
Shared SFT Preprocessor & Collation Module.

Unifies preprocessing across training and inspection:
- Standardized Qwen chat template formatting
- Tokenizer encoding with exact BPE tokenizer
- Configured truncation policy
- Strict answer-only loss masking: prompt tokens set to -100, assistant target tokens preserved
- Dynamic batch padding: padding tokens set to pad_token_id in input_ids, 0 in attention_mask, -100 in labels
- Explicit boundary assertion: verifies prompt_ids is an exact prefix of full_ids
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer


@dataclass
class SFTExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    prompt_tokens_len: int
    completion_tokens_len: int
    total_len: int
    raw_len: int = 0
    is_valid: bool = True
    has_empty_completion: bool = False
    is_truncated: bool = False
    truncated_answer: bool = False
    loss_mask_misaligned: bool = False
    error_reason: Optional[str] = None


class SFTPreprocessor:
    """Shared preprocessor for SFT data inspection and training collation."""

    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_len: int = 2048,
        truncation: bool = True,
    ):
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.truncation = truncation
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_prompt_and_completion(self, sample: Dict[str, Any]) -> Tuple[str, str, bool, Optional[str]]:
        """Extracts prompt and completion strings from various dataset schemas."""
        if not isinstance(sample, dict):
            return "", "", False, "Row is not a valid JSON dictionary"

        # 1. OpenAI messages format
        messages = sample.get("messages")
        if isinstance(messages, list):
            user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
            assistant_msgs = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
            prompt = "\n".join(user_msgs).strip()
            completion = "\n".join(assistant_msgs).strip()
            if not prompt:
                return "", completion, False, "Missing user prompt in messages"
            return prompt, completion, True, None

        # 2. Key-pair formats: prompt/completion, question/answer, instruction/output
        prompt = (
            sample.get("prompt")
            or sample.get("question")
            or sample.get("instruction")
            or ""
        )
        completion = (
            sample.get("completion")
            or sample.get("answer")
            or sample.get("output")
            or sample.get("response")
            or ""
        )

        if not str(prompt).strip():
            return "", str(completion), False, "Empty or missing prompt"

        return str(prompt).strip(), str(completion).strip(), True, None

    def process_sample(self, sample: Dict[str, Any]) -> SFTExample:
        """Processes a single raw SFT record into model-ready token IDs and answer-only labels."""
        prompt, completion, is_valid, err = self.extract_prompt_and_completion(sample)

        if not is_valid:
            return SFTExample(
                input_ids=[],
                attention_mask=[],
                labels=[],
                prompt_tokens_len=0,
                completion_tokens_len=0,
                total_len=0,
                is_valid=False,
                has_empty_completion=False,
                is_truncated=False,
                truncated_answer=False,
                loss_mask_misaligned=False,
                error_reason=err,
            )

        has_empty_completion = len(completion) == 0

        # Construct chat template texts
        prompt_messages = [{"role": "user", "content": prompt}]
        full_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion if not has_empty_completion else ""},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = self.tokenizer.encode(prompt_text, truncation=False)
        full_ids = self.tokenizer.encode(full_text, truncation=False)

        raw_total_len = len(full_ids)
        prompt_len = len(prompt_ids)

        # Explicit mask boundary verification:
        # Assert that the separately tokenized prompt is an exact prefix of the full token sequence
        assert full_ids[:prompt_len] == prompt_ids, (
            f"Tokenization boundary mismatch: prompt_ids is not an exact prefix of full_ids. "
            f"Prompt IDs len={prompt_len}, Full IDs prefix={full_ids[:prompt_len]}"
        )

        raw_target_len = max(0, raw_total_len - prompt_len)
        is_truncated = False
        truncated_answer = False

        if self.truncation and raw_total_len > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]
            is_truncated = True
            remaining_target_tokens = max(0, len(full_ids) - prompt_len)
            if raw_target_len > 0 and remaining_target_tokens < raw_target_len:
                truncated_answer = True

        total_len = len(full_ids)
        attention_mask = [1] * total_len

        # Answer-only loss masking:
        # Prompt tokens: labels = -100 (uncomputed loss)
        # Assistant completion tokens: labels = token_id (target loss)
        labels = []
        for i, token_id in enumerate(full_ids):
            if i < prompt_len:
                labels.append(-100)
            else:
                labels.append(token_id)

        # Count active assistant tokens contributing to loss
        active_target_tokens = sum(1 for lbl in labels if lbl != -100)

        # Check for loss mask misalignment:
        loss_mask_misaligned = False
        raw_labels = sample.get("labels")
        if isinstance(raw_labels, list) and raw_labels:
            # Misaligned if prompt prefix contains unmasked tokens (label != -100)
            prompt_slice = raw_labels[:prompt_len]
            if any(lbl != -100 for lbl in prompt_slice):
                loss_mask_misaligned = True
            elif -100 not in raw_labels:
                loss_mask_misaligned = True
            elif not has_empty_completion and len(raw_labels) > prompt_len and all(lbl == -100 for lbl in raw_labels[prompt_len:]):
                loss_mask_misaligned = True

        return SFTExample(
            input_ids=full_ids,
            attention_mask=attention_mask,
            labels=labels,
            prompt_tokens_len=prompt_len,
            completion_tokens_len=active_target_tokens,
            total_len=total_len,
            raw_len=raw_total_len,
            is_valid=True,
            has_empty_completion=has_empty_completion,
            is_truncated=is_truncated,
            truncated_answer=truncated_answer,
            loss_mask_misaligned=loss_mask_misaligned,
            error_reason="Empty assistant completion" if has_empty_completion else None,
        )

    def collate_batch(
        self,
        examples: List[SFTExample],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Collate a batch of SFTExamples into dynamically padded tensors on device.
        
        Returns:
            input_ids: (B, max_len)
            attention_mask: (B, max_len) with 0 for padding
            labels: (B, max_len) with -100 for prompt and padding
            useful_tokens: Total unpadded tokens in batch
            padded_tokens: Total padded tokens sent to model (max_len * B)
        """
        bs = len(examples)
        lengths = [ex.total_len for ex in examples]
        max_len = max(lengths)
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        labels = torch.full((bs, max_len), -100, dtype=torch.long, device=device)

        useful_tokens = sum(lengths)
        padded_tokens = max_len * bs

        for i, ex in enumerate(examples):
            seq_len = ex.total_len
            input_ids[i, :seq_len] = torch.tensor(ex.input_ids, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = torch.tensor(ex.attention_mask, dtype=torch.long, device=device)
            labels[i, :seq_len] = torch.tensor(ex.labels, dtype=torch.long, device=device)

        return input_ids, attention_mask, labels, useful_tokens, padded_tokens
