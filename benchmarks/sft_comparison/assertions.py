"""
Independent Custom Assertions for SFT Batches.
Verifies the declared SFT objective:
1. Only assistant-answer tokens contribute to loss (labels == -100 on prompt).
2. Padding tokens must be ignored (labels == -100 wherever attention_mask == 0).
3. Every accepted example must retain at least one answer token that contributes to shifted causal loss.
4. Active label tokens must match input_ids exactly.
5. The input sequence must conform to the declared ChatML template.
6. The answer must contain substantive content tokens beyond boundary/EOS tokens.
"""

from typing import Dict, Any, List, Optional
import torch
from transformers import PreTrainedTokenizerBase


class SFTAssertionError(AssertionError):
    """Specific assertion failure for SFT batch validation."""
    def __init__(self, check_name: str, message: str, row_id: Optional[str] = None):
        self.check_name = check_name
        self.row_id = row_id
        super().__init__(f"[{check_name}] (Row: {row_id or 'unknown'}): {message}")


def assert_sft_batch(
    batch: Dict[str, torch.Tensor],
    prompt_lengths: List[int],
    tokenizer: PreTrainedTokenizerBase,
    row_ids: Optional[List[str]] = None,
    expected_template_prefix: str = "<|im_start|>user",
    expected_template_suffix: str = "<|im_start|>assistant\n",
) -> None:
    """Runs independent verification checks across every example in an SFT batch."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch.get("attention_mask")

    batch_size, seq_len = input_ids.shape
    row_ids = row_ids or [f"row_{i:03d}" for i in range(batch_size)]

    for b in range(batch_size):
        r_id = row_ids[b]
        b_inputs = input_ids[b]
        b_labels = labels[b]
        b_mask = attention_mask[b] if attention_mask is not None else torch.ones_like(b_inputs)
        p_len = prompt_lengths[b]

        # 1. Non-empty Supervision Check
        active_labels = (b_labels != -100).nonzero(as_tuple=True)[0]
        if len(active_labels) == 0:
            raise SFTAssertionError(
                "NO_SUPERVISED_TOKENS",
                "Example has zero supervised tokens (all labels are -100).",
                row_id=r_id,
            )

        # 2. Causal Shift Loss Contribution Check:
        # In causal LM, token i is predicted from token i-1. Tokens at index 0 cannot be predicted.
        # At least one supervised token must be at index >= 1.
        causal_supervised = [idx.item() for idx in active_labels if idx.item() >= 1]
        if len(causal_supervised) == 0:
            raise SFTAssertionError(
                "NO_CAUSAL_SUPERVISED_TOKENS",
                "Supervised tokens exist only at index 0; no token contributes to shifted causal LM loss.",
                row_id=r_id,
            )

        # 3. Answer-Only Supervision Check (Prompt Must Be Ignored)
        prompt_slice_labels = b_labels[:p_len]
        unmasked_prompt_indices = (prompt_slice_labels != -100).nonzero(as_tuple=True)[0]
        if len(unmasked_prompt_indices) > 0:
            violating_idx = unmasked_prompt_indices[0].item()
            tok_id = b_inputs[violating_idx].item()
            tok_text = tokenizer.decode([tok_id])
            raise SFTAssertionError(
                "PROMPT_LABELS_UNMASKED",
                f"Prompt token at position {violating_idx} ({tok_id}: {repr(tok_text)}) has label != -100. "
                "Prompt tokens must not contribute to loss.",
                row_id=r_id,
            )

        # 4. Zero Supervision on Padding Positions
        pad_indices = (b_mask == 0).nonzero(as_tuple=True)[0]
        if len(pad_indices) > 0:
            unmasked_pad = [idx.item() for idx in pad_indices if b_labels[idx] != -100]
            if len(unmasked_pad) > 0:
                p_idx = unmasked_pad[0]
                raise SFTAssertionError(
                    "PADDING_LABELS_UNMASKED",
                    f"Padded position {p_idx} (attention_mask == 0) has label {b_labels[p_idx].item()} != -100. "
                    "Padding tokens must never contribute to loss.",
                    row_id=r_id,
                )

        # 5. Label-to-Token Identity Check
        for idx in active_labels:
            i = idx.item()
            if b_labels[i].item() != b_inputs[i].item():
                raise SFTAssertionError(
                    "LABEL_TOKEN_MISMATCH",
                    f"Label at index {i} ({b_labels[i].item()}) does not match input token ({b_inputs[i].item()}).",
                    row_id=r_id,
                )

        # 6. Template Delimiter Verification
        decoded_prompt = tokenizer.decode(b_inputs[:p_len], skip_special_tokens=False)
        if expected_template_prefix not in decoded_prompt or expected_template_suffix not in decoded_prompt:
            raise SFTAssertionError(
                "TEMPLATE_MISMATCH",
                f"Prompt does not contain required ChatML delimiters ({expected_template_prefix} ... {expected_template_suffix}). "
                f"Decoded prompt prefix: {repr(decoded_prompt[:60])}",
                row_id=r_id,
            )

        # 7. Substantive Content Verification (Not solely EOS / closing tag)
        active_token_ids = [b_inputs[idx].item() for idx in active_labels]
        # Filter out EOS and template delimiter tokens
        content_tokens = [
            t for t in active_token_ids
            if t not in (tokenizer.eos_token_id, tokenizer.pad_token_id)
            and tokenizer.decode([t]).strip() not in ("<|im_end|>", "")
        ]
        if len(content_tokens) == 0:
            raise SFTAssertionError(
                "EMPTY_ANSWER_CONTENT",
                "Supervised tokens consist entirely of EOS or template closing delimiters without answer content.",
                row_id=r_id,
            )
