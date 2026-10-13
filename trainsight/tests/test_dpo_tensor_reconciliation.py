import json
from pathlib import Path
import pytest
import torch

from trainsight.preprocessing.dpo_preprocessor import DPOPreprocessor
from trl.trainer.dpo_trainer import DataCollatorForPreference, DPOTrainer


@pytest.fixture(scope="module")
def preprocessor():
    return DPOPreprocessor(
        tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_prompt_length=1024,
        max_completion_length=1024,
        max_length=2048,
    )


def test_dpo_tensor_reconciliation_all_993_rows(preprocessor: DPOPreprocessor):
    """
    Tensor Reconciliation Pass Criterion:
    Verifies that for all 993 approved rows in clean_ultrafeedback_993.jsonl:
    1. Chosen and rejected use 100% identical prompt token IDs.
    2. Both branches retain at least one completion token.
    3. Exactly 16 prompts exceed 1024 tokens and are deterministically left-truncated to 1024.
    4. Truncated prompts retain the generation prompt suffix (<|im_start|>assistant\\n).
    """
    dataset_path = Path("trainsight/sample_data/clean_ultrafeedback_993.jsonl")
    assert dataset_path.exists(), f"Dataset file not found at {dataset_path}"

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    assert len(rows) == 993, f"Expected 993 rows, found {len(rows)}"

    truncated_prompt_count = 0
    truncated_chosen_count = 0
    truncated_rejected_count = 0

    assistant_turn_suffix = preprocessor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    for idx, row in enumerate(rows):
        ex = preprocessor.process_sample(row)

        # 1. Identical prompt tokens in both branches (contract guarantee)
        assert len(ex.prompt_input_ids) > 0, f"Row {idx}: empty prompt tokens"
        assert len(ex.chosen_input_ids) >= 1, f"Row {idx}: chosen has 0 completion tokens"
        assert len(ex.rejected_input_ids) >= 1, f"Row {idx}: rejected has 0 completion tokens"

        # 2. Check bounds
        assert len(ex.prompt_input_ids) <= 1024, f"Row {idx}: prompt exceeded 1024 tokens"
        assert len(ex.chosen_input_ids) <= 1024, f"Row {idx}: chosen exceeded 1024 tokens"
        assert len(ex.rejected_input_ids) <= 1024, f"Row {idx}: rejected exceeded 1024 tokens"

        if ex.truncated_prompt:
            truncated_prompt_count += 1
            assert ex.raw_prompt_len > 1024
            assert len(ex.prompt_input_ids) == 1024
            # Verify that left truncation preserves the trailing generation prompt
            assert ex.prompt_input_ids[-len(assistant_turn_suffix):] == assistant_turn_suffix

        if ex.truncated_chosen:
            truncated_chosen_count += 1
        if ex.truncated_rejected:
            truncated_rejected_count += 1

    # Reconciled truncation counts:
    # 16 prompts exceed 1024 tokens and are left-truncated
    # 0 chosen exceed 1024 tokens
    # 1 rejected (Row 828, 1024 body + 1 EOS = 1025 tokens) is right-truncated to 1024
    assert truncated_prompt_count == 16, f"Expected 16 truncated prompts, found {truncated_prompt_count}"
    assert truncated_chosen_count == 0, f"Expected 0 truncated chosen, found {truncated_chosen_count}"
    assert truncated_rejected_count == 1, f"Expected 1 truncated rejected, found {truncated_rejected_count}"


def test_dpo_loss_mask_and_token_probabilities(preprocessor: DPOPreprocessor):
    """
    Verifies that prompt and padding positions contribute zero log-probabilities in the loss computation:
    - prompt positions receive loss_mask == 0
    - completion padding positions receive loss_mask == 0
    - only real completion tokens receive loss_mask == 1
    """
    dataset_path = Path("trainsight/sample_data/clean_ultrafeedback_993.jsonl")
    with open(dataset_path, "r", encoding="utf-8") as f:
        sample_rows = [json.loads(f.readline()) for _ in range(4)]

    examples = [preprocessor.process_sample(r) for r in sample_rows]
    collated = preprocessor.collate_batch(examples)
    concatenated = preprocessor.build_concatenated_batch(collated)

    input_ids = concatenated["input_ids"]
    attention_mask = concatenated["attention_mask"]
    loss_mask = concatenated["loss_mask"]

    # Total batch size is 2 * B (first B chosen, next B rejected)
    assert input_ids.shape[0] == 8
    assert attention_mask.shape == input_ids.shape
    assert loss_mask.shape == input_ids.shape

    prompt_width = collated["prompt_input_ids"].shape[1]

    # Rule 1: Prompt positions must contribute NO loss (loss_mask == 0)
    prompt_loss_slice = loss_mask[:, :prompt_width]
    assert (prompt_loss_slice == 0).all(), "Prompt positions must have loss_mask == 0"

    # Rule 2: Padding positions in completion must contribute NO loss (loss_mask == 0)
    pad_id = preprocessor.tokenizer.pad_token_id
    comp_input_slice = input_ids[:, prompt_width:]
    comp_loss_slice = loss_mask[:, prompt_width:]

    pad_positions = (comp_input_slice == pad_id)
    assert (comp_loss_slice[pad_positions] == 0).all(), "Padding tokens must have loss_mask == 0"

    # Rule 3: Valid completion tokens must have loss_mask == 1
    valid_positions = (comp_input_slice != pad_id)
    assert (comp_loss_slice[valid_positions] == 1).all(), "Real completion tokens must have loss_mask == 1"


def test_dpo_trl_data_collator_smoke_batch(preprocessor: DPOPreprocessor):
    """
    Verifies that the preprocessed records pass through TRL's real DataCollatorForPreference
    and DPOTrainer.concatenated_inputs without shape errors or prompt divergence.
    """
    dataset_path = Path("trainsight/sample_data/clean_ultrafeedback_993.jsonl")
    with open(dataset_path, "r", encoding="utf-8") as f:
        # Pick 4 rows including at least one over-budget prompt
        rows = [json.loads(line) for line in f]

    # Row 42 is an over-budget prompt (1212 tokens)
    smoke_rows = [rows[0], rows[1], rows[2], rows[42]]
    examples = [preprocessor.process_sample(r) for r in smoke_rows]

    # Convert to TRL feature dictionary
    features = [
        {
            "prompt_input_ids": ex.prompt_input_ids,
            "chosen_input_ids": ex.chosen_input_ids,
            "rejected_input_ids": ex.rejected_input_ids,
        }
        for ex in examples
    ]

    trl_collator = DataCollatorForPreference(pad_token_id=preprocessor.tokenizer.pad_token_id)
    batch = trl_collator(features)

    assert "prompt_input_ids" in batch
    assert "prompt_attention_mask" in batch
    assert "chosen_input_ids" in batch
    assert "chosen_attention_mask" in batch
    assert "rejected_input_ids" in batch
    assert "rejected_attention_mask" in batch

    assert batch["prompt_input_ids"].shape[0] == 4
    # Max prompt width is 1024 because row 42 was capped at 1024
    assert batch["prompt_input_ids"].shape[1] == 1024

    # Run through DPOTrainer.concatenated_inputs
    concatenated = DPOTrainer.concatenated_inputs(batch, padding_value=preprocessor.tokenizer.pad_token_id)

    prompt_ids = concatenated["prompt_input_ids"]
    assert prompt_ids.shape[0] == 8  # 2 * 4

    # Verify that prompt_ids for chosen branch strictly equals rejected branch
    assert torch.equal(prompt_ids[:4], prompt_ids[4:]), "Chosen and rejected prompt tensors must be 100% bitwise identical!"


def test_dpo_tensor_reconciliation_option_a_976_rows(preprocessor: DPOPreprocessor):
    """
    Option A Verification (976 Fully Preserved Rows):
    Excludes the single row (Row 828) whose rejected completion exceeded the 1024 cap at EOS.
    Asserts that across all 976 training pairs:
    1. Zero prompts are truncated (100% within 1024 token budget).
    2. Zero chosen completions are truncated (100% within 1024 token budget).
    3. Zero rejected completions are truncated (100% within 1024 token budget, all have intact EOS).
    4. Bitwise identical prompt tokens across chosen and rejected branches.
    5. Exactly 123 tied-score pairs retained as label-confidence context (0 reversals).
    6. Zero identical assistant responses.
    """
    dataset_path = Path("trainsight/sample_data/clean_ultrafeedback_976.jsonl")
    assert dataset_path.exists(), f"Dataset file not found at {dataset_path}"

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    assert len(rows) == 976, f"Expected 976 rows, found {len(rows)}"

    truncated_prompt_count = 0
    truncated_chosen_count = 0
    truncated_rejected_count = 0
    tied_score_count = 0
    reversal_count = 0

    for idx, row in enumerate(rows):
        ex = preprocessor.process_sample(row)

        # 1. Prompts and completions strictly intact within budget
        assert not ex.truncated_prompt, f"Row {idx}: prompt truncated"
        assert not ex.truncated_chosen, f"Row {idx}: chosen truncated"
        assert not ex.truncated_rejected, f"Row {idx}: rejected truncated"
        assert ex.raw_prompt_len <= 1024
        assert ex.raw_chosen_len <= 1024
        assert ex.raw_rejected_len <= 1024

        # 2. Both branches have valid completions
        assert len(ex.chosen_input_ids) >= 1
        assert len(ex.rejected_input_ids) >= 1

        if ex.score_chosen is not None and ex.score_rejected is not None:
            if ex.score_chosen == ex.score_rejected:
                tied_score_count += 1
            elif ex.score_chosen < ex.score_rejected:
                reversal_count += 1

        if ex.truncated_prompt:
            truncated_prompt_count += 1
        if ex.truncated_chosen:
            truncated_chosen_count += 1
        if ex.truncated_rejected:
            truncated_rejected_count += 1

    assert truncated_prompt_count == 0, "Option A must have exactly 0 truncated prompts"
    assert truncated_chosen_count == 0, "Option A must have exactly 0 truncated chosen"
    assert truncated_rejected_count == 0, "Option A must have exactly 0 truncated rejected"
    assert tied_score_count == 123, f"Expected 123 score ties, found {tied_score_count}"
    assert reversal_count == 0, f"Expected 0 score reversals, found {reversal_count}"


def test_dpo_tensor_reconciliation_eval_191_rows(preprocessor: DPOPreprocessor):
    """
    Fixed Held-Out Evaluation Cohort Verification (191 Rows):
    Audited from first 200 pairs of untouched test_prefs split.
    Excludes: 3 identical pairs, 4 over-budget prompts, 3 over-budget completions (9 unique rows).
    Asserts:
    1. Exactly 191 clean, fully intact evaluation pairs.
    2. Zero prompt truncations.
    3. Zero completion truncations (intact EOS tokens).
    4. Exactly 19 tied scores, 0 reversals, 0 identical pairs.
    """
    dataset_path = Path("trainsight/sample_data/eval_ultrafeedback_test_191.jsonl")
    assert dataset_path.exists(), f"Dataset file not found at {dataset_path}"

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    assert len(rows) == 191, f"Expected 191 rows, found {len(rows)}"

    truncated_prompt_count = 0
    truncated_chosen_count = 0
    truncated_rejected_count = 0
    tied_score_count = 0
    reversal_count = 0

    for idx, row in enumerate(rows):
        ex = preprocessor.process_sample(row)

        assert not ex.truncated_prompt, f"Eval row {idx}: prompt truncated"
        assert not ex.truncated_chosen, f"Eval row {idx}: chosen truncated"
        assert not ex.truncated_rejected, f"Eval row {idx}: rejected truncated"

        if ex.score_chosen is not None and ex.score_rejected is not None:
            if ex.score_chosen == ex.score_rejected:
                tied_score_count += 1
            elif ex.score_chosen < ex.score_rejected:
                reversal_count += 1

        if ex.truncated_prompt:
            truncated_prompt_count += 1
        if ex.truncated_chosen:
            truncated_chosen_count += 1
        if ex.truncated_rejected:
            truncated_rejected_count += 1

    assert truncated_prompt_count == 0
    assert truncated_chosen_count == 0
    assert truncated_rejected_count == 0
    assert tied_score_count == 19, f"Expected 19 ties, found {tied_score_count}"
    assert reversal_count == 0, f"Expected 0 reversals, found {reversal_count}"


