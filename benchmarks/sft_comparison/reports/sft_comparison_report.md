# Local SFT Comparison Report: Native TRL vs Custom Assertions vs TrainSight

**Objective**: Verify that only assistant-answer tokens contribute to loss, prompt and padding tokens are ignored, and every accepted example retains at least one answer token contributing to shifted causal loss.

### Frozen Environment & Contract

- **TRL Version**: `0.24.0`
- **Transformers Version**: `5.5.0`
- **PyTorch Version**: `2.10.0`
- **Model Tokenizer**: `Qwen/Qwen2.5-0.5B-Instruct` (Vocab Size: 151,643)
- **Chat Template**: `ChatML (<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n)`
- **Max Sequence Length**: `256` tokens
- **Collation & Padding**: Right padding with `pad_token_id=151643`, `attention_mask=0` on padding, `labels=-100` on padding and prompt.

---

## 1. Outcome Summary Across Six Cases

Outcome categories:

- **Blocked**: Execution is automatically halted with a diagnosed failure.

- **Accepted**: Pipeline continues execution without flagging.

- **Warned**: Problem is reported in logs but execution continues.

- **Unsupported**: The check cannot run for that configuration or execution boundary.


| Case | Native TRL | Custom Assertions | TrainSight | Correct Diagnosis? |
|---|---|---|---|---|
| **Clean Input** | Accepted | Accepted | Accepted | TRL, Assertions, TrainSight |
| **Empty Answer** | Accepted | Blocked | Blocked | Assertions, TrainSight |
| **Answer Removed by Truncation** | Accepted | Blocked | Blocked | Assertions, TrainSight |
| **Prompt Included in Loss** | Accepted | Blocked | Blocked | Assertions, TrainSight |
| **Template Mismatch** | Accepted | Blocked | Blocked | Assertions, TrainSight |
| **Padding Included in Loss** | Accepted | Blocked | Blocked | Assertions, TrainSight |

---

## 2. Detailed Case Diagnosis & Tensor Inspection

### Case: Clean Input
- **Fault Type**: `none` (Fixture: `clean_fixture.jsonl`)
- **Native TRL Diagnostic**: Batch collated successfully without exception.
- **Custom Assertions Diagnostic**: All custom assertions passed.
- **TrainSight Diagnostic**: TrainSight pre-flight quality gate and dataloader reconciliation passed.

**Inspected Tensor Evidence**:
- Row ID: `row_001` | Sequence Length: 50 | Prompt Length: 36
- Active Supervised Labels (`!= -100`): **8**
- Causal Loss Contributing Tokens (index $\ge 1$): **8**
- Unmasked Prompt Tokens: **0**
- Unmasked Padding Tokens: **0**
- Decoded Prompt Prefix: `<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful ass`
- Decoded Answer Preview: `The capital of France is Paris.<|im_end|><|endoftext|><|endoftext|><|endoftext|>`
- Labels Head (First 20 tokens): `[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]`

---

### Case: Empty Answer
- **Fault Type**: `dataset_empty_answer` (Fixture: `fault_empty_answer.jsonl`)
- **Native TRL Diagnostic**: Batch collated successfully without exception.
- **Custom Assertions Diagnostic**: [EMPTY_ANSWER_CONTENT] (Row: row_001): Supervised tokens consist entirely of EOS or template closing delimiters without answer content.
- **TrainSight Diagnostic**: TrainSight pre-flight quality gate failed: 1 empty completion(s).

**Inspected Tensor Evidence**:
- Row ID: `row_001` | Sequence Length: 50 | Prompt Length: 36
- Active Supervised Labels (`!= -100`): **1**
- Causal Loss Contributing Tokens (index $\ge 1$): **1**
- Unmasked Prompt Tokens: **0**
- Unmasked Padding Tokens: **0**
- Decoded Prompt Prefix: `<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful ass`
- Decoded Answer Preview: `<|im_end|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|end`
- Labels Head (First 20 tokens): `[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]`

---

### Case: Answer Removed by Truncation
- **Fault Type**: `dataset_truncated_answer` (Fixture: `fault_truncated_answer.jsonl`)
- **Native TRL Diagnostic**: Batch collated successfully without exception.
- **Custom Assertions Diagnostic**: [NO_SUPERVISED_TOKENS] (Row: row_001): Example has zero supervised tokens (all labels are -100).
- **TrainSight Diagnostic**: TrainSight pre-flight quality gate failed: 1 answer(s) eliminated by truncation.

**Inspected Tensor Evidence**:
- Row ID: `row_001` | Sequence Length: 256 | Prompt Length: 315
- Active Supervised Labels (`!= -100`): **0**
- Causal Loss Contributing Tokens (index $\ge 1$): **0**
- Unmasked Prompt Tokens: **0**
- Unmasked Padding Tokens: **0**
- Decoded Prompt Prefix: `<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful ass`
- Decoded Answer Preview: ``
- Labels Head (First 20 tokens): `[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]`

---

### Case: Prompt Included in Loss
- **Fault Type**: `batch_unmasked_prompt` (Fixture: `clean_fixture.jsonl`)
- **Native TRL Diagnostic**: Batch collated successfully without exception.
- **Custom Assertions Diagnostic**: [PROMPT_LABELS_UNMASKED] (Row: row_001): Prompt token at position 0 (151644: '<|im_start|>') has label != -100. Prompt tokens must not contribute to loss.
- **TrainSight Diagnostic**: TrainSight dataloader reconciliation failed: Row row_001: Prompt tokens unmasked in dataloader labels (labels[:36] != -100).

**Inspected Tensor Evidence**:
- Row ID: `row_001` | Sequence Length: 50 | Prompt Length: 36
- Active Supervised Labels (`!= -100`): **44**
- Causal Loss Contributing Tokens (index $\ge 1$): **43**
- Unmasked Prompt Tokens: **36**
- Unmasked Padding Tokens: **0**
- Decoded Prompt Prefix: `<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful ass`
- Decoded Answer Preview: `The capital of France is Paris.<|im_end|><|endoftext|><|endoftext|><|endoftext|>`
- Labels Head (First 20 tokens): `[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645]`

---

### Case: Template Mismatch
- **Fault Type**: `dataset_template_mismatch` (Fixture: `fault_template_mismatch.jsonl`)
- **Native TRL Diagnostic**: Batch collated successfully without exception.
- **Custom Assertions Diagnostic**: [TEMPLATE_MISMATCH] (Row: row_001): Prompt does not contain required ChatML delimiters (<|im_start|>user ... <|im_start|>assistant
). Decoded prompt prefix: '### Instruction:\nWhat is the capital of France?\n\n### Respons'
- **TrainSight Diagnostic**: TrainSight dataloader reconciliation failed: Row row_001: Dataloader tokens diverge from approved ChatML contract. Expected 51 tokens, got 21 tokens.

**Inspected Tensor Evidence**:
- Row ID: `row_001` | Sequence Length: 28 | Prompt Length: 13
- Active Supervised Labels (`!= -100`): **8**
- Causal Loss Contributing Tokens (index $\ge 1$): **8**
- Unmasked Prompt Tokens: **0**
- Unmasked Padding Tokens: **0**
- Decoded Prompt Prefix: `### Instruction:
What is the capital of France?

### Response:
`
- Decoded Answer Preview: `The capital of France is Paris.<|im_end|><|endoftext|><|endoftext|><|endoftext|>`
- Labels Head (First 20 tokens): `[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 785, 6722, 315, 9625, 374, 12095, 13]`

---

### Case: Padding Included in Loss
- **Fault Type**: `batch_unmasked_padding` (Fixture: `clean_fixture.jsonl`)
- **Native TRL Diagnostic**: Batch collated successfully without exception.
- **Custom Assertions Diagnostic**: [PADDING_LABELS_UNMASKED] (Row: row_001): Padded position 44 (attention_mask == 0) has label 42 != -100. Padding tokens must never contribute to loss.
- **TrainSight Diagnostic**: TrainSight dataloader reconciliation failed: Row row_001: Padded positions supervised in dataloader labels (attention_mask == 0 but label != -100).

**Inspected Tensor Evidence**:
- Row ID: `row_001` | Sequence Length: 50 | Prompt Length: 36
- Active Supervised Labels (`!= -100`): **9**
- Causal Loss Contributing Tokens (index $\ge 1$): **9**
- Unmasked Prompt Tokens: **0**
- Unmasked Padding Tokens: **1**
- Decoded Prompt Prefix: `<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful ass`
- Decoded Answer Preview: `The capital of France is Paris.<|im_end|><|endoftext|><|endoftext|><|endoftext|>`
- Labels Head (First 20 tokens): `[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]`

---

## 3. Engineering Analysis: What TrainSight Catches & Maintenance Comparison

### 1. What Native TRL Does Not Catch
Native TRL focuses on syntactic collation without semantic contract guarantees:
- **Empty Answer**: TRL silently trains on the trailing EOS token (`<|im_end|>`), teaching the model to terminate immediately.
- **Truncated Answer**: When a prompt exceeds `max_seq_len`, TRL truncates the sequence. If all answer tokens are eliminated, all labels become `-100`, producing zero loss updates or NaN gradients without raising an error.
- **Prompt & Padding Leakage**: TRL's collator does not self-verify its label output; if prompt or padded tokens have unmasked labels, cross-entropy is calculated over them without detection.
- **Template Mismatch**: TRL tokenizes whatever text is passed, lacking an approved contract reference to detect that a dataset uses an unintended formatting convention.

### 2. Custom Assertions vs. TrainSight
- **Custom Assertions** successfully match TrainSight's coverage by asserting invariants on the post-collation batch.
- **Maintenance Trade-off**: Writing custom assertions requires maintaining 5-7 separate tensor checks across every new training script, managing prompt boundary offsets, and handling causal shifts manually.
- **TrainSight Value**: TrainSight bundles pre-flight file inspection (catching corruptions before spinning up GPU infrastructure) with trainer dataloader reconciliation into an automated, fail-closed CLI gate.

