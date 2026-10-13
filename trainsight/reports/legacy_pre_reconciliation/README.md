# Legacy Pre-Reconciliation Reports

This directory preserves diagnostic records generated prior to the TrainSight DPO tokenizer contract and tensor reconciliation milestone.

## Superseded Assets

### `dpo_exact_per_pair_val_reconciliation.json`
- **Date**: October 13, 2026 (pre-alignment)
- **Deficiencies Resolved in Corrected Milestone**:
  1. **Tokenizer Formatting Gap**: Used raw string concatenation (`prompt + chosen`) rather than Qwen ChatML conversational templating (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`).
  2. **Metric Conflation**: Compared base model raw conditional log-likelihood ranking (105 wins / 191) against trained DPO policy reward margin ranking (112 wins / 191), claiming an ungrounded "+15.7 percentage point preference shift".
  3. **Batch Aggregation Artifact**: Inferred 112 GPU wins from TRL's 58.85% batch accuracy without accounting for terminal singleton weighting (95 pairs of size 2, 1 pair of size 1).
- **Current Canonical Replacement**: `trainsight/reports/corrected_milestone/dpo_run_clean_ultrafeedback_976.json` and `trainsight/reports/corrected_milestone/evaluator_reconciliation_diagnostic.json`.
