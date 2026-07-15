# TakaSecure synthetic banking SFT dataset

This package generates exactly 7,000 English-language chat examples for a
fictional bank-policy assistant. Every policy, case, institution, identifier,
and operational rule is synthetic. Nothing in this dataset should be treated
as Bangladesh Bank guidance, law, or production banking policy.

## Purpose

The dataset teaches model behavior for a private, RAG-grounded banking
assistant:

- evidence-grounded single-policy answers;
- multi-policy synthesis;
- abstention when context is insufficient;
- role-based access denial without leaking restricted content;
- resistance to instructions embedded in retrieved documents;
- current-versus-superseded policy resolution;
- deterministic calculation-tool routing.

It is designed for supervised fine-tuning of response behavior. Confidential
facts and real policies should remain in the RAG system, not model weights.

## Generate

```bash
python generate_banking_sft.py
```

Outputs are written to `generated/`:

- `banking_sft_train.jsonl`: 5,600 rows
- `banking_sft_validation.jsonl`: 700 rows
- `banking_sft_test.jsonl`: 700 rows
- `banking_sft_all.jsonl`: all 7,000 rows
- `policy_catalog.json`: the fictional policy catalog
- `quality_report.json`: automated validation results and distributions

## Dataset composition

| Task | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Single-policy grounding | 1,300 | 1,040 | 130 | 130 |
| Multi-policy synthesis | 1,300 | 1,040 | 130 | 130 |
| Insufficient-context abstention | 900 | 720 | 90 | 90 |
| Access-control refusal | 900 | 720 | 90 | 90 |
| Prompt-injection resistance | 900 | 720 | 90 | 90 |
| Policy-version conflict | 700 | 560 | 70 | 70 |
| Deterministic tool routing | 1,000 | 800 | 100 | 100 |
| **Total** | **7,000** | **5,600** | **700** | **700** |

Narrative targets use multiple deterministic response templates to reduce
surface-form repetition. All 7,000 examples remain synthetic. A separate
expert-authored evaluation set is still required for final claims.
## Split strategy

Policy families are isolated by split. A policy family used in training never
appears in validation or test, including its scope variants. This is stricter
than randomly splitting paraphrases of the same clause.

## Important limitations

- The content has not been reviewed by banking, legal, compliance, fraud, or
  information-security professionals.
- Automated checks validate structure and provenance consistency, not whether a
  policy would be appropriate for a real institution.
- Template-generated diversity is not a substitute for expert-authored cases.
- Keep a separate human-written evaluation set for final model claims.
- Never mix real customer information into this public synthetic package.

Before production use, replace the fictional catalog with versioned,
access-controlled, institution-approved documents in RAG and have domain
experts review training behavior and evaluation scenarios.
