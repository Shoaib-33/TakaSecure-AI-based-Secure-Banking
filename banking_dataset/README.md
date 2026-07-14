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
python banking_dataset/generate_banking_sft.py
```

Outputs are written to `banking_dataset/generated/`:

- `banking_sft_train.jsonl`: 5,600 rows
- `banking_sft_validation.jsonl`: 700 rows
- `banking_sft_test.jsonl`: 700 rows
- `banking_sft_all.jsonl`: all 7,000 rows
- `policy_catalog.json`: the fictional policy catalog
- `quality_report.json`: automated validation results and distributions

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
