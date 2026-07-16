# Data layout

All data in this repository is synthetic.

- `sft/` contains the train, validation, and test splits, policy catalog, and generator quality report.
- `rag/v4/` contains the current enterprise policy corpus, manifest, and 166-question benchmark.

Regenerate the SFT data with `python scripts/generate_banking_sft.py`.
Regenerate the current RAG corpus with `python scripts/generate_enterprise_rag_corpus.py`.
