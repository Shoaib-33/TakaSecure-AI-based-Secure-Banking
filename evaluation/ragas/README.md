# RAGAS evaluation artifacts

The `smoke/` directory contains the latest valid end-to-end evaluation. It is
intentionally labelled as a smoke run because it contains one golden example,
not the complete 50-case benchmark.

## Latest valid smoke result

| Metric | Score |
|---|---:|
| Successful requests | 1 / 1 |
| Grounded response rate | 1.000 |
| Verification pass rate | 1.000 |
| Citation recall pass rate | 1.000 |
| Citation precision pass rate | 1.000 |
| Legacy exclusion pass rate | 1.000 |
| RAGAS faithfulness | 1.000 |
| RAGAS context recall | 1.000 |
| RAGAS context precision | 0.909 |
| End-to-end latency | 19.679 seconds |

The evaluator was RAGAS 0.4.3 using `base-llama` on the same vLLM deployment
as the `takasecure` generator. This is not an independent judge, and a
single-example smoke score must not be represented as the full benchmark.

## Full-run outputs

When the live vLLM service is available, run:

```powershell
.\bank_venv\Scripts\python.exe -u scripts\run_ragas_evaluation.py
```

A successful 50-case run writes these files alongside this README:

- `golden_50_generations.jsonl`
- `ragas_per_example.csv`
- `summary.json`

Failed infrastructure-only runs are not retained as model evaluation results.
