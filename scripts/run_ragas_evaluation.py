from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from takasecure_rag.config import Settings
from takasecure_rag.graph import AdaptiveRAG
from takasecure_rag.schemas import ChatRequest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "data/rag/v4/TakaSecure_RAG_Golden_50_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation/ragas"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def generate(golden_rows: list[dict[str, Any]], output_path: Path) -> list[dict]:
    settings = Settings()
    settings = settings.model_copy(
        update={
            "upstash_redis_rest_url": None,
            "upstash_redis_rest_token": None,
        }
    )
    rag = AdaptiveRAG(settings)
    existing = {
        row["golden_id"]: row
        for row in read_jsonl(output_path)
    } if output_path.exists() else {}

    for position, golden in enumerate(golden_rows, start=1):
        if golden["golden_id"] in existing and not existing[golden["golden_id"]]["error"]:
            print(f"[{position}/{len(golden_rows)}] cached {golden['golden_id']}")
            continue
        started = time.perf_counter()
        try:
            state = rag.graph.invoke(
                {
                    "request": ChatRequest(
                        question=golden["question"],
                        user_role=golden["user_role"],
                        department=golden["department"],
                    )
                }
            )
            response = state["response"]
            contexts = [document.page_content for document in state.get("documents", [])]
            record = {
                **golden,
                "response": response["answer"],
                "retrieved_contexts": contexts,
                "citations": response["citations"],
                "grounded": response["grounded"],
                "access_denied": response.get("access_denied", False),
                "escalation_required": response["escalation_required"],
                "verification_passed": response["verification"]["passed"],
                "retrieval_strategy": response["retrieval_strategy"],
                "correction_attempts": response["correction_attempts"],
                "tool_name": response.get("tool_name"),
                "latency_seconds": round(time.perf_counter() - started, 3),
                "error": None,
            }
        except Exception as error:
            message = " ".join(str(error).split())
            if len(message) > 300:
                message = message[:297] + "..."
            record = {
                **golden,
                "response": "",
                "retrieved_contexts": [],
                "citations": [],
                "grounded": False,
                "access_denied": False,
                "escalation_required": True,
                "verification_passed": False,
                "retrieval_strategy": "error",
                "correction_attempts": 0,
                "tool_name": None,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(error).__name__}: {message}",
            }
        existing[golden["golden_id"]] = record
        ordered = [existing[row["golden_id"]] for row in golden_rows if row["golden_id"] in existing]
        write_jsonl(output_path, ordered)
        print(
            f"[{position}/{len(golden_rows)}] {golden['golden_id']} "
            f"{record['latency_seconds']:.1f}s citations={len(record['citations'])} "
            f"error={bool(record['error'])}"
        )
    return [existing[row["golden_id"]] for row in golden_rows]


def deterministic_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row["error"]]
    citation_recall = []
    citation_precision = []
    legacy_pass = []
    tool_pass = []
    for row in successful:
        expected = set(row["expected_policy_ids"])
        found = set(row["citations"])
        citation_recall.append(expected.issubset(found))
        citation_precision.append(found.issubset(expected) and bool(found))
        forbidden = set(row["forbidden_policy_ids"])
        legacy_pass.append(not bool(found & forbidden))
        if row["expected_tool"]:
            tool_pass.append(row["tool_name"] == row["expected_tool"])
    latencies = [row["latency_seconds"] for row in successful]
    return {
        "total": len(rows),
        "successful": len(successful),
        "error_rate": 1 - (len(successful) / len(rows)),
        "grounded_rate": mean_bool(row["grounded"] for row in successful),
        "verification_pass_rate": mean_bool(
            row["verification_passed"] for row in successful
        ),
        "citation_recall_pass_rate": mean_bool(citation_recall),
        "citation_precision_pass_rate": mean_bool(citation_precision),
        "legacy_exclusion_pass_rate": mean_bool(legacy_pass),
        "tool_name_accuracy": mean_bool(tool_pass),
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
        },
        "task_distribution": dict(Counter(row["task"] for row in rows)),
    }


def mean_bool(values) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def install_ragas_vertexai_compatibility_shim() -> None:
    """Work around RAGAS 0.4.3 importing a removed optional LangChain module."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        module = types.ModuleType(module_name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = module


def ragas_scores(
    rows: list[dict[str, Any]],
    output_dir: Path,
    settings: Settings,
) -> dict[str, Any]:
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    install_ragas_vertexai_compatibility_shim()
    import ragas
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig

    eligible = [
        row
        for row in rows
        if not row["error"] and row["response"] and row["retrieved_contexts"]
    ]
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["response"],
                "retrieved_contexts": row["retrieved_contexts"],
                "reference": row["reference"],
            }
            for row in eligible
        ]
    )
    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.ragas_evaluator_model,
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            temperature=0,
            max_tokens=512,
        )
    )
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=judge),
            LLMContextRecall(llm=judge),
            LLMContextPrecisionWithReference(llm=judge),
        ],
        run_config=RunConfig(
            timeout=240,
            max_retries=2,
            max_wait=30,
            max_workers=1,
            seed=20260717,
        ),
        raise_exceptions=False,
        show_progress=True,
        batch_size=1,
    )
    frame = result.to_pandas()
    frame.insert(0, "golden_id", [row["golden_id"] for row in eligible])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "ragas_per_example.csv", index=False)
    metric_columns = [
        "faithfulness",
        "context_recall",
        "llm_context_precision_with_reference",
    ]
    aggregate = {}
    valid_counts = {}
    for column in metric_columns:
        if column not in frame:
            continue
        values = frame[column].dropna()
        aggregate[column] = float(values.mean()) if len(values) else None
        valid_counts[column] = len(values)
    return {
        "ragas_version": ragas.__version__,
        "judge_model": settings.ragas_evaluator_model,
        "judge_endpoint": "same vLLM deployment as generator",
        "independent_judge": False,
        "eligible_examples": len(eligible),
        "metrics": aggregate,
        "valid_score_counts": valid_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    golden_rows = read_jsonl(args.golden)
    if args.limit:
        golden_rows = golden_rows[: args.limit]
    generation_path = args.output_dir / "golden_50_generations.jsonl"
    if args.skip_generation:
        generated = read_jsonl(generation_path)
        requested_ids = {row["golden_id"] for row in golden_rows}
        generated = [row for row in generated if row["golden_id"] in requested_ids]
    else:
        generated = generate(golden_rows, generation_path)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden_dataset": str(args.golden.relative_to(ROOT)),
        "deterministic": deterministic_scores(generated),
    }
    if not args.skip_ragas:
        summary["ragas"] = ragas_scores(generated, args.output_dir, Settings())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
