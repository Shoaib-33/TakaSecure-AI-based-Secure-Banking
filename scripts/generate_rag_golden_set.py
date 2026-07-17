from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = (
    ROOT / "data/rag/v4/TakaSecure_RAG_Evaluation_Benchmark_v4.jsonl"
)
CATALOG_PATH = ROOT / "data/sft/policy_catalog.json"
OUTPUT_PATH = ROOT / "data/rag/v4/TakaSecure_RAG_Golden_50_v1.jsonl"
REPORT_PATH = ROOT / "data/rag/v4/TakaSecure_RAG_Golden_50_v1.report.json"
SEED = 20260717
TASK_TARGETS = {
    "single_hop": 20,
    "evidence": 20,
    "temporal_conflict": 5,
    "tool_routing": 5,
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_rank(row: dict) -> str:
    identity = f"{SEED}:{row['id']}".encode()
    return hashlib.sha256(identity).hexdigest()


def balanced_selection(rows: list[dict], count: int) -> list[dict]:
    by_department: dict[str, deque[dict]] = defaultdict(deque)
    for row in sorted(rows, key=stable_rank):
        by_department[row["department"]].append(row)
    departments = sorted(
        by_department,
        key=lambda name: hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest(),
    )
    selected: list[dict] = []
    while len(selected) < count:
        progressed = False
        for department in departments:
            if by_department[department] and len(selected) < count:
                selected.append(by_department[department].popleft())
                progressed = True
        if not progressed:
            raise RuntimeError(f"Only {len(selected)} rows available for target {count}")
    return selected


def reference_for(row: dict, catalog: dict[str, dict]) -> str:
    statements = []
    for policy_id in row["expected_policy_ids"]:
        policy = catalog[policy_id]
        statements.append(f"{policy_id}: {policy['clause']}")
    if row["task"] == "tool_routing":
        statements.append(f"Approved tool: {row['expected_tool']}.")
    if row["task"] == "temporal_conflict":
        excluded = ", ".join(row.get("must_exclude", []))
        statements.append(
            "The current policy version governs on the stated date; "
            f"superseded policy {excluded} must not govern."
        )
    return " ".join(statements)


def main() -> None:
    benchmark = read_jsonl(BENCHMARK_PATH)
    catalog_rows = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog = {row["policy_id"]: row for row in catalog_rows}
    selected = []
    for task, target in TASK_TARGETS.items():
        candidates = [row for row in benchmark if row["task"] == task]
        selected.extend(balanced_selection(candidates, target))
    selected.sort(key=lambda row: row["id"])

    golden_rows = []
    for position, row in enumerate(selected, start=1):
        golden_rows.append(
            {
                "golden_id": f"golden-{position:03d}",
                "benchmark_id": row["id"],
                "task": row["task"],
                "department": row["department"],
                "scope": row["scope"],
                "user_role": row["allowed_roles"][0],
                "question": row["question"],
                "reference": reference_for(row, catalog),
                "expected_policy_ids": row["expected_policy_ids"],
                "forbidden_policy_ids": row.get("must_exclude", []),
                "expected_tool": row.get("expected_tool"),
            }
        )

    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in golden_rows
    )
    OUTPUT_PATH.write_text(serialized, encoding="utf-8")
    report = {
        "valid": len(golden_rows) == 50,
        "version": "1.0",
        "seed": SEED,
        "source_benchmark": str(BENCHMARK_PATH.relative_to(ROOT)),
        "source_benchmark_rows": len(benchmark),
        "golden_rows": len(golden_rows),
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "task_distribution": dict(Counter(row["task"] for row in golden_rows)),
        "department_distribution": dict(
            Counter(row["department"] for row in golden_rows)
        ),
        "unique_questions": len({row["question"] for row in golden_rows}),
        "unique_benchmark_ids": len({row["benchmark_id"] for row in golden_rows}),
    }
    if not report["valid"] or report["unique_benchmark_ids"] != 50:
        raise RuntimeError(f"Golden-set validation failed: {report}")
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
