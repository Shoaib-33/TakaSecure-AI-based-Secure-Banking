import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


POLICY_ID_PATTERN = re.compile(r"TSB-[A-Z-]+-\d{2}-\d(?:-LEGACY)?")


def canonical_policy_id(policy_id: str) -> str:
    return policy_id.removesuffix("-LEGACY")


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


class PolicyCatalog:
    """Deterministic policy authorization and approved-tool metadata."""

    def __init__(self, path: Path):
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.policies = {row["policy_id"]: row for row in rows}
        self.known_roles = {
            role for row in rows for role in row.get("allowed_roles", [])
        }

    @staticmethod
    def policy_ids(text: str) -> list[str]:
        return list(dict.fromkeys(POLICY_ID_PATTERN.findall(text)))

    def _policy(self, policy_id: str) -> dict | None:
        return self.policies.get(canonical_policy_id(policy_id))

    def authorize_request(self, role: str, department: str | None) -> AccessDecision:
        if role not in self.known_roles:
            return AccessDecision(
                allowed=False,
                reason=f"Role '{role}' is not authorized for the internal policy corpus.",
            )
        accessible = [
            row
            for row in self.policies.values()
            if role in row.get("allowed_roles", [])
            and (not department or row.get("department") == department)
        ]
        if not accessible:
            scope = f" department '{department}'" if department else " requested scope"
            return AccessDecision(
                allowed=False,
                reason=f"Role '{role}' has no policy access for{scope}.",
            )
        return AccessDecision(allowed=True, reason="Role access validated against policy metadata.")

    def filter_documents(
        self,
        documents: Iterable[Document],
        role: str,
        department: str | None,
    ) -> list[Document]:
        authorized: list[Document] = []
        for document in documents:
            ids = self.policy_ids(document.page_content)
            policies = [self._policy(policy_id) for policy_id in ids]
            policies = [policy for policy in policies if policy]
            if not policies:
                continue
            if not all(role in policy.get("allowed_roles", []) for policy in policies):
                continue
            if department and not all(
                policy.get("department") == department for policy in policies
            ):
                continue
            document.metadata["policy_ids"] = ids
            document.metadata["authorized_role"] = role
            authorized.append(document)
        return authorized

    def approved_tools(self, documents: Iterable[Document]) -> dict[str, str]:
        tools: dict[str, str] = {}
        for document in documents:
            for policy_id in self.policy_ids(document.page_content):
                policy = self._policy(policy_id)
                if policy and policy.get("calculation_tool"):
                    tools[policy_id] = policy["calculation_tool"]
        return tools
