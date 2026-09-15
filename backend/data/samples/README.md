# Sample corpus

Committed sample documents used for Phase 3+ demos, manual verification, and as
fixtures for evaluation (Phase 6). Unlike `data/raw/` (gitignored, for real
private company docs), this folder is version-controlled so the demo corpus is
reproducible.

To ingest these, copy them into `data/raw/` and run the ingestion CLI on the GPU
VM, or point `INGESTION_DATA_DIR` at this folder:

```bash
INGESTION_DATA_DIR=data/samples uv run python scripts/run_ingestion.py --doc-type SOP --sop-version v3.0
```

## Contents

| File | Type | Purpose |
|---|---|---|
| `expense_approval_sop_v2.md` | SOP v2.1 | Superseded version (for the version-warning case) |
| `expense_approval_sop_v3.md` | SOP v3.0 | Current version; supersedes v2.1 |
| `onboarding_sop_v1.md` | SOP v1.4 | Graph traversal: process → step → role/system, compliance |
| `product_atlas_router.md` | Product info | Product → feature/spec, compatibility edges |

## Graph-traversal questions these support

1. Which role owns the Finance Approval step of Expense Approval? (Step → OWNED_BY → Role)
2. Which system does the Finance Approval step require? (Step → REQUIRES → System)
3. What compliance requirement must the Employee Onboarding process satisfy? (Process → MUST_SATISFY → ComplianceRequirement)
4. Which steps require the Concur system? (Step → REQUIRES → System)
5. What features does the Atlas 5000 Router have? (Product → HAS_FEATURE → Feature)
6. What is the Atlas 5000 Router compatible with? (Product → COMPATIBLE_WITH → Product)

## Superseded-SOP case

Ingesting `expense_approval_sop_v3.md` after `expense_approval_sop_v2.md` creates
a `(v3.0)-[:SUPERSEDES]->(v2.1)` edge and flips the v2.1 chunks to
`is_current_version = false`, so any answer citing the v2.1 SOP surfaces a
version warning.
