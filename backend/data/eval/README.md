# Evaluation golden set

`golden_set.jsonl` is the curated, version-controlled ground truth for Phase 6
(RAGAS metrics + the four-stage ablation study). It is committed on purpose,
unlike `data/eval/reports/` (generated run output, gitignored).

Each line is one `GoldenItem` (see `src/evaluation/golden_set.py`):

| field | meaning |
|---|---|
| `id` | stable unique id for the case |
| `question` | the user question |
| `ground_truth_answer` | the reference answer (RAGAS `reference`) |
| `ground_truth_contexts` | supporting context strings (quote the source text) |
| `category` | `graph` / `table_chart` / `web_search` / `superseded_sop` / `general` |
| `expected_tool` | `internal_search` / `web_search` / `answer_directly` (optional) |
| `contexts_are_external` | `true` for web-search cases whose ground truth is web-derived |
| `notes` | free-form provenance note |

## Provenance

Items are grounded in the real sample corpus under `data/samples/`:

- `expense_approval_sop_v3.md` (v3.0, current) and `expense_approval_sop_v2.md`
  (v2.1, superseded) — the Expense Approval process.
- `onboarding_sop_v1.md` (v1.4) — the Employee Onboarding process.
- `product_atlas_router.md` — the Atlas 5000 Router product info.
- `chart.pdf` — an image-only PDF containing (a) an employee table with columns
  First name, Last Name, Position, Office Branch, Age, Salary and (b) a grouped
  bar chart of Dogs vs Cats counts for January–May. These values are only
  recoverable through the multimodal (table/image_caption) ingestion path, which
  is why they anchor the `table_chart` cases.

## Required coverage (SPEC §11, enforced by the loader)

The loader (`load_golden_set`) rejects a set that is outside 30–50 items or that
is short on any required category:

- `graph` ≥ 5 — relational / multi-hop questions.
- `table_chart` ≥ 5 — answerable only from `chart.pdf` tables/figures (and the
  product spec table).
- `web_search` ≥ 2 — **external by design**, NOT answerable from internal docs.
  These items set `contexts_are_external: true` and carry a web-derived context
  string plus a `notes` field. They exist to exercise the web-search path; their
  internal-retrieval metrics are expected to be weak.
- `superseded_sop` ≥ 1 — the correct answer comes from the current SOP (v3.0);
  citing the superseded v2.1 should surface a version warning.

## Superseded-SOP case

The expense SOP changed between v2.1 and v3.0 (e.g. reimbursement SLA 10 → 5
business days; a new $1000 director sign-off threshold). The `superseded_sop`
items take their ground truth from v3.0.
