"""Centralized prompt templates.

Phase 3 adds the graph-extraction and entity-extraction prompts. Router,
synthesis, and follow-up-rewrite prompts are centralized here in Phase 5;
until then the synthesis prompt lives in ``generation/synthesizer.py``.
"""

from __future__ import annotations

# Inlining the schema keeps the model from inventing labels/relations: it is
# instructed to emit only the listed node labels and relationship types, as
# strict JSON, and to skip anything that does not fit.
GRAPH_EXTRACTION_SYSTEM_PROMPT = """You extract a knowledge graph from a single \
chunk of a company document (an SOP or product info sheet).

Return ONLY a JSON object of the form:
{"triples": [{"subject": {"label": <NodeLabel>, "name": <str>, "properties": {}},
              "relation": <RelationType>,
              "object": {"label": <NodeLabel>, "name": <str>, "properties": {}}}]}

Allowed NodeLabel values: Process, Step, Role, System, ComplianceRequirement,
Product, Feature, Spec.

Allowed relationships (subject label -> relation -> object label ONLY):
- Process -> HAS_STEP -> Step
- Step -> OWNED_BY -> Role
- Step -> REQUIRES -> System
- Process -> MUST_SATISFY -> ComplianceRequirement
- Product -> HAS_FEATURE -> Feature
- Product -> HAS_SPEC -> Spec
- Product -> COMPATIBLE_WITH -> Product

Rules:
- Use ONLY the labels and relationships above. Never invent new ones.
- If a triple does not fit an allowed shape exactly, omit it.
- "name" is a short canonical entity name (e.g. "Expense Approval", "IT Admin",
  "SAP"), not a whole sentence.
- Optional properties you may set when clearly present: Step.order (int),
  ComplianceRequirement.regulation_ref (str), Spec.value (str), Spec.unit (str),
  Product.sku (str).
- Do NOT emit SUPERSEDES; SOP version chains are created separately.
- If the chunk contains no extractable facts, return {"triples": []}.
- Output JSON only. No prose, no markdown fences."""

# The entity-extraction prompt for query-time graph search (deviation D-9:
# LLM-based, not local NER). Reasoning is disabled by the caller for speed.
ENTITY_EXTRACTION_SYSTEM_PROMPT = """Extract the named entities from the user's \
question that could match nodes in a company knowledge graph (processes, SOP \
names, steps, roles, systems, compliance requirements, products, features, or \
specs).

Return ONLY a JSON object: {"entities": [<str>, ...]}.
- Each entity is a short noun phrase (e.g. "expense approval", "SAP", "IT Admin").
- Do not include generic words, verbs, or question words.
- If there are no such entities, return {"entities": []}.
- Output JSON only. No prose, no markdown fences."""
