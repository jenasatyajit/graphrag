"""Knowledge-graph schema contracts (SPEC 6.2).

These Pydantic models are the single source of truth for the node labels and
relationship types the graph accepts. The LLM extractor (Phase 3) must emit
triples that validate against these models; anything off-schema is rejected so
the model cannot invent new labels or edge types. Both the extractor and the
Neo4j writer import the constants below so the Cypher and the prompt never
drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --- Node labels (SPEC 6.2) --------------------------------------------------

NodeLabel = Literal[
    "Process",
    "Step",
    "Role",
    "System",
    "ComplianceRequirement",
    "Product",
    "Feature",
    "Spec",
]

NODE_LABELS: frozenset[str] = frozenset(
    (
        "Process",
        "Step",
        "Role",
        "System",
        "ComplianceRequirement",
        "Product",
        "Feature",
        "Spec",
    )
)

# --- Relationship types (SPEC 6.2) -------------------------------------------

RelationType = Literal[
    "HAS_STEP",
    "OWNED_BY",
    "REQUIRES",
    "MUST_SATISFY",
    "SUPERSEDES",
    "HAS_FEATURE",
    "HAS_SPEC",
    "COMPATIBLE_WITH",
]

RELATION_TYPES: frozenset[str] = frozenset(
    (
        "HAS_STEP",
        "OWNED_BY",
        "REQUIRES",
        "MUST_SATISFY",
        "SUPERSEDES",
        "HAS_FEATURE",
        "HAS_SPEC",
        "COMPATIBLE_WITH",
    )
)

# Allowed (subject_label, relation, object_label) shapes. Constraining the
# endpoints keeps the graph coherent and lets us reject nonsense like
# ``(Role)-[:HAS_STEP]->(System)`` that a language model might hallucinate.
_ALLOWED_EDGE_SHAPES: frozenset[tuple[str, str, str]] = frozenset(
    (
        ("Process", "HAS_STEP", "Step"),
        ("Step", "OWNED_BY", "Role"),
        ("Step", "REQUIRES", "System"),
        ("Process", "MUST_SATISFY", "ComplianceRequirement"),
        ("Process", "SUPERSEDES", "Process"),
        ("Product", "HAS_FEATURE", "Feature"),
        ("Product", "HAS_SPEC", "Spec"),
        ("Product", "COMPATIBLE_WITH", "Product"),
    )
)


def is_allowed_edge(subject_label: str, relation: str, object_label: str) -> bool:
    """Return whether an edge shape is permitted by SPEC 6.2."""
    return (subject_label, relation, object_label) in _ALLOWED_EDGE_SHAPES


class GraphNode(BaseModel):
    """A typed graph node keyed by ``(label, name)``.

    ``name`` is the natural key used in ``MERGE`` so re-ingesting the same entity
    updates rather than duplicates it. ``properties`` holds the optional
    schema-specific fields (e.g. ``order`` for Step, ``regulation_ref`` for a
    ComplianceRequirement); they are kept loose here and applied verbatim on the
    node so the schema can evolve without a migration.
    """

    label: NodeLabel
    name: str = Field(min_length=1)
    properties: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Node name cannot be blank.")
        return cleaned


class GraphTriple(BaseModel):
    """A validated (subject)-[relation]->(object) fact extracted from a chunk.

    Endpoint labels and the relation are constrained to SPEC 6.2. An invalid
    combination raises a ``ValidationError`` so the extractor can drop or repair
    the triple instead of persisting an off-schema edge.
    """

    subject: GraphNode
    relation: RelationType
    object: GraphNode

    @model_validator(mode="after")
    def _check_edge_shape(self) -> GraphTriple:
        if not is_allowed_edge(self.subject.label, self.relation, self.object.label):
            raise ValueError(
                f"Edge ({self.subject.label})-[:{self.relation}]->({self.object.label}) "
                "is not permitted by the graph schema."
            )
        return self

    def as_text(self) -> str:
        """Render the triple as a human-readable fact for the synthesis context."""
        return f"{self.subject.name} {self.relation.replace('_', ' ').lower()} {self.object.name}"


class GraphFact(BaseModel):
    """A traversal result: a triple plus the chunk(s) that support it.

    ``chunk_ids`` come from the endpoint nodes' ``chunk_ids`` property, so a
    graph-derived claim is always traceable back to a real ingested chunk and
    can carry a chunk-based citation (per the Phase 3 citation decision).
    """

    subject_name: str
    relation: RelationType
    object_name: str
    chunk_ids: list[str] = Field(default_factory=list)

    def as_text(self) -> str:
        return f"{self.subject_name} {self.relation.replace('_', ' ').lower()} {self.object_name}"
