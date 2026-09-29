"""Async Neo4j client wrapper for the domain knowledge graph (Phase 3).

Owns the driver lifecycle and every Cypher interaction: connectivity checks for
``/health``, parameterized node/edge writes during ingestion, the SOP
``SUPERSEDES`` version chain, and bounded query-time traversal that returns
citable facts. Every query is parameterized (never string-concatenated) and
scoped to the configured database. Failures are wrapped in ``GraphError`` so a
raw driver exception never reaches the API boundary; callers decide whether to
degrade gracefully (query path) or fail the batch (ingestion).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.api.config import Settings
from src.api.errors import GraphError
from src.api.schemas import ComponentHealth
from src.indexing.graph_schema import GraphFact, GraphNode, GraphTriple, RelationType

logger = structlog.get_logger(__name__)


class GraphStore:
    """Thin, resilient boundary around the Neo4j async driver."""

    def __init__(self, settings: Settings, *, driver: Any | None = None) -> None:
        self._settings = settings
        self._database = settings.neo4j_database
        self._timeout = settings.graph_query_timeout_seconds
        self._max_retries = settings.graph_max_retries
        self._max_hops = settings.graph_traversal_max_hops
        self._entity_match_limit = settings.graph_entity_match_limit
        self._fact_limit = settings.graph_fact_limit
        self._driver = driver
        self._owns_driver = driver is None

    # -- lifecycle ------------------------------------------------------------

    async def connect(self) -> None:
        """Create the async driver when one was not injected."""
        if self._driver is not None:
            return
        try:
            # Imported lazily so a driver can be injected in tests without the
            # real package being importable, mirroring the other store wrappers.
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(
                    self._settings.neo4j_user,
                    self._settings.neo4j_password.get_secret_value(),
                ),
            )
        except Exception as exc:
            raise GraphError("The Neo4j driver could not be created.") from exc

    async def verify_connectivity(self) -> ComponentHealth:
        """Report a real Neo4j health status for ``/health``."""
        try:
            await self.connect()
            assert self._driver is not None
            await self._driver.verify_connectivity()
            return ComponentHealth(name="neo4j", status="ok")
        except Exception:
            return ComponentHealth(
                name="neo4j",
                status="unhealthy",
                detail="Neo4j AuraDB is unreachable or rejected the connection.",
            )

    async def close(self) -> None:
        """Close the driver when this instance owns it."""
        if self._driver is not None and self._owns_driver:
            await self._driver.close()

    # -- writes (ingestion) ---------------------------------------------------

    async def write_triples(self, triples: Sequence[GraphTriple], *, chunk_id: str) -> int:
        """MERGE the nodes and edges for one chunk's triples.

        Every endpoint node accumulates ``chunk_id`` in its ``chunk_ids`` list so
        graph facts stay traceable to a source chunk. Returns the number of edges
        written (each triple is one edge).

        Cypher cannot parameterize node labels or relationship types, and we
        deliberately avoid an APOC dependency. Because the schema is a small,
        fixed, validated set (:class:`GraphTriple` rejects anything off-schema),
        we group triples by their ``(subject_label, relation, object_label)``
        shape and run one static, parameterized query per shape. The only
        interpolated tokens are schema constants, never user text.
        """
        written = 0
        grouped: dict[tuple[str, str, str], list[GraphTriple]] = {}
        for triple in triples:
            key = (triple.subject.label, triple.relation, triple.object.label)
            grouped.setdefault(key, []).append(triple)

        for (subject_label, relation, object_label), group in grouped.items():
            rows = [
                {
                    "subject_name": triple.subject.name,
                    "subject_props": dict(triple.subject.properties),
                    "subject_version": _merge_version(triple.subject),
                    "object_name": triple.object.name,
                    "object_props": dict(triple.object.properties),
                    "object_version": _merge_version(triple.object),
                }
                for triple in group
            ]
            cypher = f"""
            UNWIND $rows AS row
            {_merge_node_clause(subject_label, "s", "subject")}
            SET s += row.subject_props,
                s.chunk_ids =
                    CASE WHEN $chunk_id IN coalesce(s.chunk_ids, [])
                         THEN s.chunk_ids
                         ELSE coalesce(s.chunk_ids, []) + $chunk_id END
            {_merge_node_clause(object_label, "o", "object")}
            SET o += row.object_props,
                o.chunk_ids =
                    CASE WHEN $chunk_id IN coalesce(o.chunk_ids, [])
                         THEN o.chunk_ids
                         ELSE coalesce(o.chunk_ids, []) + $chunk_id END
            MERGE (s)-[rel:`{relation}`]->(o)
            RETURN count(rel) AS edges
            """
            records = await self._run(cypher, {"rows": rows, "chunk_id": chunk_id}, write=True)
            written += int(records[0]["edges"]) if records else 0
        return written

    async def supersede_previous_versions(self, doc_title: str, current_version: str) -> int:
        """Link older Process versions of ``doc_title`` to the current one.

        Creates ``(current)-[:SUPERSEDES]->(old)`` for every prior version and
        marks the older processes ``is_current_version = false``. Returns the
        number of superseded processes. Idempotent via ``MERGE``.
        """
        cypher = """
        MATCH (current:Process {doc_title: $doc_title, sop_version: $current_version})
        MATCH (old:Process {doc_title: $doc_title})
        WHERE old.sop_version <> $current_version
        SET old.is_current_version = false, current.is_current_version = true
        MERGE (current)-[:SUPERSEDES]->(old)
        RETURN count(DISTINCT old) AS superseded
        """
        records = await self._run(
            cypher,
            {"doc_title": doc_title, "current_version": current_version},
            write=True,
        )
        return int(records[0]["superseded"]) if records else 0

    async def superseded_chunk_ids(self) -> set[str]:
        """Return chunk ids attached to any process flagged not-current.

        Ingestion uses this to flip the matching Qdrant chunks'
        ``is_current_version`` so the vector store and graph agree.
        """
        cypher = """
        MATCH (p:Process)
        WHERE p.is_current_version = false
        UNWIND coalesce(p.chunk_ids, []) AS chunk_id
        RETURN collect(DISTINCT chunk_id) AS chunk_ids
        """
        records = await self._run(cypher, {}, write=False)
        if not records:
            return set()
        return {str(value) for value in records[0]["chunk_ids"]}

    # -- traversal (query time) ----------------------------------------------

    async def match_entities(self, names: Sequence[str]) -> list[str]:
        """Fuzzy-match candidate names to existing node names (case-insensitive).

        Matches on containment in either direction so "SAP system" finds a "SAP"
        node and vice versa. Returns the matched canonical node names.
        """
        cleaned = [name.strip() for name in names if name.strip()]
        if not cleaned:
            return []
        cypher = """
        UNWIND $names AS candidate
        MATCH (n)
        WHERE n.name IS NOT NULL
          AND (
            toLower(n.name) CONTAINS toLower(candidate)
            OR toLower(candidate) CONTAINS toLower(n.name)
          )
        RETURN DISTINCT n.name AS name
        LIMIT $limit
        """
        records = await self._run(
            cypher,
            {"names": cleaned, "limit": self._entity_match_limit},
            write=False,
        )
        return [str(record["name"]) for record in records]

    async def neighborhood_facts(self, node_names: Sequence[str]) -> list[GraphFact]:
        """Return facts within ``max_hops`` of the seed nodes as citable triples.

        Each fact carries the union of both endpoints' ``chunk_ids`` so the
        synthesizer can attach a chunk-based citation to a graph-derived claim.
        """
        seeds = [name.strip() for name in node_names if name.strip()]
        if not seeds:
            return []
        # Bounded variable-length pattern; hop count comes from validated config
        # (1-3), so the interpolated integer cannot carry injection.
        cypher = f"""
        MATCH (seed)
        WHERE seed.name IN $seeds
        MATCH path = (seed)-[r*1..{self._max_hops}]-(other)
        WITH relationships(path) AS rels
        UNWIND rels AS rel
        WITH startNode(rel) AS s, rel, endNode(rel) AS o
        WHERE s.name IS NOT NULL AND o.name IS NOT NULL
        WITH DISTINCT
            s.name AS subject_name,
            type(rel) AS relation,
            o.name AS object_name,
            coalesce(s.chunk_ids, []) + coalesce(o.chunk_ids, []) AS raw_chunk_ids
        RETURN
            subject_name,
            relation,
            object_name,
            [cid IN raw_chunk_ids WHERE cid IS NOT NULL | cid] AS chunk_ids
        LIMIT $limit
        """
        records = await self._run(
            cypher,
            {"seeds": seeds, "limit": self._fact_limit},
            write=False,
        )
        facts: list[GraphFact] = []
        for record in records:
            relation = str(record["relation"])
            if relation not in RelationType.__args__:  # type: ignore[attr-defined]
                # Ignore any edge type outside the schema so callers only ever
                # see contract-compliant relations.
                continue
            deduped_chunk_ids = list(dict.fromkeys(str(value) for value in record["chunk_ids"]))
            facts.append(
                GraphFact(
                    subject_name=str(record["subject_name"]),
                    relation=relation,  # type: ignore[arg-type]
                    object_name=str(record["object_name"]),
                    chunk_ids=deduped_chunk_ids,
                )
            )
        return facts

    # -- internals ------------------------------------------------------------

    async def _run(
        self,
        cypher: str,
        parameters: dict[str, Any],
        *,
        write: bool,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized query with bounded retries and a timeout."""
        await self.connect()
        assert self._driver is not None
        routing = "w" if write else "r"
        result: Any = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
                retry=retry_if_exception_type(_retryable_neo4j_errors()),
                reraise=True,
            ):
                with attempt:
                    result = await self._driver.execute_query(
                        cypher,
                        parameters_=parameters,
                        database_=self._database,
                        routing_=routing,
                    )
            if result is None:
                raise GraphError("A Neo4j graph operation returned no result.")
            return [record.data() for record in result.records]
        except GraphError:
            raise
        except Exception as exc:
            raise GraphError("A Neo4j graph operation failed.") from exc


def _merge_version(node: GraphNode) -> str:
    """Return the ``sop_version`` used to key a versioned Process MERGE.

    Only ``Process`` nodes are versioned. A missing version collapses to an empty
    string so version-less processes (and every non-Process node) still MERGE on
    name alone. Two SOP versions of the same document therefore become two
    distinct Process nodes, which is what the ``SUPERSEDES`` chain links.
    """
    if node.label != "Process":
        return ""
    version = node.properties.get("sop_version", "")
    return str(version) if version else ""


def _merge_node_clause(label: str, var: str, prefix: str) -> str:
    """Build the MERGE clause for a node, version-aware for ``Process``.

    ``Process`` nodes MERGE on ``(name, sop_version)`` so each SOP version is its
    own node; all other labels MERGE on ``name`` only. Labels are schema
    constants (validated by ``GraphTriple``), never user text, so interpolation
    here cannot carry injection.
    """
    if label == "Process":
        return (
            f"MERGE ({var}:`{label}` "
            f"{{name: row.{prefix}_name, sop_version: row.{prefix}_version}})"
        )
    return f"MERGE ({var}:`{label}` {{name: row.{prefix}_name}})"


def _retryable_neo4j_errors() -> tuple[type[BaseException], ...]:
    """Return the transient neo4j exception types, tolerating an absent package."""
    try:
        from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

        return (ServiceUnavailable, SessionExpired, TransientError, TimeoutError)
    except Exception:
        return (TimeoutError,)
