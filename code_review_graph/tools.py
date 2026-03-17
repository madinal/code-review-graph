"""MCP tool definitions for the Code Review Graph server.

Exposes 8 tools:
1. build_or_update_graph  - full or incremental build
2. get_impact_radius      - blast radius from changed files
3. query_graph            - predefined graph queries
4. get_review_context     - focused subgraph + review prompt
5. semantic_search_nodes  - keyword + vector search across nodes
6. list_graph_stats       - aggregate statistics
7. embed_graph            - compute vector embeddings for semantic search
8. get_docs_section       - token-optimized documentation retrieval
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingStore, embed_all_nodes, semantic_search
from .graph import GraphStore, edge_to_dict, node_to_dict
from .incremental import (
    find_project_root,
    full_build,
    get_changed_files,
    get_db_path,
    get_staged_and_unstaged,
    incremental_update,
)


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _to_snake_case(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _looks_like_test_file(path: str) -> bool:
    posix = Path(path).as_posix()
    name = Path(path).name
    return "/tests/" in posix or name.startswith("test_") or name.endswith("_test.py")


def _validate_repo_root(path: Path) -> Path:
    """Validate that a path is a plausible project root.

    Ensures the path is an existing directory that contains a ``.git``
    or ``.code-review-graph`` directory, preventing arbitrary file-system
    traversal via the ``repo_root`` parameter.
    """
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"repo_root is not an existing directory: {resolved}"
        )
    if not (resolved / ".git").exists() and not (resolved / ".code-review-graph").exists():
        raise ValueError(
            f"repo_root does not look like a project root (no .git or "
            f".code-review-graph directory found): {resolved}"
        )
    return resolved


def _get_store(repo_root: str | None = None) -> tuple[GraphStore, Path]:
    """Resolve repo root and open the graph store."""
    root = _validate_repo_root(Path(repo_root)) if repo_root else find_project_root()
    db_path = get_db_path(root)
    return GraphStore(db_path), root


# ---------------------------------------------------------------------------
# Tool 1: build_or_update_graph
# ---------------------------------------------------------------------------


def build_or_update_graph(
    full_rebuild: bool = False,
    repo_root: str | None = None,
    base: str = "HEAD~1",
) -> dict[str, Any]:
    """Build or incrementally update the code knowledge graph.

    Args:
        full_rebuild: If True, re-parse every file. If False (default),
                      only re-parse files changed since `base`.
        repo_root: Path to the repository root. Auto-detected if omitted.
        base: Git ref for incremental diff (default: HEAD~1).

    Returns:
        Summary with files_parsed/updated, node/edge counts, and errors.
    """
    store, root = _get_store(repo_root)
    try:
        if full_rebuild:
            result = full_build(root, store)
            return {
                "status": "ok",
                "build_type": "full",
                "summary": (
                    f"Full build complete: parsed {result['files_parsed']} files, "
                    f"created {result['total_nodes']} nodes and {result['total_edges']} edges."
                ),
                **result,
            }
        else:
            result = incremental_update(root, store, base=base)
            if result["files_updated"] == 0:
                return {
                    "status": "ok",
                    "build_type": "incremental",
                    "summary": "No changes detected. Graph is up to date.",
                    **result,
                }
            return {
                "status": "ok",
                "build_type": "incremental",
                "summary": (
                    f"Incremental update: {result['files_updated']} files re-parsed, "
                    f"{result['total_nodes']} nodes and {result['total_edges']} edges updated. "
                    f"Changed: {result['changed_files']}. "
                    f"Dependents also updated: {result['dependent_files']}."
                ),
                **result,
            }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 2: get_impact_radius
# ---------------------------------------------------------------------------


def get_impact_radius(
    changed_files: list[str] | None = None,
    max_depth: int = 2,
    repo_root: str | None = None,
    base: str = "HEAD~1",
) -> dict[str, Any]:
    """Analyze the blast radius of changed files.

    Args:
        changed_files: Explicit list of changed file paths (relative to repo root).
                       If omitted, auto-detects from git diff.
        max_depth: How many hops to traverse in the graph (default: 2).
        repo_root: Repository root path. Auto-detected if omitted.
        base: Git ref for auto-detecting changes (default: HEAD~1).

    Returns:
        Changed nodes, impacted nodes, impacted files, and connecting edges.
    """
    store, root = _get_store(repo_root)
    try:
        if changed_files is None:
            changed_files = get_changed_files(root, base)
            if not changed_files:
                changed_files = get_staged_and_unstaged(root)

        if not changed_files:
            return {
                "status": "ok",
                "summary": "No changed files detected.",
                "changed_nodes": [],
                "impacted_nodes": [],
                "impacted_files": [],
            }

        # Convert to absolute paths for graph lookup
        abs_files = [str(root / f) for f in changed_files]
        result = store.get_impact_radius(abs_files, max_depth=max_depth)

        changed_dicts = [node_to_dict(n) for n in result["changed_nodes"]]
        impacted_dicts = [node_to_dict(n) for n in result["impacted_nodes"]]
        edge_dicts = [edge_to_dict(e) for e in result["edges"]]

        summary_parts = [
            f"Blast radius for {len(changed_files)} changed file(s):",
            f"  - {len(changed_dicts)} nodes directly changed",
            f"  - {len(impacted_dicts)} nodes impacted (within {max_depth} hops)",
            f"  - {len(result['impacted_files'])} additional files affected",
        ]

        return {
            "status": "ok",
            "summary": "\n".join(summary_parts),
            "changed_files": changed_files,
            "changed_nodes": changed_dicts,
            "impacted_nodes": impacted_dicts,
            "impacted_files": result["impacted_files"],
            "edges": edge_dicts,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 3: query_graph
# ---------------------------------------------------------------------------

_QUERY_PATTERNS = {
    "callers_of": "Find all functions that call a given function",
    "callees_of": "Find all functions called by a given function",
    "imports_of": "Find all imports of a given file or module",
    "importers_of": "Find all files that import a given file or module",
    "children_of": "Find all nodes contained in a file or class",
    "tests_for": "Find all tests for a given function or class",
    "inheritors_of": "Find all classes that inherit from a given class",
    "file_summary": "Get a summary of all nodes in a file",
}


def query_graph(
    pattern: str,
    target: str,
    repo_root: str | None = None,
) -> dict[str, Any]:
    """Run a predefined graph query.

    Args:
        pattern: Query pattern. One of: callers_of, callees_of, imports_of,
                 importers_of, children_of, tests_for, inheritors_of, file_summary.
        target: The node name, qualified name, or file path to query about.
        repo_root: Repository root path. Auto-detected if omitted.

    Returns:
        Matching nodes and edges for the query.
    """
    store, root = _get_store(repo_root)
    try:
        if pattern not in _QUERY_PATTERNS:
            return {
                "status": "error",
                "error": f"Unknown pattern '{pattern}'. Available: {list(_QUERY_PATTERNS.keys())}",
            }

        results: list[dict] = []
        edges_out: list[dict] = []

        # Resolve target - try as-is, then as absolute path, then search
        candidates = []
        exact_non_test = []
        node = store.get_node(target)
        if not node:
            abs_target = str(root / target)
            node = store.get_node(abs_target)
        if not node:
            # Search by name
            candidates = store.search_nodes(target, limit=50)
            exact_non_test = [c for c in candidates if c.name == target and not c.is_test]
            if len(exact_non_test) == 1:
                node = exact_non_test[0]
                target = node.qualified_name
            elif len(candidates) == 1:
                node = candidates[0]
                target = node.qualified_name
            elif len(exact_non_test) > 1 and pattern != "tests_for":
                return {
                    "status": "ambiguous",
                    "summary": f"Multiple matches for '{target}'. Please use a qualified name.",
                    "candidates": [node_to_dict(c) for c in exact_non_test],
                }
            elif len(candidates) > 1 and pattern != "tests_for":
                return {
                    "status": "ambiguous",
                    "summary": f"Multiple matches for '{target}'. Please use a qualified name.",
                    "candidates": [node_to_dict(c) for c in candidates],
                }

        if not node and not candidates and pattern != "file_summary":
            return {
                "status": "not_found",
                "summary": f"No node found matching '{target}'.",
            }

        qn = node.qualified_name if node else target

        if pattern == "callers_of":
            call_edges = [e for e in store.get_edges_by_target(qn) if e.kind == "CALLS"]
            callers = store.get_nodes_by_qualified([e.source_qualified for e in call_edges])
            for e in call_edges:
                caller = callers.get(e.source_qualified)
                if caller:
                    results.append(node_to_dict(caller))
                edges_out.append(edge_to_dict(e))

        elif pattern == "callees_of":
            call_edges = [e for e in store.get_edges_by_source(qn) if e.kind == "CALLS"]
            callees = store.get_nodes_by_qualified([e.target_qualified for e in call_edges])
            for e in call_edges:
                callee = callees.get(e.target_qualified)
                if callee:
                    results.append(node_to_dict(callee))
                edges_out.append(edge_to_dict(e))

        elif pattern == "imports_of":
            for e in store.get_edges_by_source(qn):
                if e.kind == "IMPORTS_FROM":
                    results.append({"import_target": e.target_qualified})
                    edges_out.append(edge_to_dict(e))

        elif pattern == "importers_of":
            # Find edges where target matches this file
            abs_target = str(root / target) if node is None else node.file_path
            for e in store.get_edges_by_target(abs_target):
                if e.kind == "IMPORTS_FROM":
                    results.append({"importer": e.source_qualified, "file": e.file_path})
                    edges_out.append(edge_to_dict(e))

        elif pattern == "children_of":
            if node and node.kind == "File":
                file_nodes = sorted(
                    [
                        candidate for candidate in store.get_nodes_by_file(node.file_path)
                        if candidate.qualified_name != node.qualified_name
                    ],
                    key=lambda candidate: (candidate.line_start, -candidate.line_end, candidate.id),
                )
                open_scopes = []
                for candidate in file_nodes:
                    while open_scopes and candidate.line_start > open_scopes[-1].line_end:
                        open_scopes.pop()
                    if not open_scopes:
                        results.append(node_to_dict(candidate))
                    open_scopes.append(candidate)
            else:
                for e in store.get_edges_by_source(qn):
                    if e.kind == "CONTAINS":
                        child = store.get_node(e.target_qualified)
                        if child:
                            results.append(node_to_dict(child))

        elif pattern == "tests_for":
            target_candidates = [node] if node else exact_non_test
            explicit_targets = [c.qualified_name for c in target_candidates]
            seen = set()

            def add_result(candidate) -> None:
                if not candidate or not candidate.is_test:
                    return
                if candidate.qualified_name in seen:
                    return
                seen.add(candidate.qualified_name)
                results.append(node_to_dict(candidate))

            for target_qn in explicit_targets:
                for e in store.get_edges_by_target(target_qn):
                    if e.kind == "TESTED_BY":
                        add_result(store.get_node(e.source_qualified))
                    elif e.kind == "CALLS":
                        add_result(store.get_node(e.source_qualified))

            names = {node.name if node else target}

            anchor_files: set[str] = set()
            anchor_classes: set[tuple[str, str]] = set()
            matched_nodes = []
            search_terms = set()
            for name in names:
                snake = _to_snake_case(name)
                search_terms.update({name, snake, snake.replace("_", "")})
                search_terms.update({f"test_{snake}", f"Test{name}", f"{name}Test", f"{name}Tests"})

            for term in search_terms:
                if term:
                    matched_nodes.extend(store.search_nodes(term, limit=50))

            normalized_names = {_normalize_identifier(name) for name in names if name}
            for candidate in matched_nodes:
                norm_name = _normalize_identifier(candidate.name)
                norm_parent = _normalize_identifier(candidate.parent_name or "")
                norm_file = _normalize_identifier(Path(candidate.file_path).stem)
                file_match = any(wanted and wanted in norm_file for wanted in normalized_names)
                name_match = any(wanted and wanted in norm_name for wanted in normalized_names)
                parent_match = any(wanted and wanted in norm_parent for wanted in normalized_names)
                if not (file_match or name_match or parent_match):
                    continue
                if _looks_like_test_file(candidate.file_path):
                    if file_match:
                        anchor_files.add(candidate.file_path)
                    if candidate.kind == "Class" and name_match:
                        anchor_classes.add((candidate.file_path, candidate.name))
                add_result(candidate)

            for file_path in anchor_files:
                for test_node in store.get_nodes_by_file(file_path):
                    add_result(test_node)

            for file_path, class_name in anchor_classes:
                for test_node in store.get_nodes_by_file(file_path):
                    if test_node.parent_name == class_name:
                        add_result(test_node)

        elif pattern == "inheritors_of":
            inheritance_edges = [
                e for e in store.get_edges_by_target(qn)
                if e.kind in ("INHERITS", "IMPLEMENTS")
            ]
            inheritors = store.get_nodes_by_qualified([e.source_qualified for e in inheritance_edges])
            for e in inheritance_edges:
                child = inheritors.get(e.source_qualified)
                if child:
                    results.append(node_to_dict(child))
                edges_out.append(edge_to_dict(e))

        elif pattern == "file_summary":
            abs_path = str(root / target)
            for n in store.get_node_occurrences_by_file(abs_path):
                results.append(node_to_dict(n))

        return {
            "status": "ok",
            "pattern": pattern,
            "target": target,
            "description": _QUERY_PATTERNS[pattern],
            "summary": f"Found {len(results)} result(s) for {pattern}('{target}')",
            "results": results,
            "edges": edges_out,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 4: get_review_context
# ---------------------------------------------------------------------------


def get_review_context(
    changed_files: list[str] | None = None,
    max_depth: int = 2,
    include_source: bool = True,
    max_lines_per_file: int = 200,
    repo_root: str | None = None,
    base: str = "HEAD~1",
) -> dict[str, Any]:
    """Generate a focused review context from changed files.

    Builds a token-optimized subgraph + source snippets for code review.

    Args:
        changed_files: Files to review (auto-detected from git diff if omitted).
        max_depth: Impact radius depth (default: 2).
        include_source: Whether to include source code snippets (default: True).
        max_lines_per_file: Max source lines per file in output (default: 200).
        repo_root: Repository root path. Auto-detected if omitted.
        base: Git ref for change detection (default: HEAD~1).

    Returns:
        Structured review context with subgraph, source snippets, and review guidance.
    """
    store, root = _get_store(repo_root)
    try:
        # Get impact radius first
        if changed_files is None:
            changed_files = get_changed_files(root, base)
            if not changed_files:
                changed_files = get_staged_and_unstaged(root)

        if not changed_files:
            return {
                "status": "ok",
                "summary": "No changes detected. Nothing to review.",
                "context": {},
            }

        abs_files = [str(root / f) for f in changed_files]
        impact = store.get_impact_radius(abs_files, max_depth=max_depth)

        # Build review context
        context: dict[str, Any] = {
            "changed_files": changed_files,
            "impacted_files": impact["impacted_files"],
            "graph": {
                "changed_nodes": [node_to_dict(n) for n in impact["changed_nodes"]],
                "impacted_nodes": [node_to_dict(n) for n in impact["impacted_nodes"]],
                "edges": [edge_to_dict(e) for e in impact["edges"]],
            },
        }

        # Add source snippets for changed files
        if include_source:
            snippets = {}
            for rel_path in changed_files:
                full_path = root / rel_path
                if full_path.is_file():
                    try:
                        lines = full_path.read_text(errors="replace").splitlines()
                        if len(lines) > max_lines_per_file:
                            # Include only the relevant functions/classes
                            relevant_lines = _extract_relevant_lines(
                                lines, impact["changed_nodes"], str(full_path), max_lines_per_file
                            )
                            snippets[rel_path] = relevant_lines
                        else:
                            snippets[rel_path] = "\n".join(
                                f"{i+1}: {line}" for i, line in enumerate(lines)
                            )
                    except (OSError, UnicodeDecodeError):
                        snippets[rel_path] = "(could not read file)"
            context["source_snippets"] = snippets

        # Generate review guidance
        guidance = _generate_review_guidance(impact, changed_files)
        context["review_guidance"] = guidance

        summary_parts = [
            f"Review context for {len(changed_files)} changed file(s):",
            f"  - {len(impact['changed_nodes'])} directly changed nodes",
            f"  - {len(impact['impacted_nodes'])} impacted nodes"
            f" in {len(impact['impacted_files'])} files",
            "",
            "Review guidance:",
            guidance,
        ]

        return {
            "status": "ok",
            "summary": "\n".join(summary_parts),
            "context": context,
        }
    finally:
        store.close()


def _extract_relevant_lines(
    lines: list[str], nodes: list, file_path: str, max_lines: int
) -> str:
    """Extract only the lines relevant to changed nodes."""
    ranges = []
    for n in nodes:
        if n.file_path == file_path and n.kind != "File":
            start = max(0, n.line_start - 3)  # 2 lines context before
            end = min(len(lines), n.line_end + 2)  # 1 line context after
            ranges.append((start, end))

    if not ranges:
        # Show first N lines as fallback
        return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines[:max_lines]))

    # Merge overlapping ranges
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: list[str] = []
    emitted = 0
    for start, end in merged:
        if emitted >= max_lines:
            break
        if parts:
            parts.append("...")
        available = max_lines - emitted
        if available <= 0:
            break
        for i in range(start, min(end, start + available)):
            parts.append(f"{i+1}: {lines[i]}")
            emitted += 1
        if end > start + available:
            parts.append("...")
            break

    return "\n".join(parts)


def _generate_review_guidance(impact: dict, changed_files: list[str]) -> str:
    """Generate review guidance based on the impact analysis."""
    guidance_parts = []

    # Check for test coverage
    changed_funcs = [
        n for n in impact["changed_nodes"] if n.kind == "Function"
    ]
    test_edges = [e for e in impact["edges"] if e.kind == "TESTED_BY"]
    tested_funcs = {e.source_qualified for e in test_edges}

    untested = [
        f for f in changed_funcs
        if f.qualified_name not in tested_funcs and not f.is_test
    ]
    if untested:
        guidance_parts.append(
            f"- {len(untested)} changed function(s) lack test coverage: "
            + ", ".join(n.name for n in untested[:5])
        )

    # Check for wide blast radius
    if len(impact["impacted_nodes"]) > 20:
        guidance_parts.append(
            f"- Wide blast radius: {len(impact['impacted_nodes'])} nodes impacted. "
            "Review callers and dependents carefully."
        )

    # Check for inheritance changes
    inheritance_edges = [e for e in impact["edges"] if e.kind in ("INHERITS", "IMPLEMENTS")]
    if inheritance_edges:
        guidance_parts.append(
            f"- {len(inheritance_edges)} inheritance/implementation relationship(s) affected. "
            "Check for Liskov substitution violations."
        )

    # Check for cross-file impact
    impacted_file_count = len(impact["impacted_files"])
    if impacted_file_count > 3:
        guidance_parts.append(
            f"- Changes impact {impacted_file_count} other files."
            " Consider splitting into smaller PRs."
        )

    if not guidance_parts:
        guidance_parts.append("- Changes appear well-contained with minimal blast radius.")

    return "\n".join(guidance_parts)


# ---------------------------------------------------------------------------
# Tool 5: semantic_search_nodes
# ---------------------------------------------------------------------------


def semantic_search_nodes(
    query: str,
    kind: str | None = None,
    limit: int = 20,
    repo_root: str | None = None,
) -> dict[str, Any]:
    """Search for nodes by name, keyword, or semantic similarity.

    Uses vector embeddings for semantic search if available (install with
    `pip install code-review-graph[embeddings]`). Falls back to keyword
    matching otherwise.

    Args:
        query: Search string to match against node names and qualified names.
        kind: Optional filter by node kind (File, Class, Function, Type, Test).
        limit: Maximum results to return (default: 20).
        repo_root: Repository root path. Auto-detected if omitted.

    Returns:
        Ranked list of matching nodes.
    """
    store, root = _get_store(repo_root)
    try:
        db_path = get_db_path(root)
        emb_store = EmbeddingStore(db_path)
        search_mode = "keyword"

        try:
            if emb_store.available and emb_store.count() > 0:
                # Vector search
                search_mode = "semantic"
                raw = semantic_search(query, store, emb_store, limit=limit * 2)
                if kind:
                    raw = [r for r in raw if r.get("kind") == kind]
                raw = raw[:limit]
                return {
                    "status": "ok",
                    "query": query,
                    "search_mode": search_mode,
                    "summary": f"Found {len(raw)} node(s) matching '{query}' via semantic search"
                    + (f" (kind={kind})" if kind else ""),
                    "results": raw,
                }
        finally:
            emb_store.close()

        # Keyword fallback
        results = store.search_nodes(query, limit=limit * 2)

        if kind:
            results = [r for r in results if r.kind == kind]

        def score(node):
            name_lower = node.name.lower()
            q_lower = query.lower()
            if name_lower == q_lower:
                return 0
            if name_lower.startswith(q_lower):
                return 1
            return 2

        results.sort(key=score)
        results = results[:limit]

        return {
            "status": "ok",
            "query": query,
            "search_mode": search_mode,
            "summary": f"Found {len(results)} node(s) matching '{query}'" + (
                f" (kind={kind})" if kind else ""
            ),
            "results": [node_to_dict(r) for r in results],
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 6: list_graph_stats
# ---------------------------------------------------------------------------


def list_graph_stats(repo_root: str | None = None) -> dict[str, Any]:
    """Get aggregate statistics about the knowledge graph.

    Args:
        repo_root: Repository root path. Auto-detected if omitted.

    Returns:
        Total nodes, edges, breakdown by kind, languages, and last update time.
    """
    store, root = _get_store(repo_root)
    try:
        stats = store.get_stats()

        summary_parts = [
            f"Graph statistics for {root.name}:",
            f"  Files: {stats.files_count}",
            f"  Total nodes: {stats.total_nodes}",
            f"  Total edges: {stats.total_edges}",
            f"  Languages: {', '.join(stats.languages) if stats.languages else 'none'}",
            f"  Last updated: {stats.last_updated or 'never'}",
            "",
            "Nodes by kind:",
        ]
        for kind, count in sorted(stats.nodes_by_kind.items()):
            summary_parts.append(f"  {kind}: {count}")
        summary_parts.append("")
        summary_parts.append("Edges by kind:")
        for kind, count in sorted(stats.edges_by_kind.items()):
            summary_parts.append(f"  {kind}: {count}")

        # Add embedding info if available
        emb_store = EmbeddingStore(get_db_path(root))
        try:
            emb_count = emb_store.count()
            summary_parts.append("")
            summary_parts.append(f"Embeddings: {emb_count} nodes embedded")
            if not emb_store.available:
                summary_parts.append("  (install sentence-transformers for semantic search)")
        finally:
            emb_store.close()

        return {
            "status": "ok",
            "summary": "\n".join(summary_parts),
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "nodes_by_kind": stats.nodes_by_kind,
            "edges_by_kind": stats.edges_by_kind,
            "languages": stats.languages,
            "files_count": stats.files_count,
            "last_updated": stats.last_updated,
            "embeddings_count": emb_count,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 7: embed_graph
# ---------------------------------------------------------------------------


def embed_graph(repo_root: str | None = None) -> dict[str, Any]:
    """Compute vector embeddings for all graph nodes to enable semantic search.

    Requires: `pip install code-review-graph[embeddings]`
    Uses the all-MiniLM-L6-v2 model (fast, 384-dim).

    Only embeds nodes that don't already have up-to-date embeddings.

    Args:
        repo_root: Repository root path. Auto-detected if omitted.

    Returns:
        Number of nodes embedded and total embedding count.
    """
    store, root = _get_store(repo_root)
    db_path = get_db_path(root)
    emb_store = EmbeddingStore(db_path)
    try:
        if not emb_store.available:
            return {
                "status": "error",
                "error": (
                    "sentence-transformers is not installed. "
                    "Install with: pip install code-review-graph[embeddings]"
                ),
            }

        newly_embedded = embed_all_nodes(store, emb_store)
        total = emb_store.count()

        return {
            "status": "ok",
            "summary": (
                f"Embedded {newly_embedded} new node(s). "
                f"Total embeddings: {total}. "
                "Semantic search is now active."
            ),
            "newly_embedded": newly_embedded,
            "total_embeddings": total,
        }
    finally:
        emb_store.close()
        store.close()


# ---------------------------------------------------------------------------
# Tool 8: get_docs_section
# ---------------------------------------------------------------------------

# Search paths for the LLM-optimized reference file
_REFERENCE_PATHS = [
    "docs/LLM-OPTIMIZED-REFERENCE.md",
]


def get_docs_section(section_name: str) -> dict[str, Any]:
    """Return a specific section from the LLM-optimized reference.

    Used by skills and Claude Code to load only the exact documentation
    section needed, keeping token usage minimal (90%+ savings).

    Args:
        section_name: Exact section name. One of: usage, review-delta,
                      review-pr, commands, legal, watch, embeddings,
                      languages, troubleshooting.

    Returns:
        The section content, or an error if not found.
    """
    import re as _re

    # Try package-relative path first (works even outside a git repo)
    pkg_dir = Path(__file__).resolve().parent.parent
    search_roots = [pkg_dir]

    # Also try repo root if inside a git repo
    try:
        _, root = _get_store()
        if root not in search_roots:
            search_roots.append(root)
    except RuntimeError:
        pass

    for search_root in search_roots:
        for rel_path in _REFERENCE_PATHS:
            full_path = search_root / rel_path
            if full_path.exists():
                content = full_path.read_text()
                match = _re.search(
                    rf'<section name="{_re.escape(section_name)}">'
                    r"(.*?)</section>",
                    content,
                    _re.DOTALL | _re.IGNORECASE,
                )
                if match:
                    return {
                        "status": "ok",
                        "section": section_name,
                        "content": match.group(1).strip(),
                    }

    available = [
        "usage", "review-delta", "review-pr", "commands",
        "legal", "watch", "embeddings", "languages", "troubleshooting",
    ]
    return {
        "status": "not_found",
        "error": (
            f"Section '{section_name}' not found. "
            f"Available: {', '.join(available)}"
        ),
    }
