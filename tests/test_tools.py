"""Tests for MCP tool functions."""

import tempfile
from pathlib import Path

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, get_db_path
from code_review_graph.parser import NodeInfo, EdgeInfo
from code_review_graph.tools import (
    get_review_context,
    list_graph_stats,
    query_graph,
    semantic_search_nodes,
)


class TestTools:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.store = GraphStore(self.tmp.name)
        self._seed_data()

    def teardown_method(self):
        self.store.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def _seed_data(self):
        """Seed the store with test data."""
        # File nodes
        self.store.upsert_node(NodeInfo(
            kind="File", name="/repo/auth.py", file_path="/repo/auth.py",
            line_start=1, line_end=50, language="python",
        ))
        self.store.upsert_node(NodeInfo(
            kind="File", name="/repo/main.py", file_path="/repo/main.py",
            line_start=1, line_end=30, language="python",
        ))
        # Class
        self.store.upsert_node(NodeInfo(
            kind="Class", name="AuthService", file_path="/repo/auth.py",
            line_start=5, line_end=40, language="python",
        ))
        # Functions
        self.store.upsert_node(NodeInfo(
            kind="Function", name="login", file_path="/repo/auth.py",
            line_start=10, line_end=20, language="python",
            parent_name="AuthService",
        ))
        self.store.upsert_node(NodeInfo(
            kind="Function", name="process", file_path="/repo/main.py",
            line_start=5, line_end=15, language="python",
        ))
        # Test
        self.store.upsert_node(NodeInfo(
            kind="Test", name="test_login", file_path="/repo/test_auth.py",
            line_start=1, line_end=10, language="python", is_test=True,
        ))

        # Edges
        self.store.upsert_edge(EdgeInfo(
            kind="CONTAINS", source="/repo/auth.py",
            target="/repo/auth.py::AuthService", file_path="/repo/auth.py",
        ))
        self.store.upsert_edge(EdgeInfo(
            kind="CONTAINS", source="/repo/auth.py::AuthService",
            target="/repo/auth.py::AuthService.login", file_path="/repo/auth.py",
        ))
        self.store.upsert_edge(EdgeInfo(
            kind="CALLS", source="/repo/main.py::process",
            target="/repo/auth.py::AuthService.login", file_path="/repo/main.py", line=10,
        ))
        self.store.commit()

    def test_search_nodes(self):
        # Direct call to store (tools need repo_root, which is harder to mock)
        results = self.store.search_nodes("login")
        names = {r.name for r in results}
        assert "login" in names

    def test_search_nodes_by_kind(self):
        results = self.store.search_nodes("auth")
        # Should find both AuthService class and auth.py file
        kinds = {r.kind for r in results}
        assert len(results) >= 1

    def test_stats(self):
        stats = self.store.get_stats()
        assert stats.total_nodes == 6
        assert stats.total_edges == 3
        assert stats.files_count == 2
        assert "python" in stats.languages

    def test_impact_from_auth(self):
        result = self.store.get_impact_radius(["/repo/auth.py"], max_depth=2)
        # Changing auth.py should impact main.py (which calls login)
        impacted_files = result["impacted_files"]
        impacted_qns = {n.qualified_name for n in result["impacted_nodes"]}
        # process() in main.py calls login(), so it should be impacted
        assert "/repo/main.py::process" in impacted_qns or "/repo/main.py" in impacted_qns

    def test_query_children_of(self):
        edges = self.store.get_edges_by_source("/repo/auth.py")
        contains = [e for e in edges if e.kind == "CONTAINS"]
        assert len(contains) >= 1

    def test_query_callers(self):
        edges = self.store.get_edges_by_target("/repo/auth.py::AuthService.login")
        callers = [e for e in edges if e.kind == "CALLS"]
        assert len(callers) == 1
        assert callers[0].source_qualified == "/repo/main.py::process"


def test_query_graph_callers_of_end_to_end(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    db_path = get_db_path(repo_root)
    store = GraphStore(str(db_path))

    auth_path = str((repo_root / "auth.py").resolve())
    main_path = str((repo_root / "main.py").resolve())

    store.upsert_node(NodeInfo(
        kind="File", name=auth_path, file_path=auth_path,
        line_start=1, line_end=20, language="python",
    ))
    store.upsert_node(NodeInfo(
        kind="File", name=main_path, file_path=main_path,
        line_start=1, line_end=20, language="python",
    ))
    store.upsert_node(NodeInfo(
        kind="Class", name="AuthService", file_path=auth_path,
        line_start=1, line_end=10, language="python",
    ))
    store.upsert_node(NodeInfo(
        kind="Function", name="login", file_path=auth_path,
        line_start=2, line_end=5, language="python",
        parent_name="AuthService",
    ))
    store.upsert_node(NodeInfo(
        kind="Function", name="process", file_path=main_path,
        line_start=2, line_end=5, language="python",
    ))
    store.upsert_edge(EdgeInfo(
        kind="CALLS",
        source=f"{main_path}::process",
        target=f"{auth_path}::AuthService.login",
        file_path=main_path,
        line=4,
    ))
    store.commit()
    store.close()

    result = query_graph(
        pattern="callers_of",
        target=f"{auth_path}::AuthService.login",
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    assert result["summary"] == (
        f"Found 1 result(s) for callers_of('{auth_path}::AuthService.login')"
    )
    assert [node["qualified_name"] for node in result["results"]] == [f"{main_path}::process"]
    assert [edge["source"] for edge in result["edges"]] == [f"{main_path}::process"]


def test_query_graph_importers_of_end_to_end_with_local_python_imports(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)

    imported = repo_root / "lib.py"
    imported.write_text(
        "def helper():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    importer = repo_root / "consumer.py"
    importer.write_text(
        "from lib import helper\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    full_build(repo_root, store)
    store.close()

    result = query_graph(
        pattern="importers_of",
        target="lib.py",
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    assert [row["importer"] for row in result["results"]] == [str(importer.resolve())]


def test_query_graph_tests_for_uses_test_file_and_class_heuristics(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)

    shared_dir = repo_root / "shared"
    tests_dir = shared_dir / "tests"
    agent_dir = repo_root / "agent"
    shared_dir.mkdir()
    tests_dir.mkdir(parents=True)
    agent_dir.mkdir()

    (shared_dir / "config.py").write_text(
        "class LLMConfig:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (agent_dir / "utilities.py").write_text(
        "class LLMConfig:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tests_dir / "test_llm.py").write_text(
        "from shared.config import LLMConfig\n\n"
        "class LLMConfigResolutionTests:\n"
        "    def test_from_env(self):\n"
        "        assert LLMConfig is not None\n\n"
        "    def test_for_variant(self):\n"
        "        assert LLMConfig is not None\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    full_build(repo_root, store)
    store.close()

    result = query_graph(
        pattern="tests_for",
        target="LLMConfig",
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    qualified_names = {row["qualified_name"] for row in result["results"]}
    assert qualified_names == {
        f"{(tests_dir / 'test_llm.py').resolve()}::LLMConfigResolutionTests.test_from_env",
        f"{(tests_dir / 'test_llm.py').resolve()}::LLMConfigResolutionTests.test_for_variant",
    }


def test_query_graph_tests_for_uses_test_call_edges(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)

    package_dir = repo_root / "shared"
    tests_dir = repo_root / "tests"
    package_dir.mkdir()
    tests_dir.mkdir()

    (package_dir / "__init__.py").write_text(
        "from .base import RawData\n",
        encoding="utf-8",
    )
    (package_dir / "base.py").write_text(
        "class RawData:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tests_dir / "test_raw_data.py").write_text(
        "from shared import RawData\n\n"
        "def test_build_raw_data():\n"
        "    RawData()\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    full_build(repo_root, store)
    store.close()

    result = query_graph(
        pattern="tests_for",
        target="RawData",
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    assert {row["qualified_name"] for row in result["results"]} == {
        f"{(tests_dir / 'test_raw_data.py').resolve()}::test_build_raw_data",
    }


def test_query_graph_children_of_file_excludes_nested_definitions(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)

    module_file = repo_root / "nested.py"
    module_file.write_text(
        "def outer():\n"
        "    class Inner:\n"
        "        pass\n"
        "    def helper():\n"
        "        return 1\n"
        "    return helper()\n\n"
        "class TopLevel:\n"
        "    pass\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    full_build(repo_root, store)
    store.close()

    result = query_graph(
        pattern="children_of",
        target="nested.py",
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    assert {row["name"] for row in result["results"]} == {"outer", "TopLevel"}


def test_query_graph_importers_of_handles_from_package_import_submodule(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)

    package_dir = repo_root / "main"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "settings.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (repo_root / "consumer.py").write_text(
        "from main import settings\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    full_build(repo_root, store)
    store.close()

    result = query_graph(
        pattern="importers_of",
        target="main/settings.py",
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    assert [row["file"] for row in result["results"]] == [
        str((repo_root / "consumer.py").resolve()),
    ]


def test_query_graph_tests_for_uses_python_symbol_references(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)

    package_dir = repo_root / "clients"
    tests_dir = repo_root / "tests"
    package_dir.mkdir()
    tests_dir.mkdir()

    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "factory.py").write_text(
        "class FooClient:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tests_dir / "test_factory.py").write_text(
        "from clients.factory import FooClient\n\n"
        "def test_factory_returns_client():\n"
        "    assert FooClient is not None\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    full_build(repo_root, store)
    store.close()

    result = query_graph(
        pattern="tests_for",
        target="FooClient",
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    assert {row["qualified_name"] for row in result["results"]} == {
        f"{(tests_dir / 'test_factory.py').resolve()}::test_factory_returns_client",
    }


def test_query_graph_file_summary_preserves_duplicate_definitions(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)

    module_file = repo_root / "duplicate_defs.py"
    module_file.write_text(
        "class Example:\n"
        "    def save_value(self):\n"
        "        return 1\n\n"
        "    def save_value(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    full_build(repo_root, store)
    store.close()

    result = query_graph(
        pattern="file_summary",
        target="duplicate_defs.py",
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    assert [row["kind"] for row in result["results"]] == ["File", "Class", "Function", "Function"]
    assert [row["name"] for row in result["results"] if row["kind"] == "Function"] == [
        "save_value",
        "save_value",
    ]


def test_get_review_context_limits_snippets_to_relevant_non_file_ranges(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)

    module_file = repo_root / "module.py"
    module_file.write_text(
        "\n".join(
            [f"value_{line_no} = {line_no}" for line_no in range(1, 90)]
            + [
                "",
                "def important_change():",
                "    return value_10 + value_20",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    full_build(repo_root, store)
    store.close()

    result = get_review_context(
        changed_files=["module.py"],
        max_depth=1,
        include_source=True,
        max_lines_per_file=8,
        repo_root=str(repo_root),
    )

    assert result["status"] == "ok"
    snippet = result["context"]["source_snippets"]["module.py"]
    assert "def important_change():" in snippet
    assert "1: value_1 = 1" not in snippet
    assert len([line for line in snippet.splitlines() if line != "..."]) <= 8
