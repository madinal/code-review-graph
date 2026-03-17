"""Tests for the incremental graph update module."""

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import (
    _GraphWatchCoordinator,
    _refresh_tested_by_edges,
    _is_binary,
    _load_ignore_patterns,
    _should_ignore,
    find_project_root,
    find_repo_root,
    full_build,
    get_all_tracked_files,
    get_changed_files,
    get_db_path,
    get_staged_and_unstaged,
    incremental_update,
)
from code_review_graph.parser import CodeParser


class TestFindRepoRoot:
    def test_finds_git_dir(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert find_repo_root(tmp_path) == tmp_path

    def test_finds_parent_git_dir(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert find_repo_root(sub) == tmp_path

    def test_returns_none_without_git(self, tmp_path):
        sub = tmp_path / "no_git"
        sub.mkdir()
        assert find_repo_root(sub) is None


class TestFindProjectRoot:
    def test_returns_git_root(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert find_project_root(tmp_path) == tmp_path

    def test_falls_back_to_start(self, tmp_path):
        sub = tmp_path / "no_git"
        sub.mkdir()
        assert find_project_root(sub) == sub


class TestGetDbPath:
    def test_creates_directory_and_db_path(self, tmp_path):
        db_path = get_db_path(tmp_path)
        assert db_path == tmp_path / ".code-review-graph" / "graph.db"
        assert (tmp_path / ".code-review-graph").is_dir()

    def test_creates_gitignore(self, tmp_path):
        get_db_path(tmp_path)
        gi = tmp_path / ".code-review-graph" / ".gitignore"
        assert gi.exists()
        assert "*\n" in gi.read_text()

    def test_migrates_legacy_db(self, tmp_path):
        legacy = tmp_path / ".code-review-graph.db"
        legacy.write_text("legacy data")
        db_path = get_db_path(tmp_path)
        assert db_path.exists()
        assert not legacy.exists()
        assert db_path.read_text() == "legacy data"

    def test_cleans_legacy_side_files(self, tmp_path):
        legacy = tmp_path / ".code-review-graph.db"
        legacy.write_text("data")
        for suffix in ("-wal", "-shm", "-journal"):
            (tmp_path / f".code-review-graph.db{suffix}").write_text("side")
        get_db_path(tmp_path)
        for suffix in ("-wal", "-shm", "-journal"):
            assert not (tmp_path / f".code-review-graph.db{suffix}").exists()


class TestIgnorePatterns:
    def test_default_patterns_loaded(self, tmp_path):
        patterns = _load_ignore_patterns(tmp_path)
        assert "node_modules/**" in patterns
        assert ".git/**" in patterns
        assert "__pycache__/**" in patterns

    def test_custom_ignore_file(self, tmp_path):
        ignore = tmp_path / ".code-review-graphignore"
        ignore.write_text("custom/**\n# comment\n\nvendor/**\n")
        patterns = _load_ignore_patterns(tmp_path)
        assert "custom/**" in patterns
        assert "vendor/**" in patterns
        # Comments and blanks should be skipped
        assert "# comment" not in patterns
        assert "" not in patterns

    def test_should_ignore_matches(self):
        patterns = ["node_modules/**", "*.pyc", ".git/**"]
        assert _should_ignore("node_modules/foo/bar.js", patterns)
        assert _should_ignore("test.pyc", patterns)
        assert _should_ignore(".git/HEAD", patterns)
        assert not _should_ignore("src/main.py", patterns)


class TestIsBinary:
    def test_text_file_is_not_binary(self, tmp_path):
        f = tmp_path / "text.py"
        f.write_text("print('hello')\n")
        assert not _is_binary(f)

    def test_binary_file_is_binary(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"header\x00binary data")
        assert _is_binary(f)

    def test_missing_file_is_binary(self, tmp_path):
        f = tmp_path / "missing.txt"
        assert _is_binary(f)


class TestGitOperations:
    @patch("code_review_graph.incremental.subprocess.run")
    def test_get_changed_files(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="src/a.py\nsrc/b.py\n",
        )
        result = get_changed_files(tmp_path)
        assert result == ["src/a.py", "src/b.py"]
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "git" in call_args[0][0]
        assert call_args[1].get("timeout") == 30

    @patch("code_review_graph.incremental.subprocess.run")
    def test_get_changed_files_fallback(self, mock_run, tmp_path):
        # First call fails, second succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout=""),
            MagicMock(returncode=0, stdout="staged.py\n"),
        ]
        result = get_changed_files(tmp_path)
        assert result == ["staged.py"]
        assert mock_run.call_count == 2

    @patch("code_review_graph.incremental.subprocess.run")
    def test_get_changed_files_timeout(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.TimeoutExpired("git", 30)
        result = get_changed_files(tmp_path)
        assert result == []

    @patch("code_review_graph.incremental.subprocess.run")
    def test_get_staged_and_unstaged(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" M src/a.py\n?? new.py\nR  old.py -> new_name.py\n",
        )
        result = get_staged_and_unstaged(tmp_path)
        assert "src/a.py" in result
        assert "new.py" in result
        assert "old.py" in result
        assert "new_name.py" in result

    @patch("code_review_graph.incremental.subprocess.run")
    def test_get_all_tracked_files(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="a.py\nb.py\nc.go\n",
        )
        result = get_all_tracked_files(tmp_path)
        assert result == ["a.py", "b.py", "c.go"]


class TestFullBuild:
    def test_full_build_parses_files(self, tmp_path):
        # Create a simple Python file
        py_file = tmp_path / "sample.py"
        py_file.write_text("def hello():\n    pass\n")
        (tmp_path / ".git").mkdir()

        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        try:
            mock_target = "code_review_graph.incremental.get_all_tracked_files"
            with patch(mock_target, return_value=["sample.py"]):
                result = full_build(tmp_path, store)
            assert result["files_parsed"] == 1
            assert result["total_nodes"] > 0
            assert result["errors"] == []
            assert store.get_metadata("last_build_type") == "full"
        finally:
            store.close()


class TestIncrementalUpdate:
    def test_incremental_with_no_changes(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        try:
            result = incremental_update(tmp_path, store, changed_files=[])
            assert result["files_updated"] == 0
        finally:
            store.close()

    def test_incremental_with_changed_file(self, tmp_path):
        py_file = tmp_path / "mod.py"
        py_file.write_text("def greet():\n    return 'hi'\n")

        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        try:
            result = incremental_update(
                tmp_path, store, changed_files=["mod.py"]
            )
            assert result["files_updated"] >= 1
            assert result["total_nodes"] > 0
        finally:
            store.close()

    def test_incremental_deleted_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        try:
            # Pre-populate with a file
            py_file = tmp_path / "old.py"
            py_file.write_text("x = 1\n")
            result = incremental_update(tmp_path, store, changed_files=["old.py"])
            assert result["total_nodes"] > 0

            # Now delete the file and run incremental
            py_file.unlink()
            incremental_update(tmp_path, store, changed_files=["old.py"])
            # File should have been removed from graph
            nodes = store.get_nodes_by_file(str(tmp_path / "old.py"))
            assert len(nodes) == 0
        finally:
            store.close()

    def test_incremental_update_refreshes_only_changed_and_dependent_tests(self, tmp_path):
        (tmp_path / ".git").mkdir()
        app_dir = tmp_path / "app"
        tests_dir = tmp_path / "tests"
        app_dir.mkdir()
        tests_dir.mkdir()

        (app_dir / "helpers.py").write_text(
            "def helper():\n"
            "    return 1\n",
            encoding="utf-8",
        )
        (tests_dir / "test_helper.py").write_text(
            "from app.helpers import helper\n\n"
            "def test_helper():\n"
            "    assert helper() == 1\n",
            encoding="utf-8",
        )
        (tests_dir / "test_other.py").write_text(
            "def test_other():\n"
            "    assert True\n",
            encoding="utf-8",
        )

        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        try:
            with patch("code_review_graph.incremental.get_all_tracked_files", return_value=[
                "app/helpers.py", "tests/test_helper.py", "tests/test_other.py",
            ]):
                full_build(tmp_path, store)

            (app_dir / "helpers.py").write_text(
                "def helper():\n"
                "    return 2\n",
                encoding="utf-8",
            )

            with patch("code_review_graph.incremental._refresh_tested_by_edges", return_value=0) as refresh:
                incremental_update(tmp_path, store, changed_files=["app/helpers.py"])

            refresh_candidates = refresh.call_args.kwargs["candidate_files"]
            assert str((tmp_path / "app" / "helpers.py").resolve()) in refresh_candidates
            assert str((tmp_path / "tests" / "test_helper.py").resolve()) in refresh_candidates
            assert str((tmp_path / "tests" / "test_other.py").resolve()) not in refresh_candidates
        finally:
            store.close()


def test_refresh_tested_by_edges_falls_back_to_calls_when_ast_parse_fails(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    get_db_path(repo_root)
    target_file = repo_root / "lib.py"
    test_file = repo_root / "tests" / "test_lib.py"
    test_file.parent.mkdir()

    target_file.write_text(
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    test_file.write_text(
        "from lib import helper\n\n"
        "def test_helper():\n"
        "    return helper()\n",
        encoding="utf-8",
    )

    store = GraphStore(str(get_db_path(repo_root)))
    try:
        with patch("code_review_graph.incremental.get_all_tracked_files", return_value=[
            "lib.py", "tests/test_lib.py",
        ]):
            full_build(repo_root, store)

        parser = CodeParser()
        with patch("code_review_graph.incremental.ast.parse", side_effect=SyntaxError):
            edge_count = _refresh_tested_by_edges(
                repo_root,
                store,
                parser,
                candidate_files={str(test_file.resolve())},
            )

        tested_by_edges = [
            edge for edge in store.get_edges_by_source(f"{test_file.resolve()}::test_helper")
            if edge.kind == "TESTED_BY"
        ]
        assert edge_count == 1
        assert [edge.target_qualified for edge in tested_by_edges] == [
            f"{target_file.resolve()}::helper",
        ]
    finally:
        store.close()


def test_watch_coordinator_handles_atomic_save_delete_then_move(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    module_file = repo_root / "module.py"
    module_file.write_text(
        "def original():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    store = GraphStore(repo_root / "test.db")
    try:
        with patch("code_review_graph.incremental.get_all_tracked_files", return_value=["module.py"]):
            full_build(repo_root, store)

        tmp_save_file = repo_root / ".module.py.tmp"
        tmp_save_file.write_text(
            "def original():\n"
            "    return 1\n\n"
            "def added_via_atomic_save():\n"
            "    return 2\n",
            encoding="utf-8",
        )
        tmp_save_file.replace(module_file)

        coordinator = _GraphWatchCoordinator(
            repo_root,
            store,
            CodeParser(),
            _load_ignore_patterns(repo_root),
            debounce_seconds=999,
        )
        coordinator.handle_deleted(str(module_file))
        coordinator.handle_moved(str(tmp_save_file), str(module_file))
        coordinator._flush()

        node_names = {node.name for node in store.get_nodes_by_file(str(module_file.resolve()))}
        assert "added_via_atomic_save" in node_names
        assert str(module_file.resolve()) in {
            node.qualified_name for node in store.get_nodes_by_file(str(module_file.resolve()))
        }
    finally:
        store.close()


def test_watch_coordinator_handles_true_rename(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    old_file = repo_root / "old_name.py"
    new_file = repo_root / "new_name.py"
    old_file.write_text(
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    store = GraphStore(repo_root / "test.db")
    try:
        with patch("code_review_graph.incremental.get_all_tracked_files", return_value=["old_name.py"]):
            full_build(repo_root, store)

        old_file.replace(new_file)

        coordinator = _GraphWatchCoordinator(
            repo_root,
            store,
            CodeParser(),
            _load_ignore_patterns(repo_root),
            debounce_seconds=999,
        )
        coordinator.handle_moved(str(old_file), str(new_file))
        coordinator._flush()

        assert store.get_nodes_by_file(str(old_file.resolve())) == []
        node_names = {node.name for node in store.get_nodes_by_file(str(new_file.resolve()))}
        assert "helper" in node_names
        assert str(new_file.resolve()) in {
            node.qualified_name for node in store.get_nodes_by_file(str(new_file.resolve()))
        }
    finally:
        store.close()


def test_watch_coordinator_handles_modify_then_delete(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    module_file = repo_root / "module.py"
    module_file.write_text(
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    store = GraphStore(repo_root / "test.db")
    try:
        with patch("code_review_graph.incremental.get_all_tracked_files", return_value=["module.py"]):
            full_build(repo_root, store)

        coordinator = _GraphWatchCoordinator(
            repo_root,
            store,
            CodeParser(),
            _load_ignore_patterns(repo_root),
            debounce_seconds=999,
        )
        coordinator.handle_modified(str(module_file))
        module_file.unlink()
        coordinator.handle_deleted(str(module_file))
        coordinator._flush()

        assert store.get_nodes_by_file(str(module_file.resolve())) == []
    finally:
        store.close()


def test_watch_coordinator_handles_read_failure_as_removal(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    module_file = repo_root / "module.py"
    module_file.write_text(
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    store = GraphStore(repo_root / "test.db")
    try:
        with patch("code_review_graph.incremental.get_all_tracked_files", return_value=["module.py"]):
            full_build(repo_root, store)

        coordinator = _GraphWatchCoordinator(
            repo_root,
            store,
            CodeParser(),
            _load_ignore_patterns(repo_root),
            debounce_seconds=999,
        )

        module_file.unlink()
        path_type = type(module_file)
        with patch.object(path_type, "read_bytes", side_effect=FileNotFoundError):
            with patch.object(path_type, "is_file", return_value=True):
                coordinator.handle_modified(str(module_file))
                coordinator._flush()

        assert store.get_nodes_by_file(str(module_file.resolve())) == []
    finally:
        store.close()


def test_watch_coordinator_handles_directory_move(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    old_dir = repo_root / "pkg"
    new_dir = repo_root / "pkg2"
    old_dir.mkdir()
    old_file = old_dir / "mod.py"
    old_file.write_text(
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    store = GraphStore(repo_root / "test.db")
    try:
        with patch("code_review_graph.incremental.get_all_tracked_files", return_value=["pkg/mod.py"]):
            full_build(repo_root, store)

        old_dir.replace(new_dir)
        new_file = new_dir / "mod.py"

        coordinator = _GraphWatchCoordinator(
            repo_root,
            store,
            CodeParser(),
            _load_ignore_patterns(repo_root),
            debounce_seconds=999,
        )
        coordinator.handle_directory_moved(str(old_dir), str(new_dir))
        coordinator._flush()

        assert store.get_nodes_by_file(str(old_file.resolve())) == []
        node_names = {node.name for node in store.get_nodes_by_file(str(new_file.resolve()))}
        assert "helper" in node_names
    finally:
        store.close()
