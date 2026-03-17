"""Tests for the Tree-sitter parser module."""

from pathlib import Path

from code_review_graph.parser import CodeParser, NodeInfo, EdgeInfo

FIXTURES = Path(__file__).parent / "fixtures"


class TestCodeParser:
    def setup_method(self):
        self.parser = CodeParser()

    def test_detect_language_python(self):
        assert self.parser.detect_language(Path("foo.py")) == "python"

    def test_detect_language_typescript(self):
        assert self.parser.detect_language(Path("foo.ts")) == "typescript"

    def test_detect_language_unknown(self):
        assert self.parser.detect_language(Path("foo.txt")) is None

    def test_parse_python_file(self):
        nodes, edges = self.parser.parse_file(FIXTURES / "sample_python.py")

        # Should have File node
        file_nodes = [n for n in nodes if n.kind == "File"]
        assert len(file_nodes) == 1

        # Should find classes
        classes = [n for n in nodes if n.kind == "Class"]
        class_names = {c.name for c in classes}
        assert "BaseService" in class_names
        assert "AuthService" in class_names

        # Should find functions
        funcs = [n for n in nodes if n.kind == "Function"]
        func_names = {f.name for f in funcs}
        assert "__init__" in func_names
        assert "authenticate" in func_names
        assert "create_auth_service" in func_names
        assert "process_request" in func_names

    def test_parse_python_edges(self):
        nodes, edges = self.parser.parse_file(FIXTURES / "sample_python.py")

        edge_kinds = {e.kind for e in edges}
        assert "CONTAINS" in edge_kinds
        assert "IMPORTS_FROM" in edge_kinds
        assert "CALLS" in edge_kinds

        # Should detect inheritance
        inherits = [e for e in edges if e.kind == "INHERITS"]
        assert len(inherits) >= 1
        assert any("AuthService" in e.source and "BaseService" in e.target for e in inherits)

    def test_parse_python_imports(self):
        nodes, edges = self.parser.parse_file(FIXTURES / "sample_python.py")
        imports = [e for e in edges if e.kind == "IMPORTS_FROM"]
        import_targets = {e.target for e in imports}
        assert "os" in import_targets
        assert "pathlib" in import_targets

    def test_parse_python_calls(self):
        nodes, edges = self.parser.parse_file(FIXTURES / "sample_python.py")
        calls = [e for e in edges if e.kind == "CALLS"]
        call_targets = {e.target for e in calls}
        assert "_validate_token" in call_targets
        assert "authenticate" in call_targets

    def test_parse_typescript_file(self):
        nodes, edges = self.parser.parse_file(FIXTURES / "sample_typescript.ts")

        classes = [n for n in nodes if n.kind == "Class"]
        class_names = {c.name for c in classes}
        assert "UserRepository" in class_names
        assert "UserService" in class_names

        funcs = [n for n in nodes if n.kind == "Function"]
        func_names = {f.name for f in funcs}
        assert "findById" in func_names or "handleGetUser" in func_names

    def test_parse_test_file(self):
        nodes, edges = self.parser.parse_file(FIXTURES / "test_sample.py")

        # Test functions should be detected
        tests = [n for n in nodes if n.kind == "Test"]
        test_names = {t.name for t in tests}
        assert "test_authenticate_valid" in test_names
        assert "test_process_request_ok" in test_names

    def test_calls_edge_same_file_resolution(self):
        """Call targets defined in the same file should be qualified."""
        nodes, edges = self.parser.parse_file(FIXTURES / "sample_python.py")
        calls = [e for e in edges if e.kind == "CALLS"]
        file_path = str(FIXTURES / "sample_python.py")

        # create_auth_service() calls AuthService() — a class defined in the same file
        auth_service_calls = [
            e for e in calls if e.target == f"{file_path}::AuthService"
        ]
        assert len(auth_service_calls) >= 1

    def test_calls_edge_cross_file_resolution(self):
        """Call targets imported from another file should resolve to that file's qualified name."""
        _, edges = self.parser.parse_file(FIXTURES / "caller_example.py")
        calls = [e for e in edges if e.kind == "CALLS"]

        sample_path = str((FIXTURES / "sample_python.py").resolve())
        # setup_and_run() calls create_auth_service(), imported from sample_python
        resolved_calls = [
            e for e in calls if e.target == f"{sample_path}::create_auth_service"
        ]
        assert len(resolved_calls) == 1

    def test_unresolved_calls_stay_bare(self):
        """Method calls and unknown calls should remain as bare names."""
        _, edges = self.parser.parse_file(FIXTURES / "sample_python.py")
        calls = [e for e in edges if e.kind == "CALLS"]
        # self._validate_token() is a method call — can't resolve the target file
        bare_calls = [e for e in calls if e.target == "_validate_token"]
        assert len(bare_calls) >= 1

    def test_calls_edge_decorated_function_resolution(self):
        """Decorated functions should be in defined_names and resolvable as call targets."""
        _, edges = self.parser.parse_file(FIXTURES / "sample_python.py")
        calls = [e for e in edges if e.kind == "CALLS"]
        file_path = str(FIXTURES / "sample_python.py")

        # guarded_process() calls process_request() — both in the same file,
        # but guarded_process is wrapped in a decorated_definition node
        resolved = [e for e in calls if e.target == f"{file_path}::process_request"
                    and "guarded_process" in e.source]
        assert len(resolved) == 1

    def test_multiple_calls_to_same_function(self):
        """Multiple calls to the same function on different lines should each produce an edge."""
        _, edges = self.parser.parse_file(FIXTURES / "multi_call_example.py")
        calls = [e for e in edges if e.kind == "CALLS" and "_internal_request" in e.target]
        assert len(calls) == 2
        lines = {e.line for e in calls}
        assert len(lines) == 2  # distinct line numbers

    def test_parse_nonexistent_file(self):
        nodes, edges = self.parser.parse_file(Path("/nonexistent/file.py"))
        assert nodes == []
        assert edges == []

    def test_parse_unsupported_extension(self):
        nodes, edges = self.parser.parse_file(Path("readme.txt"))
        assert nodes == []
        assert edges == []


def test_calls_edge_cross_file_resolution_preserves_original_name_for_alias(tmp_path):
    target_file = tmp_path / "alias_target.py"
    target_file.write_text(
        "def helper():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    caller_file = tmp_path / "alias_caller.py"
    caller_file.write_text(
        "from alias_target import helper as renamed\n\n"
        "def caller():\n"
        "    return renamed()\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(caller_file)
    calls = [e for e in edges if e.kind == "CALLS"]

    assert any(e.target == f"{target_file.resolve()}::helper" for e in calls)


def test_calls_edge_cross_file_resolution_for_module_alias_member_call(tmp_path):
    target_file = tmp_path / "module_target.py"
    target_file.write_text(
        "def helper():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    caller_file = tmp_path / "module_alias_caller.py"
    caller_file.write_text(
        "import module_target as mod\n\n"
        "def caller():\n"
        "    return mod.helper()\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(caller_file)
    calls = [e for e in edges if e.kind == "CALLS"]

    assert any(e.target == f"{target_file.resolve()}::helper" for e in calls)


def test_calls_edge_relative_import_resolution(tmp_path):
    pkg_dir = tmp_path / "relative_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    helper_file = pkg_dir / "helpers.py"
    helper_file.write_text(
        "def helper():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    caller_file = pkg_dir / "caller.py"
    caller_file.write_text(
        "from .helpers import helper\n\n"
        "def caller():\n"
        "    return helper()\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(caller_file)
    calls = [e for e in edges if e.kind == "CALLS"]

    assert any(e.target == f"{helper_file.resolve()}::helper" for e in calls)


def test_module_resolution_cache_is_scoped_per_parse(tmp_path):
    pkg_a = tmp_path / "pkg_a"
    pkg_b = tmp_path / "pkg_b"
    pkg_a.mkdir()
    pkg_b.mkdir()

    helper_a = pkg_a / "helpers.py"
    helper_a.write_text(
        "def helper():\n"
        "    return 'a'\n",
        encoding="utf-8",
    )
    helper_b = pkg_b / "helpers.py"
    helper_b.write_text(
        "def helper():\n"
        "    return 'b'\n",
        encoding="utf-8",
    )

    caller_a = pkg_a / "caller.py"
    caller_a.write_text(
        "from helpers import helper\n\n"
        "def caller():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    caller_b = pkg_b / "caller.py"
    caller_b.write_text(
        "from helpers import helper\n\n"
        "def caller():\n"
        "    return helper()\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    parser.parse_file(caller_a)
    _, edges = parser.parse_file(caller_b)
    calls = [e for e in edges if e.kind == "CALLS"]

    assert any(e.target == f"{helper_b.resolve()}::helper" for e in calls)


def test_calls_edge_resolution_uses_function_local_imports(tmp_path):
    target_file = tmp_path / "resolver.py"
    target_file.write_text(
        "def resolve_content():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    caller_file = tmp_path / "tools.py"
    caller_file.write_text(
        "def outer():\n"
        "    def inner():\n"
        "        from resolver import resolve_content\n"
        "        return resolve_content()\n"
        "    return inner()\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(caller_file)
    calls = [e for e in edges if e.kind == "CALLS"]

    assert any(e.target == f"{target_file.resolve()}::resolve_content" for e in calls)


def test_import_edges_include_aliased_import_statements_and_local_file_targets(tmp_path):
    target_file = tmp_path / "local_mod.py"
    target_file.write_text(
        "def helper():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    caller_file = tmp_path / "imports.py"
    caller_file.write_text(
        "import redis.asyncio as aioredis\n"
        "import local_mod as mod\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(caller_file)
    imports = [e.target for e in edges if e.kind == "IMPORTS_FROM"]

    assert "redis.asyncio" in imports
    assert str(target_file.resolve()) in imports


def test_import_edges_include_future_import_statements(tmp_path):
    module_file = tmp_path / "future_example.py"
    module_file.write_text(
        "from __future__ import annotations\n"
        "import os\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(module_file)
    imports = [e.target for e in edges if e.kind == "IMPORTS_FROM"]

    assert "__future__" in imports
    assert "os" in imports


def test_absolute_import_does_not_resolve_to_current_file(tmp_path):
    module_file = tmp_path / "redis.py"
    module_file.write_text(
        "import redis\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(module_file)
    imports = [e.target for e in edges if e.kind == "IMPORTS_FROM"]

    assert imports == ["redis"]


def test_calls_edge_resolution_follows_package_re_exports(tmp_path):
    pkg_dir = tmp_path / "shared"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "from .base import RawData\n",
        encoding="utf-8",
    )
    base_file = pkg_dir / "base.py"
    base_file.write_text(
        "class RawData:\n"
        "    pass\n",
        encoding="utf-8",
    )
    caller_file = tmp_path / "consumer.py"
    caller_file.write_text(
        "from shared import RawData\n\n"
        "def build():\n"
        "    return RawData()\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(caller_file)
    calls = [e for e in edges if e.kind == "CALLS"]

    assert any(e.target == f"{base_file.resolve()}::RawData" for e in calls)


def test_absolute_import_resolution_is_case_sensitive(tmp_path):
    package_dir = tmp_path / "Newspaper"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    module_file = tmp_path / "consumer.py"
    module_file.write_text(
        "from newspaper import Article\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(module_file)
    imports = [e.target for e in edges if e.kind == "IMPORTS_FROM"]

    assert imports == ["newspaper"]


def test_from_import_of_local_submodule_resolves_to_submodule_file(tmp_path):
    pkg_dir = tmp_path / "main"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    settings_file = pkg_dir / "settings.py"
    settings_file.write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    module_file = tmp_path / "consumer.py"
    module_file.write_text(
        "from main import settings\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(module_file)
    imports = [e.target for e in edges if e.kind == "IMPORTS_FROM"]

    assert imports == [str(settings_file.resolve())]


def test_absolute_imports_use_repo_source_roots_before_nested_ancestors(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    app_dir = repo_root / "app"
    spider_dir = app_dir / "scrapy_projects" / "scrapy" / "generic_scrapy" / "generic_scrapy" / "spiders"
    spider_dir.mkdir(parents=True)
    (app_dir / "main").mkdir()
    (app_dir / "main" / "__init__.py").write_text("", encoding="utf-8")
    settings_file = app_dir / "main" / "settings.py"
    settings_file.write_text("VALUE = 1\n", encoding="utf-8")
    nested_scrapy = app_dir / "scrapy_projects" / "scrapy" / "__init__.py"
    nested_scrapy.parent.mkdir(parents=True, exist_ok=True)
    nested_scrapy.write_text("", encoding="utf-8")

    module_file = spider_dir / "generic_spider.py"
    module_file.write_text(
        "import scrapy\n"
        "from main import settings\n",
        encoding="utf-8",
    )

    parser = CodeParser()
    _, edges = parser.parse_file(module_file)
    imports = [e.target for e in edges if e.kind == "IMPORTS_FROM"]

    assert "scrapy" in imports
    assert str(settings_file.resolve()) in imports
    assert str(nested_scrapy.resolve()) not in imports
