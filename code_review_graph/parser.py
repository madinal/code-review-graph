"""Tree-sitter based multi-language code parser.

Extracts structural nodes (classes, functions, imports, types) and edges
(calls, inheritance, contains) from source files.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tree_sitter_language_pack as tslp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models for extracted entities
# ---------------------------------------------------------------------------


@dataclass
class NodeInfo:
    kind: str  # File, Class, Function, Type, Test
    name: str
    file_path: str
    line_start: int
    line_end: int
    language: str = ""
    parent_name: Optional[str] = None  # enclosing class/module
    params: Optional[str] = None
    return_type: Optional[str] = None
    modifiers: Optional[str] = None
    is_test: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class EdgeInfo:
    kind: str  # CALLS, IMPORTS_FROM, INHERITS, IMPLEMENTS, CONTAINS, TESTED_BY, DEPENDS_ON
    source: str  # qualified name or path
    target: str  # qualified name or path
    file_path: str
    line: int = 0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ImportBinding:
    """A locally bound import and the symbol/module it points to."""

    module: str
    exported_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Language extension mapping
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".php": "php",
}

# Tree-sitter node type mappings per language
# Maps (language) -> dict of semantic role -> list of TS node types
_CLASS_TYPES: dict[str, list[str]] = {
    "python": ["class_definition"],
    "javascript": ["class_declaration", "class"],
    "typescript": ["class_declaration", "class"],
    "tsx": ["class_declaration", "class"],
    "go": ["type_declaration"],
    "rust": ["struct_item", "enum_item", "impl_item"],
    "java": ["class_declaration", "interface_declaration", "enum_declaration"],
    "c": ["struct_specifier", "type_definition"],
    "cpp": ["class_specifier", "struct_specifier"],
    "csharp": [
        "class_declaration", "interface_declaration",
        "enum_declaration", "struct_declaration",
    ],
    "ruby": ["class", "module"],
    "kotlin": ["class_declaration", "object_declaration"],
    "swift": ["class_declaration", "struct_declaration", "protocol_declaration"],
    "php": ["class_declaration", "interface_declaration"],
}

_FUNCTION_TYPES: dict[str, list[str]] = {
    "python": ["function_definition"],
    "javascript": ["function_declaration", "method_definition", "arrow_function"],
    "typescript": ["function_declaration", "method_definition", "arrow_function"],
    "tsx": ["function_declaration", "method_definition", "arrow_function"],
    "go": ["function_declaration", "method_declaration"],
    "rust": ["function_item"],
    "java": ["method_declaration", "constructor_declaration"],
    "c": ["function_definition"],
    "cpp": ["function_definition"],
    "csharp": ["method_declaration", "constructor_declaration"],
    "ruby": ["method", "singleton_method"],
    "kotlin": ["function_declaration"],
    "swift": ["function_declaration"],
    "php": ["function_definition", "method_declaration"],
}

_IMPORT_TYPES: dict[str, list[str]] = {
    "python": ["import_statement", "import_from_statement", "future_import_statement"],
    "javascript": ["import_statement"],
    "typescript": ["import_statement"],
    "tsx": ["import_statement"],
    "go": ["import_declaration"],
    "rust": ["use_declaration"],
    "java": ["import_declaration"],
    "c": ["preproc_include"],
    "cpp": ["preproc_include"],
    "csharp": ["using_directive"],
    "ruby": ["call"],  # require/require_relative
    "kotlin": ["import_header"],
    "swift": ["import_declaration"],
    "php": ["namespace_use_declaration"],
}

_CALL_TYPES: dict[str, list[str]] = {
    "python": ["call"],
    "javascript": ["call_expression", "new_expression"],
    "typescript": ["call_expression", "new_expression"],
    "tsx": ["call_expression", "new_expression"],
    "go": ["call_expression"],
    "rust": ["call_expression", "macro_invocation"],
    "java": ["method_invocation", "object_creation_expression"],
    "c": ["call_expression"],
    "cpp": ["call_expression"],
    "csharp": ["invocation_expression", "object_creation_expression"],
    "ruby": ["call", "method_call"],
    "kotlin": ["call_expression"],
    "swift": ["call_expression"],
    "php": ["function_call_expression", "member_call_expression"],
}

# Patterns that indicate a test function
_TEST_PATTERNS = [
    re.compile(r"^test_"),
    re.compile(r"^Test"),
    re.compile(r"_test$"),
    re.compile(r"\.test\."),
    re.compile(r"\.spec\."),
    re.compile(r"_spec$"),
]

_TEST_FILE_PATTERNS = [
    re.compile(r"test_.*\.py$"),
    re.compile(r".*_test\.py$"),
    re.compile(r".*\.test\.[jt]sx?$"),
    re.compile(r".*\.spec\.[jt]sx?$"),
    re.compile(r".*_test\.go$"),
    re.compile(r"tests?/"),
]


def _is_test_file(path: str) -> bool:
    return any(p.search(path) for p in _TEST_FILE_PATTERNS)


def _is_test_function(name: str, file_path: str) -> bool:
    """A function is a test only if its name matches test patterns.
    Being in a test file alone is not sufficient (test files contain helpers too).
    """
    return any(p.search(name) for p in _TEST_PATTERNS)


def file_hash(path: Path) -> str:
    """SHA-256 hash of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class CodeParser:
    """Parses source files using Tree-sitter and extracts structural information."""

    def __init__(self) -> None:
        self._parsers: dict[str, object] = {}
        self._module_file_cache: dict[str, Optional[str]] = {}
        self._file_scope_cache: dict[tuple[str, str], tuple[dict[str, ImportBinding], set[str]]] = {}

    def _get_parser(self, language: str):  # type: ignore[arg-type]
        if language not in self._parsers:
            try:
                self._parsers[language] = tslp.get_parser(language)  # type: ignore[arg-type]
            except Exception:
                return None
        return self._parsers[language]

    def detect_language(self, path: Path) -> Optional[str]:
        return EXTENSION_TO_LANGUAGE.get(path.suffix.lower())

    def parse_file(self, path: Path) -> tuple[list[NodeInfo], list[EdgeInfo]]:
        """Parse a single file and return extracted nodes and edges."""
        try:
            source = path.read_bytes()
        except (OSError, PermissionError):
            return [], []
        return self.parse_bytes(path, source)

    def parse_bytes(self, path: Path, source: bytes) -> tuple[list[NodeInfo], list[EdgeInfo]]:
        """Parse pre-read bytes and return extracted nodes and edges.

        This avoids re-reading the file from disk, eliminating TOCTOU gaps
        when the caller has already read the bytes (e.g. for hashing).
        """
        # Module resolution depends on the caller being parsed, so keep this
        # cache scoped to a single parse session.
        self._module_file_cache.clear()
        self._file_scope_cache.clear()

        language = self.detect_language(path)
        if not language:
            return [], []

        parser = self._get_parser(language)
        if not parser:
            return [], []

        tree = parser.parse(source)
        nodes: list[NodeInfo] = []
        edges: list[EdgeInfo] = []
        file_path_str = str(path)

        # File node
        nodes.append(NodeInfo(
            kind="File",
            name=file_path_str,
            file_path=file_path_str,
            line_start=1,
            line_end=source.count(b"\n") + 1,
            language=language,
        ))

        # Pre-scan for import mappings and defined names
        import_map, defined_names = self._collect_file_scope(
            tree.root_node, language, source,
        )

        # Walk the tree
        self._extract_from_tree(
            tree.root_node, source, language, file_path_str, nodes, edges,
            import_map=import_map, defined_names=defined_names,
        )

        return nodes, edges

    def _extract_from_tree(
        self,
        root,
        source: bytes,
        language: str,
        file_path: str,
        nodes: list[NodeInfo],
        edges: list[EdgeInfo],
        enclosing_class: Optional[str] = None,
        enclosing_func: Optional[str] = None,
        import_map: Optional[dict[str, ImportBinding]] = None,
        defined_names: Optional[set[str]] = None,
    ) -> None:
        """Recursively walk the AST and extract nodes/edges."""
        class_types = set(_CLASS_TYPES.get(language, []))
        func_types = set(_FUNCTION_TYPES.get(language, []))
        import_types = set(_IMPORT_TYPES.get(language, []))
        call_types = set(_CALL_TYPES.get(language, []))
        current_import_map = dict(import_map or {})

        for child in root.children:
            node_type = child.type

            # --- Classes ---
            if node_type in class_types:
                name = self._get_name(child, language, "class")
                if name:
                    node = NodeInfo(
                        kind="Class",
                        name=name,
                        file_path=file_path,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language=language,
                        parent_name=enclosing_class,
                    )
                    nodes.append(node)

                    # CONTAINS edge
                    edges.append(EdgeInfo(
                        kind="CONTAINS",
                        source=file_path,
                        target=self._qualify(name, file_path, enclosing_class),
                        file_path=file_path,
                        line=child.start_point[0] + 1,
                    ))

                    # Inheritance edges
                    bases = self._get_bases(child, language, source)
                    for base in bases:
                        edges.append(EdgeInfo(
                            kind="INHERITS",
                            source=self._qualify(name, file_path, enclosing_class),
                            target=base,
                            file_path=file_path,
                            line=child.start_point[0] + 1,
                        ))

                    # Recurse into class body
                    self._extract_from_tree(
                        child, source, language, file_path, nodes, edges,
                        enclosing_class=name, enclosing_func=None,
                        import_map=current_import_map, defined_names=defined_names,
                    )
                    continue

            # --- Functions ---
            if node_type in func_types:
                name = self._get_name(child, language, "function")
                if name:
                    is_test = _is_test_function(name, file_path)
                    kind = "Test" if is_test else "Function"
                    qualified = self._qualify(name, file_path, enclosing_class)
                    params = self._get_params(child, language, source)
                    ret_type = self._get_return_type(child, language, source)

                    node = NodeInfo(
                        kind=kind,
                        name=name,
                        file_path=file_path,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language=language,
                        parent_name=enclosing_class,
                        params=params,
                        return_type=ret_type,
                        is_test=is_test,
                    )
                    nodes.append(node)

                    # CONTAINS edge
                    container = (
                        self._qualify(enclosing_class, file_path, None)
                        if enclosing_class
                        else file_path
                    )
                    edges.append(EdgeInfo(
                        kind="CONTAINS",
                        source=container,
                        target=qualified,
                        file_path=file_path,
                        line=child.start_point[0] + 1,
                    ))

                    # Recurse to find calls inside the function
                    self._extract_from_tree(
                        child, source, language, file_path, nodes, edges,
                        enclosing_class=enclosing_class, enclosing_func=name,
                        import_map=current_import_map, defined_names=defined_names,
                    )
                    continue

            # --- Imports ---
            if node_type in import_types:
                self._collect_import_names(child, language, source, current_import_map)
                if language == "python" and node_type in ("import_from_statement", "future_import_statement"):
                    imports = self._resolve_python_import_targets(
                        child, file_path, language, source,
                    )
                else:
                    imports = self._extract_import(child, language, source)
                for imp_target in imports:
                    resolved_target = imp_target
                    if not (language == "python" and node_type in ("import_from_statement", "future_import_statement")):
                        resolved_target = self._resolve_module_to_file(
                            imp_target, file_path, language,
                        ) or imp_target
                    edges.append(EdgeInfo(
                        kind="IMPORTS_FROM",
                        source=file_path,
                        target=resolved_target,
                        file_path=file_path,
                        line=child.start_point[0] + 1,
                    ))
                continue

            # --- Calls ---
            if node_type in call_types:
                call_ref = self._get_call_reference(child, language, source)
                if call_ref and enclosing_func:
                    caller = self._qualify(enclosing_func, file_path, enclosing_class)
                    target = self._resolve_call_target(
                        call_ref, file_path, language,
                        current_import_map, defined_names or set(),
                    )
                    edges.append(EdgeInfo(
                        kind="CALLS",
                        source=caller,
                        target=target,
                        file_path=file_path,
                        line=child.start_point[0] + 1,
                    ))

            # Recurse for other node types
            self._extract_from_tree(
                child, source, language, file_path, nodes, edges,
                enclosing_class=enclosing_class, enclosing_func=enclosing_func,
                import_map=current_import_map, defined_names=defined_names,
            )

    def _collect_file_scope(
        self, root, language: str, source: bytes,
    ) -> tuple[dict[str, ImportBinding], set[str]]:
        """Pre-scan top-level AST to collect import mappings and defined names.

        Returns:
            (import_map, defined_names) where import_map maps imported names
            to their source module/path, and defined_names is the set of
            function/class names defined at file scope.
        """
        import_map: dict[str, ImportBinding] = {}
        defined_names: set[str] = set()

        class_types = set(_CLASS_TYPES.get(language, []))
        func_types = set(_FUNCTION_TYPES.get(language, []))
        import_types = set(_IMPORT_TYPES.get(language, []))

        # Node types that wrap a class/function with decorators/annotations
        decorator_wrappers = {"decorated_definition", "decorator"}

        for child in root.children:
            node_type = child.type

            # Unwrap decorator wrappers to reach the inner definition
            target = child
            if node_type in decorator_wrappers:
                for inner in child.children:
                    if inner.type in func_types or inner.type in class_types:
                        target = inner
                        break

            target_type = target.type

            # Collect defined function/class names
            if target_type in func_types or target_type in class_types:
                name = self._get_name(target, language,
                                      "class" if target_type in class_types else "function")
                if name:
                    defined_names.add(name)

            # Collect import mappings: imported_name → module_path
            if node_type in import_types:
                self._collect_import_names(child, language, source, import_map)

        return import_map, defined_names

    def _collect_import_names(
        self, node, language: str, source: bytes, import_map: dict[str, ImportBinding],
    ) -> None:
        """Extract imported names and their source modules into import_map."""
        if language == "python":
            seen_import_keyword = False
            if node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        module = child.text.decode("utf-8", errors="replace")
                        local_name = module.split(".", 1)[0]
                        import_map[local_name] = ImportBinding(module=module)
                    elif child.type == "aliased_import":
                        module = None
                        alias = None
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                module = sub.text.decode("utf-8", errors="replace")
                            elif sub.type == "identifier":
                                alias = sub.text.decode("utf-8", errors="replace")
                        if module and alias:
                            import_map[alias] = ImportBinding(module=module)

            if node.type == "import_from_statement":
                # from X.Y import A, B → {A: X.Y, B: X.Y}
                module = None
                seen_import_keyword = False
                for child in node.children:
                    if child.type in ("dotted_name", "relative_import") and not seen_import_keyword:
                        module = child.text.decode("utf-8", errors="replace")
                    elif child.type == "import":
                        seen_import_keyword = True
                    elif seen_import_keyword and module:
                        if child.type in ("identifier", "dotted_name"):
                            name = child.text.decode("utf-8", errors="replace")
                            import_map[name] = self._make_python_import_binding(module, name)
                        elif child.type == "aliased_import":
                            imported_name = None
                            alias = None
                            for sub in child.children:
                                if sub.type in ("identifier", "dotted_name"):
                                    if imported_name is None:
                                        imported_name = sub.text.decode("utf-8", errors="replace")
                                    else:
                                        alias = sub.text.decode("utf-8", errors="replace")
                            if imported_name and alias:
                                import_map[alias] = self._make_python_import_binding(
                                    module, imported_name,
                                )

        elif language in ("javascript", "typescript", "tsx"):
            # import { A, B } from './path' → {A: ./path, B: ./path}
            module = None
            for child in node.children:
                if child.type == "string":
                    module = child.text.decode("utf-8", errors="replace").strip("'\"")
            if module:
                for child in node.children:
                    if child.type == "import_clause":
                        self._collect_js_import_names(child, module, import_map)

    def _collect_js_import_names(
        self, clause_node, module: str, import_map: dict[str, ImportBinding],
    ) -> None:
        """Walk JS/TS import_clause to extract named and default imports."""
        for child in clause_node.children:
            if child.type == "identifier":
                # Default import
                name = child.text.decode("utf-8", errors="replace")
                import_map[name] = ImportBinding(module=module)
            elif child.type == "named_imports":
                for spec in child.children:
                    if spec.type == "import_specifier":
                        # Could be: name or name as alias
                        names = [
                            s.text.decode("utf-8", errors="replace")
                            for s in spec.children
                            if s.type in ("identifier", "property_identifier")
                        ]
                        # Last identifier is the local name
                        if names:
                            import_map[names[-1]] = ImportBinding(
                                module=module,
                                exported_name=names[0],
                            )

    def _make_python_import_binding(self, module: str, imported_name: str) -> ImportBinding:
        """Build a Python import binding from a module and imported name."""
        if module.replace(".", "") == "":
            return ImportBinding(module=f"{module}{imported_name}")
        return ImportBinding(module=module, exported_name=imported_name)

    def _get_file_scope(
        self, file_path: str, language: str,
    ) -> tuple[dict[str, ImportBinding], set[str]]:
        """Return cached top-level imports and definitions for a source file."""
        cache_key = (str(Path(file_path).resolve()), language)
        if cache_key in self._file_scope_cache:
            return self._file_scope_cache[cache_key]

        path = Path(file_path)
        try:
            source = path.read_bytes()
        except OSError:
            scope = ({}, set())
            self._file_scope_cache[cache_key] = scope
            return scope

        parser = self._get_parser(language)
        if not parser:
            scope = ({}, set())
            self._file_scope_cache[cache_key] = scope
            return scope

        tree = parser.parse(source)
        scope = self._collect_file_scope(tree.root_node, language, source)
        self._file_scope_cache[cache_key] = scope
        return scope

    def _resolve_exported_symbol(
        self,
        module: str,
        exported_name: str,
        file_path: str,
        language: str,
        seen: Optional[set[tuple[str, str]]] = None,
    ) -> Optional[str]:
        """Resolve an imported symbol, following Python package re-exports."""
        resolved = self._resolve_module_to_file(module, file_path, language)
        if not resolved:
            return None

        if seen is None:
            seen = set()

        key = (resolved, exported_name)
        if key in seen:
            return None
        seen.add(key)

        import_map, defined_names = self._get_file_scope(resolved, language)
        if exported_name in defined_names:
            return self._qualify(exported_name, resolved, None)

        rebound = import_map.get(exported_name)
        if rebound and rebound.exported_name:
            return self._resolve_exported_symbol(
                rebound.module, rebound.exported_name, resolved, language, seen,
            )

        return None

    def _resolve_existing_path(self, base_dir: Path, relative_path: Path) -> Optional[Path]:
        """Resolve a relative path using exact path-component casing."""
        current = base_dir.resolve()
        for part in relative_path.parts:
            if part in ("", "."):
                continue
            if part == "..":
                current = current.parent
                continue
            try:
                children = {child.name: child for child in current.iterdir()}
            except OSError:
                return None
            current = children.get(part)
            if current is None:
                return None
        return current

    def _python_import_roots(self, file_path: str) -> list[Path]:
        """Find likely Python source roots for absolute imports in a repo."""
        caller_file = Path(file_path).resolve()
        repo_root = None
        for parent in caller_file.parents:
            if (parent / ".git").exists() or (parent / ".code-review-graph").exists():
                repo_root = parent
                break

        if repo_root is None:
            return []

        roots = []
        for child in ("app", "src", "lib"):
            candidate = repo_root / child
            if candidate.is_dir():
                roots.append(candidate)
        roots.append(repo_root)

        seen = set()
        ordered = []
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)
        return ordered

    def _resolve_python_import_targets(
        self, node, file_path: str, language: str, source: bytes,
    ) -> list[str]:
        """Resolve Python ``from X import Y`` edges to local submodules when possible."""
        imports = []
        base_module = None
        seen_import_keyword = False

        for child in node.children:
            if child.type in ("dotted_name", "relative_import", "__future__") and not seen_import_keyword:
                base_module = child.text.decode("utf-8", errors="replace")
            elif child.type == "import":
                seen_import_keyword = True
            elif seen_import_keyword:
                imported_name = None
                if child.type in ("identifier", "dotted_name"):
                    imported_name = child.text.decode("utf-8", errors="replace")
                elif child.type == "aliased_import":
                    for sub in child.children:
                        if sub.type in ("identifier", "dotted_name"):
                            imported_name = sub.text.decode("utf-8", errors="replace")
                            break

                if imported_name and base_module:
                    candidate = self._join_module_path(base_module, imported_name)
                    resolved_candidate = self._resolve_module_to_file(
                        candidate, file_path, language,
                    )
                    if resolved_candidate:
                        imports.append(resolved_candidate)
                    else:
                        resolved_base = self._resolve_module_to_file(
                            base_module, file_path, language,
                        )
                        imports.append(resolved_base or base_module)

        if imports:
            return imports

        return self._extract_import(node, language, source)

    def _resolve_module_to_file(
        self, module: str, file_path: str, language: str,
    ) -> Optional[str]:
        """Resolve a module/import path to an absolute file path.

        Uses self._module_file_cache to avoid repeated filesystem lookups.
        """
        cache_key = f"{language}:{Path(file_path).resolve()}:{module}"
        if cache_key in self._module_file_cache:
            return self._module_file_cache[cache_key]

        resolved = self._do_resolve_module(module, file_path, language)
        self._module_file_cache[cache_key] = resolved
        return resolved

    def _do_resolve_module(
        self, module: str, file_path: str, language: str,
    ) -> Optional[str]:
        """Language-aware module-to-file resolution."""
        caller_dir = Path(file_path).parent
        caller_file = Path(file_path).resolve()

        if language == "python":
            leading_dots = len(module) - len(module.lstrip("."))
            module_name = module.lstrip(".")

            rel_parts = [p for p in module_name.split(".") if p]
            candidates = []
            if rel_parts:
                rel_path = Path(*rel_parts)
                candidates = [rel_path.with_suffix(".py"), rel_path / "__init__.py"]

            if leading_dots:
                current = caller_dir
                for _ in range(max(leading_dots - 1, 0)):
                    if current == current.parent:
                        break
                    current = current.parent
                for candidate in candidates:
                    target = self._resolve_existing_path(current, candidate)
                    if target and target.is_file():
                        return str(target)
            else:
                import_roots = self._python_import_roots(file_path)
                if import_roots:
                    for root in import_roots:
                        for candidate in candidates:
                            target = self._resolve_existing_path(root, candidate)
                            if target and target.is_file() and target.resolve() != caller_file:
                                return str(target)

                    designated_roots = [
                        root for root in import_roots if root.name in ("app", "src", "lib")
                    ]
                    if designated_roots and any(
                        root == caller_dir or root in caller_dir.parents for root in designated_roots
                    ):
                        return None

                # Fall back to ancestor walk when we cannot infer a repo root,
                # or when repo-level roots do not contain the imported module.
                current = caller_dir
                while True:
                    for candidate in candidates:
                        target = self._resolve_existing_path(current, candidate)
                        if target and target.is_file() and target.resolve() != caller_file:
                            return str(target)
                    if current == current.parent:
                        break
                    current = current.parent

        elif language in ("javascript", "typescript", "tsx"):
            if module.startswith("."):
                # Relative import — resolve from caller's directory
                base = caller_dir / module
                extensions = [".ts", ".tsx", ".js", ".jsx"]
                # Try exact path first (might already have extension)
                if base.is_file():
                    return str(base.resolve())
                # Try with extensions
                for ext in extensions:
                    target = base.with_suffix(ext)
                    if target.is_file():
                        return str(target.resolve())
                # Try index file in directory
                if base.is_dir():
                    for ext in extensions:
                        target = base / f"index{ext}"
                        if target.is_file():
                            return str(target.resolve())

        return None

    def _resolve_call_target(
        self,
        call_ref: list[str],
        file_path: str,
        language: str,
        import_map: dict[str, ImportBinding],
        defined_names: set[str],
    ) -> str:
        """Resolve a bare call name to a qualified target, with fallback."""
        leaf_name = call_ref[-1]

        if len(call_ref) == 1:
            if leaf_name in defined_names:
                return self._qualify(leaf_name, file_path, None)

            binding = import_map.get(leaf_name)
            if binding and binding.exported_name:
                resolved = self._resolve_exported_symbol(
                    binding.module, binding.exported_name, file_path, language,
                )
                if resolved:
                    return resolved

            return leaf_name

        base_name = call_ref[0]
        binding = import_map.get(base_name)
        if not binding:
            return leaf_name

        attribute_parts = self._trim_module_prefix(
            binding.module, base_name, call_ref[1:],
        )
        if not attribute_parts:
            return leaf_name

        module = binding.module
        if binding.exported_name:
            module = self._join_module_path(binding.module, binding.exported_name)

        resolved = self._resolve_exported_symbol(
            module, attribute_parts[-1], file_path, language,
        )
        if resolved:
            return resolved

        return leaf_name

    def _join_module_path(self, module: str, suffix: str) -> str:
        """Append a child module name while preserving relative-import prefixes."""
        if module.endswith("."):
            return f"{module}{suffix}"
        return f"{module}.{suffix}"

    def _trim_module_prefix(
        self, module: str, base_name: str, attribute_parts: list[str],
    ) -> list[str]:
        """Remove attribute segments that are part of the imported module path."""
        if not attribute_parts:
            return []

        module_parts = [p for p in module.lstrip(".").split(".") if p]
        if module_parts and module_parts[0] == base_name:
            module_parts = module_parts[1:]

        remaining = list(attribute_parts)
        while module_parts and remaining and module_parts[0] == remaining[0]:
            module_parts.pop(0)
            remaining.pop(0)
        return remaining

    def _qualify(self, name: str, file_path: str, enclosing_class: Optional[str]) -> str:
        """Create a qualified name: file_path::ClassName.name or file_path::name."""
        if enclosing_class:
            return f"{file_path}::{enclosing_class}.{name}"
        return f"{file_path}::{name}"

    def _get_name(self, node, language: str, kind: str) -> Optional[str]:
        """Extract the name from a class/function definition node."""
        # For C/C++: function names are inside function_declarator/pointer_declarator
        # Check these first to avoid matching the return type_identifier
        if language in ("c", "cpp") and kind == "function":
            for child in node.children:
                if child.type in ("function_declarator", "pointer_declarator"):
                    result = self._get_name(child, language, kind)
                    if result:
                        return result
        # Most languages use a 'name' child
        for child in node.children:
            if child.type in (
                "identifier", "name", "type_identifier", "property_identifier",
                "simple_identifier", "constant",
            ):
                return child.text.decode("utf-8", errors="replace")
        # For Go type declarations, look for type_spec
        if language == "go" and node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    return self._get_name(child, language, kind)
        return None

    def _get_params(self, node, language: str, source: bytes) -> Optional[str]:
        """Extract parameter list as a string."""
        for child in node.children:
            if child.type in ("parameters", "formal_parameters", "parameter_list"):
                return child.text.decode("utf-8", errors="replace")
        return None

    def _get_return_type(self, node, language: str, source: bytes) -> Optional[str]:
        """Extract return type annotation if present."""
        for child in node.children:
            if child.type in ("type", "return_type", "type_annotation"):
                return child.text.decode("utf-8", errors="replace")
        # Python: look for -> annotation
        if language == "python":
            for i, child in enumerate(node.children):
                if child.type == "->" and i + 1 < len(node.children):
                    return node.children[i + 1].text.decode("utf-8", errors="replace")
        return None

    def _get_bases(self, node, language: str, source: bytes) -> list[str]:
        """Extract base classes / implemented interfaces."""
        bases = []
        if language == "python":
            for child in node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type in ("identifier", "attribute"):
                            bases.append(arg.text.decode("utf-8", errors="replace"))
        elif language in ("java", "csharp", "kotlin"):
            # Look for superclass/interfaces in extends/implements clauses
            for child in node.children:
                if child.type in (
                    "superclass", "super_interfaces", "extends_type",
                    "implements_type", "type_identifier", "supertype",
                    "delegation_specifier",
                ):
                    text = child.text.decode("utf-8", errors="replace")
                    bases.append(text)
        elif language == "cpp":
            # C++: base_class_clause contains type_identifiers
            for child in node.children:
                if child.type == "base_class_clause":
                    for sub in child.children:
                        if sub.type == "type_identifier":
                            bases.append(sub.text.decode("utf-8", errors="replace"))
        elif language in ("typescript", "javascript", "tsx"):
            # extends clause
            for child in node.children:
                if child.type in ("extends_clause", "implements_clause"):
                    for sub in child.children:
                        if sub.type in ("identifier", "type_identifier", "nested_identifier"):
                            bases.append(sub.text.decode("utf-8", errors="replace"))
        elif language == "go":
            # Embedded structs / interface composition
            for child in node.children:
                if child.type == "type_spec":
                    for sub in child.children:
                        if sub.type in ("struct_type", "interface_type"):
                            for field_node in sub.children:
                                if field_node.type == "field_declaration_list":
                                    for f in field_node.children:
                                        if f.type == "type_identifier":
                                            bases.append(f.text.decode("utf-8", errors="replace"))
        return bases

    def _extract_import(self, node, language: str, source: bytes) -> list[str]:
        """Extract import targets as module/path strings."""
        imports = []
        text = node.text.decode("utf-8", errors="replace").strip()

        if language == "python":
            # import x.y.z  or  from x.y import z
            if node.type in ("import_from_statement", "future_import_statement"):
                for child in node.children:
                    if child.type in ("dotted_name", "relative_import", "__future__"):
                        imports.append(child.text.decode("utf-8", errors="replace"))
                        break
            else:
                for child in node.children:
                    if child.type == "dotted_name":
                        imports.append(child.text.decode("utf-8", errors="replace"))
                    elif child.type == "aliased_import":
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                imports.append(sub.text.decode("utf-8", errors="replace"))
                                break
        elif language in ("javascript", "typescript", "tsx"):
            # import ... from 'module'
            for child in node.children:
                if child.type == "string":
                    val = child.text.decode("utf-8", errors="replace").strip("'\"")
                    imports.append(val)
        elif language == "go":
            for child in node.children:
                if child.type == "import_spec_list":
                    for spec in child.children:
                        if spec.type == "import_spec":
                            for s in spec.children:
                                if s.type == "interpreted_string_literal":
                                    val = s.text.decode("utf-8", errors="replace")
                                    imports.append(val.strip('"'))
                elif child.type == "import_spec":
                    for s in child.children:
                        if s.type == "interpreted_string_literal":
                            val = s.text.decode("utf-8", errors="replace")
                            imports.append(val.strip('"'))
        elif language == "rust":
            # use crate::module::item
            imports.append(text.replace("use ", "").rstrip(";").strip())
        elif language in ("c", "cpp"):
            # #include <header> or #include "header"
            for child in node.children:
                if child.type in ("system_lib_string", "string_literal"):
                    val = child.text.decode("utf-8", errors="replace").strip("<>\"")
                    imports.append(val)
        elif language in ("java", "csharp"):
            # import/using package.Class
            parts = text.split()
            if len(parts) >= 2:
                imports.append(parts[-1].rstrip(";"))
        elif language == "ruby":
            # require 'module' or require_relative 'path'
            if "require" in text:
                match = re.search(r"""['"](.*?)['"]""", text)
                if match:
                    imports.append(match.group(1))
        else:
            # Fallback: just record the text
            imports.append(text)

        return imports

    def _get_call_name(self, node, language: str, source: bytes) -> Optional[str]:
        """Extract the function/method name being called."""
        ref = self._get_call_reference(node, language, source)
        if ref:
            return ref[-1]
        return None

    def _get_call_reference(self, node, language: str, source: bytes) -> Optional[list[str]]:
        """Extract the call reference as ordered identifier parts."""
        if not node.children:
            return None

        first = node.children[0]

        # Simple call: func_name(args)
        if first.type == "identifier":
            return [first.text.decode("utf-8", errors="replace")]

        # Method call: obj.method(args)
        member_types = (
            "attribute", "member_expression",
            "field_expression", "selector_expression",
        )
        if first.type in member_types:
            parts = self._get_reference_parts(first)
            return parts or [first.text.decode("utf-8", errors="replace")]

        # Scoped call (e.g., Rust path::func())
        if first.type in ("scoped_identifier", "qualified_name"):
            parts = self._get_reference_parts(first)
            return parts or [first.text.decode("utf-8", errors="replace")]

        return None

    def _get_reference_parts(self, node) -> list[str]:
        """Collect ordered identifier-like parts from a dotted/scoped reference."""
        identifier_types = (
            "identifier", "property_identifier", "field_identifier",
            "field_name", "type_identifier", "name",
        )
        nested_types = (
            "attribute", "member_expression", "field_expression",
            "selector_expression", "scoped_identifier", "qualified_name",
            "nested_identifier", "dotted_name",
        )

        if node.type in identifier_types:
            return [node.text.decode("utf-8", errors="replace")]

        parts: list[str] = []
        for child in node.children:
            if child.type in identifier_types:
                parts.append(child.text.decode("utf-8", errors="replace"))
            elif child.type in nested_types:
                parts.extend(self._get_reference_parts(child))
        return parts
