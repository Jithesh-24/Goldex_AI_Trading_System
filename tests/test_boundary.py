"""python3 tests/test_boundary.py -- app/ must never transitively import
learning/ or research/. This is the automated enforcement of the
production/research boundary (spec §4)."""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN_ROOTS = {"learning", "research"}


def _module_imports(path):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _walk_py_files(pkg_dir):
    for root, _, files in os.walk(pkg_dir):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def test_app_never_imports_learning_or_research():
    violations = []
    for path in _walk_py_files(os.path.join(BASE, "app")):
        for name in _module_imports(path):
            top = name.split(".")[0]
            if top in FORBIDDEN_ROOTS:
                violations.append((path, name))
    assert not violations, f"app/ imports research-only code: {violations}"


if __name__ == "__main__":
    test_app_never_imports_learning_or_research()
    print("tests/test_boundary.py: OK")
