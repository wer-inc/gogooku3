#!/usr/bin/env python3
"""
Audit potentially unused files and modules in this repo.

What it does (non-destructive):
- Parses Python imports to find unimported modules under src/gogooku3.
- Greps (pure Python) for filename mentions to flag maybe-unreferenced configs.
- Writes a Markdown report to archive/ by default.

Usage examples:
  python scripts/audit_unused.py
  python scripts/audit_unused.py --no-output --verbose
  python scripts/audit_unused.py --fast  # skip non-Python reference scan

Notes:
- Static analysis can miss dynamic imports (importlib, CLI entrypoints).
- Treat results as candidates; quarantine first, then verify with tests.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_ROOTS = [
    Path("src/gogooku3"),
    Path("scripts"),
    Path("configs"),
    Path("tests"),
    Path("dagster_repo"),
]

DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "archive",
    "data",
    "output",
    "cache",
    "_logs",
}

TEXT_EXTS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".conf",
    ".sh",
    ".md",
}

# Target non-Python files to check for cross-references
NONPY_TARGET_EXTS = {
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".conf",
    ".sh",
}


def iter_files(
    roots: Iterable[Path], excludes: Set[str]
) -> Iterator[Tuple[Path, Path]]:
    """Yield (abs_path, rel_path) for files under roots, skipping excluded dirs."""
    for root in roots:
        abs_root = (REPO_ROOT / root).resolve()
        if not abs_root.exists():
            continue
        for p in abs_root.rglob("*"):
            if p.is_dir():
                # Skip excluded directories early
                if p.name in excludes:
                    # prune: skip walking into this directory
                    # rglob can't be pruned directly; skip via path check later
                    pass
                continue
            rel = p.relative_to(REPO_ROOT)
            if any(part in excludes for part in rel.parts):
                continue
            yield p, rel


def module_name_for_path(rel_path: Path) -> Optional[str]:
    """Compute module name for repository-relative path if within src/gogooku3 or scripts/tests.

    Returns dotted module path (e.g., gogooku3.training.loop) or None if not a python module file.
    """
    if rel_path.suffix != ".py":
        return None
    parts = rel_path.parts
    # Package under src/gogooku3
    if len(parts) >= 2 and parts[0] == "src" and parts[1] == "gogooku3":
        pkg_parts = list(parts[1:])  # include 'gogooku3'
        if pkg_parts[-1] == "__init__.py":
            pkg_parts = pkg_parts[:-1]
        else:
            pkg_parts[-1] = pkg_parts[-1].replace(".py", "")
        return ".".join(pkg_parts)
    # Script modules
    if parts[0] in {"scripts", "tests"}:
        mod_parts = list(parts)
        if mod_parts[-1] == "__init__.py":
            mod_parts = mod_parts[:-1]
        else:
            mod_parts[-1] = mod_parts[-1].replace(".py", "")
        return ".".join(mod_parts)
    return None


def resolve_from_import(current_module: str, module: Optional[str], level: int) -> Optional[str]:
    """Resolve a from-import to an absolute module path.

    current_module is the module path of the file doing the import.
    If it's a package module (no trailing name), resolution still works.
    """
    if level == 0:
        return module or None
    # Compute base package for relative import
    cur_parts = current_module.split(".")
    # If current module points to a module file (not package), drop last segment
    # Heuristic: assume last segment is a module unless current path is a package __init__
    # We can't know here; dropping one extra later if needed is okay.
    base_parts = cur_parts[:-1]
    # Ascend 'level-1' additional levels
    if level > 1:
        base_parts = base_parts[: - (level - 1)] if (level - 1) <= len(base_parts) else []
    if module:
        return ".".join([*(base_parts), module]) if base_parts else module
    return ".".join(base_parts) if base_parts else None


def collect_imports(py_path: Path, module_name: str) -> Set[str]:
    """Parse a Python file and return a set of imported module names (absolute)."""
    used: Set[str] = set()
    try:
        src = py_path.read_text(encoding="utf-8")
    except Exception:
        return used
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return used

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    used.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            abs_mod = resolve_from_import(module_name, mod, node.level)
            if abs_mod:
                used.add(abs_mod)
        elif isinstance(node, ast.Call):
            # importlib.import_module("pkg.sub") or import_module("pkg.sub")
            func = node.func
            name: Optional[str] = None
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "importlib" and func.attr == "import_module":
                    name = None
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if isinstance(node.args[0].value, str):
                            name = node.args[0].value
            elif isinstance(func, ast.Name) and func.id == "import_module":
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        name = node.args[0].value
            if name:
                used.add(name)

    return used


def build_module_map(py_files: Iterable[Tuple[Path, Path]]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for abs_p, rel_p in py_files:
        mod = module_name_for_path(rel_p)
        if mod:
            mapping[mod] = abs_p
    return mapping


def longest_existing_prefix(mod: str, mapping: dict[str, Path]) -> Optional[str]:
    """Return the longest dotted prefix of mod present in mapping."""
    parts = mod.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in mapping:
            return cand
    return None


def find_unimported_modules(
    py_files: Iterable[Tuple[Path, Path]], include_tests: bool = True
) -> list[tuple[str, Path]]:
    mapping = build_module_map(py_files)
    used_modules: Set[str] = set()

    for abs_p, rel_p in py_files:
        mod = module_name_for_path(rel_p)
        if not mod:
            continue
        # Skip counting imports from modules outside our package if desired later
        imports = collect_imports(abs_p, mod)
        for imp in imports:
            # Normalize by resolving to longest prefix that exists
            resolved = longest_existing_prefix(imp, mapping)
            if resolved:
                used_modules.add(resolved)

    candidates: list[tuple[str, Path]] = []
    for mod, path in mapping.items():
        # Only flag modules under gogooku3 package
        if not mod.startswith("gogooku3"):
            continue
        # Consider scripts/tests as entrypoints; keep them out
        if ".tests" in mod or mod.startswith("tests."):
            if include_tests:
                used_modules.add(mod)
            continue
        # Heuristic: treat packages with __init__ as used if any submodule is used
        if any(u == mod or u.startswith(mod + ".") for u in used_modules):
            continue
        candidates.append((mod, path))

    return sorted(candidates, key=lambda x: str(x[1]))


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def maybe_unreferenced_nonpy(
    all_files: Iterable[Tuple[Path, Path]], fast: bool = False, verbose: bool = False
) -> list[Path]:
    targets: list[Tuple[Path, Path]] = []
    searchable: list[Tuple[Path, Path]] = []
    for abs_p, rel_p in all_files:
        if rel_p.suffix in NONPY_TARGET_EXTS:
            targets.append((abs_p, rel_p))
        if rel_p.suffix in TEXT_EXTS:
            searchable.append((abs_p, rel_p))

    # Preload contents for faster scanning
    content_map: dict[Path, str] = {}
    if not fast:
        for abs_p, _ in searchable:
            content_map[abs_p] = load_text(abs_p)

    unref: list[Path] = []
    for abs_p, rel_p in targets:
        if fast:
            # In fast mode, skip costly search
            continue
        name = rel_p.name
        rel_str = str(rel_p).replace("\\", "/")
        found = False
        for s_abs, s_rel in searchable:
            if s_abs == abs_p:
                continue
            text = content_map.get(s_abs, "")
            if not text:
                continue
            if name in text or rel_str in text:
                found = True
                if verbose:
                    print(f"reference to {rel_p} found in {s_rel}")
                break
        if not found:
            unref.append(rel_p)
    return sorted(unref, key=str)


def write_report(
    unused_modules: list[tuple[str, Path]],
    unref_files: list[Path],
    output: Optional[Path],
) -> None:
    lines: list[str] = []
    lines.append("# Unused Files Audit Report")
    lines.append("")
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Generated: {ts}")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Unimported Python modules (src/gogooku3): {len(unused_modules)}"
    )
    lines.append(f"- Maybe-unreferenced non-Python files: {len(unref_files)}")
    lines.append("")
    if unused_modules:
        lines.append("## Unimported Python Modules (candidates)")
        for mod, path in unused_modules:
            rel = path.relative_to(REPO_ROOT)
            lines.append(f"- `{rel}`  (module: `{mod}`)")
        lines.append("")
    if unref_files:
        lines.append("## Maybe-unreferenced Non-Python Files (candidates)")
        for rel in unref_files:
            lines.append(f"- `{rel}`")
        lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Static analysis; confirm by quarantining and running tests/pipelines."
    )
    lines.append(
        "- Dynamic imports, CLI entrypoints, and DAG tools may hide usage."
    )

    content = "\n".join(lines) + "\n"

    # Always print to stdout
    print(content)
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Report written to {output}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--roots",
        nargs="*",
        default=[str(p) for p in DEFAULT_ROOTS],
        help="Root directories to scan",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDES),
        help="Directory names to exclude",
    )
    ap.add_argument(
        "--include-tests",
        action="store_true",
        help="Count tests as usage (default off for module unused check)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Skip non-Python reference scanning (faster)",
    )
    ap.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write report file to archive/",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (prints reference hits)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any candidates are found",
    )
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    excludes = set(args.exclude)

    all_files = list(iter_files(roots, excludes))
    py_files = [(a, r) for a, r in all_files if r.suffix == ".py"]

    unused_modules = find_unimported_modules(
        py_files, include_tests=args.include_tests
    )
    unref_files = maybe_unreferenced_nonpy(
        all_files, fast=bool(args.fast), verbose=bool(args.verbose)
    )

    date_tag = dt.datetime.now().strftime("%Y-%m-%d")
    output = None
    if not args.no_output:
        output = REPO_ROOT / "archive" / f"unused_report_{date_tag}.md"

    write_report(unused_modules, [Path(p) for p in unref_files], output)

    if args.strict and (unused_modules or unref_files):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

