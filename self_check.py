#!/usr/bin/env python3
"""Lightweight startup check for the scripts/ directory.

Goals:
- Ensure all Python scripts in scripts/ parse (syntax check via py_compile)
- Print a compact inventory of scripts

This is safe to run at gateway startup (fast, no network calls).
"""

from __future__ import annotations

import py_compile
from pathlib import Path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    py_files = sorted(p for p in script_dir.glob("*.py") if p.name != Path(__file__).name)

    print(f"scripts/: found {len(py_files)} python files")
    for p in py_files:
        try:
            py_compile.compile(str(p), doraise=True)
            status = "ok"
        except Exception as e:
            status = f"FAIL: {e}"
        print(f"- {p.name}: {status}")

    # Non-zero exit if any failed
    failures = [p for p in py_files if _compile_ok(p) is False]
    return 1 if failures else 0


def _compile_ok(p: Path) -> bool:
    try:
        py_compile.compile(str(p), doraise=True)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
