"""Verifica que el motor no importe librerías de interfaz. Sale con código 1 si encuentra alguna.

  python scripts/check_engine_imports.py
"""
import ast
import sys
from pathlib import Path

UI_LIBS = {"streamlit", "gradio", "flask", "fastapi", "dash", "tkinter", "PyQt5", "PySide6", "telegram", "plotly"}
pkg = Path(__file__).resolve().parent.parent / "vozregistro"
bad = []
for f in sorted(pkg.rglob("*.py")):
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            if n.split(".")[0] in UI_LIBS:
                bad.append(f"{f.relative_to(pkg.parent)}:{node.lineno} importa {n}")
print("\n".join(bad) if bad else f"OK: {len(list(pkg.rglob('*.py')))} archivos del motor, ninguna librería de UI")
sys.exit(1 if bad else 0)
