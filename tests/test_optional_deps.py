"""Feature-scoped imports must degrade with an actionable message, not a traceback.

`pypdfium2` + `Pillow` are only used by the page renderer (and therefore the citation highlight
PNGs), and ReportLab only by the synthetic demo-corpus generator. All three are declared in the
base dependency set, but they are small, self-contained features, so a *partial* install (or a
trimmed container image) must still leave the core — extraction, indexing, retrieval, answer
building, numerical verification — importable and usable, and must say what to reinstall rather
than raising `ModuleNotFoundError` from inside a request.

The absence is simulated with an import blocker on ``sys.meta_path`` (plus
evicting the cached modules), because simply setting ``sys.modules[name] = None``
does not stop ``from reportlab.lib import colors`` — the cached submodule is
found first. The first test double-checks the same claim in a *fresh* interpreter.
"""
from __future__ import annotations

import contextlib
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


class _Blocker:
    """Meta-path finder that pretends the given top-level packages do not exist."""

    def __init__(self, roots: tuple[str, ...]) -> None:
        self.roots = roots

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102
        if fullname.split(".")[0] in self.roots:
            raise ImportError(f"No module named {fullname!r}")
        return None


@contextlib.contextmanager
def blocked(*roots: str):
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in roots}
    for k in saved:
        del sys.modules[k]
    finder = _Blocker(roots)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k.split(".")[0] in roots]:
            del sys.modules[k]
        sys.modules.update(saved)


def test_core_imports_without_render_or_demo_extras():
    """A fresh interpreter can import the service layer with those packages blocked."""
    code = (
        "import sys\n"
        "class B:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('pypdfium2', 'PIL', 'reportlab'):\n"
        "            raise ImportError('No module named ' + repr(fullname))\n"
        "sys.meta_path.insert(0, B())\n"
        "for k in [k for k in list(sys.modules) if k.split('.')[0] in ('pypdfium2','PIL','reportlab')]:\n"
        "    del sys.modules[k]\n"
        "import engrag, engrag.db, engrag.service, engrag.rag, engrag.verify.numbers, "
        "engrag.verify.claims, engrag.retrieval.hybrid, engrag.ingestion.extract\n"
        "loaded = [k for k in sys.modules if k.split('.')[0] in ('pypdfium2','PIL','reportlab')]\n"
        "print('LOADED', loaded)\n"
        "print('CORE_OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=SRC, capture_output=True, text=True,
                         env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"}, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "CORE_OK" in res.stdout
    assert "LOADED []" in res.stdout, f"core pulled in optional deps: {res.stdout}"


def test_page_renderer_explains_the_missing_extra(tmp_path):
    from engrag.rendering import render_page_png

    with blocked("pypdfium2"):
        with pytest.raises(SystemExit) as excinfo:
            render_page_png(tmp_path / "missing.pdf", 1, 1,
                            renders_dir=tmp_path / "renders", scale=1.5)
    msg = str(excinfo.value)
    assert "pypdfium2" in msg and "Pillow" in msg
    assert "pip install" in msg                              # actionable, not just a name
    assert "Text search, Q&A and verification do not need them" in msg


def test_demo_generator_explains_the_missing_and_leaves_no_files(tmp_path):
    import engrag.demo_corpus as dc

    out = tmp_path / "corpus"
    with blocked("reportlab"):
        poisoned = importlib.reload(dc)
        assert poisoned.DEPS_ERROR and "ReportLab" in poisoned.DEPS_ERROR
        with pytest.raises(SystemExit) as excinfo:
            poisoned.generate_all(out)
    assert 'pip install -e ".[dev]"' in str(excinfo.value)
    assert not out.exists()                                  # failed before writing anything
    importlib.reload(dc)                                     # restore for the rest of the suite


def test_modules_are_fully_usable_again_after_the_blocked_import(tmp_path):
    """The guard must not leave the module poisoned for later tests / reruns."""
    import engrag.demo_corpus as dc
    from engrag.rendering import render_page_png

    docs = dc.generate_all(tmp_path)
    assert len(docs) == 5 and all(d.path.exists() and d.pages > 0 for d in docs)
    png = render_page_png(docs[0].path, 777, 1, renders_dir=tmp_path / "renders")
    assert png and png[:8] == b"\x89PNG\r\n\x1a\n"
