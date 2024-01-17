"""Static consistency checks for the Streamlit frontend.

The UI is the one part of this repository that a `pytest` run does not execute
(it needs a browser), which let a whole class of bugs hide: `st.tabs([...])`
and `st.columns([...])` return a tuple, so an unpack list of the wrong length
crashes the *entire view* at render time. These tests read the source with the
`ast` module and assert that every such unpack matches, without starting
Streamlit. `scripts/take_screenshots.py` additionally fails on any rendered
exception box, so a live check exists too.
"""
from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "src" / "engrag" / "frontend" / "app.py"
UIKIT = Path(__file__).resolve().parents[1] / "src" / "engrag" / "frontend" / "uikit.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _func_calls(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == name):
            yield node


def test_tab_and_column_unpacks_match_their_labels():
    tree = _tree(APP)
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.Tuple)):
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
            if len(targets) != 1 or not isinstance(targets[0], ast.Tuple):
                continue
            unpack, call = targets[0], node.value
        else:                                   # bare `a, b = ...` inside Assign only
            continue
        if not isinstance(call, ast.Call) or not call.args:
            continue
        fname = getattr(call.func, "attr", "")
        if fname not in {"tabs", "columns", "expander"}:
            continue
        arg = call.args[0]
        if isinstance(arg, ast.List):
            n = len(arg.elts)
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            n = arg.value                        # st.columns(3)
        else:
            continue
        if len(unpack.elts) != n:
            problems.append(f"{APP.name}:{node.lineno} st.{fname}() has {n} items "
                            f"but {len(unpack.elts)} targets")
    assert not problems, "frontend unpack mismatch:\n  " + "\n  ".join(problems)


def test_every_tab_variable_used_is_defined():
    """`with t3:` after `t1, t2 = st.tabs(...)` is a NameError at render time."""
    tree = _tree(APP)
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            if isinstance(t, ast.Tuple):
                defined.update(e.id for e in t.elts if isinstance(e, ast.Name))
            elif isinstance(t, ast.Name):
                defined.add(t.id)
        elif isinstance(node, (ast.For, ast.withitem)):
            pass
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Name):
                    used.add(ctx.id)
    missing = sorted(u for u in used if u.startswith("t") and u[1:].isdigit()
                     and u not in defined)
    assert not missing, f"`with <name>` used but never assigned: {missing}"


def test_all_eight_views_are_routed():
    tree = _tree(APP)
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for view in ("page_ask", "page_documents", "page_search", "page_viewer",
                 "page_values", "page_settings", "page_eval", "page_history"):
        assert view in funcs, f"{view} missing from app.py"

    views = pages = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "VIEWS":
                views = node.value
            elif node.targets[0].id == "pages":
                pages = node.value
    assert isinstance(views, ast.List), "VIEWS list not found"
    labels = {e.value for e in views.elts if isinstance(e, ast.Constant)}
    assert labels == {"ask", "documents", "search", "citations", "values",
                      "settings", "evaluation", "history"}, labels
    assert isinstance(pages, ast.Dict), "pages routing dict not found"
    keys = {k.value for k in pages.keys if isinstance(k, ast.Constant)}
    assert keys == labels, f"sidebar offers {sorted(labels)} but routes {sorted(keys)}"
    routed = {v.id for v in pages.values if isinstance(v, ast.Name)}
    assert routed <= funcs, f"routes to undefined functions: {routed - funcs}"


def test_uikit_width_helper_falls_back_for_old_streamlit():
    from engrag.frontend.uikit import WIDTH_KW
    assert set(WIDTH_KW) <= {"width", "use_container_width"}


def test_ui_never_opens_a_network_client_directly():
    """All provider traffic must go through llm/providers.py and only on action.

    The UI file must not import an HTTP client or a vendor SDK: that is what
    keeps the "no call without an explicit choice" promise checkable by grep.
    """
    tree = _tree(APP)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"httpx", "requests", "urllib3", "aiohttp", "openai", "anthropic"}
    assert not (imported & banned), f"frontend imports network SDKs: {imported & banned}"
