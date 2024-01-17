import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sys as _sys  # noqa: E402
if _sys.getrecursionlimit() < 6000:     # see engrag.config — re.compile guard
    _sys.setrecursionlimit(8000)

import pytest                          # noqa: E402

from engrag.config import load_settings  # noqa: E402
from engrag.service import Service        # noqa: E402


@pytest.fixture(scope="session")
def svc(tmp_path_factory):
    """Service over a freshly built synthetic demo corpus (offline)."""
    tmp = tmp_path_factory.mktemp("engrag-data")
    settings = load_settings({"ENGRAG_DATA_DIR": str(tmp)})
    service = Service(settings)
    from engrag.eval import build_service
    service = build_service(settings, tmp / "gen")
    return service


@pytest.fixture(scope="session")
def collection_id(svc):
    rows = svc.db.list_collections()
    return rows[0]["id"] if rows else None
