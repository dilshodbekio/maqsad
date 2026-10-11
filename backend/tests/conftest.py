import os
import struct
import zlib
from dataclasses import replace

import pytest

os.environ["ML_FORCE_STUB"] = "1"  # tests run against the stub regardless of ml/ state

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import ml_stub  # noqa: E402
from backend.app.config import Settings  # noqa: E402
from backend.app.main import create_app  # noqa: E402


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_FORCE_STUB", "1")
    monkeypatch.delenv("REQUIRE_REAL_ML", raising=False)

    def _make(adapter=None, **overrides) -> TestClient:
        base = dict(data_dir=tmp_path / "data", frontend_dist=tmp_path / "no_dist", admin_token="",
                    llm_base_url="", llm_model="", llm_api_key="", llm_rpm=600)
        base.update(overrides)
        ml_stub.reset_sandbox(None)
        app = create_app(settings=replace(Settings.from_env(), **base), adapter=adapter)
        assert app.state.ml.wait_warmup(30), "ml warm-up did not finish"
        return TestClient(app)

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


@pytest.fixture
def loan_id(client) -> str:
    return client.get("/api/queue").json()["items"][0]["loan_id"]


def png_bytes(rgb=(200, 30, 30)) -> bytes:
    """Valid 1x1 PNG; different colors -> different bytes."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" + bytes(rgb))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def draft(c, loan_id, *, timeout=15.0, **body):
    """POST /api/draft_report and, for a 202, poll the job until it is done. Returns the final DraftReport dict
    (or the job dict when it ended in an error)."""
    import time
    r = c.post("/api/draft_report", json={"loan_id": loan_id, **body})
    if r.status_code == 200:
        return r.json()
    assert r.status_code == 202, r.text
    job_id, end = r.json()["job_id"], time.monotonic() + timeout
    while time.monotonic() < end:
        j = c.get(f"/api/draft_report/job/{job_id}")
        assert j.status_code == 200, j.text
        j = j.json()
        if j["status"] == "done":
            return j["result"]
        if j["status"] == "error":
            return j
        time.sleep(0.02)
    raise AssertionError("draft job did not finish")
