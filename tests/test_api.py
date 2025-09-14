import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GEMINI_API_KEY", "test-api-key")
os.environ.setdefault("GEMINI_MODEL", "test-model")

import app as fastapi_app  # noqa: E402
import customer_support  # noqa: E402


def test_support_endpoint(monkeypatch):
    """POST /support returns workflow output."""

    def dummy_invoke(inputs):
        assert inputs == {"query": "Need help with billing"}
        return {
            "category": "Billing",
            "sentiment": "Neutral",
            "response": "Billing help provided",
        }

    monkeypatch.setattr(customer_support.app, "invoke", dummy_invoke)

    client = TestClient(fastapi_app.app)
    resp = client.post("/support", json={"query": "Need help with billing"})

    assert resp.status_code == 200
    assert resp.json() == {
        "category": "Billing",
        "sentiment": "Neutral",
        "response": "Billing help provided",
    }

