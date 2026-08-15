import pytest


@pytest.fixture(autouse=True)
def test_only_watermark_environment(monkeypatch):
    """Allow importing app.main without ever supplying a production key."""
    monkeypatch.setenv("WM_HASH_KEY", "17")
