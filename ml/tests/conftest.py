import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests (large generation)")
    config.addinivalue_line("markers", "clip: tests that run the CLIP asset check (need the CLIP weights)")


@pytest.fixture(autouse=True)
def _no_clip_unless_requested(request, monkeypatch):
    """Only tests marked `clip` run the CLIP asset check inside check_photo. All other tests upload arbitrary images for
    arbitrary loans; with CLIP on, those would (correctly) trigger PHOTO_ASSET_MISMATCH and hide what they test."""
    if request.node.get_closest_marker("clip"):
        return
    from ml.photo import checker
    monkeypatch.setattr(checker, "_asset_match", lambda rgb, purpose: None)
