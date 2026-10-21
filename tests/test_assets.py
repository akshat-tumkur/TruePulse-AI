"""Tests for truepulse.assets, using a local HTTP server (no internet needed)."""

import functools
import hashlib
import http.server
import threading

import pytest

from truepulse.assets import AssetError, ensure_file, sha256_of

PAYLOAD = b"pretend model bytes" * 1000
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def server(tmp_path):
    """Serve tmp_path/served over HTTP and yield the base URL."""
    served = tmp_path / "served"
    served.mkdir()
    (served / "model.bin").write_bytes(PAYLOAD)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_sha256_of_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD)
    assert sha256_of(path) == PAYLOAD_SHA256


def test_downloads_and_verifies(server, tmp_path):
    dest = tmp_path / "models" / "model.bin"
    assert ensure_file(f"{server}/model.bin", dest, PAYLOAD_SHA256) == dest
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_name("model.bin.part").exists()


def test_existing_valid_file_is_reused_without_network(tmp_path):
    dest = tmp_path / "model.bin"
    dest.write_bytes(PAYLOAD)
    # An unreachable URL proves no download is attempted.
    assert ensure_file("http://127.0.0.1:9/unreachable", dest, PAYLOAD_SHA256) == dest


def test_wrong_checksum_download_is_rejected_and_not_kept(server, tmp_path):
    dest = tmp_path / "model.bin"
    with pytest.raises(AssetError, match="SHA-256"):
        ensure_file(f"{server}/model.bin", dest, "0" * 64)
    assert not dest.exists()
    assert not dest.with_name("model.bin.part").exists()


def test_corrupt_existing_file_raises_and_is_left_untouched(tmp_path):
    dest = tmp_path / "model.bin"
    dest.write_bytes(b"corrupted")
    with pytest.raises(AssetError, match="Delete the file"):
        ensure_file("http://127.0.0.1:9/unreachable", dest, PAYLOAD_SHA256)
    assert dest.read_bytes() == b"corrupted"


def test_missing_file_on_server_raises_clear_error(server, tmp_path):
    with pytest.raises(AssetError, match="Could not download"):
        ensure_file(f"{server}/missing.bin", tmp_path / "x.bin", PAYLOAD_SHA256)
