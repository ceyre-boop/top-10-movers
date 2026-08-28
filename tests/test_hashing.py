from __future__ import annotations

from pathlib import Path

from top10 import hashing
from top10.config import DOCS


def test_hash_file_matches_hashlib(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello world")
    import hashlib

    assert hashing.hash_file(p) == hashlib.sha256(b"hello world").hexdigest()


def test_hash_spec_order_independent():
    a = {"b": 1, "a": 2, "c": {"y": 1, "x": 2}}
    b = {"a": 2, "c": {"x": 2, "y": 1}, "b": 1}
    assert hashing.hash_spec(a) == hashing.hash_spec(b)


def test_hash_spec_differs_for_different_values():
    assert hashing.hash_spec({"a": 1}) != hashing.hash_spec({"a": 2})


def test_verify_spec_hash_bare_hex(tmp_path):
    target = tmp_path / "content.txt"
    target.write_bytes(b"some content")
    digest = hashing.hash_file(target)

    hash_file_path = tmp_path / "content.sha256"
    hash_file_path.write_text(digest)

    assert hashing.verify_spec_hash(target, hash_file_path) is True


def test_verify_spec_hash_sha256sum_format(tmp_path):
    target = tmp_path / "content.txt"
    target.write_bytes(b"some content")
    digest = hashing.hash_file(target)

    hash_file_path = tmp_path / "content.sha256"
    hash_file_path.write_text(f"{digest}  content.txt\n")

    assert hashing.verify_spec_hash(target, hash_file_path) is True


def test_verify_spec_hash_mismatch(tmp_path):
    target = tmp_path / "content.txt"
    target.write_bytes(b"some content")

    hash_file_path = tmp_path / "content.sha256"
    hash_file_path.write_text("0" * 64)

    assert hashing.verify_spec_hash(target, hash_file_path) is False


def test_verify_spec_hash_against_committed_label_spec():
    label_spec = DOCS / "LABEL_SPEC.md"
    expected_hash_path = DOCS / "LABEL_SPEC.sha256"
    assert hashing.verify_spec_hash(label_spec, expected_hash_path) is True
