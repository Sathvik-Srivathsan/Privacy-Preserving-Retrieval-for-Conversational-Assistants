# -*- coding: utf-8 -*-
"""
Store tests (Phase-1 T2) —— LocalStore round-trip + GoogleDriveStore.

LocalStore: put/get/list_ids integrity, encrypted-at-rest, missing-key returns None.
GoogleDriveStore: get()/list_ids() unit-tested OFFLINE against an in-memory fake
Drive service (no creds/no network; live round-trip is optional/time-boxed).

Run:  python tests\test_store.py      (from repo root)
Exit 0 iff ALL_OK.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO / "src"))

from store import GoogleDriveStore, LocalStore, StoredChunk  # noqa: E402


# --------------------------------------------------------------------------- #
# LocalStore
# --------------------------------------------------------------------------- #


def _tmp_store():
    tmp = tempfile.mkdtemp(prefix="saferag_store_")
    return LocalStore(tmp), tmp


def _sample_chunk(cid="c1", blob=b"\x00\x01\x02 raw ciphertext bytes",
                  tag="role:doctor", meta=None):
    return StoredChunk(cid, blob, tag, meta or {"dim": 8})


def test_local_roundtrip():
    s, _ = _tmp_store()
    c = _sample_chunk()
    s.put(c)
    got = s.get(c.chunk_id)
    assert got is not None
    assert got.chunk_id == c.chunk_id
    assert got.blob == c.blob
    assert got.tag == c.tag
    assert got.meta == c.meta


def test_local_get_missing_returns_none():
    s, _ = _tmp_store()
    assert s.get("nope") is None


def test_local_list_ids():
    s, _ = _tmp_store()
    s.put(_sample_chunk("a"))
    s.put(_sample_chunk("b"))
    s.put(_sample_chunk("c"))
    assert sorted(s.list_ids()) == ["a", "b", "c"]


def test_local_list_ids_ignores_foreign_files():
    s, root = _tmp_store()
    s.put(_sample_chunk("a"))
    with open(Path(root) / "notes.txt", "wb") as f:
        f.write(b"not a store file")
    assert s.list_ids() == ["a"]


def test_local_encrypted_at_rest():
    s, root = _tmp_store()
    blob = b"plaintext-equivalent payload"
    s.put(_sample_chunk("x", blob=blob))
    raw = (Path(root) / "x.blob").read_bytes()
    assert blob not in raw, "Fernet must hide the plaintext at rest"


def test_local_key_persists_across_instances():
    # simulates "build once, query later" in a later session/process: a NEW
    # LocalStore over the same directory (key=None) must reuse store.key
    root = tempfile.mkdtemp(prefix="saferag_store_")
    first = LocalStore(root)
    first.put(_sample_chunk("c1", b"persisted payload", "role:doctor", {"dim": 8}))
    assert (Path(root) / LocalStore.KEY_FILENAME).exists()

    second = LocalStore(root)  # fresh instance, same dir, no explicit key
    assert second.key == first.key
    got = second.get("c1")
    assert got is not None
    assert got.blob == b"persisted payload"
    assert got.tag == "role:doctor"
    assert got.meta == {"dim": 8}


def test_local_explicit_key_overrides_and_wins():
    import base64
    key = base64.urlsafe_b64encode(b"\x00" * 32)  # valid Fernet key
    root = tempfile.mkdtemp(prefix="saferag_store_")
    s = LocalStore(root, key=key)
    s.put(_sample_chunk("c1", b"custom key payload"))
    assert s.key == key
    got = LocalStore(root, key=key).get("c1")
    assert got is not None and got.blob == b"custom key payload"


def test_local_multiple_roundtrip_distinct():
    s, _ = _tmp_store()
    s.put(_sample_chunk("one", b"bb-q", tag="role:nurse"))
    s.put(_sample_chunk("two", b"bb-z"))
    assert s.get("one").blob == b"bb-q" and s.get("one").tag == "role:nurse"
    assert s.get("two").blob == b"bb-z"
    assert s.get("one") != s.get("two")


# --------------------------------------------------------------------------- #
# GoogleDriveStore — OFFLINE fake (no creds, no network)
# --------------------------------------------------------------------------- #


class _FakeRequest:
    def __init__(self, resp):
        self._resp = resp

    def execute(self, **_kw):
        return self._resp

    def __call__(self):
        return self


class FakeDrive:
    """In-memory Drive v3 stub shaped like googleapiclient discovery objects."""

    def __init__(self):
        self._files = {}      # id -> {"name": ..., "description": ..., "parents": id|None}
        self._blobs = {}      # id -> bytes
        self._next_id = 1
        self._folder_id = None

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return f"id{i}"

    def files(self):
        return _FakeFiles(self)


class _FakeFiles:
    def __init__(self, drive):
        self._d = drive

    def create(self, body=None, media_body=None, fields=None):
        d = dict(body or {})
        fid = self._d._new_id()
        if d.get("mimeType") == "application/vnd.google-apps.folder":
            self._d._folder_id = fid
            self._d._files[fid] = {"name": d["name"], "description": "",
                                   "parents": None}
            return _FakeRequest({"id": fid})
        self._d._files[fid] = {"name": d.get("name", ""),
                               "description": d.get("description", ""),
                               "parents": (d.get("parents") or ["?"])[0]}
        raw = b""
        if isinstance(media_body, bytes):
            raw = media_body
        else:
            fd = getattr(media_body, "_fd", None)
            if hasattr(fd, "getvalue"):
                raw = bytes(fd.getvalue())
            elif isinstance(getattr(media_body, "_body", None), bytes):
                raw = media_body._body
        self._d._blobs[fid] = raw
        return _FakeRequest({"id": fid, "webViewLink": f"https://fake/{fid}"})

    def list(self, q=None, fields=None, pageToken=None):
        # AND semantics: honour BOTH name='..' and 'parent' in parents when
        # both appear in q (mirrors Drive scoping, so get() can't leak a
        # same-named file from another folder).
        out = []
        name_want = parent_want = None
        if q:
            m = re.search(r"name='([^']*)'", q)
            if m:
                name_want = m.group(1)
            m = re.search(r"'([^']+)' in parents", q)
            if m:
                parent_want = m.group(1)
        if name_want is None and parent_want is None:
            return _FakeRequest({"files": []})
        for fid, rec in self._d._files.items():
            if name_want is not None and rec.get("name") != name_want:
                continue
            if parent_want is not None and rec.get("parents") != parent_want:
                continue
            out.append({"id": fid, "name": rec["name"],
                        "description": rec.get("description", "")})
        if fields == "id,webViewLink":
            return _FakeRequest([{"id": f["id"], "webViewLink": f"https://fake/{f['id']}"}
                                 for f in out])
        if fields and "description" not in fields:
            out = [{"id": f["id"], "name": f["name"]} for f in out]
        return _FakeRequest({"files": out})

    def get_media(self, fileId=None):
        return _FakeRequest(self._d._blobs.get(fileId, b""))


def _gd_store(fake):
    st = GoogleDriveStore(creds_path="gdrive/service_account.json")
    st._svc = fake
    return st


def test_gdrive_put_get_roundtrip():
    st = _gd_store(FakeDrive())
    c = StoredChunk("c1", b"\xde\xad\xbe\xef", "role:doctor", {"dim": 8})
    url = st.put(c)
    assert url.startswith("https://fake/")
    got = st.get("c1")
    assert got is not None
    assert got.chunk_id == "c1"
    assert got.blob == b"\xde\xad\xbe\xef"
    assert got.tag == "role:doctor"
    assert got.meta == {"dim": 8}


def test_gdrive_get_missing_returns_none():
    st = _gd_store(FakeDrive())
    assert st.get("unknown") is None


def test_gdrive_list_ids():
    fake = FakeDrive()
    st = _gd_store(fake)
    st.put(StoredChunk("a", b"1", "role:doctor"))
    st.put(StoredChunk("b", b"2", "role:nurse"))
    assert sorted(st.list_ids()) == ["a", "b"]


def test_gdrive_list_ids_ignores_foreign_files():
    fake = FakeDrive()
    st = _gd_store(fake)
    st.put(StoredChunk("a", b"1", "role:doctor"))
    fid = fake._new_id()
    fake._files[fid] = {"name": "notes.txt", "description": "", "parents": fake._folder_id}
    assert st.list_ids() == ["a"]


def test_gdrive_get_respects_folder_scoping():
    # same file NAME in two folders: get() must scope to the store's folder;
    # with a broken fake (OR matching) this test would return the wrong blob.
    fake = FakeDrive()
    a = fake.files().create(body={"name": "A", "mimeType": "application/vnd.google-apps.folder"},
                            fields="id").execute()["id"]
    b = fake.files().create(body={"name": "B", "mimeType": "application/vnd.google-apps.folder"},
                            fields="id").execute()["id"]
    for fid, name, parent, blob in ((fake._new_id(), "c1.blob", a, b"aa"),
                                    (fake._new_id(), "c1.blob", b, b"bb")):
        fake._files[fid] = {"name": name, "description": "{}", "parents": parent}
        fake._blobs[fid] = blob

    st_a = GoogleDriveStore(creds_path="x")
    st_a._svc = fake
    st_a._ensure_folder = lambda svc: a
    st_b = GoogleDriveStore(creds_path="x")
    st_b._svc = fake
    st_b._ensure_folder = lambda svc: b

    assert st_a.get("c1").blob == b"aa"
    assert st_b.get("c1").blob == b"bb"
    assert st_a.get("nope") is None


def test_gdrive_get_non_json_description_tolerated():
    st = _gd_store(FakeDrive())
    st.put(StoredChunk("c1", b"blob", "tag-from-desc", {"k": 1}))
    # corrupt the description after put, get should still return the blob
    for rec in st._svc._files.values():
        if rec["name"] == "c1.blob":
            rec["description"] = "not json {"
    got = st.get("c1")
    assert got is not None and got.blob == b"blob"
    assert got.tag == "" and got.meta == {}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append((name, repr(exc)))
    report = {
        "module": "store",
        "tests": len(tests),
        "passed": len(tests) - len(fails),
        "failed": len(fails),
        "failures": [f[0] for f in fails],
        "all_ok": not fails,
    }
    print(json.dumps(report, indent=2))
    print(f"ALL_OK: {not fails}   tests: {len(tests)}   passed: {len(tests) - len(fails)}")
    if fails:
        for name, exc in fails:
            print(f"FAIL {name}: {exc}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())