# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Encrypted corpus store: local (Fernet) + Google Drive (honest-but-curious).

SafeRAG outsources encrypted document blobs to an honest-but-curious cloud
(Section II threat model). Google Drive's free tier is exactly that: our
Phase-1 ShardStore writes encrypted chunks + policy tags; nothing plaintext
leaves the laptop. Drive wiring is Phase-1 (priority 5) so we declare the
adapter interface now and pop the key question when the rig is run.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

try:
    from cryptography.fernet import Fernet
    _HAS_FERNET = True
except ImportError:
    _HAS_FERNET = False


@dataclass
class StoredChunk:
    chunk_id: str
    blob: bytes        # encrypted
    tag: str           # policy/attribute tag string
    meta: dict = field(default_factory=dict)


class LocalStore:
    """Local dev-store: Fernet-encrypted blobs on disk (no cloud cost).

    Hard-requires the ``cryptography`` package (Fernet): this store's whole
    job is encrypted-at-rest, so running unencrypted is REFUSED rather than
    silently degrading (a no-Fernet fallback would drop the chunk ``tag``/``meta``
    — the access-control metadata — and so is not allowed).

    Key persistence: the first time a store directory is opened, a key is
    generated and saved next to the blobs as ``store.key``. Later openings of
    the same directory reuse that key, so a corpus written in one session
    stays readable in a later session (build once, query later). Callers may
    override by passing an explicit ``key`` (their responsibility to keep it).
    """

    KEY_FILENAME = "store.key"

    def __init__(self, root: str, key: Optional[bytes] = None,
                 key_file: Optional[str] = None):
        if not _HAS_FERNET:
            raise RuntimeError(
                "LocalStore requires the 'cryptography' package (Fernet); "
                "refusing to run with the insecure plaintext fallback")
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.key_file = key_file or os.path.join(root, self.KEY_FILENAME)
        if key is not None:
            self.key = key
        elif os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            self.save_key()
        self._fernet = Fernet(self.key)

    def save_key(self, path: Optional[str] = None) -> None:
        """Persist the current Fernet key (default: ``store.key`` in the root)."""
        with open(path or self.key_file, "wb") as f:
            f.write(self.key)

    def put(self, chunk: StoredChunk) -> None:
        p = os.path.join(self.root, f"{chunk.chunk_id}.blob")
        payload = self._fernet.encrypt(
            json.dumps({"b": chunk.blob.decode("latin-1"), "tag": chunk.tag,
                        "meta": chunk.meta}).encode("utf-8"))
        with open(p, "wb") as f:
            f.write(payload)

    def get(self, chunk_id: str) -> StoredChunk | None:
        p = os.path.join(self.root, f"{chunk_id}.blob")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            raw = f.read()
        d = json.loads(self._fernet.decrypt(raw).decode("utf-8"))
        return StoredChunk(chunk_id, d["b"].encode("latin-1"), d["tag"], d["meta"])

    def list_ids(self) -> list[str]:
        return [f[:-5] for f in os.listdir(self.root) if f.endswith(".blob")]


class GoogleDriveStore:
    """
    Honest-but-curious cloud adapter (Phase-1 target). Writes only encrypted
    blobs; Drive sees ciphertext + tag index (no plaintext, no embeddings).
    Requires a Service-Account credentials JSON (free) placed in ./gdrive/.
    """

    def __init__(self, creds_path: str = "gdrive/service_account.json",
                 folder_name: str = "SafeRAGCorpus"):
        self.creds_path = creds_path
        self.folder_name = folder_name
        self._svc = None

    def _connect(self):
        if self._svc is not None:
            return self._svc
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            self.creds_path,
            scopes=["https://www.googleapis.com/auth/drive.file"])
        self._svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._svc

    def connected(self) -> bool:
        return os.path.exists(self.creds_path)

    def put(self, chunk: StoredChunk) -> str:
        svc = self._connect()
        folder = self._ensure_folder(svc)
        body = {"name": f"{chunk.chunk_id}.blob", "parents": [folder],
                "description": json.dumps({"tag": chunk.tag, "meta": chunk.meta})}
        media = upload_media(chunk.blob, "*/*", chunksize=256 * 1024)
        fl = svc.files().create(body=body, media_body=media,
                                fields="id,webViewLink").execute()
        return fl.get("webViewLink")

    def get(self, chunk_id: str) -> StoredChunk | None:
        svc = self._connect()
        folder = self._ensure_folder(svc)
        q = (f"name='{chunk_id}.blob' and '{folder}' in parents and trashed=false")
        res = svc.files().list(q=q, fields="files(id,name,description)").execute()
        files = res.get("files", [])
        if not files:
            return None
        blob = svc.files().get_media(fileId=files[0]["id"]).execute()
        tag, meta = "", {}
        try:
            d = json.loads(files[0].get("description", "{}"))
            tag = str(d.get("tag", ""))
            m = d.get("meta", {})
            if isinstance(m, dict):
                meta = m
        except (TypeError, ValueError):
            pass
        return StoredChunk(chunk_id, blob, tag, meta)

    def list_ids(self) -> list[str]:
        svc = self._connect()
        folder = self._ensure_folder(svc)
        names, page_token = [], None
        while True:
            res = svc.files().list(
                q=f"'{folder}' in parents and trashed=false",
                fields="nextPageToken,files(name)",
                pageToken=page_token).execute()
            for fl in res.get("files", []):
                name = fl.get("name", "")
                if name.endswith(".blob"):
                    names.append(name[:-5])
            page_token = res.get("nextPageToken")
            if not page_token:
                break
        return names

    def _ensure_folder(self, svc) -> str:
        q = (f"name='{self.folder_name}' and mimeType='application/vnd.google-apps.folder' "
             f"and trashed=false")
        res = svc.files().list(q=q, fields="files(id,name)").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        f = svc.files().create(body={"name": self.folder_name,
                                     "mimeType": "application/vnd.google-apps.folder"},
                              fields="id").execute()
        return f.get("id")


# re-export upload media lazily (heavy import only when GDrive used).
# NOTE: newer google-api-python-client has MediaInMemoryUpload (bytes),
# not the old MediaIoUpload — this callback name is intentionally gone.
def upload_media(data: bytes, mime_type: str, chunksize: int):
    from googleapiclient.http import MediaInMemoryUpload as _M
    return _M(data, mime_type, chunksize=chunksize)
