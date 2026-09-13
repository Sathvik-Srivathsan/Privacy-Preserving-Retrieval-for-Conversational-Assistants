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

import hashlib
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
    """Local dev-store: Fernet-encrypted blobs on disk (no cloud cost)."""

    def __init__(self, root: str, key: Optional[bytes] = None):
        self.root = root
        os.makedirs(root, exist_ok=True)
        if key is None:
            key = Fernet.generate_key() if _HAS_FERNET else _dev_key()
        self.key = key
        if _HAS_FERNET:
            self._fernet = Fernet(key)

    def put(self, chunk: StoredChunk) -> None:
        p = os.path.join(self.root, f"{chunk.chunk_id}.blob")
        payload = chunk.blob if not _HAS_FERNET else self._fernet.encrypt(
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
        if _HAS_FERNET:
            d = json.loads(self._fernet.decrypt(raw).decode("utf-8"))
            return StoredChunk(chunk_id, d["b"].encode("latin-1"), d["tag"], d["meta"])
        return StoredChunk(chunk_id, raw, "tag:" + hashlib.sha256(raw).hexdigest()[:12])

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
                "description": "encrypted SafeRAG chunk (ciphertext only)"}
        media = MediaIoUpload(chunk.blob, "*/*", chunksize=256 * 1024)
        fl = svc.files().create(body=body, media_body=media,
                                fields="id,webViewLink").execute()
        return fl.get("webViewLink")

    def get(self, chunk_id: str) -> StoredChunk | None:
        raise NotImplementedError("Drive read path wired in Phase-2 harness")

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


def _dev_key() -> bytes:
    return hashlib.sha256(b"SafeRAG-Improved-dev").digest()


# re-export MediaIoUpload lazily (heavy import only when GDrive used)
def MediaIoUpload(data, mime_type, chunksize):
    from googleapiclient.http import MediaIoUpload as _M
    return _M(data, mime_type, chunksize=chunksize)
