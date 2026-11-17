"""Thread-safe bounded in-memory LRU cache for synthesized voice audio clips."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredAudio:
    """An audio payload stored in the bounded audio cache."""

    audio_id: str
    audio_bytes: bytes
    media_type: str


class InMemoryAudioStore:
    """Thread-safe LRU cache holding generated TTS audio clips for GET /api/v1/voice/audio/{id}.

    Enforces a strict entry cap (`max_entries`) so server RAM stays bounded even
    under sustained voice traffic.
    """

    def __init__(self, max_entries: int = 50) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._store: OrderedDict[str, StoredAudio] = OrderedDict()

    def put(self, audio_bytes: bytes, media_type: str = "audio/wav") -> str:
        """Store audio bytes and return a unique audio_id, evicting the oldest clip if full."""
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")
        audio_id = f"aud_{uuid.uuid4().hex[:12]}"
        item = StoredAudio(audio_id=audio_id, audio_bytes=audio_bytes, media_type=media_type)
        with self._lock:
            self._store[audio_id] = item
            self._store.move_to_end(audio_id)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
        return audio_id

    def get(self, audio_id: str) -> StoredAudio | None:
        """Retrieve a stored audio clip by ID and refresh its LRU position."""
        with self._lock:
            item = self._store.get(audio_id)
            if item is not None:
                self._store.move_to_end(audio_id)
            return item

    def count(self) -> int:
        """Return number of audio clips currently cached."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Remove all cached audio clips."""
        with self._lock:
            self._store.clear()

