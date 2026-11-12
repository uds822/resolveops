from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class EmbeddingModel(ABC):
    """Every embedding model must implement these two methods."""

    model_name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks (for indexing)."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question (for search)."""
        ...


class LocalEmbeddingModel(EmbeddingModel):
    """Runs embedding models locally using sentence-transformers.

    No API key needed. Free. Works offline.

    Args:
        model_name: HuggingFace model identifier. Default: "all-MiniLM-L6-v2"
          (fast, 384 dims) Upgrade: "BAAI/bge-base-en-v1.5" (better quality, 768
          dims)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local models. "
                "Install it with: pip install sentence-transformers"
            )
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class RemoteEmbeddingModel(EmbeddingModel):
    """Calls OpenAI-compatible embedding APIs (such as NVIDIA NIM or OpenRouter).

    Args:
        model_name: Model identifier (e.g. "nvidia/nv-embedqa-e5-v5",
          "snowflake/arctic-embed-l")
        base_url: e.g. "https://integrate.api.nvidia.com/v1" or
          "https://openrouter.ai/api/v1"
        api_key: API secret key. If omitted, reads from environment.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ValueError(
                f"API key is required for remote embedding model '{model_name}'. "
                "Set it in .env or pass api_key explicitly."
            )
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        import requests

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, object] = {
            "input": texts,
            "model": self.model_name,
        }
        # NVIDIA NIM API supports specifying input_type ("query" vs "passage")
        if "nvidia" in self.base_url:
            payload["input_type"] = input_type

        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding API request failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        items = sorted(data["data"], key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = 32
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._post(batch, input_type="passage"))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        results = self._post([text], input_type="query")
        return results[0]


def make_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Factory function creating local or cloud embedding models.

    Supported prefixes:
      - "nvidia:<model>"     -> Uses NVIDIA NIM API (reads NVIDIA_API_KEY from .env)
                                Default model: "nvidia/nv-embedqa-e5-v5"
      - "openrouter:<model>" -> Uses OpenRouter API (reads OPENROUTER_API_KEY from .env)
      - "<huggingface-name>" -> Uses local SentenceTransformers (e.g. "all-MiniLM-L6-v2")
    """
    if model_name.startswith("nvidia:"):
        actual_model = model_name.removeprefix("nvidia:") or "nvidia/nv-embedqa-e5-v5"
        api_key = os.environ.get("NVIDIA_API_KEY")
        return RemoteEmbeddingModel(
            model_name=actual_model,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )

    if model_name.startswith("openrouter:"):
        actual_model = model_name.removeprefix("openrouter:") or "openai/text-embedding-3-small"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        return RemoteEmbeddingModel(
            model_name=actual_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    return LocalEmbeddingModel(model_name)
