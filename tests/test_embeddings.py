from unittest.mock import MagicMock

import numpy as np
import pytest

from resolveops.retrieval.embeddings import (
    EmbeddingModel,
    LocalEmbeddingModel,
    RemoteEmbeddingModel,
    make_embedding_model,
)


class FakeSentenceTransformer:
    """A fast fake model for tests that doesn't need internet or GPU."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)
        return np.array(
            [[0.1, 0.2, 0.3] for _ in texts],
            dtype=np.float32,
        )


def test_make_embedding_model_returns_instance(monkeypatch):
    """Verify factory returns an EmbeddingModel instance."""
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
    )
    model = make_embedding_model("test-model")
    assert isinstance(model, EmbeddingModel)
    assert model.model_name == "test-model"


def test_embed_documents_returns_list_of_float_vectors(monkeypatch):
    """Verify embed_documents converts list of strings into 2D float list."""
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
    )
    model = LocalEmbeddingModel("test-model")
    chunks = ["First paragraph about DynamoDB.", "Second paragraph about limits."]

    vectors = model.embed_documents(chunks)

    assert len(vectors) == 2
    assert vectors[0] == pytest.approx([0.1, 0.2, 0.3])
    assert vectors[1] == pytest.approx([0.1, 0.2, 0.3])
    assert isinstance(vectors[0], list)
    assert isinstance(vectors[0][0], float)


def test_embed_query_returns_single_float_vector(monkeypatch):
    """Verify embed_query converts single string into 1D float list."""
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
    )
    model = LocalEmbeddingModel("test-model")

    vector = model.embed_query("What is the max item size?")

    assert len(vector) == 3
    assert vector == pytest.approx([0.1, 0.2, 0.3])
    assert isinstance(vector, list)
    assert isinstance(vector[0], float)


def test_missing_sentence_transformers_raises_helpful_error(monkeypatch):
    """Verify helpful message if dependency is missing."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="sentence-transformers is required"):
        LocalEmbeddingModel()


def test_remote_embedding_requires_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        RemoteEmbeddingModel(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key=None,
        )


def test_remote_embedding_calls_api(monkeypatch):
    import requests

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.5, 0.6, 0.7]},
            {"index": 1, "embedding": [0.8, 0.9, 1.0]},
        ]
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    model = RemoteEmbeddingModel(
        model_name="nvidia/nv-embedqa-e5-v5",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="test_api_key",
    )

    doc_vectors = model.embed_documents(["doc 1", "doc 2"])
    assert len(doc_vectors) == 2
    assert doc_vectors[0] == [0.5, 0.6, 0.7]

    query_mock_response = MagicMock()
    query_mock_response.status_code = 200
    query_mock_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: query_mock_response)

    query_vector = model.embed_query("query text")
    assert query_vector == [0.1, 0.2, 0.3]


def test_make_embedding_model_with_nvidia_prefix(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv_test_key_123")
    model = make_embedding_model("nvidia:nvidia/nv-embedqa-e5-v5")
    assert isinstance(model, RemoteEmbeddingModel)
    assert model.model_name == "nvidia/nv-embedqa-e5-v5"
    assert model.base_url == "https://integrate.api.nvidia.com/v1"



def test_remote_embedding_requires_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        RemoteEmbeddingModel(
            model_name="test-model",
            base_url="https://api.test.com/v1",
            api_key=None,
        )


def test_remote_embedding_calls_api(monkeypatch):
    import requests

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.5, 0.6, 0.7]},
            {"index": 1, "embedding": [0.8, 0.9, 1.0]},
        ]
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

    model = RemoteEmbeddingModel(
        model_name="nvidia/nv-embedqa-e5-v5",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="test_api_key",
    )

    doc_vectors = model.embed_documents(["doc 1", "doc 2"])
    assert len(doc_vectors) == 2
    assert doc_vectors[0] == [0.5, 0.6, 0.7]

    query_mock_response = MagicMock()
    query_mock_response.status_code = 200
    query_mock_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: query_mock_response)

    query_vector = model.embed_query("query text")
    assert query_vector == [0.1, 0.2, 0.3]


def test_make_embedding_model_with_nvidia_prefix(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv_test_key_123")
    model = make_embedding_model("nvidia:nvidia/nv-embedqa-e5-v5")
    assert isinstance(model, RemoteEmbeddingModel)
    assert model.model_name == "nvidia/nv-embedqa-e5-v5"
    assert model.base_url == "https://integrate.api.nvidia.com/v1"