import json
from pathlib import Path

import pytest

from resolveops.ingestion.models import Block, BlockType, Document, SourceType
from resolveops.jobs import pdf_job


class FakeBlob:
    def __init__(self):
        self.payload = None
        self.upload_options = None

    def download_to_filename(self, filename):
        Path(filename).write_bytes(b"PDF bytes")

    def upload_from_string(self, payload, **options):
        self.payload = payload
        self.upload_options = options


class FakeStorage:
    def __init__(self):
        self.blobs = {}

    def bucket(self, name):
        storage = self

        class Bucket:
            def blob(self, path):
                return storage.blobs.setdefault((name, path), FakeBlob())

        return Bucket()


def test_pdf_job_saves_bounded_parser_output(monkeypatch):
    client = FakeStorage()

    def fake_parse(path, *, page_range, source_id):
        assert path.is_file()
        assert page_range == (58, 58)
        assert source_id == "gs://input/guide.pdf"
        return Document(
            id="doc-1", source=source_id, source_type=SourceType.PDF,
            blocks=[Block(id="p1", type=BlockType.PARAGRAPH,
                          content="A paragraph", order=0, page_number=58)],
            metadata={"file_path": str(path)},
        )

    monkeypatch.setattr(pdf_job, "parse_pdf_with_docling", fake_parse)
    result = pdf_job.run_pdf_job(
        "gs://input/guide.pdf", "gs://output/page-58.json",
        page_start=58, page_end=58, storage_client=client,
    )

    blob = client.blobs[("output", "page-58.json")]
    saved = json.loads(blob.payload)
    assert result["status"] == "written"
    assert saved["schema_version"] == 2
    assert saved["document"]["source"] == "gs://input/guide.pdf"
    assert "file_path" not in saved["document"]["metadata"]
    assert "chunks" not in saved
    assert saved["validation"] == {
        "state": "passed", "review_reasons": [], "flagged_table_ids": []
    }
    assert saved["metrics"]["pages_processed"] == 1
    assert saved["metrics"]["table_count"] == 0
    assert blob.upload_options["if_generation_match"] == 0


def test_pdf_job_flags_suspicious_table_without_chunking(monkeypatch):
    client = FakeStorage()

    def fake_parse(path, *, page_range, source_id):
        return Document(
            id="doc-1", source=source_id, source_type=SourceType.PDF,
            blocks=[Block(
                id="table-1", type=BlockType.TABLE, content="",
                order=0, page_number=58,
                metadata={"table_extraction": {"validation_state": "needs_review"}},
            )],
        )

    monkeypatch.setattr(pdf_job, "parse_pdf_with_docling", fake_parse)
    pdf_job.run_pdf_job(
        "gs://input/guide.pdf", "gs://output/page-58.json",
        page_start=58, page_end=58, storage_client=client,
    )

    saved = json.loads(client.blobs[("output", "page-58.json")].payload)
    assert saved["validation"] == {
        "state": "needs_review",
        "review_reasons": ["table_extraction_needs_review"],
        "flagged_table_ids": ["table-1"],
    }
    assert saved["metrics"]["tables_needing_review"] == 1
    assert "chunks" not in saved


def test_pdf_job_rejects_unbounded_work():
    with pytest.raises(ValueError, match="exceeds max_pages"):
        pdf_job.run_pdf_job(
            "gs://input/guide.pdf", "gs://output/result.json",
            page_start=1, page_end=11, storage_client=FakeStorage(),
        )
