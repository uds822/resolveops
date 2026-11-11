"""Bounded Docling PDF pilot for a Cloud Run Job."""

import json
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from resolveops.ingestion.docling_pdf_parser import parse_pdf_with_docling
from resolveops.ingestion.models import BlockType


def _gcs_location(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected a GCS object URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def run_pdf_job(
    input_uri: str,
    output_uri: str,
    *,
    page_start: int,
    page_end: int,
    max_pages: int = 10,
    storage_client=None,
) -> dict:
    """Parse a small inclusive page range and create one retry-safe JSON object."""
    if page_start < 1 or page_end < page_start:
        raise ValueError("Invalid page range")
    if max_pages < 1 or page_end - page_start + 1 > max_pages:
        raise ValueError(f"Page range exceeds max_pages={max_pages}")
    input_bucket, input_name = _gcs_location(input_uri)
    output_bucket, output_name = _gcs_location(output_uri)
    if input_uri == output_uri:
        raise ValueError("Input and output objects must differ")

    if storage_client is None:
        from google.cloud import storage

        storage_client = storage.Client()

    started = time.monotonic()
    input_blob = storage_client.bucket(input_bucket).blob(input_name)
    output_blob = storage_client.bucket(output_bucket).blob(output_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.pdf"
        input_blob.download_to_filename(str(input_path))
        document = parse_pdf_with_docling(
            input_path,
            page_range=(page_start, page_end),
            source_id=input_uri,
        )
        # The worker's temporary path has no meaning once this job finishes.
        document.metadata.pop("file_path", None)
        table_blocks = [block for block in document.blocks if block.type == BlockType.TABLE]
        flagged_tables = [
            block for block in table_blocks
            if block.metadata.get("table_extraction", {}).get("validation_state") != "passed"
        ]
        review_reasons = []
        if not document.blocks:
            review_reasons.append("empty_document")
        if flagged_tables:
            review_reasons.append("table_extraction_needs_review")
        if any(
            block.page_number is None or not page_start <= block.page_number <= page_end
            for block in document.blocks
        ):
            review_reasons.append("block_page_outside_requested_range")
        validation = {
            "state": "needs_review" if review_reasons else "passed",
            "review_reasons": review_reasons,
            "flagged_table_ids": [block.id for block in flagged_tables],
        }
        peak_rss_mib = None
        if os.name == "posix":
            import resource

            peak_rss_mib = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        metrics = {
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_rss_mib": peak_rss_mib,
            "input_bytes": input_path.stat().st_size,
            "pages_processed": page_end - page_start + 1,
            "block_count": len(document.blocks),
            "table_count": len(table_blocks),
            "tables_needing_review": len(flagged_tables),
        }
        result = {
            "schema_version": 2,
            "document": document.model_dump(mode="json"),
            "validation": validation,
            "metrics": metrics,
        }
        payload = json.dumps(result, ensure_ascii=False).encode("utf-8")

    try:
        output_blob.upload_from_string(
            payload,
            content_type="application/json",
            if_generation_match=0,
        )
    except Exception as error:
        from google.api_core.exceptions import PreconditionFailed

        if not isinstance(error, PreconditionFailed):
            raise
        return {"status": "already_exists", "output_uri": output_uri}

    return {"status": "written", "output_uri": output_uri, "metrics": metrics}


def main() -> None:
    result = run_pdf_job(
        os.environ["INPUT_URI"],
        os.environ["OUTPUT_URI"],
        page_start=int(os.environ["PAGE_START"]),
        page_end=int(os.environ["PAGE_END"]),
        max_pages=int(os.environ.get("MAX_PAGES", "10")),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
