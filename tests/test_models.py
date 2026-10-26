import pytest
from pydantic import ValidationError

from resolveops.ingestion.models import (
    Block,
    BlockType,
    Document,
    SourceType,
)

def test_document_accepts_valid_structure():
    block = Block(
        id="block-1",
        type=BlockType.HEADING,
        content="Getting started",
        order=0,
        heading_level=1,
    )

    document = Document(
        id="document-1",
        source="guide.md",
        source_type=SourceType.MARKDOWN,
        title="Developer Guide",
        blocks=[block],
    )

    assert document.source_type == SourceType.MARKDOWN
    assert document.blocks[0].content == "Getting started"
    assert document.blocks[0].order == 0


def test_block_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Block(
            id="block-1",
            type=BlockType.PARAGRAPH,
            content="Example paragraph",
            order=0,
            unexpected_field="invalid",
        )


def test_block_rejects_invalid_page_number():
    with pytest.raises(ValidationError):
        Block(
            id="block-1",
            type=BlockType.PAGE_TEXT,
            content="Page content",
            order=0,
            page_number=0,
        )
