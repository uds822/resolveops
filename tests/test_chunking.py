from resolveops.ingestion.chunking import chunk_document
from resolveops.ingestion.models import Block, BlockType, Document, SourceType, TableCell


def test_chunker_keeps_table_whole_with_heading_and_citation():
    document = Document(
        id="doc-1", source="gs://input/guide.pdf", source_type=SourceType.PDF,
        title="Guide", blocks=[
            Block(id="h1", type=BlockType.HEADING, content="Access", order=0,
                  page_number=2, heading_level=1),
            Block(id="p1", type=BlockType.PARAGRAPH, content="Before table", order=1,
                  page_number=2),
            Block(id="t1", type=BlockType.TABLE,
                  content="Name | Value\nA | 1", order=2, page_number=3,
                  table_data=[["Name", "Value"], ["A", "1"]],
                  table_cells=[TableCell(row_index=0, column_index=0,
                                         text="Name", column_header=True)],
                  metadata={"bbox": [1, 2, 3, 4], "table_extraction": {
                      "validation_state": "passed"}}),
            Block(id="p2", type=BlockType.PARAGRAPH, content="After table", order=3,
                  page_number=3),
        ],
    )

    chunks = chunk_document(document, max_chars=15)

    assert [chunk.kind for chunk in chunks] == ["text", "table", "text"]
    assert chunks[1].content == "Access\n\nName | Value\nA | 1"
    assert chunks[1].page_numbers == [3]
    assert chunks[1].context_block_ids == ["h1"]
    assert chunks[1].table_data == [["Name", "Value"], ["A", "1"]]
    assert chunks[1].table_cells[0].column_header is True
    assert chunks[1].indexable is True
    assert [chunk.id for chunk in chunk_document(document, max_chars=15)] == [
        chunk.id for chunk in chunks
    ]


def test_chunker_marks_review_table_unindexable():
    document = Document(
        id="doc-1", source="source", source_type=SourceType.PDF,
        blocks=[Block(id="t1", type=BlockType.TABLE, content="", order=0,
                      metadata={"table_extraction": {
                          "validation_state": "needs_review",
                          "review_reasons": ["empty_table"]}})],
    )

    chunk = chunk_document(document)[0]
    assert chunk.kind == "table"
    assert chunk.indexable is False
    assert chunk.metadata["table_extraction"]["review_reasons"] == ["empty_table"]
