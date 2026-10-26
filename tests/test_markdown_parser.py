
from resolveops.ingestion.markdown_parser import parse_markdown
from resolveops.ingestion.models import BlockType, SourceType


def test_markdown_extraction(tmp_path):
    path = tmp_path / "sample.md"

    path.write_text(
        "# S3 Versioning\n\n"
        "Versioning protects objects.\n\n"
        "- Enable versioning\n"
        "- Check permissions\n\n"
        "```bash\n"
        "aws s3 ls\n"
        "```\n\n"
        "| Service | Status |\n"
        "| --- | --- |\n"
        "| S3 | Enabled |\n",
        encoding="utf-8",
    )

    result = parse_markdown(path)
    blocks = result.blocks

    assert result.source_type == SourceType.MARKDOWN
    assert result.title == "S3 Versioning"
    assert len(blocks) == 6
    assert blocks[0].type == BlockType.HEADING
    assert blocks[0].content == "S3 Versioning"
    assert blocks[0].heading_level == 1
    assert blocks[1].type == BlockType.PARAGRAPH
    assert blocks[2].type == BlockType.LIST_ITEM
    assert blocks[3].type == BlockType.LIST_ITEM
    assert blocks[4].type == BlockType.CODE
    assert blocks[4].content == "aws s3 ls"
    assert blocks[4].language == "bash"
    assert blocks[5].type == BlockType.TABLE
    assert blocks[5].table_data == [
        ["Service", "Status"],
        ["S3", "Enabled"],
    ]
    assert [block.order for block in blocks] == list(range(6))


def test_markdown_ids_are_deterministic(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("# AWS\n\nDocumentation.", encoding="utf-8")

    first = parse_markdown(path)
    second = parse_markdown(path)

    assert first.id == second.id
    assert [block.id for block in first.blocks] == [
        block.id for block in second.blocks
    ]


def test_code_block_without_language(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("```\nplain code\n```\n", encoding="utf-8")

    result = parse_markdown(path)

    assert result.blocks[0].type == BlockType.CODE
    assert result.blocks[0].content == "plain code"
    assert result.blocks[0].language is None
