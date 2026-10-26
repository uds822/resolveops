
from pathlib import Path

from markdown_it import MarkdownIt

from resolveops.ingestion.models import Block, BlockType, Document, SourceType
from resolveops.ingestion.normalization import (
    build_block,
    make_document_id,
    resolve_source,
    table_to_text,
)


def _extract_table(tokens, start_index: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    current_row: list[str] | None = None
    index = start_index + 1

    while index < len(tokens):
        token = tokens[index]

        if token.type == "tr_open":
            current_row = []
        elif token.type in {"th_open", "td_open"}:
            cell_text = ""
            if index + 1 < len(tokens):
                cell_text = tokens[index + 1].content.strip()
            if current_row is not None:
                current_row.append(cell_text)
        elif token.type == "tr_close" and current_row is not None:
            rows.append(current_row)
            current_row = None
        elif token.type == "table_close":
            return rows, index

        index += 1

    return rows, index


def parse_markdown(
    file_path: str | Path,
    *,
    source_id: str | None = None,
) -> Document:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8-sig")
    source = resolve_source(path, source_id)
    document_id = make_document_id(SourceType.MARKDOWN, source)

    parser = MarkdownIt("commonmark").enable("table")
    tokens = parser.parse(content)

    blocks: list[Block] = []
    list_depth = 0
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token.type == "list_item_open":
            list_depth += 1
        elif token.type == "list_item_close":
            list_depth -= 1
        elif token.type == "table_open":
            rows, table_end = _extract_table(tokens, index)

            if rows:
                text = table_to_text(rows)
                blocks.append(
                    build_block(
                        document_id=document_id,
                        block_type=BlockType.TABLE,
                        content=text,
                        order=len(blocks),
                        table_data=rows,
                    )
                )

            index = table_end + 1
            continue
        elif token.type in {"heading_open", "paragraph_open"}:
            text = tokens[index + 1].content.strip()

            if not text:
                index += 1
                continue

            if token.type == "heading_open":
                block_type = BlockType.HEADING
                heading_level = int(token.tag[1:])
            elif list_depth > 0:
                block_type = BlockType.LIST_ITEM
                heading_level = None
            else:
                block_type = BlockType.PARAGRAPH
                heading_level = None

            blocks.append(
                build_block(
                    document_id=document_id,
                    block_type=block_type,
                    content=text,
                    order=len(blocks),
                    heading_level=heading_level,
                )
            )
        elif token.type in {"fence", "code_block"}:
            text = token.content.rstrip("\n")
            if text:
                info = token.info.strip()
                language = info.split(maxsplit=1)[0] if info else None
                blocks.append(
                    build_block(
                        document_id=document_id,
                        block_type=BlockType.CODE,
                        content=text,
                        order=len(blocks),
                        language=language,
                    )
                )

        index += 1

    title = next(
        (
            block.content
            for block in blocks
            if block.type == BlockType.HEADING
        ),
        path.stem,
    )

    return Document(
        id=document_id,
        source=source,
        source_type=SourceType.MARKDOWN,
        title=title,
        blocks=blocks,
        metadata={"file_path": str(path)},
    )
