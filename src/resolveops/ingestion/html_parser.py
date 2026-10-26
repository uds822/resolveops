
import re
from pathlib import Path

from bs4 import BeautifulSoup

from resolveops.ingestion.models import Block, BlockType, Document, SourceType
from resolveops.ingestion.normalization import (
    build_block,
    make_document_id,
    resolve_source,
    table_to_text,
)


def _code_language(tag) -> str | None:
    code = tag.find("code")
    if code is None:
        return None

    for class_name in code.get("class", []):
        if class_name.startswith("language-"):
            return class_name.removeprefix("language-") or None

    return None

#This function parses an HTML file and extracts its content into a structured Document object. It identifies the main content area of the HTML, removes unnecessary elements, and processes various HTML tags to create blocks of text, headings, lists, code snippets, and tables. Each block is assigned a unique ID and relevant metadata. The function returns a Document object containing all extracted blocks and associated information.
def parse_html(
    file_path: str | Path,
    *,
    source_id: str | None = None,
) -> Document:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    html = path.read_text(encoding="utf-8")
    source = resolve_source(path, source_id)
    document_id = make_document_id(SourceType.HTML, source)
    soup = BeautifulSoup(html, "html.parser")

    main = None
    selectors = [
        "#main-col-body",
        "#main-content",
        "article",
        "main",
        '[role="main"]'
    ]

    for selector in selectors:
        main = soup.select_one(selector)
        if main:
            break

    if main is None:
        raise ValueError("Main article not found")

    for tag in main.select("script, style, nav, footer, aside"):
        tag.decompose()

    blocks: list[Block] = []
    tags = [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "pre",
        "table",
    ]

    for tag in main.find_all(tags):
        if tag.find_parent(["pre", "table"]):
            continue

        if tag.name == "p" and tag.find_parent("li"):
            continue

        if tag.name == "li" and tag.find_parent("li"):
            continue

        if tag.name == "table":
            rows: list[list[str]] = []

            for tr in tag.find_all("tr"):
                cells = [
                    c.get_text(" ", strip=True)
                    for c in tr.find_all(
                        ["th", "td"], recursive=False
                    )
                ]
                if cells:
                    rows.append(cells)

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
            continue

        if tag.name.startswith("h"):
            block_type = BlockType.HEADING
            heading_level = int(tag.name[1:])
        elif tag.name == "pre":
            block_type = BlockType.CODE
            heading_level = None
        elif tag.name == "li":
            block_type = BlockType.LIST_ITEM
            heading_level = None
        else:
            block_type = BlockType.PARAGRAPH
            heading_level = None

        if block_type == BlockType.CODE:
            text = tag.get_text().strip()
        else:
            text = tag.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text)

        if text:
            blocks.append(
                build_block(
                    document_id=document_id,
                    block_type=block_type,
                    content=text,
                    order=len(blocks),
                    heading_level=heading_level,
                    language=(
                        _code_language(tag)
                        if block_type == BlockType.CODE
                        else None
                    ),
                )
            )

    title = soup.title.get_text(" ", strip=True) if soup.title else path.stem

    return Document(
        id=document_id,
        source=source,
        source_type=SourceType.HTML,
        title=title,
        blocks=blocks,
        metadata={"file_path": str(path)},
    )
