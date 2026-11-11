from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#defines the classes that hold our document data using Pydantic.

class SourceType(StrEnum):
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CODE = "code"
    TABLE = "table"
    PAGE_TEXT = "page_text"

#Preserving complex table structure
class TableCell(BaseModel):
    """One source cell in a table, retaining spans and header semantics."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    text: str

    column_header: bool = False
    row_header: bool = False
    row_section: bool = False
    fillable: bool = False

    bbox: list[float] | None = None
    bbox_origin: str | None = None


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: BlockType
    content: str
    order: int = Field(ge=0)

    page_number: int | None = Field(default=None, ge=1)
    heading_level: int | None = Field(default=None, ge=1, le=6)
    language: str | None = None
    table_data: list[list[str]] | None = None
    table_cells: list[TableCell] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    source_type: SourceType
    title: str | None = None

    blocks: list[Block] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
