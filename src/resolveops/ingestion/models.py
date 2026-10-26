from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    source_type: SourceType
    title: str | None = None

    blocks: list[Block] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)