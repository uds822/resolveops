
import pytest

from resolveops.ingestion.html_parser import parse_html
from resolveops.ingestion.models import BlockType, SourceType



def test_whitespace_cleaning(tmp_path):
    path = tmp_path / "sample.html"

    path.write_text("""
    <main>
        <p>Access denied
           because of missing permissions.</p>
        <pre>line1
    line2</pre>
    </main>
    """, encoding="utf-8")

    result = parse_html(path)
    blocks = result.blocks

    assert result.source_type == SourceType.HTML
    assert blocks[0].content == (
        "Access denied because of missing permissions."
    )

    assert blocks[1].type == BlockType.CODE
    assert "\n" in blocks[1].content


def test_html_preserves_structure(tmp_path):
    path = tmp_path / "sample.html"
    path.write_text(
        "<html><head><title>AWS Guide</title></head><body><main>"
        "<h2>Permissions</h2>"
        "<pre><code class='language-python'>print('ok')</code></pre>"
        "<table><tr><th>Role</th><th>Status</th></tr>"
        "<tr><td>Admin</td><td>Allowed</td></tr></table>"
        "</main></body></html>",
        encoding="utf-8",
    )

    result = parse_html(path)

    assert result.title == "AWS Guide"
    assert result.blocks[0].type == BlockType.HEADING
    assert result.blocks[0].heading_level == 2
    assert result.blocks[1].type == BlockType.CODE
    assert result.blocks[1].language == "python"
    assert result.blocks[2].type == BlockType.TABLE
    assert result.blocks[2].table_data == [
        ["Role", "Status"],
        ["Admin", "Allowed"],
    ]


def test_html_requires_main_content(tmp_path):
    path = tmp_path / "sample.html"
    path.write_text("<html><body><p>Outside article</p></body></html>")

    with pytest.raises(ValueError, match="Main article not found"):
        parse_html(path)
