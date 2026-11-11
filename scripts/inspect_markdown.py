
from pathlib import Path
from collections import Counter

from resolveops.ingestion.markdown_parser import parse_markdown

folder = Path("data/raw/aws/markdown")
files = sorted(folder.glob("*.md"))

print("Total Markdown files:", len(files))

for path in files:
    print("\nDocument:", path.name)

    try:
        result = parse_markdown(path)
        blocks = result.blocks

        counts = Counter(block.type.value for block in blocks)

        print("Total blocks:", len(blocks))
        print("Block types:", dict(counts))

        headings = [
            block.content for block in blocks
            if block.type.value == "heading"
        ]

        print("First headings:", headings[:3])

        if not blocks:
            print("WARNING: No content extracted")

        raw = path.read_text(encoding="utf-8-sig")

        table_lines = [
            line for line in raw.splitlines()
            if line.strip().startswith("|")
        ]

        if table_lines:
            print("REVIEW: Possible Markdown table found")

    except Exception as e:
        print("FAILED:", e)
