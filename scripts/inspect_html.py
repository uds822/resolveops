
from pathlib import Path
from collections import Counter
from resolveops.ingestion.html_parser import parse_html

folder = Path("data/raw/aws/html")

for path in sorted(folder.glob("*.html")):
    print("\nDocument:", path.name)

    try:
        result = parse_html(path)
        blocks = result.blocks

        counts = Counter(block.type.value for block in blocks)

        print("Title:", result.title)
        print("Total blocks:", len(blocks))
        print("Types:", dict(counts))

        for block in blocks[:3]:
            print(str(block.model_dump(mode="json"))[:200])

    except Exception as e:
        print("FAILED:", e)
