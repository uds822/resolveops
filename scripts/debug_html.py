
from pathlib import Path
from resolveops.ingestion.html_parser import parse_html

folder = Path("data/raw/aws/html")

path = next(folder.glob("Troubleshoot access denied*.html"))

result = parse_html(path)

for block in result.blocks:
    text = block.content

    if "implicit denial" in text.lower():
        print("EXTRACTED TEXT:")
        print(repr(text))
        break

html = path.read_text(encoding="utf-8")

pos = html.lower().find("implicit denial")

if pos != -1:
    print("\nORIGINAL HTML:")
    print(html[pos:pos + 500])
else:
    print("\nPhrase not found directly in raw HTML")
