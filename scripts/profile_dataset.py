
from pathlib import Path
from collections import Counter
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/raw/aws"
SUPPORTED = {".pdf", ".html", ".htm", ".md"}

#collect all supported files in the data directory
files = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in SUPPORTED
]

if not files:
    raise SystemExit("No supported documents found.")

print("Total documents:", len(files))
print("Formats:", dict(Counter(p.suffix.lower() for p in files)))

for path in sorted(files):
    size_mb = path.stat().st_size / (1024 * 1024)
     

    if path.suffix.lower() == ".pdf":
        try:
            with pymupdf.open(str(path)) as pdf:
                pages = pdf.page_count
                print("Pages:", pages)

                if pages:
                    sample_ids = sorted({0, pages // 2, pages - 1})
                    for i in sample_ids:
                        page = pdf.load_page(i)
                        if page is None:
                            raise RuntimeError(
                                f"Could not load page {i + 1} of {pages}"
                            )
                        text = page.get_text("text")
                        print(f"  Page {i + 1}: {len(text.strip())} text chars")
        except Exception as exc:
            print("PDF inspection failed:", exc)
