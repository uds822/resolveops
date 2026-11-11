
from pathlib import Path
import time

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "data/raw/aws/pdf"
output = ROOT / "data/profiling/pdf"
output.mkdir(parents=True, exist_ok=True)
# This script extracts text from a sample of pages in the first PDF file found
# in the specified folder and saves the extracted text to separate files in the
# output directory. It also measures and prints the extraction time per page.

files = sorted(folder.glob("*.pdf"))

if not files:
    raise SystemExit("No PDF files found")

file = files[0]

#open the PDF file and extract text from selected pages using pymupdf. The selected pages are the first page, the last page, and the middle page of the document. The extracted text is saved to separate text files in the output directory, and the time taken for extraction is printed for each page.
with pymupdf.open(str(file)) as pdf:
    total = pdf.page_count
    if total <= 0:
        raise RuntimeError(f"PDF contains no pages: {file}")

    pages = sorted({
        min(10, total - 1),
        min(99, total - 1),
        total // 2
    })

    print("Document:", file.name)

    for i in pages:
        start = time.perf_counter()

        page = pdf.load_page(i)
        if page is None:
            raise RuntimeError(
                f"Could not load page {i + 1} of {total} from {file}"
            )

        text = page.get_text("text")

        duration = time.perf_counter() - start

        path = output / f"page_{i + 1}.txt"
        path.write_text(text, encoding="utf-8")

        print("Page:", i + 1)
        print("Characters:", len(text))
        print("Time:", round(duration, 4), "seconds")
        print("Saved:", path)
        print()
