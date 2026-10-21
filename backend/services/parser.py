# backend/services/parser.py
import fitz
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# Create executor for CPU-bound work
_executor = ThreadPoolExecutor(max_workers=2)

def _extract_sync(file_bytes):
    """Synchronous PDF extraction"""
    pages_text = []
    with fitz.open(stream=file_bytes, filetype='pdf') as doc:
        for page in doc:
            pages_text.append(page.get_text())
    return "\n".join(pages_text).strip()

async def extract_text_from_pdf(file_bytes):
    """Async wrapper for PDF extraction"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _extract_sync, file_bytes)


def extract_resume_style(file_bytes):
    """Read the two most prominent chromatic text colors from an uploaded PDF."""
    colors = Counter()
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = (span.get("text") or "").strip()
                        if not text:
                            continue
                        color = int(span.get("color", 0))
                        red, green, blue = (color >> 16) & 255, (color >> 8) & 255, color & 255
                        high, low = max(red, green, blue), min(red, green, blue)
                        saturation = (high - low) / high if high else 0
                        luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
                        if saturation >= 0.18 and 0.10 <= luminance <= 0.88:
                            colors[f"{red:02X}{green:02X}{blue:02X}"] += len(text)

    selected = [color for color, _ in colors.most_common(2)]
    if not selected:
        selected = ["222222"]
    if len(selected) == 1:
        selected.append(selected[0])
    return {"accent_color": selected[0], "secondary_color": selected[1]}
