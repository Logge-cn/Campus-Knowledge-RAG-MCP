"""Create the declared image-only acceptance PDF from an official native source."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image, ImageEnhance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--contrast", type=float, default=0.96)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    source = pymupdf.open(args.input_pdf)
    output = pymupdf.open()
    scale = args.dpi / 72
    for source_page in source:
        pixmap = source_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY)
        image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
        image = ImageEnhance.Contrast(image).enhance(args.contrast)
        encoded = BytesIO()
        image.save(encoded, format="JPEG", quality=args.jpeg_quality, optimize=True)
        output_page = output.new_page(width=source_page.rect.width, height=source_page.rect.height)
        output_page.insert_image(output_page.rect, stream=encoded.getvalue())

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output_pdf, garbage=4, deflate=True)
    output.close()
    source.close()


if __name__ == "__main__":
    main()
