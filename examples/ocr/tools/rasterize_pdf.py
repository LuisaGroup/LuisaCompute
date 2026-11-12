#!/usr/bin/env python3
"""rasterize_pdf.py -- rasterize PDF pages to images OUTSIDE the inference process.

Offline tool. Uses an external PDF library (pypdfium2, imported lazily) with
strict file/page/pixel limits and NO network access. The inference engine
never parses PDFs in-process; it consumes the page images this tool produces
plus a document manifest (see ocr/document.h).

Standard-library-only manifest writing; the PDF library is required only for
actual rasterization and is never downloaded by this tool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _die(msg: str):
    print(f"rasterize_pdf: error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rasterize_pdf.py",
        description="Rasterize PDF pages to PNG images + a document manifest (offline, external PDF library).",
    )
    p.add_argument("--pdf", required=True, help="Local PDF path (no download).")
    p.add_argument("--output-dir", required=True, help="Directory for page images + manifest.")
    p.add_argument("--dpi", type=int, default=150, help="Rasterization DPI (default 150).")
    p.add_argument("--max-pages", type=int, default=64, help="Maximum pages to rasterize.")
    p.add_argument("--max-file-bytes", type=int, default=128 * 1024 * 1024,
                   help="Maximum PDF file size.")
    p.add_argument("--max-pixels", type=int, default=64 * 1024 * 1024,
                   help="Maximum pixels per rendered page.")
    p.add_argument("--pages", default="", help="Optional 0-based page range 'a-b' (inclusive).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.pdf):
        _die(f"PDF not found: {args.pdf}")
    file_size = os.path.getsize(args.pdf)
    if file_size > args.max_file_bytes:
        _die(f"PDF exceeds the file-size limit ({file_size} > {args.max_file_bytes})")
    if args.dpi <= 0 or args.dpi > 1200:
        _die("DPI must be in (0, 1200]")
    if args.max_pages <= 0:
        _die("--max-pages must be positive")
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError:
        _die("pypdfium2 is required for rasterization (pip install pypdfium2); "
             "this tool never downloads it")
    os.makedirs(args.output_dir, exist_ok=True)
    pdf = pdfium.PdfDocument(args.pdf)
    try:
        total = len(pdf)
        first, last = 0, total - 1
        if args.pages:
            try:
                a, b = args.pages.split("-")
                first, last = int(a), int(b)
            except ValueError:
                _die("--pages must be 'a-b' (0-based, inclusive)")
            if first < 0 or last >= total or first > last:
                _die(f"page range out of bounds (document has {total} pages)")
        if last - first + 1 > args.max_pages:
            _die(f"requested page count exceeds --max-pages ({args.max_pages})")
        pages = []
        scale = args.dpi / 72.0
        for index in range(first, last + 1):
            page = pdf[index]
            width_pt, height_pt = page.get_size()
            est_pixels = int(width_pt * scale) * int(height_pt * scale)
            if est_pixels > args.max_pixels:
                _die(f"page {index} would exceed the pixel limit ({est_pixels} > {args.max_pixels})")
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            name = f"page_{index:04d}.png"
            path = os.path.join(args.output_dir, name)
            pil_image.save(path)
            pages.append({
                "source_page_index": index,
                "image": name,
                "width": pil_image.width,
                "height": pil_image.height,
            })
        manifest = {
            "format_version": 1,
            "producer": "rasterize_pdf.py",
            "source": os.path.basename(args.pdf),
            "rasterizer": f"pypdfium2 {getattr(pdfium, '__version__', 'unknown')}",
            "dpi": args.dpi,
            "pages": pages,
        }
        manifest_path = os.path.join(args.output_dir, "document_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"rasterize_pdf: wrote {len(pages)} pages to {args.output_dir}")
    finally:
        pdf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())