"""Gemini-based OCR pipeline for invoice_reorder.

Drop-in alternative to reorder_invoices.run() that calls Google Gemini instead
of Anthropic Claude. All shared logic (PDF rendering, matching, output writing)
is imported from reorder_invoices — nothing is duplicated here.

Requires GOOGLE_API_KEY in the environment.
"""
from __future__ import annotations

import gc as _gc
import json as _json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Optional

import google.ai.generativelanguage as glm
import google.generativeai as genai

import reorder_invoices as ri
from gemini.pricing import GEMINI_DEFAULT_MODEL, GeminiUsageTotals

log = logging.getLogger("gemini.pipeline")

# Plain dict schemas for Gemini response_schema — Pydantic models with
# default=None fields crash the google-generativeai protobuf converter.
_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "supplier":  {"type": "string", "nullable": True},
        "amount":    {"type": "string", "nullable": True},
        "date":      {"type": "string", "nullable": True},
        "id_number": {"type": "string", "nullable": True},
    },
}

_TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "supplier":  {"type": "string", "nullable": True},
                    "amount":    {"type": "string", "nullable": True},
                    "date_raw":  {"type": "string", "nullable": True},
                    "id_number": {"type": "string", "nullable": True},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Fatal-error detection for Gemini (mirrors ri._is_fatal_api_error)
# ---------------------------------------------------------------------------
def _is_fatal_gemini_error(e: Exception) -> bool:
    """Return True for errors that will not improve with retries."""
    try:
        import google.auth.exceptions as _gauth
        import google.api_core.exceptions as _gcore
        if isinstance(e, (_gauth.TransportError, _gcore.PermissionDenied,
                          _gcore.Unauthenticated, _gcore.ResourceExhausted)):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return "api key not valid" in msg or "quota exceeded" in msg or "permission denied" in msg


# ---------------------------------------------------------------------------
# Single-page invoice extraction
# ---------------------------------------------------------------------------
def extract_fields_from_image_gemini(
    invoice_model: genai.GenerativeModel,
    image_bytes: bytes,
    model_name: str,
    usage: GeminiUsageTotals,
) -> ri._ExtractedInvoice:
    """Send one page image to Gemini and return parsed invoice fields."""
    mime_type = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
    image_part = glm.Part(inline_data=glm.Blob(mime_type=mime_type, data=image_bytes))
    response = invoice_model.generate_content([image_part, "Extract the invoice fields."])
    usage.add(getattr(response, "usage_metadata", None), model_name)
    return ri._ExtractedInvoice.model_validate_json(response.text)


def _try_extract_page_gemini(
    invoice_model: genai.GenerativeModel,
    img_bytes: bytes,
    page_index: int,
    *,
    model_name: str,
    usage: GeminiUsageTotals,
) -> ri.OcrPage:
    """Extract one page; on non-fatal error returns an all-null OcrPage."""
    try:
        ex = extract_fields_from_image_gemini(invoice_model, img_bytes, model_name, usage)
    except Exception as e:
        if _is_fatal_gemini_error(e):
            raise
        log.error("page %d extraction failed (%s): %s", page_index, type(e).__name__, e)
        return ri.OcrPage(page_index=page_index, supplier=None, amount=None, date=None, id_number=None)
    return ri.OcrPage(
        page_index=page_index,
        supplier=ex.supplier,
        amount=ri._norm_amount(ex.amount),
        date=ri._norm_date(ex.date),
        id_number=ex.id_number,
    )


# ---------------------------------------------------------------------------
# Full scanned-PDF extraction
# ---------------------------------------------------------------------------
_BATCH_SIZE = 10  # pages per fitz open/close cycle — keeps mmap'd PDF data bounded
_MAX_IMG_PIXELS = 1240 * 1754  # ~150 DPI A4; extracted images above this are resized


def _malloc_trim() -> None:
    """Return free C-heap pages to the OS. No-op on non-Linux."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _page_to_jpeg(doc, page_index: int, dpi: int) -> bytes:
    """Extract or render one PDF page as JPEG bytes, capped at _MAX_IMG_PIXELS.

    For scanned PDFs with a single JPEG/PNG image per page, extracts the
    original compressed bytes directly (no pixmap, far less memory). High-res
    images (> _MAX_IMG_PIXELS) are downscaled so gRPC send-buffers stay
    bounded across 100+ pages.

    Falls back to fitz rendering for CCITT/JBIG2/multi-image pages — fitz
    handles those formats natively even though PIL does not.
    """
    import fitz
    import math as _math
    from PIL import Image as _PILImage
    import io as _io

    page = doc[page_index]
    img_bytes: bytes | None = None
    images = page.get_images(full=True)
    if len(images) == 1:
        try:
            img_data = doc.extract_image(images[0][0])
            ext = (img_data.get("ext") or "").lower()
            if ext in ("jpg", "jpeg", "png"):
                img_bytes = img_data["image"]
                log.debug("page %d: direct-extracted %s (%d bytes)", page_index, ext, len(img_bytes))
            else:
                log.debug("page %d: unsupported embedded format %r, using fitz render", page_index, ext)
        except Exception as e:
            log.debug("page %d: extract_image failed (%s), using fitz render", page_index, e)

    if img_bytes is not None:
        try:
            im = _PILImage.open(_io.BytesIO(img_bytes))
            w, h = im.size
            if w * h > _MAX_IMG_PIXELS:
                scale = _math.sqrt(_MAX_IMG_PIXELS / (w * h))
                new_w = max(100, int(w * scale))
                new_h = max(140, int(h * scale))
                log.debug("page %d: resizing %dx%d → %dx%d", page_index, w, h, new_w, new_h)
                im_small = im.resize((new_w, new_h), _PILImage.LANCZOS)
                im.close(); del im
                buf = _io.BytesIO()
                im_small.save(buf, format="JPEG", quality=85)
                im_small.close(); del im_small
                return buf.getvalue()
            im.close(); del im
            return img_bytes
        except Exception as e:
            log.debug("page %d: PIL failed (%s), falling back to fitz render", page_index, e)

    # Fallback: render the page via fitz at target DPI.
    # Handles CCITT, JBIG2, multi-image pages, and any PIL-unsupported format.
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    result = pix.tobytes(output="jpeg")
    del pix
    return result

def extract_pages_with_gemini(
    pdf_path: str | Path,
    invoice_model: genai.GenerativeModel,
    model_name: str = GEMINI_DEFAULT_MODEL,
    dpi: int = ri.RENDER_DPI,
    max_pages: Optional[int] = None,
    retry_with_rotation: bool = True,
    usage: Optional[GeminiUsageTotals] = None,
) -> list[ri.OcrPage]:
    """Render each page of the scanned PDF and extract fields via Gemini vision.

    The PDF is opened and closed every _BATCH_SIZE pages so the OS can reclaim
    the memory-mapped PDF data between batches. Without this, a 100-page scanned
    PDF can hold 100-300 MB of mmap'd data in RAM throughout the entire run.
    """
    import fitz
    if usage is None:
        usage = GeminiUsageTotals()

    # Quick open just to get the page count, then close immediately.
    try:
        _tmp = fitz.open(str(pdf_path))
        n = len(_tmp)
        _tmp.close()
        del _tmp
    except Exception as e:
        log.error("extract_pages_with_gemini: failed to open %s: %s", pdf_path, e)
        return []

    limit = min(n, max_pages) if max_pages and max_pages > 0 else n
    pages: list[ri.OcrPage] = []

    for batch_start in range(0, limit, _BATCH_SIZE):
        batch_end = min(batch_start + _BATCH_SIZE, limit)
        log.info("extract_pages_with_gemini: batch pages %d–%d / %d", batch_start, batch_end - 1, limit)
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            log.error("extract_pages_with_gemini: reopen failed at batch %d: %s", batch_start, e)
            break
        try:
            for i in range(batch_start, batch_end):
                try:
                    img = _page_to_jpeg(doc, i, dpi)
                except Exception as e:
                    log.error("page %d image failed: %s (%s)", i, e, type(e).__name__)
                    pages.append(ri.OcrPage(page_index=i, supplier=None, amount=None, date=None, id_number=None))
                    continue
                page = _try_extract_page_gemini(invoice_model, img, i, model_name=model_name, usage=usage)
                if retry_with_rotation and ri._likely_misread(page):
                    best = page
                    log.info("page %d looked misread (non-null=%d) — trying rotations", i, ri._count_non_null(page))
                    for rotation in (180, 90, 270):
                        try:
                            rotated_bytes = ri._rotate_png_bytes(img, rotation)
                        except Exception as e:
                            log.warning("page %d rotation %d° render failed: %s", i, rotation, e)
                            continue
                        candidate = _try_extract_page_gemini(invoice_model, rotated_bytes, i,
                                                             model_name=model_name, usage=usage)
                        del rotated_bytes
                        if ri._count_non_null(candidate) > ri._count_non_null(best):
                            log.info("page %d: rotation %d° improved (%d -> %d non-null)",
                                     i, rotation, ri._count_non_null(best), ri._count_non_null(candidate))
                            best = candidate
                        if ri._count_non_null(best) >= 3:
                            break
                    page = best
                del img
                pages.append(page)
                _gc.collect()
                _malloc_trim()
                log.info("page %d: supplier=%r amount=%s date=%s id=%s",
                         i, page.supplier, page.amount, page.date, page.id_number)
        finally:
            doc.close()
            del doc
        _gc.collect()
        _malloc_trim()

    return pages


# ---------------------------------------------------------------------------
# Table PDF extraction
# ---------------------------------------------------------------------------
def parse_table_pdf_with_gemini(
    pdf_path: str | Path,
    table_model: genai.GenerativeModel,
    model_name: str = GEMINI_DEFAULT_MODEL,
    dpi: int = ri.RENDER_DPI,
    usage: Optional[GeminiUsageTotals] = None,
) -> list[ri.TableRow]:
    """Read a Hebrew payment-order PDF via Gemini vision."""
    if usage is None:
        usage = GeminiUsageTotals()
    images = ri.render_pdf_pages_to_png(pdf_path, dpi=dpi)
    all_rows: list[ri.TableRow] = []
    for page_i, img_bytes in enumerate(images):
        img_part = glm.Part(inline_data=glm.Blob(mime_type="image/png", data=img_bytes))
        try:
            response = table_model.generate_content(
                [img_part, "Extract every invoice row from this table page."]
            )
            usage.add(getattr(response, "usage_metadata", None), model_name)
        except Exception as e:
            if _is_fatal_gemini_error(e):
                raise
            log.error("table page %d extraction failed (%s): %s", page_i, type(e).__name__, e)
            continue
        try:
            extraction = ri._TableExtraction.model_validate_json(response.text)
        except Exception as e:
            log.error("table page %d JSON parse failed: %s", page_i, e)
            continue
        page_rows = extraction.rows or []
        for r in page_rows:
            if not any([r.supplier, r.amount, r.date_raw, r.id_number]):
                continue
            all_rows.append(ri.TableRow(
                index=len(all_rows),
                supplier=(r.supplier or "").strip(),
                amount=ri._norm_amount(r.amount),
                date=ri._parse_table_date(r.date_raw),
                id_number=(r.id_number or "").strip() or None,
                raw={"page": page_i, "from_vision": True, "date_raw": r.date_raw},
            ))
        log.info("table page %d: %d rows (running total %d)", page_i, len(page_rows), len(all_rows))
    log.info("parsed %d total table rows from %s (Gemini vision)", len(all_rows), pdf_path)
    return all_rows


# ---------------------------------------------------------------------------
# Top-level run — drop-in for reorder_invoices.run()
# ---------------------------------------------------------------------------
def gemini_run(
    table_pdf: str | Path,
    scanned_pdf: str | Path,
    out_dir: str | Path,
    max_pages: Optional[int] = None,
    vat_rate: Decimal = ri.DEFAULT_VAT_RATE,
    model_name: str = GEMINI_DEFAULT_MODEL,
) -> dict:
    """Process invoices using Gemini vision. Drop-in for reorder_invoices.run().

    Produces the same output files: output_sorted.pdf, match_report.csv,
    all_invoices_extracted.csv, table_extracted.csv, usage.json, run_report.md.

    Requires GOOGLE_API_KEY in the environment.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    usage = GeminiUsageTotals()

    # Build two models: one for single-invoice pages, one for table pages.
    invoice_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=ri.EXTRACTION_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=_INVOICE_SCHEMA,
            max_output_tokens=1024,
        ),
    )
    table_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=ri.TABLE_EXTRACTION_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=_TABLE_SCHEMA,
            max_output_tokens=8000,
        ),
    )

    # Try pdfplumber first (fast, free). Fall back to Gemini vision if quality is poor.
    rows = ri.parse_table_pdf(table_pdf)
    useful = sum(1 for r in rows if r.supplier and r.amount) if rows else 0
    quality_ok = rows and useful / len(rows) >= 0.15
    if not quality_ok:
        log.info(
            "pdfplumber got %d rows but only %d useful — retrying with Gemini vision",
            len(rows), useful,
        )
        rows = parse_table_pdf_with_gemini(table_pdf, table_model,
                                            model_name=model_name, usage=usage)
    else:
        log.info("pdfplumber extracted %d rows (%d useful)", len(rows), useful)

    pages = extract_pages_with_gemini(
        scanned_pdf, invoice_model, model_name=model_name,
        max_pages=max_pages, usage=usage, dpi=150,
    )

    matches = ri.match(rows, pages, vat_rate=vat_rate)

    out_pdf        = out_dir / "output_sorted.pdf"
    out_csv        = out_dir / "match_report.csv"
    extracted_csv  = out_dir / "all_invoices_extracted.csv"
    table_csv      = out_dir / "table_extracted.csv"

    ri.write_sorted_pdf(scanned_pdf, out_pdf, matches, total_pages=len(pages))
    ri.write_report_csv(matches, out_csv, pages=pages)
    ri.write_extracted_csv(pages, extracted_csv)
    ri.write_table_csv(rows, table_csv)

    # Persist usage — same schema as Claude's usage.json so run_report reads it identically.
    try:
        usage_snapshot = {
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "model": usage.model,
            "cost_usd": str(usage.cost_usd()),
            "vat_rate": str(vat_rate),
        }
        (out_dir / "usage.json").write_text(_json.dumps(usage_snapshot, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("usage.json write failed: %s", e)

    try:
        ri.generate_run_report(out_dir, vat_rate=vat_rate)
    except Exception as e:
        log.warning("run_report generation failed: %s", e)

    return {
        "rows": len(rows),
        "pages": len(pages),
        "matched": sum(1 for m in matches if m.status == "matched"),
        "low_confidence": sum(1 for m in matches if m.status == "low_confidence"),
        "unmatched": sum(1 for m in matches if m.status == "unmatched"),
        "out_pdf": str(out_pdf),
        "out_csv": str(out_csv),
        "extracted_csv": str(extracted_csv),
        "table_csv": str(table_csv),
    }
