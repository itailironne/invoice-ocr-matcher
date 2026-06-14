"""Gemini-based OCR pipeline for invoice_reorder.

Drop-in alternative to reorder_invoices.run() that calls Google Gemini instead
of Anthropic Claude. All shared logic (PDF rendering, matching, output writing)
is imported from reorder_invoices — nothing is duplicated here.

Uses the Gemini REST API directly (urllib, Python stdlib) instead of the
google-generativeai SDK. This eliminates the gRPC / protobuf C extensions
(~100 MB of baseline RAM) that caused OOM on the Render 512 MB Starter plan.

Requires GOOGLE_API_KEY in the environment.
"""
from __future__ import annotations

import base64 as _b64
import gc as _gc
import json as _json
import logging
import os
import urllib.error as _urlerr
import urllib.request as _urlreq
from decimal import Decimal
from pathlib import Path
from typing import Optional

import reorder_invoices as ri
from gemini.pricing import GEMINI_DEFAULT_MODEL, GeminiUsageTotals

log = logging.getLogger("gemini.pipeline")

# Gemini REST endpoint (v1beta supports response_schema / JSON mode).
_GEMINI_REST_URL = "https://generativelanguage.googleapis.com/v1beta/models"

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
# REST API helper
# ---------------------------------------------------------------------------
def _gemini_generate(
    api_key: str,
    model_name: str,
    system_instruction: str,
    image_bytes: bytes,
    prompt: str,
    response_schema: dict,
    max_output_tokens: int,
    usage: GeminiUsageTotals,
) -> str:
    """POST one image to the Gemini generateContent REST endpoint.

    Returns the text from the first candidate part. Raises RuntimeError on
    HTTP errors (caller decides whether to retry or skip the page).
    401/403 are re-raised as PermissionError so _is_fatal_gemini_error() can
    catch them and abort the whole run immediately.
    """
    mime_type = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
    body = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime_type,
                                 "data": _b64.b64encode(image_bytes).decode()}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingBudget": 1024},
        },
    }
    url = f"{_GEMINI_REST_URL}/{model_name}:generateContent?key={api_key}"
    req = _urlreq.Request(
        url,
        data=_json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
    except _urlerr.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        if e.code in (401, 403):
            raise PermissionError(f"Gemini auth error {e.code}: {body_text}") from e
        raise RuntimeError(f"Gemini HTTP {e.code}: {body_text}") from e

    usage.add_http(data.get("usageMetadata"), model_name)

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {data}")
    finish = candidates[0].get("finishReason")
    if finish == "MAX_TOKENS":
        meta = data.get("usageMetadata") or {}
        log.warning(
            "Gemini hit MAX_TOKENS (maxOutputTokens=%d, thinking=%s, output=%s) — "
            "response is TRUNCATED and JSON will not parse. Raise max_output_tokens.",
            max_output_tokens,
            meta.get("thoughtsTokenCount"),
            meta.get("candidatesTokenCount"),
        )
    parts = (candidates[0].get("content") or {}).get("parts") or []
    if not parts:
        raise ValueError(f"Gemini returned empty content parts: {candidates[0]}")
    return parts[0].get("text", "")


# ---------------------------------------------------------------------------
# Fatal-error detection
# ---------------------------------------------------------------------------
def _is_fatal_gemini_error(e: Exception) -> bool:
    """Return True for errors that will not improve with retries."""
    if isinstance(e, PermissionError):
        return True
    msg = str(e).lower()
    return "api key not valid" in msg or "quota exceeded" in msg or "permission denied" in msg


# ---------------------------------------------------------------------------
# Single-page invoice extraction
# ---------------------------------------------------------------------------
def extract_fields_from_image_gemini(
    image_bytes: bytes,
    model_name: str,
    api_key: str,
    usage: GeminiUsageTotals,
) -> ri._ExtractedInvoice:
    """Send one page image to Gemini and return parsed invoice fields."""
    text = _gemini_generate(
        api_key=api_key,
        model_name=model_name,
        system_instruction=ri.EXTRACTION_SYSTEM_PROMPT,
        image_bytes=image_bytes,
        prompt="Extract the invoice fields.",
        response_schema=_INVOICE_SCHEMA,
        max_output_tokens=4096,
        usage=usage,
    )
    return ri._ExtractedInvoice.model_validate_json(text)


def _try_extract_page_gemini(
    img_bytes: bytes,
    page_index: int,
    *,
    model_name: str,
    api_key: str,
    usage: GeminiUsageTotals,
) -> ri.OcrPage:
    """Extract one page; on non-fatal error returns an all-null OcrPage."""
    try:
        ex = extract_fields_from_image_gemini(img_bytes, model_name, api_key, usage)
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
_MAX_IMG_PIXELS = 1240 * 1754  # ~150 DPI A4; resize extracted images above this


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
    original compressed bytes directly — no pixmap, no 6-12 MB uncompressed
    buffer. High-res images are downscaled. Falls back to fitz rendering for
    CCITT/JBIG2/multi-image pages (fitz handles those natively).
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
    model_name: str = GEMINI_DEFAULT_MODEL,
    api_key: str = "",
    dpi: int = ri.RENDER_DPI,
    max_pages: Optional[int] = None,
    retry_with_rotation: bool = True,
    usage: Optional[GeminiUsageTotals] = None,
) -> list[ri.OcrPage]:
    """Render each page of the scanned PDF and extract fields via Gemini vision.

    The PDF is opened and closed every _BATCH_SIZE pages so the OS can reclaim
    the memory-mapped PDF data between batches.
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
                page = _try_extract_page_gemini(img, i, model_name=model_name,
                                                api_key=api_key, usage=usage)
                if retry_with_rotation and ri._likely_misread(page):
                    best = page
                    log.info("page %d looked misread (non-null=%d) — trying rotations", i, ri._count_non_null(page))
                    for rotation in (180, 90, 270):
                        try:
                            rotated_bytes = ri._rotate_png_bytes(img, rotation)
                        except Exception as e:
                            log.warning("page %d rotation %d° render failed: %s", i, rotation, e)
                            continue
                        candidate = _try_extract_page_gemini(rotated_bytes, i,
                                                             model_name=model_name,
                                                             api_key=api_key, usage=usage)
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
    model_name: str = GEMINI_DEFAULT_MODEL,
    api_key: str = "",
    dpi: int = ri.RENDER_DPI,
    usage: Optional[GeminiUsageTotals] = None,
) -> list[ri.TableRow]:
    """Read a Hebrew payment-order PDF via Gemini vision."""
    if usage is None:
        usage = GeminiUsageTotals()
    images = ri.render_pdf_pages_to_png(pdf_path, dpi=dpi)
    all_rows: list[ri.TableRow] = []
    for page_i, img_bytes in enumerate(images):
        try:
            text = _gemini_generate(
                api_key=api_key,
                model_name=model_name,
                system_instruction=ri.TABLE_EXTRACTION_SYSTEM_PROMPT,
                image_bytes=img_bytes,
                prompt="Extract every invoice row from this table page.",
                response_schema=_TABLE_SCHEMA,
                # 8000 truncated dense Hebrew pages mid-JSON (thinkingBudget eats
                # into this too); 16000 leaves headroom so rows aren't dropped.
                max_output_tokens=16000,
                usage=usage,
            )
        except Exception as e:
            if _is_fatal_gemini_error(e):
                raise
            log.error("table page %d extraction failed (%s): %s", page_i, type(e).__name__, e)
            continue
        try:
            extraction = ri._TableExtraction.model_validate_json(text)
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

    api_key = os.environ["GOOGLE_API_KEY"]
    usage = GeminiUsageTotals()

    # Try pdfplumber first (fast, free). Fall back to Gemini vision if quality is poor.
    rows = ri.parse_table_pdf(table_pdf)
    useful = sum(1 for r in rows if r.supplier and r.amount) if rows else 0
    quality_ok = rows and useful / len(rows) >= 0.15
    if not quality_ok:
        log.info(
            "pdfplumber got %d rows but only %d useful — retrying with Gemini vision",
            len(rows), useful,
        )
        rows = parse_table_pdf_with_gemini(table_pdf, model_name=model_name,
                                            api_key=api_key, usage=usage)
    else:
        log.info("pdfplumber extracted %d rows (%d useful)", len(rows), useful)

    pages = extract_pages_with_gemini(
        scanned_pdf, model_name=model_name, api_key=api_key,
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
