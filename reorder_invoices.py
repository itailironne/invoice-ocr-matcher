"""Reorder a scanned-invoices PDF to match the row order of an invoice-table PDF.

Pipeline: parse the table PDF (pdfplumber) -> render each scanned page to an image
(PyMuPDF) -> extract supplier/amount/date/id_number via Claude vision (structured
output) -> fuzzy-match pages to rows -> write a sorted PDF, a match-report CSV,
and an extracted-invoices CSV.

Usable two ways:
  - CLI:        python reorder_invoices.py --table T.pdf --scanned S.pdf --out-dir OUT
  - As module:  from reorder_invoices import parse_table_pdf, extract_pages_with_claude, ...

Requires ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import base64
import csv
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional

import anthropic
import pdfplumber
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from thefuzz import fuzz
from dateutil import parser as dateparser

log = logging.getLogger("reorder_invoices")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
CLAUDE_MODEL = "claude-opus-4-7"  # Opus 4.7 has high-res vision (2576px long edge)
RENDER_DPI = 200                  # 200 DPI on A4 -> ~1654x2339, comfortably under 4.7's 2576px cap

SCORE_WEIGHTS = {"supplier": 0.4, "amount": 0.25, "date": 0.15, "id_number": 0.2}
MATCH_THRESHOLD = 60.0  # below this -> unmatched

# Israeli VAT (מע"מ) added to table amounts before matching against scanned receipts.
# הוראת תשלום (the payment-order summary) lists amounts PRE-VAT; the scanned receipts
# show amounts WITH VAT included. So a 1:1 match by amount requires:
#   table_amount * (1 + VAT) ≈ scan_amount
#
# This is the DEFAULT only. Every function that uses it accepts a `vat_rate` override,
# and the CLI / web app expose a per-run override. Update this constant when the
# statutory rate changes; or pass a different value to `run()` / the form for one-offs.
DEFAULT_VAT_RATE = Decimal("0.18")

HEB_COL_SUPPLIER = ("ספק", "שם ספק", "נותן שירות", "תיאור", "שם המוטב")
HEB_COL_AMOUNT = ("סכום", "סה\"כ", "לתשלום", "סך הכל", "מחיר")
HEB_COL_DATE = ("תאריך", "תאריך חשבונית", "עבור תאריך", "עבור")
HEB_COL_ID = ("עוסק מורשה", "ח.פ", "ח\"פ", "מס' עוסק", "ע.מ", "ע\"מ", "קוד הספק")

# Location prefixes that appear in front of supplier names in some PDFs (e.g. "בת ים-טי אנד אם אור").
# Stripped before fuzzy-matching against scanned receipts that wouldn't include the location.
SUPPLIER_LOCATION_PREFIXES = (
    "בת ים-", "בת-ים-", "בת ים -", "בת-ים -", "בת ים  ", "בת ים ", "בת ים",
    "בת-ים", "בתים-", "בתים ",
    "תל אביב-", "ת\"א-", "ת'א-", "ת.א.-", "ת״א-",
    "ירושלים-", "חיפה-", "ראשון לציון-", "פתח תקווה-",
)

EXTRACTION_SYSTEM_PROMPT = """You extract structured fields from a single scanned Israeli invoice / receipt (קבלה / חשבונית).

Return JSON with exactly these fields:
- supplier: the business name as written on the receipt (keep Hebrew as Hebrew). Pick the main vendor header, not the customer name. Null if you cannot see it.
- amount: the GRAND TOTAL as a plain numeric string (e.g. "910" or "3700.50"). No currency symbol, no thousands separators. If multiple amounts appear, prefer the line labeled סה"כ / לתשלום / סך הכל. Null if no total visible.
- date: the invoice/receipt date in ISO format YYYY-MM-DD. The original is usually DD.MM.YY or DD/MM/YYYY — Israeli convention is day-first. Null if no date.
- id_number: the 9-digit Israeli business identifier (עוסק מורשה / ח.פ / ע.מ / מס׳ עוסק). Always 8 or 9 digits. Null if absent.

Rules:
- The image may be rotated; read it correctly regardless of orientation.
- Return null for any field you cannot read with confidence. Do not guess.
- The supplier name often appears at the top of the page in a header box.
- The id number is often printed near the supplier name, sometimes prefixed by "עוסק מורשה".
"""

TABLE_EXTRACTION_SYSTEM_PROMPT = """You extract a table of invoice/payment rows from a Hebrew payment-order document (הוראת תשלום).

Return JSON with `rows`, a list. Each row has:
- supplier: the supplier / payee name as printed on that row (Hebrew). The column header is usually "תיאור" or "ספק" or "שם הספק". If the name has a location prefix like "בת ים-X", keep it as-is. Null if blank.
- amount: the row's price as a numeric string (no currency, no thousands separators). Header usually "מחיר" or "סכום" or "סה\"כ". Negative numbers stay negative.
- date_raw: return the date cell EXACTLY AS PRINTED, character-for-character. Do not convert, reformat, or normalize. The column header is "עבור תאריך" or "תאריך". Examples of literal cell contents you might see and must return verbatim:
    "03/26"        ← return "03/26" exactly
    "04/26"        ← return "04/26" exactly
    "11/25"        ← return "11/25" exactly
    "14/05/2026"   ← return "14/05/2026" exactly
    "11.7.23"      ← return "11.7.23" exactly
  Do NOT swap day/month. Do NOT pad with year. Do NOT output ISO format. Just the raw cell text. If the cell is blank, null.
- id_number: the supplier's Israeli business ID (8-9 digits) if a column for it exists ("עוסק מורשה" / "ח.פ" / "ע.מ" / "קוד הספק"). Null otherwise. The single supplier code in the document header (e.g. "קוד הספק: 51002049") is NOT a per-row id — leave null for individual rows unless the row itself has one.

Rules:
- Only include real invoice / payment rows. Skip totals (סה"כ, מע"מ), headers, signature lines, page numbers, "department" / "approved by" annotations.
- The document may span multiple pages — return only the rows on THIS PAGE image.
- Return null for any cell you can't read confidently. Do not guess.
- Preserve original supplier names character-for-character (including "בת ים-" prefixes); do not normalize them.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class _ExtractedInvoice(BaseModel):
    """Structured-output schema passed to client.messages.parse()."""
    supplier: Optional[str] = Field(None, description="Business name from the receipt header")
    amount: Optional[str] = Field(None, description="Grand total as numeric string, no currency")
    date: Optional[str] = Field(None, description="Invoice date in YYYY-MM-DD")
    id_number: Optional[str] = Field(None, description="9-digit Israeli business id")


class _ExtractedTableRow(BaseModel):
    supplier: Optional[str] = None
    amount: Optional[str] = None
    date_raw: Optional[str] = Field(None, description="Literal date cell as printed, e.g. '03/26', '14/05/2026'")
    id_number: Optional[str] = None


class _TableExtraction(BaseModel):
    rows: list[_ExtractedTableRow] = Field(default_factory=list)


_MM_YY_RE = re.compile(r"^\s*(\d{1,2})\s*/\s*(\d{2})\s*$")


def _parse_table_date(raw: Optional[str]) -> Optional[date]:
    """Parse a date cell from a Hebrew payment-order table.

    Handles two common formats explicitly:
    - MM/YY (billing period, e.g. '03/26' = March 2026)  → first day of that month
    - DD/MM/YY or DD/MM/YYYY                             → that exact day, day-first
    Falls back to dateutil with dayfirst=True for anything else.
    """
    if not raw:
        return None
    s = str(raw).strip()
    m = _MM_YY_RE.match(s)
    if m:
        month = int(m.group(1))
        yy = int(m.group(2))
        year = 2000 + yy if yy < 70 else 1900 + yy
        try:
            return date(year, month, 1)
        except ValueError:
            return None
    return _norm_date(s)


@dataclass
class TableRow:
    index: int
    supplier: str
    amount: Optional[Decimal]
    date: Optional[date]
    id_number: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class OcrPage:
    page_index: int
    supplier: Optional[str]
    amount: Optional[Decimal]
    date: Optional[date]
    id_number: Optional[str]
    text: str = ""  # kept for backwards compat; Claude vision doesn't return raw OCR text


@dataclass
class InvoiceCluster:
    """One logical invoice that may span multiple consecutive PDF pages."""
    cluster_index: int
    page_indices: list[int]  # all source pages, in order
    supplier: Optional[str]
    amount: Optional[Decimal]
    date: Optional[date]
    id_number: Optional[str]


@dataclass
class Match:
    table_row_index: int
    supplier: str
    amount: Optional[Decimal]
    date: Optional[date]
    id_number: Optional[str]
    matched_page: Optional[int]
    match_score: float
    status: str  # "matched" | "low_confidence" | "unmatched"


# ---------------------------------------------------------------------------
# Field normalization (shared by table PDF parse + Claude output)
# ---------------------------------------------------------------------------
def _norm_amount(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = re.sub(r"[₪\s]|ש\"?ח|NIS", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if re.search(r",\d{1,2}$", s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


_ISO_DATE_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")


def _norm_date(value: object) -> Optional[date]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Unambiguous ISO 8601 (YYYY-MM-DD) is interpreted directly — `dayfirst=True`
    # would otherwise flip it to DD-MM-YY (e.g. "2026-03-01" -> Jan 3, 2026).
    m = _ISO_DATE_RE.match(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    try:
        return dateparser.parse(s, dayfirst=True, fuzzy=True).date()
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Stage 1: parse table PDF
# ---------------------------------------------------------------------------
def _detect_columns(header: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, cell in enumerate(header or []):
        if cell is None:
            continue
        c = str(cell).strip()
        if not c:
            continue
        if any(label in c for label in HEB_COL_SUPPLIER) and "supplier" not in out:
            out["supplier"] = i
        elif any(label in c for label in HEB_COL_AMOUNT) and "amount" not in out:
            out["amount"] = i
        elif any(label in c for label in HEB_COL_DATE) and "date" not in out:
            out["date"] = i
        elif any(label in c for label in HEB_COL_ID) and "id" not in out:
            out["id"] = i
    return out


def parse_table_pdf(path: str | Path) -> list[TableRow]:
    """Extract invoice rows from the table PDF via pdfplumber."""
    rows: list[TableRow] = []
    cols: dict[str, int] = {}
    saw_header = False

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table:
                    continue
                for raw_row in table:
                    cells = [("" if c is None else str(c)).strip() for c in raw_row]
                    if not any(cells):
                        continue
                    if not saw_header:
                        detected = _detect_columns(cells)
                        if detected:
                            cols = detected
                            saw_header = True
                            continue
                    if not cols:
                        cols = {"supplier": 0, "amount": 1, "date": 2}
                    supplier = cells[cols["supplier"]] if cols.get("supplier") is not None and cols["supplier"] < len(cells) else ""
                    amount_cell = cells[cols["amount"]] if cols.get("amount") is not None and cols["amount"] < len(cells) else ""
                    date_cell = cells[cols["date"]] if cols.get("date") is not None and cols["date"] < len(cells) else ""
                    id_cell = cells[cols["id"]] if cols.get("id") is not None and cols["id"] < len(cells) else ""
                    if not supplier and not amount_cell and not date_cell:
                        continue
                    id_number = None
                    if id_cell:
                        m = re.search(r"(\d{8,10})", id_cell)
                        id_number = m.group(1) if m else None
                    rows.append(TableRow(
                        index=len(rows),
                        supplier=supplier,
                        amount=_norm_amount(amount_cell),
                        date=_norm_date(date_cell),
                        id_number=id_number,
                        raw={"cells": cells},
                    ))
    log.info("parsed %d table rows from %s", len(rows), path)
    return rows


def parse_table_pdf_with_claude(
    pdf_path: str | Path,
    client: Optional[anthropic.Anthropic] = None,
    dpi: int = RENDER_DPI,
    model: str = CLAUDE_MODEL,
) -> list[TableRow]:
    """Read a Hebrew payment-order / invoice-list PDF via Claude vision.

    Use this when pdfplumber can't handle the table (RTL Hebrew comes out reversed,
    cell boundaries get lost, etc). Each page is sent to Claude with the
    TABLE_EXTRACTION_SYSTEM_PROMPT; the model returns a JSON list of rows. Rows
    from all pages are concatenated and indexed sequentially.
    """
    client = client or anthropic.Anthropic()
    images = render_pdf_pages_to_png(pdf_path, dpi=dpi)
    all_rows: list[TableRow] = []
    for page_i, img_bytes in enumerate(images):
        b64 = base64.standard_b64encode(img_bytes).decode("ascii")
        try:
            response = client.messages.parse(
                model=model,
                max_tokens=8000,
                system=[{
                    "type": "text",
                    "text": TABLE_EXTRACTION_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": "Extract every invoice row from this table page."},
                    ],
                }],
                output_format=_TableExtraction,
            )
        except anthropic.APIError as e:
            log.warning("table page %d extraction failed: %s", page_i, e)
            continue
        page_rows = response.parsed_output.rows or []
        for r in page_rows:
            # Skip rows with no useful content.
            if not any([r.supplier, r.amount, r.date_raw, r.id_number]):
                continue
            all_rows.append(TableRow(
                index=len(all_rows),
                supplier=(r.supplier or "").strip(),
                amount=_norm_amount(r.amount),
                date=_parse_table_date(r.date_raw),
                id_number=(r.id_number or "").strip() or None,
                raw={"page": page_i, "from_vision": True, "date_raw": r.date_raw},
            ))
        log.info("table page %d: %d rows (running total %d)", page_i, len(page_rows), len(all_rows))
    log.info("parsed %d total table rows from %s (Claude vision)", len(all_rows), pdf_path)
    return all_rows


# ---------------------------------------------------------------------------
# Stage 2: extract fields from each scanned page via Claude vision
# ---------------------------------------------------------------------------
def render_pdf_pages_to_png(pdf_path: str | Path, dpi: int = RENDER_DPI) -> list[bytes]:
    """Render every page of `pdf_path` to PNG bytes. Returns one bytes blob per page."""
    images: list[bytes] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(pix.tobytes(output="png"))
    finally:
        doc.close()
    return images


def extract_fields_from_image(
    client: anthropic.Anthropic,
    image_bytes: bytes,
    media_type: str = "image/png",
    model: str = CLAUDE_MODEL,
) -> _ExtractedInvoice:
    """Send a single page image to Claude and return the parsed fields.

    The system prompt is cached (5-min TTL) so subsequent pages in the same run
    pay ~0.1x for the prompt rather than the full price.
    """
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": EXTRACTION_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                },
                {"type": "text", "text": "Extract the invoice fields."},
            ],
        }],
        output_format=_ExtractedInvoice,
    )
    return response.parsed_output


def _rotate_png_bytes(img_bytes: bytes, degrees: int) -> bytes:
    """Rotate PNG bytes by N degrees (CW). Returns new PNG bytes."""
    from PIL import Image
    import io as _io
    img = Image.open(_io.BytesIO(img_bytes))
    rotated = img.rotate(-degrees, expand=True)  # PIL rotates CCW; negate for CW
    buf = _io.BytesIO()
    rotated.save(buf, "PNG")
    return buf.getvalue()


def _count_non_null(page: OcrPage) -> int:
    return sum(1 for v in (page.supplier, page.amount, page.date, page.id_number) if v is not None)


def _likely_misread(page: OcrPage) -> bool:
    """Heuristic: a page is suspect if the supplier is missing or fewer than 2 fields came back."""
    return page.supplier is None or _count_non_null(page) <= 1


def extract_pages_with_claude(
    pdf_path: str | Path,
    client: Optional[anthropic.Anthropic] = None,
    dpi: int = RENDER_DPI,
    model: str = CLAUDE_MODEL,
    max_pages: Optional[int] = None,
    retry_with_rotation: bool = True,
) -> list[OcrPage]:
    """Render each page of the scanned PDF and extract fields via Claude vision.

    If `max_pages` is given, only the first N pages are processed.
    If `retry_with_rotation` is True (default), pages that look like
    misreads (no supplier, mostly null fields) are retried at 90°/180°/270°
    rotation and the best result is kept.
    """
    client = client or anthropic.Anthropic()
    images = render_pdf_pages_to_png(pdf_path, dpi=dpi)
    if max_pages is not None and max_pages > 0:
        images = images[:max_pages]
    pages: list[OcrPage] = []
    for i, img in enumerate(images):
        page = _try_extract_page(client, img, i, model=model)
        if retry_with_rotation and _likely_misread(page):
            best = page
            best_img = img
            log.info("page %d looked misread (non-null=%d) — trying rotations", i, _count_non_null(page))
            for rotation in (180, 90, 270):  # 180° first — most common phone-photo case
                try:
                    rotated_bytes = _rotate_png_bytes(img, rotation)
                except Exception as e:
                    log.warning("page %d rotation %d° render failed: %s", i, rotation, e)
                    continue
                candidate = _try_extract_page(client, rotated_bytes, i, model=model)
                if _count_non_null(candidate) > _count_non_null(best):
                    log.info("page %d: rotation %d° improved extraction (%d -> %d non-null)",
                             i, rotation, _count_non_null(best), _count_non_null(candidate))
                    best = candidate
                    best_img = rotated_bytes
                # Stop early once we have a solid read.
                if _count_non_null(best) >= 3:
                    break
            page = best
        pages.append(page)
        log.info("page %d: supplier=%r amount=%s date=%s id=%s",
                 i, page.supplier, page.amount, page.date, page.id_number)
    return pages


def _try_extract_page(client, img_bytes, page_index, *, model=CLAUDE_MODEL) -> OcrPage:
    """Helper: extract one page; on API error returns an all-null OcrPage."""
    try:
        ex = extract_fields_from_image(client, img_bytes, model=model)
    except anthropic.APIError as e:
        log.warning("page %d extraction API error: %s", page_index, e)
        return OcrPage(page_index=page_index, supplier=None, amount=None, date=None, id_number=None)
    return OcrPage(
        page_index=page_index,
        supplier=ex.supplier,
        amount=_norm_amount(ex.amount),
        date=_norm_date(ex.date),
        id_number=ex.id_number,
    )


def load_extracted_csv(csv_path: str | Path) -> list[OcrPage]:
    """Load an all_invoices_extracted.csv back into OcrPage objects."""
    pages: list[OcrPage] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pages.append(OcrPage(
                page_index=int(row["page"]),
                supplier=row["supplier"] or None,
                amount=_norm_amount(row["amount"]) if row["amount"] else None,
                date=_norm_date(row["date"]) if row["date"] else None,
                id_number=row["id_number"] or None,
            ))
    return pages


def retry_failed_pages_with_rotation(
    scanned_pdf: str | Path,
    existing_pages: list[OcrPage],
    client: Optional[anthropic.Anthropic] = None,
    dpi: int = RENDER_DPI,
    model: str = CLAUDE_MODEL,
) -> tuple[list[OcrPage], list[int]]:
    """Run rotation retry on previously-failed pages from an existing extraction.

    Returns (updated_pages, recovered_indices). Pages that look like misreads
    (per `_likely_misread`) are re-rendered, rotated to 180°/90°/270°, and
    re-extracted; the rotation that yields the most non-null fields wins.
    Good pages are left untouched.
    """
    client = client or anthropic.Anthropic()
    suspect_positions = [i for i, p in enumerate(existing_pages) if _likely_misread(p)]
    if not suspect_positions:
        log.info("no suspect pages found — nothing to retry")
        return existing_pages, []

    log.info("retrying %d suspect pages with rotation", len(suspect_positions))
    images = render_pdf_pages_to_png(scanned_pdf, dpi=dpi)
    updated = list(existing_pages)
    recovered: list[int] = []
    for list_idx in suspect_positions:
        page = existing_pages[list_idx]
        i = page.page_index
        if i >= len(images):
            continue
        img = images[i]
        best = page
        for rotation in (180, 90, 270):
            try:
                rotated = _rotate_png_bytes(img, rotation)
            except Exception as e:
                log.warning("page %d rotate %d° failed: %s", i, rotation, e)
                continue
            candidate = _try_extract_page(client, rotated, i, model=model)
            if _count_non_null(candidate) > _count_non_null(best):
                log.info("page %d: rotation %d° improved (%d -> %d non-null fields)",
                         i, rotation, _count_non_null(best), _count_non_null(candidate))
                best = candidate
            if _count_non_null(best) >= 3:
                break
        if _count_non_null(best) > _count_non_null(page):
            recovered.append(i)
        updated[list_idx] = best
    log.info("rotation retry: %d/%d pages recovered", len(recovered), len(suspect_positions))
    return updated, recovered


# ---------------------------------------------------------------------------
# Stage 3: match
# ---------------------------------------------------------------------------
def _strip_location_prefix(name: Optional[str]) -> Optional[str]:
    """Remove leading location markers like 'בת ים-' from supplier names."""
    if not name:
        return name
    s = name.strip()
    for prefix in SUPPLIER_LOCATION_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):].strip()
    return s


def _score_supplier(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    # Strip location prefixes from both sides — table entries often carry a
    # location like "בת ים-X" while the scanned receipt has just "X".
    a_clean = _strip_location_prefix(a)
    b_clean = _strip_location_prefix(b)
    return float(fuzz.token_set_ratio(a_clean, b_clean))


def _score_amount(
    table_amount: Optional[Decimal],
    scan_amount: Optional[Decimal],
    vat_rate: Decimal = DEFAULT_VAT_RATE,
) -> float:
    """Score amount similarity. Adds VAT to the table side before comparing,
    since הוראת תשלום reports pre-VAT amounts and scanned receipts report
    amounts that include VAT.

    Order of arguments: (table_amount, scan_amount).
    """
    if table_amount is None or scan_amount is None:
        return 0.0
    if table_amount == 0 or scan_amount == 0:
        return 0.0
    # Apply VAT to the table amount (preserving sign for credits/refunds).
    table_with_vat = table_amount * (Decimal(1) + vat_rate)
    a = abs(table_with_vat)
    b = abs(scan_amount)
    diff = abs(a - b) / max(a, b)
    if diff <= Decimal("0.01"):
        return 100.0
    if diff <= Decimal("0.05"):
        return 50.0
    return 0.0


def _score_date(a: Optional[date], b: Optional[date]) -> float:
    """Score date proximity. Same month+year -> 100 (handles MM/YY billing
    periods that the table side often uses); adjacent month -> 50; else 0."""
    if a is None or b is None:
        return 0.0
    if a.year == b.year and a.month == b.month:
        return 100.0
    if a.year == b.year and abs(a.month - b.month) == 1:
        return 50.0
    # Cross-year December/January edge case.
    if abs((a.year - b.year)) == 1 and {a.month, b.month} == {1, 12}:
        return 50.0
    return 0.0


def _score_id(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    return 100.0 if a == b else 0.0


def _pair_score(row: TableRow, page: OcrPage, vat_rate: Decimal = DEFAULT_VAT_RATE) -> float:
    return (
        SCORE_WEIGHTS["supplier"] * _score_supplier(row.supplier, page.supplier)
        + SCORE_WEIGHTS["amount"] * _score_amount(row.amount, page.amount, vat_rate)
        + SCORE_WEIGHTS["date"] * _score_date(row.date, page.date)
        + SCORE_WEIGHTS["id_number"] * _score_id(row.id_number, page.id_number)
    )


def order_scans_by_table(
    rows: list[TableRow],
    pages: list[OcrPage],
    threshold: float = 50.0,
) -> tuple[list[int], dict[int, tuple[Optional[int], float]]]:
    """Order each scan page by which row of the table it best matches.

    Unlike `match()` (1:1 greedy), this is many-to-one: many scans can share
    a single table row. Used when the table is a summary (e.g. payment order
    where one row = many invoices), so the scans should be GROUPED by supplier
    and ORDERED by the table's row order, not 1:1 matched by amount.

    Scoring: supplier 50% + date (month/year) 40% + id 10%. Amount is ignored
    because the table totals don't equal per-scan amounts in summary tables.

    Returns:
        (ordered_page_indices, assignments)
        where assignments maps page_index -> (assigned_row_index or None, score).
    """
    rows = list(rows)
    pages = list(pages)
    assignments: dict[int, tuple[Optional[int], float]] = {}

    # Hard floor: a scan must share a supplier with a candidate row before any
    # date/id bonus can pull it in. This prevents date-only false matches
    # (e.g. a Rentokil scan from April landing in the מוקד אמון row just because
    # both happen to be April).
    SUPPLIER_FLOOR = 60.0

    for page in pages:
        best_idx: Optional[int] = None
        best_score = -1.0
        for row in rows:
            sup = _score_supplier(row.supplier, page.supplier)
            if sup < SUPPLIER_FLOOR:
                continue
            s = (
                0.5 * sup
                + 0.4 * _score_date(row.date, page.date)
                + 0.1 * _score_id(row.id_number, page.id_number)
            )
            if s > best_score:
                best_score = s
                best_idx = row.index
        if best_idx is not None and best_score >= threshold:
            assignments[page.page_index] = (best_idx, round(best_score, 2))
        else:
            assignments[page.page_index] = (None, round(max(best_score, 0.0), 2))

    def _key(page: OcrPage):
        row_idx, score = assignments[page.page_index]
        if row_idx is None:
            return (1, page.page_index)
        return (0, row_idx, -score, page.page_index)

    ordered = sorted(pages, key=_key)
    return [p.page_index for p in ordered], assignments


def cluster_consecutive_pages(
    pages: list[OcrPage],
    supplier_match_threshold: float = 80.0,
) -> list[InvoiceCluster]:
    """Group consecutive PDF pages that likely belong to the SAME invoice.

    A continuation page (page 2+ of a multi-page invoice) is identified by:
      - Same supplier as the previous page (fuzzy ≥ threshold), AND
      - Significantly less data than the previous page (typically no amount).

    Importantly, two consecutive pages from the same supplier that BOTH
    carry a full amount are treated as SEPARATE invoices (the supplier just
    happens to have multiple receipts in a row).

    Each cluster collapses to one logical invoice. The representative fields:
      - supplier: longest non-null supplier name in the cluster
      - amount:   the cluster's primary amount (the only non-null amount; if
                  the cluster is a single page, that page's amount)
      - date:     earliest non-null date
      - id_number: most-common non-null id
    """
    if not pages:
        return []
    sorted_pages = sorted(pages, key=lambda p: p.page_index)
    raw_clusters: list[list[OcrPage]] = [[sorted_pages[0]]]
    for p in sorted_pages[1:]:
        prev = raw_clusters[-1][-1]
        attach = False

        # Required: supplier match (or absent supplier as continuation).
        supplier_matches = False
        if p.supplier and prev.supplier:
            supplier_matches = _score_supplier(prev.supplier, p.supplier) >= supplier_match_threshold
        elif prev.supplier and not p.supplier:
            supplier_matches = True

        if supplier_matches:
            # Continuation only if the current page is significantly less
            # informative than the previous one — typically missing the amount.
            curr_has_amount = p.amount is not None
            prev_has_amount = prev.amount is not None
            curr_nonnull = _count_non_null(p)
            prev_nonnull = _count_non_null(prev)
            # Continuation rules:
            #   1. Curr has no amount and prev does → continuation (typical 2-page invoice)
            #   2. Curr has ≤1 non-null field → continuation (almost-blank back side)
            #   3. Both have amounts that are EQUAL → same invoice, duplicated total line
            if (prev_has_amount and not curr_has_amount) or curr_nonnull <= 1:
                attach = True
            elif curr_has_amount and prev_has_amount and p.amount == prev.amount:
                attach = True
            # Else: same supplier but both have distinct full data → separate invoices.

        if attach:
            raw_clusters[-1].append(p)
        else:
            raw_clusters.append([p])

    clusters: list[InvoiceCluster] = []
    for i, cluster_pages in enumerate(raw_clusters):
        # Pick representative fields.
        suppliers = [p.supplier for p in cluster_pages if p.supplier]
        ids = [p.id_number for p in cluster_pages if p.id_number]
        amounts = [p.amount for p in cluster_pages if p.amount is not None]
        dates = [p.date for p in cluster_pages if p.date is not None]
        # Longest supplier name (multi-page often has shortened headers on cont. pages).
        supplier = max(suppliers, key=len) if suppliers else None
        # Largest absolute amount (the total typically sits on the page with the biggest number).
        amount = max(amounts, key=lambda a: abs(a)) if amounts else None
        date_val = min(dates) if dates else None
        # Most common id; if tied, first non-null.
        id_number = max(set(ids), key=ids.count) if ids else None
        clusters.append(InvoiceCluster(
            cluster_index=i,
            page_indices=[p.page_index for p in cluster_pages],
            supplier=supplier,
            amount=amount,
            date=date_val,
            id_number=id_number,
        ))
    return clusters


def match_clusters_to_table(
    rows: Iterable[TableRow],
    clusters: list[InvoiceCluster],
    vat_rate: Decimal = DEFAULT_VAT_RATE,
) -> list[Match]:
    """1:1 greedy match of table rows to invoice clusters using two keys
    (supplier name + amount with VAT applied to table). Same scoring as
    `match_two_keys` but operating on clusters."""
    rows = list(rows)
    used: set[int] = set()
    matches: list[Match] = []
    for row in rows:
        best_cluster: Optional[InvoiceCluster] = None
        best_score = -1.0
        for c in clusters:
            if c.cluster_index in used:
                continue
            sup = _score_supplier(row.supplier, c.supplier)
            amt = _score_amount(row.amount, c.amount, vat_rate)
            if sup < 40 or amt < 1:
                continue
            s = 0.5 * sup + 0.5 * amt
            if s > best_score:
                best_score = s
                best_cluster = c
        if best_cluster is not None and best_score >= 70:
            used.add(best_cluster.cluster_index)
            status = "matched" if best_score >= 90 else "low_confidence"
            # Match.matched_page points to the FIRST page of the cluster.
            first_page = best_cluster.page_indices[0]
            matches.append(Match(
                table_row_index=row.index, supplier=row.supplier, amount=row.amount,
                date=row.date, id_number=row.id_number, matched_page=first_page,
                match_score=round(best_score, 2), status=status,
            ))
        else:
            matches.append(Match(
                table_row_index=row.index, supplier=row.supplier, amount=row.amount,
                date=row.date, id_number=row.id_number, matched_page=None,
                match_score=round(max(best_score, 0.0), 2), status="unmatched",
            ))
    return matches


def second_pass_amount_only(
    rows: list[TableRow],
    clusters: list[InvoiceCluster],
    matches: list[Match],
    *,
    amount_threshold: float = 95.0,
    vat_rate: Decimal = DEFAULT_VAT_RATE,
) -> int:
    """Second-pass rescue for table rows where supplier didn't match (e.g.
    Hebrew↔English transliteration like 'וויטסנט' vs 'WhiteScent').

    For each still-unmatched row, find AVAILABLE clusters whose amount matches
    the row exactly (≥ `amount_threshold`). If exactly one such cluster exists
    AND no other unmatched row is also competing for it with the same amount,
    promote the row to low_confidence with that cluster.

    Mutates `matches` in place. Returns the number of rows newly matched.
    """
    used = {m.matched_page for m in matches if m.matched_page is not None}
    available = [c for c in clusters if c.page_indices[0] not in used]

    # First, group unmatched rows by their amount so we can detect ties.
    unmatched_rows = []
    for m in matches:
        if m.status != "unmatched":
            continue
        row = next(r for r in rows if r.index == m.table_row_index)
        if row.amount is None:
            continue
        unmatched_rows.append((m, row))

    recovered = 0
    for m, row in unmatched_rows:
        exact_candidates = [c for c in available if _score_amount(row.amount, c.amount, vat_rate) >= amount_threshold]
        if len(exact_candidates) != 1:
            continue
        cluster = exact_candidates[0]
        # Don't steal: skip if another unmatched row matches this same cluster equally well.
        competing = sum(
            1 for m2, r2 in unmatched_rows
            if m2 is not m and r2.amount is not None
            and _score_amount(r2.amount, cluster.amount) >= amount_threshold
        )
        if competing > 0:
            continue
        m.matched_page = cluster.page_indices[0]
        m.match_score = 50.0  # supplier=0, amount=100 -> weighted = 50 (50/50)
        m.status = "low_confidence"
        available.remove(cluster)
        recovered += 1
    return recovered


def match_two_keys(
    rows: Iterable[TableRow],
    pages: Iterable[OcrPage],
    vat_rate: Decimal = DEFAULT_VAT_RATE,
) -> list[Match]:
    """1:1 greedy matching using only two keys: supplier name + amount (VAT-adjusted).

    For each table row, pick the still-unmatched scan that best aligns on these
    two keys. Date and id are ignored. Scoring:
      score = 0.5 * supplier_score + 0.5 * amount_score
    Status:
      - score >= 90 (sup ≥ 80 AND amount within 1%)         -> matched
      - score >= 70 (sup ≥ 60 AND amount within 5%)         -> low_confidence
      - otherwise                                            -> unmatched
    """
    rows = list(rows)
    pages = list(pages)
    used: set[int] = set()
    matches: list[Match] = []
    for row in rows:
        best_idx: Optional[int] = None
        best_score = -1.0
        for p in pages:
            if p.page_index in used:
                continue
            sup = _score_supplier(row.supplier, p.supplier)
            amt = _score_amount(row.amount, p.amount, vat_rate)
            # Hard floor: a real two-key match requires SOMETHING on both keys.
            if sup < 40 or amt < 1:
                continue
            s = 0.5 * sup + 0.5 * amt
            if s > best_score:
                best_score = s
                best_idx = p.page_index
        if best_idx is not None and best_score >= 70:
            used.add(best_idx)
            status = "matched" if best_score >= 90 else "low_confidence"
            matches.append(Match(
                table_row_index=row.index, supplier=row.supplier, amount=row.amount,
                date=row.date, id_number=row.id_number, matched_page=best_idx,
                match_score=round(best_score, 2), status=status,
            ))
        else:
            matches.append(Match(
                table_row_index=row.index, supplier=row.supplier, amount=row.amount,
                date=row.date, id_number=row.id_number, matched_page=None,
                match_score=round(max(best_score, 0.0), 2), status="unmatched",
            ))
    return matches


def match(
    rows: Iterable[TableRow],
    pages: Iterable[OcrPage],
    threshold: float = MATCH_THRESHOLD,
    vat_rate: Decimal = DEFAULT_VAT_RATE,
) -> list[Match]:
    rows = list(rows)
    pages = list(pages)
    used: set[int] = set()
    matches: list[Match] = []
    for row in rows:
        best_idx: Optional[int] = None
        best_score = -1.0
        for p in pages:
            if p.page_index in used:
                continue
            s = _pair_score(row, p, vat_rate)
            if s > best_score:
                best_score = s
                best_idx = p.page_index
        if best_idx is not None and best_score >= threshold:
            used.add(best_idx)
            matches.append(Match(
                table_row_index=row.index, supplier=row.supplier, amount=row.amount,
                date=row.date, id_number=row.id_number, matched_page=best_idx,
                match_score=round(best_score, 2),
                status="matched" if best_score >= 75 else "low_confidence",
            ))
        else:
            matches.append(Match(
                table_row_index=row.index, supplier=row.supplier, amount=row.amount,
                date=row.date, id_number=row.id_number, matched_page=None,
                match_score=round(max(best_score, 0.0), 2), status="unmatched",
            ))
    return matches


# ---------------------------------------------------------------------------
# Stage 4: outputs
# ---------------------------------------------------------------------------
def write_sorted_pdf(scanned_pdf: str | Path, out_pdf: str | Path, matches: list[Match], total_pages: int) -> None:
    used = [m.matched_page for m in matches if m.matched_page is not None]
    used_set = set(used)
    leftover = [i for i in range(total_pages) if i not in used_set]
    order = used + leftover

    src = fitz.open(str(scanned_pdf))
    try:
        dst = fitz.open()
        try:
            for p in order:
                dst.insert_pdf(src, from_page=p, to_page=p)
            dst.save(str(out_pdf))
        finally:
            dst.close()
    finally:
        src.close()
    log.info("wrote %s (%d pages, %d leftover appended)", out_pdf, len(order), len(leftover))


def write_report_csv(matches: list[Match], path: str | Path) -> None:
    fieldnames = ["table_row_index", "supplier", "amount", "date", "id_number", "matched_page", "match_score", "status"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in matches:
            row = asdict(m)
            row["amount"] = "" if row["amount"] is None else str(row["amount"])
            row["date"] = "" if row["date"] is None else row["date"].isoformat()
            row["id_number"] = "" if row["id_number"] is None else row["id_number"]
            row["matched_page"] = "" if row["matched_page"] is None else row["matched_page"]
            writer.writerow(row)
    log.info("wrote %s (%d rows)", path, len(matches))


def write_pdf_from_csv_order(
    csv_path: str | Path,
    source_pdf: str | Path,
    output_pdf: str | Path,
    page_column: str = "page",
) -> int:
    """Build a new PDF whose page order matches the row order of a CSV.

    Use case: edit `all_invoices_extracted.csv` to reorder rows however you
    like (sort by supplier, by amount, by date, manually drag rows), then
    regenerate the PDF in that exact order.

    The CSV's `page_column` (default "page") gives the 0-indexed source page
    for each row. Returns the number of pages written.
    """
    page_order: list[int] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get(page_column, "").strip()
            if raw == "":
                continue
            page_order.append(int(raw))

    src = fitz.open(str(source_pdf))
    try:
        dst = fitz.open()
        try:
            for p in page_order:
                if 0 <= p < src.page_count:
                    dst.insert_pdf(src, from_page=p, to_page=p)
                else:
                    log.warning("page index %d out of range (PDF has %d pages) — skipped", p, src.page_count)
            dst.save(str(output_pdf))
        finally:
            dst.close()
    finally:
        src.close()
    log.info("wrote %s (%d pages in CSV order)", output_pdf, len(page_order))
    return len(page_order)


def write_table_csv(rows: list[TableRow], path: str | Path) -> None:
    """Flat CSV of the parsed table rows: index, supplier, amount, date, id_number."""
    fieldnames = ["index", "supplier", "amount", "date", "id_number"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "index": r.index,
                "supplier": r.supplier or "",
                "amount": "" if r.amount is None else str(r.amount),
                "date": "" if r.date is None else r.date.isoformat(),
                "id_number": r.id_number or "",
            })
    log.info("wrote %s (%d rows)", path, len(rows))


def write_extracted_csv(pages: list[OcrPage], path: str | Path) -> None:
    fieldnames = ["page", "supplier", "amount", "date", "id_number"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pages:
            writer.writerow({
                "page": p.page_index,
                "supplier": p.supplier or "",
                "amount": "" if p.amount is None else str(p.amount),
                "date": "" if p.date is None else p.date.isoformat(),
                "id_number": p.id_number or "",
            })
    log.info("wrote %s (%d rows)", path, len(pages))


def pages_to_dataframe(pages: list[OcrPage]):
    import pandas as pd
    return pd.DataFrame([
        {"page": p.page_index, "supplier": p.supplier, "amount": p.amount, "date": p.date, "id_number": p.id_number}
        for p in pages
    ])


def matches_to_dataframe(matches: list[Match]):
    import pandas as pd
    return pd.DataFrame([
        {
            "row": m.table_row_index, "supplier": m.supplier, "amount": m.amount,
            "date": m.date, "id_number": m.id_number, "matched_page": m.matched_page,
            "score": m.match_score, "status": m.status,
        }
        for m in matches
    ])


# ---------------------------------------------------------------------------
# Combine arbitrary files (PDFs + images) into a single sorted PDF
# ---------------------------------------------------------------------------
def _file_to_pdf_bytes(file_path: Path) -> bytes:
    """Return PDF bytes for a file. PDFs pass through; images are wrapped in a 1-page PDF."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return file_path.read_bytes()
    from PIL import Image
    import io as _io
    img = Image.open(file_path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = _io.BytesIO()
    img.save(buf, "PDF", resolution=150)
    return buf.getvalue()


def combine_files_to_pdf(file_paths: list[Path], out_pdf: str | Path) -> int:
    """Concatenate a list of files (PDFs and/or images) into one PDF in the given order.

    Returns the number of source files combined. Each input contributes one or more pages
    (PDFs keep their page count; images contribute exactly one page).
    """
    dst = fitz.open()
    try:
        for p in file_paths:
            pdf_bytes = _file_to_pdf_bytes(Path(p))
            src = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                dst.insert_pdf(src)
            finally:
                src.close()
        dst.save(str(out_pdf))
    finally:
        dst.close()
    return len(file_paths)


def batch_extract_and_combine(
    file_paths: list[Path],
    out_dir: str | Path,
    sort_by: str = "date",
    client: Optional[anthropic.Anthropic] = None,
) -> dict:
    """Extract fields from each file via Claude vision, sort by `sort_by`, and combine
    all files into one PDF in that order.

    Returns a dict with: combined_pdf path, csv path, rows (list of {filename, supplier,
    amount, date, id_number} in sorted order).
    """
    client = client or anthropic.Anthropic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_paths = [Path(p) for p in file_paths]
    rows: list[dict] = []
    for i, p in enumerate(file_paths):
        try:
            fields = extract_single_file(p, client=client)
        except Exception as e:
            log.warning("extract failed for %s: %s", p.name, e)
            fields = {"supplier": None, "amount": None, "date": None, "id_number": None}
        rows.append({
            "_upload_order": i,
            "filename": p.name,
            "supplier": fields["supplier"],
            "amount": fields["amount"],
            "date": fields["date"],
            "id_number": fields["id_number"],
            "_path": p,
        })

    # Sort rows. Items with missing keys fall to the end, preserving upload order.
    def _key(row):
        v = row.get(sort_by)
        if v is None or v == "":
            return (1, row["_upload_order"])
        if sort_by == "date":
            try:
                d = dateparser.parse(str(v), dayfirst=True, fuzzy=True).date()
                return (0, d.isoformat())
            except (ValueError, OverflowError):
                return (1, row["_upload_order"])
        if sort_by == "amount":
            d = _norm_amount(v)
            return (0, float(d)) if d is not None else (1, row["_upload_order"])
        return (0, str(v))
    rows.sort(key=_key)

    combined_pdf = out_dir / "combined_sorted.pdf"
    combine_files_to_pdf([r["_path"] for r in rows], combined_pdf)

    csv_path = out_dir / "extracted_invoices.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "supplier", "amount", "date", "id_number"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ("filename", "supplier", "amount", "date", "id_number")})

    return {
        "combined_pdf": str(combined_pdf),
        "csv": str(csv_path),
        "rows": [
            {k: r.get(k) for k in ("filename", "supplier", "amount", "date", "id_number")}
            for r in rows
        ],
        "sort_by": sort_by,
        "count": len(rows),
    }


# ---------------------------------------------------------------------------
# Single-image helper (for the test page in the web app)
# ---------------------------------------------------------------------------
def extract_single_file(
    path: str | Path,
    client: Optional[anthropic.Anthropic] = None,
    dpi: int = RENDER_DPI,
    model: str = CLAUDE_MODEL,
) -> dict:
    """Extract fields from one file (PDF or image). Returns a flat dict."""
    client = client or anthropic.Anthropic()
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        images = render_pdf_pages_to_png(path, dpi=dpi)
        if not images:
            raise ValueError(f"{path} contains no pages")
        img_bytes = images[0]
        media_type = "image/png"
    elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        img_bytes = path.read_bytes()
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), f"image/{suffix.lstrip('.')}")
    else:
        raise ValueError(f"unsupported file type: {suffix}")
    extracted = extract_fields_from_image(client, img_bytes, media_type=media_type, model=model)
    return {
        "supplier": extracted.supplier,
        "amount": extracted.amount,
        "date": extracted.date,
        "id_number": extracted.id_number,
    }


# ---------------------------------------------------------------------------
# Stage 6: audit / run report (Hebrew markdown)
# ---------------------------------------------------------------------------
def _load_table_csv(path: Path) -> list[TableRow]:
    """Read a table_extracted.csv back into TableRow objects."""
    rows: list[TableRow] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append(TableRow(
                    index=int(r.get("index", len(rows))),
                    supplier=(r.get("supplier") or "").strip(),
                    amount=Decimal(r["amount"]) if r.get("amount") else None,
                    date=_norm_date(r["date"]) if r.get("date") else None,
                    id_number=(r.get("id_number") or "").strip() or None,
                ))
            except (ValueError, InvalidOperation):
                continue
    return rows


def _classify_unmatched(
    row: TableRow,
    clusters: list[InvoiceCluster],
    used_cluster_ids: set[int],
    vat_rate: Decimal,
    table_rows: list[TableRow],
) -> tuple[str, list[dict]]:
    """Return (reason_hebrew, top3_candidates) for an unmatched row."""
    # Score against every cluster
    scored = []
    for c in clusters:
        sup = _score_supplier(row.supplier, c.supplier)
        amt = _score_amount(row.amount, c.amount, vat_rate)
        scored.append({
            "cluster": c,
            "sup": sup,
            "amt": amt,
            "in_use": c.cluster_index in used_cluster_ids,
        })
    scored.sort(key=lambda x: (x["sup"] + x["amt"]), reverse=True)
    top3 = scored[:3]

    # Categorize the cause based on the top candidate
    if not top3 or top3[0]["sup"] == 0 and top3[0]["amt"] == 0:
        return "אין מועמד מתאים כלל בסריקות", top3

    top = top3[0]
    if top["amt"] == 100 and top["sup"] == 0 and top["in_use"]:
        return "סכום תואם בדיוק לאשכול אחר — סבירות גבוהה שזו שורת טבלה כפולה (התרסקות עם שורה אחרת)", top3
    if top["amt"] == 100 and top["sup"] == 0:
        return "סכום תואם אך השמות שונים — סבירות גבוהה שמדובר בתעתיק עברית↔אנגלית (לדוגמה וויטסנט ↔ WhiteScent)", top3
    if top["sup"] >= 80 and top["amt"] == 0:
        # Check if it's because of duplicate table row
        same_amount_rows = [r for r in table_rows if r.index != row.index and r.amount == row.amount and _score_supplier(r.supplier, row.supplier) >= 80]
        if same_amount_rows:
            return f"שורת טבלה כפולה — שורה {same_amount_rows[0].index} בעלת אותו ספק וסכום זכתה באשכול המתאים היחיד", top3
        return "ספק זהה אך אף סריקה לא תואמת בסכום (אולי הסריקה חסרת סכום או הסכום שגוי)", top3
    if top["sup"] >= 80 and 0 < top["amt"] < 100:
        return f"ספק זהה אך הסכום חורג מטווח הסבילות ({100 - int(top['amt'])}% פער)", top3
    if top["sup"] < 60:
        return "אף סריקה לא תואמת בשם הספק (סף מינימלי 60)", top3
    return "התאמה חלקית בלבד — לא מספיק לעבור את הסף", top3


def _classify_low_confidence(match_row: dict, vat_rate: Decimal) -> str:
    """Explain WHY a row got low_confidence status (vs full match)."""
    try:
        score = float(match_row.get("match_score", 0))
    except (ValueError, TypeError):
        score = 0
    scan_sup = (match_row.get("scan_supplier") or match_row.get("supplier") or "").strip()
    table_sup = (match_row.get("table_supplier") or match_row.get("supplier") or "").strip()
    scan_amt = match_row.get("scan_amount") or ""
    table_amt_pre = match_row.get("table_amount_pre_vat") or match_row.get("amount") or ""
    table_amt_with = match_row.get("table_amount_with_vat") or ""

    reasons = []
    if scan_sup and table_sup:
        s = _score_supplier(table_sup, scan_sup)
        if s < 80:
            reasons.append(f"חוסר חפיפה חלקי בשם הספק (ציון ספק {int(s)})")
    elif scan_sup and not table_sup:
        reasons.append("אין שם ספק בצד הטבלה להשוואה")

    if scan_amt and table_amt_pre:
        try:
            sa = abs(Decimal(scan_amt))
            ta = abs(Decimal(table_amt_with) if table_amt_with else Decimal(table_amt_pre) * (Decimal(1) + vat_rate))
            diff_pct = float(abs(sa - ta) / max(sa, ta)) * 100
            if diff_pct > 1:
                reasons.append(f"פער סכום של {diff_pct:.1f}% (טבלה כולל מע״מ: {ta}, סריקה: {sa})")
        except (ValueError, InvalidOperation):
            pass
    elif not scan_amt:
        reasons.append("חסר סכום בסריקה")

    if not reasons:
        if score < 90:
            reasons.append(f"ציון התאמה משוקלל ({score}) מתחת לסף ההתאמה המלאה (90)")
        else:
            reasons.append("אחד מהשדות אינו מושלם — מומלץ לבדוק ידנית")
    return "; ".join(reasons)


def generate_run_report(
    out_dir: str | Path,
    *,
    vat_rate: Decimal = DEFAULT_VAT_RATE,
) -> str:
    """Read the output CSVs in `out_dir`, synthesize a Hebrew markdown audit
    report, write `out_dir/run_report.md`, and return the text."""
    out_dir = Path(out_dir)

    table_rows = _load_table_csv(out_dir / "table_extracted.csv")
    pages: list[OcrPage] = []
    extracted_csv = out_dir / "all_invoices_extracted.csv"
    if not extracted_csv.exists():
        extracted_csv = out_dir / "extracted_invoices.csv"
    if extracted_csv.exists():
        pages = load_extracted_csv(extracted_csv)

    # Load match_report.csv — tolerate both old and new schemas.
    match_csv = out_dir / "match_report.csv"
    if not match_csv.exists():
        match_csv = out_dir / "match_report_v3.csv"
    match_rows: list[dict] = []
    if match_csv.exists():
        with open(match_csv, encoding="utf-8-sig", newline="") as f:
            match_rows = list(csv.DictReader(f))

    clusters = cluster_consecutive_pages(pages) if pages else []
    cluster_by_first_page = {c.page_indices[0]: c for c in clusters}

    # Build the set of cluster indices that are "in use" by some match row.
    used_cluster_ids: set[int] = set()
    used_first_pages: set[int] = set()
    for r in match_rows:
        pages_str = r.get("matched_cluster_pages") or r.get("matched_page") or ""
        if not pages_str:
            continue
        # Could be "61,62,63,64" or just "61"
        first = pages_str.split(",")[0].strip()
        try:
            first_idx = int(first)
            used_first_pages.add(first_idx)
            if first_idx in cluster_by_first_page:
                used_cluster_ids.add(cluster_by_first_page[first_idx].cluster_index)
        except ValueError:
            pass

    # Aggregate stats
    n_table = len(table_rows)
    n_pages = len(pages)
    n_clusters = len(clusters)
    multi_clusters = sum(1 for c in clusters if len(c.page_indices) >= 2)
    size_dist = Counter(len(c.page_indices) for c in clusters)

    # Match status counts
    status_counts: Counter[str] = Counter(r.get("status", "?") for r in match_rows)

    # Per-field null counts
    tbl_no_supplier = sum(1 for r in table_rows if not r.supplier)
    tbl_no_amount = sum(1 for r in table_rows if r.amount is None)
    tbl_no_date = sum(1 for r in table_rows if r.date is None)

    scn_no_supplier = sum(1 for p in pages if not p.supplier)
    scn_no_amount = sum(1 for p in pages if p.amount is None)
    scn_no_date = sum(1 for p in pages if p.date is None)
    scn_no_id = sum(1 for p in pages if not p.id_number)
    scn_all_null = sum(1 for p in pages if not any([p.supplier, p.amount, p.date, p.id_number]))
    scn_at_most_one = sum(1 for p in pages if sum(1 for v in [p.supplier, p.amount, p.date, p.id_number] if v) <= 1)
    scn_complete = sum(1 for p in pages if all([p.supplier, p.amount is not None, p.date, p.id_number]))

    # ----- Build markdown -----
    lines: list[str] = []
    A = lines.append
    A(f"# דוח אבחון — הרצת התאמת חשבוניות")
    A(f"_תאריך: {datetime.now().strftime('%Y-%m-%d %H:%M')}_  ")
    A(f"_תיקייה: `{out_dir.name}`_")
    A("")
    A("---")
    A("")

    # 1. Run summary
    A("## 1. סיכום הרצה")
    A("")
    A(f"- שורות שחולצו מהטבלה: **{n_table}**")
    A(f"- עמודים סרוקים: **{n_pages}**")
    A(f"- חשבוניות לוגיות (אשכולות): **{n_clusters}** (מתוכן {multi_clusters} רב־עמודיות)")
    A(f"- שיעור מע״מ שהוחל על צד הטבלה: **{int(vat_rate * 100)}%**")
    A("- אלגוריתם התאמה: שני מפתחות (ספק + סכום כולל מע״מ)")
    A("- ספים: התאמה מלאה ≥ 90, התאמה חלשה ≥ 70, אחרת לא הותאם")
    A("")

    # 2. Extraction quality
    A("## 2. איכות החילוץ")
    A("")
    A("### צד הטבלה")
    A(f"- חסר ספק: {tbl_no_supplier} שורות")
    A(f"- חסר סכום: {tbl_no_amount} שורות")
    A(f"- חסר תאריך: {tbl_no_date} שורות")
    A("")
    A("### צד הסריקות")
    A(f"- עמודים עם כל ארבעת השדות: **{scn_complete} מתוך {n_pages}** ({(scn_complete * 100 // n_pages) if n_pages else 0}%)")
    A(f"- עמודים ריקים לחלוטין: {scn_all_null}")
    A(f"- עמודים עם שדה אחד בלבד (סביר עמוד המשך): {scn_at_most_one}")
    A(f"- חסר ספק: {scn_no_supplier} עמודים")
    A(f"- חסר סכום: {scn_no_amount} עמודים  ⚠️ (אלה לא יוכלו להיות מותאמים)")
    A(f"- חסר תאריך: {scn_no_date} עמודים")
    A(f"- חסר מס׳ עוסק: {scn_no_id} עמודים")
    A("")

    # 3. Clusters
    A("## 3. אשכולות (חשבוניות רב־עמודיות)")
    A("")
    A(f"- סך אשכולות: {n_clusters}")
    A(f"- אשכולות רב־עמודיים: {multi_clusters}")
    A(f"- התפלגות גדלים: " + ", ".join(f"גודל {k}: {v}" for k, v in sorted(size_dist.items())))
    A("")
    if multi_clusters:
        A("**רשימת אשכולות רב־עמודיים:**")
        for c in clusters:
            if len(c.page_indices) >= 2:
                A(f"- עמודים `{c.page_indices}` ← {c.supplier or '(ספק לא ידוע)'}")
        A("")

    # 4. Match results
    A("## 4. תוצאות התאמה")
    A("")
    matched_n = status_counts.get("matched", 0)
    low_n = status_counts.get("low_confidence", 0)
    unmatched_n = status_counts.get("unmatched", 0)
    leftover_n = status_counts.get("leftover", 0)
    A(f"- ✅ הותאמו: **{matched_n}**")
    A(f"- ⚠️ התאמה חלשה: **{low_n}**")
    A(f"- ❌ לא הותאמו: **{unmatched_n}**")
    if leftover_n:
        A(f"- שאריות (סריקות ללא שורת טבלה תואמת): **{leftover_n}**")
    if n_table:
        A(f"- סה״כ הוקצו לשורות טבלה: **{matched_n + low_n}** מתוך {n_table} ({((matched_n + low_n) * 100 // n_table)}%)")
    A("")

    # 5. Detailed problems
    A("## 5. בעיות מפורטות")
    A("")

    # 5a. Unmatched
    unmatched_match_rows = [r for r in match_rows if r.get("status") == "unmatched" and (r.get("table_row_index") or r.get("table_row_index") == "0")]
    A(f"### שורות שלא הותאמו ({len(unmatched_match_rows)})")
    A("")
    if not unmatched_match_rows:
        A("_אין שורות לא־מותאמות. כל שורות הטבלה קיבלו הקצאה._")
        A("")
    else:
        for r in unmatched_match_rows:
            try:
                row_idx = int(r.get("table_row_index", "")) if r.get("table_row_index") else None
            except ValueError:
                row_idx = None
            sup = r.get("table_supplier") or r.get("supplier") or ""
            amt = r.get("table_amount_pre_vat") or r.get("amount") or ""
            d = r.get("table_date") or r.get("date") or ""
            A(f"#### שורה {row_idx}: `{sup}` — {amt} ₪ ({d})")

            # Find this row in table_rows for full data
            tbl_row = next((t for t in table_rows if t.index == row_idx), None) if row_idx is not None else None
            if tbl_row:
                reason, top3 = _classify_unmatched(tbl_row, clusters, used_cluster_ids, vat_rate, table_rows)
                A(f"- **סיבה**: {reason}")
                if top3:
                    A("- **מועמדים מובילים:**")
                    for i, cand in enumerate(top3, 1):
                        c = cand["cluster"]
                        in_use_mark = " ⚠️ (תפוס על־ידי שורה אחרת)" if cand["in_use"] else ""
                        A(f"  {i}. אשכול עמודים `{c.page_indices}` — ספק: `{c.supplier or '?'}`, סכום: {c.amount or '?'}{in_use_mark}")
                        A(f"     - ציון ספק: {int(cand['sup'])}, ציון סכום: {int(cand['amt'])}")
            else:
                A("- _לא הצלחתי לטעון את פרטי השורה מהטבלה._")
            A("")

    # 5b. Low confidence
    low_match_rows = [r for r in match_rows if r.get("status") == "low_confidence"]
    A(f"### שורות בהתאמה חלשה ({len(low_match_rows)})")
    A("")
    if not low_match_rows:
        A("_אין שורות בהתאמה חלשה._")
        A("")
    else:
        A("_שורות אלה הותאמו אך הציון לא הגיע ל־90. מומלץ לבדוק ידנית במיקום ה־PDF המצוין._")
        A("")
        for r in low_match_rows:
            pos = r.get("output_pdf_position") or "?"
            row_idx = r.get("table_row_index") or "?"
            tbl_sup = r.get("table_supplier") or r.get("supplier") or ""
            scn_sup = r.get("scan_supplier") or ""
            score = r.get("match_score") or "?"
            A(f"#### עמוד PDF {pos} · שורה {row_idx}: `{tbl_sup}` ↔ `{scn_sup}` (ציון {score})")
            A(f"- **מה חלש**: {_classify_low_confidence(r, vat_rate)}")
            A("")

    # 6. Scan issues
    A("## 6. בעיות בצד הסריקות")
    A("")
    problem_pages = [p for p in pages if (p.supplier is None or p.amount is None or p.date is None or p.id_number is None)]
    if not problem_pages:
        A("_כל הסריקות חולצו עם כל ארבעת השדות. מצוין._")
        A("")
    else:
        # Categorize
        blank_pages = [p for p in pages if not any([p.supplier, p.amount, p.date, p.id_number])]
        no_amount_pages = [p for p in pages if p.amount is None and p.supplier]
        no_supplier_pages = [p for p in pages if not p.supplier and (p.amount is not None or p.date or p.id_number)]

        if blank_pages:
            A(f"### עמודים ריקים לחלוטין ({len(blank_pages)})")
            A("_נראה כעמודי הפרדה / ריקים. בדוק את ה־PDF המקורי._")
            for p in blank_pages[:20]:
                A(f"- עמוד {p.page_index}")
            A("")

        if no_amount_pages:
            A(f"### עמודים שחסר בהם הסכום ({len(no_amount_pages)})")
            A("_עמודים אלה לא יוכלו להיות מותאמים בקפדנות. סבירות גבוהה שהם עמוד 2 של חשבונית רב־עמודית (אם עמוד {N-1} הוא אותו ספק) או סריקה באיכות נמוכה._")
            for p in no_amount_pages[:20]:
                A(f"- עמוד {p.page_index}: `{p.supplier}`")
            if len(no_amount_pages) > 20:
                A(f"- _...ועוד {len(no_amount_pages) - 20} עמודים_")
            A("")

        if no_supplier_pages:
            A(f"### עמודים שחסר בהם הספק ({len(no_supplier_pages)})")
            for p in no_supplier_pages[:20]:
                fields = []
                if p.amount: fields.append(f"סכום={p.amount}")
                if p.date: fields.append(f"תאריך={p.date}")
                if p.id_number: fields.append(f"ע.מ.={p.id_number}")
                A(f"- עמוד {p.page_index}: " + ", ".join(fields))
            A("")

    # 7. Recommendations
    A("## 7. המלצות")
    A("")
    recs: list[str] = []
    if n_pages and scn_no_amount / n_pages > 0.1:
        recs.append(f"📈 **העלאת רזולוציה**: ב־{scn_no_amount} עמודים מתוך {n_pages} ({scn_no_amount * 100 // n_pages}%) חסר הסכום. שקול להעלות את `RENDER_DPI` מ־200 ל־300 לחילוץ טוב יותר של מספרים קטנים.")
    transliteration_count = 0
    for r in unmatched_match_rows:
        sup = r.get("table_supplier") or r.get("supplier") or ""
        amt = r.get("table_amount_pre_vat") or r.get("amount") or ""
        if amt:
            try:
                row_idx = int(r.get("table_row_index", "-1"))
                tbl = next((t for t in table_rows if t.index == row_idx), None)
                if tbl:
                    for c in clusters:
                        if c.cluster_index in used_cluster_ids:
                            continue
                        if _score_amount(tbl.amount, c.amount, vat_rate) == 100 and _score_supplier(tbl.supplier, c.supplier) == 0:
                            transliteration_count += 1
                            break
            except ValueError:
                pass
    if transliteration_count >= 3:
        recs.append(f"🔤 **טבלת תרגום ידני**: {transliteration_count} שורות נכשלו עם סכום מושלם אך ספק שונה — סבירות גבוהה לתעתיק עברית↔אנגלית. שקול להוסיף מילון אליאסים (לדוגמה: `וויטסנט` → `WhiteScent`).")
    if unmatched_n:
        recs.append(f"👀 **בדיקה ידנית**: {unmatched_n} שורות דורשות בדיקה ידנית. ראה סעיף 5 לפרטים.")
    if low_n:
        recs.append(f"🔍 **אימות**: {low_n} שורות בהתאמה חלשה — מומלץ לפתוח את ה־PDF במיקומים המצוינים ולוודא שההתאמה נכונה.")
    if not recs:
        recs.append("✨ אין המלצות — ההרצה הצליחה ללא חריגים משמעותיים.")
    for r in recs:
        A(f"- {r}")
    A("")

    text = "\n".join(lines)
    report_path = out_dir / "run_report.md"
    report_path.write_text(text, encoding="utf-8-sig")
    log.info("wrote %s (%d chars)", report_path, len(text))
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(
    table_pdf: str | Path,
    scanned_pdf: str | Path,
    out_dir: str | Path,
    max_pages: Optional[int] = None,
    use_vision_for_table: bool = True,
    vat_rate: Decimal = DEFAULT_VAT_RATE,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    if use_vision_for_table:
        rows = parse_table_pdf_with_claude(table_pdf, client=client)
    else:
        rows = parse_table_pdf(table_pdf)
    pages = extract_pages_with_claude(scanned_pdf, client=client, max_pages=max_pages)
    matches = match(rows, pages, vat_rate=vat_rate)
    out_pdf = out_dir / "output_sorted.pdf"
    out_csv = out_dir / "match_report.csv"
    extracted_csv = out_dir / "all_invoices_extracted.csv"
    table_csv = out_dir / "table_extracted.csv"
    write_sorted_pdf(scanned_pdf, out_pdf, matches, total_pages=len(pages))
    write_report_csv(matches, out_csv)
    write_extracted_csv(pages, extracted_csv)
    write_table_csv(rows, table_csv)
    # Generate the Hebrew audit report from the CSVs that were just written.
    try:
        generate_run_report(out_dir, vat_rate=vat_rate)
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reorder scanned-invoices PDF to match a table PDF using Claude vision.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--scanned", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vat-rate", type=float, default=float(DEFAULT_VAT_RATE),
                        help=f"VAT rate to add to table amounts before matching (default: {DEFAULT_VAT_RATE})")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set in the environment.", file=sys.stderr)
        return 2

    summary = run(args.table, args.scanned, args.out_dir, vat_rate=Decimal(str(args.vat_rate)))
    print("Done.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
