# Invoice Reorder — How It Works

This document explains the full pipeline behind the **המלך CFO** invoice reconciliation app: what the inputs are, what happens to them, what comes out, and how each piece of code maps to a real-world step.

Audience: someone who wants to understand or extend the system. Hebrew is the primary UI language; this doc is in English so future engineers can follow it.

---

## TL;DR

Two PDFs go in, four files come out.

```
table.pdf            ─┐
(Hebrew payment       │       ┌─── output_sorted.pdf       (scans reordered to table order)
 order list)          │       │
                      ├──→    ├─── match_report.csv         (per-row: scan page + score + status)
all_invoices.pdf     ─┤       │
(scanned receipts,    │       ├─── all_invoices_extracted.csv   (per scan: supplier/amount/date/id)
 one per page,        │       │
 arbitrary order)     │       └─── table_extracted.csv      (per table row: supplier/amount/date/id)
                      ┘
```

Under the hood, **Claude vision (Opus 4.7)** reads every page of both PDFs as an image, and a Python matcher pairs scanned pages with table rows by supplier name, amount, date, and Israeli business ID.

---

## The Five Stages

### 1. Render

[`render_pdf_pages_to_png()`](reorder_invoices.py) — Uses **PyMuPDF (`fitz`)** to rasterize every page of a PDF to PNG bytes at 200 DPI. On A4 that produces ~1654×2339 pixel images, comfortably under Opus 4.7's 2576-px long-edge limit.

Why render at all? Claude vision works on images, not PDF byte streams. Rendering also lets us bypass `pdfplumber`, which mangles right-to-left Hebrew text by extracting characters in visual (reversed) order.

### 2. Extract — table side

[`parse_table_pdf_with_claude()`](reorder_invoices.py)

For each page image of the table PDF (e.g. `הוראת תשלום.pdf`):

- Sends the image to Claude with the `TABLE_EXTRACTION_SYSTEM_PROMPT` (cached for 5 minutes — first call writes the cache at ~1.25× cost, every subsequent call reads it at ~0.1×).
- Asks for JSON `{rows: [{supplier, amount, date, id_number}]}`, validated through a Pydantic `_TableExtraction` schema with `client.messages.parse(...)`.
- Concatenates rows across all pages and normalizes amounts to `Decimal` and dates to `date` objects.

The prompt explicitly handles:
- Hebrew column headers like `תיאור / שם הספק` for the supplier column, `מחיר / סכום / סה"כ` for amount, `עבור תאריך / תאריך` for date.
- `MM/YY` billing periods (e.g. `04/26` → `2026-04-01`).
- Skipping totals, headers, signatures, "department" annotations.

### 3. Extract — scanned side

[`extract_pages_with_claude()`](reorder_invoices.py)

For each page image of the scanned PDF:

- Sends the image to Claude with the `EXTRACTION_SYSTEM_PROMPT` (also cached).
- Asks for a single JSON object `{supplier, amount, date, id_number}` via `_ExtractedInvoice` Pydantic schema.
- Normalizes amounts and dates the same way.

#### Rotation retry (auto-fix for upside-down scans)

After the first read, [`_likely_misread()`](reorder_invoices.py) checks the result. A page is suspect if:
- Supplier came back `null`, OR
- Fewer than 2 of the 4 fields are non-null.

For suspect pages, the system retries at **180°, then 90°, then 270°** rotation ([`_rotate_png_bytes()`](reorder_invoices.py)) and keeps whichever rotation produced the most non-null fields. Stops early once a rotation yields ≥3 fields (we have enough to match cleanly).

This adds cost only when needed. A clean 100-page batch might trigger zero retries; a folder of phone photos might trigger several.

#### Retrofitting an existing run

If you already extracted a long PDF but later realize some pages failed, [`retry_failed_pages_with_rotation()`](reorder_invoices.py) loads the existing CSV ([`load_extracted_csv()`](reorder_invoices.py)) and retries **only the suspect pages** with rotation. Pays for retries, not full re-extraction.

### 4. Match (two modes)

There are two matching strategies, picked based on the shape of the table:

#### Mode A — 1:1 strict match ([`match()`](reorder_invoices.py))

Used when each table row corresponds to exactly one scanned invoice. Greedy best-pair: for each table row in original order, score every unmatched scan page and pick the best one above threshold.

| Field | Weight | Scoring |
|---|---:|---|
| `supplier` | 0.4 | `thefuzz.token_set_ratio(table, scan)` — strips "בת ים-" location prefix from both sides |
| `amount` | 0.25 | 100 if within ±1%, 50 if within ±5%, else 0 |
| `date` | 0.15 | 100 if same year+month, 50 if adjacent, else 0 |
| `id_number` | 0.20 | 100 if exact match, else 0 |

Threshold **60**. ≥75 → `matched`; 60–75 → `low_confidence`; below → `unmatched`.

#### Mode B — many-to-one grouping ([`order_scans_by_table()`](reorder_invoices.py))

Used when the table is a **summary** — e.g. a payment-order where one row sums up many individual invoices. The amounts don't match 1:1 with the scans, but each scan still belongs to some row's supplier.

For each scan, score it against every table row using:

| Field | Weight | Why higher than Mode A |
|---|---:|---|
| `supplier` | 0.5 | Primary anchor when amounts don't align |
| `date` | 0.4 | Distinguishes same-supplier rows that differ only by month |
| `id_number` | 0.1 | Often missing per-row in payment orders |
| `amount` | 0 | Ignored — the table totals are aggregates |

Each scan is assigned to its highest-scoring row (threshold **40**). Multiple scans can share a row. Scans below threshold go to the end of the output PDF in upload order.

This is the right mode when the table is `הוראת תשלום` (payment instruction) and the scans are the individual receipts that justify those payments.

Both modes are deterministic and greedy. For thousands of pages, `scipy.optimize.linear_sum_assignment` would give globally-optimal 1:1 assignments — not needed at the current scale.

### 5. Write outputs

| Function | Output | Format |
|---|---|---|
| [`write_sorted_pdf()`](reorder_invoices.py) | `output_sorted.pdf` | PyMuPDF: matched pages in table order, unmatched pages appended at the end |
| [`write_report_csv()`](reorder_invoices.py) | `match_report.csv` | UTF-8 BOM (so Excel renders Hebrew correctly) |
| [`write_extracted_csv()`](reorder_invoices.py) | `all_invoices_extracted.csv` | Per-scan-page extraction (flat table) |
| [`write_table_csv()`](reorder_invoices.py) | `table_extracted.csv` | Per-table-row extraction |

---

## Three Ways To Use It

### A. Web app (recommended)

```
python app.py
```

Open <http://127.0.0.1:5000/> and use one of three pages:

| Route | Purpose |
|---|---|
| `/` (`מיון לפי טבלה`) | Upload `table.pdf` + `scanned.pdf` + optional max-pages limit. Full pipeline. |
| `/batch` (`איחוד אצווה`) | Upload many individual invoice files (PDF or image). Extract + sort by date/supplier/amount → one combined PDF. No table needed. |
| `/test` (`בדיקה בודדת`) | Upload one invoice. Get the four fields back as a row + CSV. Spot-check / QA before a big batch. |
| `/runs` (`היסטוריה`) | Browse past runs (web + CLI), open any of them. |
| `/results/<job>` | Render a results page for any output folder. |

All outputs land under `web_outputs/<timestamp-hash>/` and stay there for repeat downloads.

### B. CLI (good for one-off large batches)

```
python reorder_invoices.py ^
  --table   "הוראת תשלום.pdf" ^
  --scanned "all_invoices.pdf" ^
  --out-dir "full_run_output" ^
  -v
```

Same pipeline. Best for long runs where you don't want the browser idle for 15 min. Outputs land in `full_run_output/`. After it finishes, open <http://127.0.0.1:5000/results/full_run_output> to view in the browser.

### C. Notebook (`reorder_invoices.ipynb`)

Cells expose each stage. Useful for tuning thresholds, regex, or matcher weights while iterating on real data.

---

## Costs

Cost per page depends mostly on image size; with the defaults (200 DPI, ~2000 input tokens including a ~600-token cached system prompt) on Opus 4.7:

| Operation | Per page | 101-page batch |
|---|---:|---:|
| Scanned page extraction | ~$0.02 | ~$2.02 |
| Table page extraction | ~$0.02 | (3 pages ≈ $0.06) |
| Rotation retry (when triggered) | ~$0.02 × rotations tried | varies |

For cheaper runs, change `CLAUDE_MODEL` in [reorder_invoices.py](reorder_invoices.py) to `claude-sonnet-4-6` (~40% cheaper) or `claude-haiku-4-5` (~80% cheaper). Vision quality is still solid on clean printed receipts; handwriting and rotation tolerance drop slightly.

---

## File Map

```
invoice_reorder/
├── app.py                          # Flask web app
├── reorder_invoices.py             # Core pipeline (all 5 stages)
├── reorder_invoices.ipynb          # Tuning notebook
├── requirements.txt
├── PROCESS.md                      # This document
├── README.md                       # Quick-start guide for users
├── templates/                      # Jinja2 templates (Hebrew RTL, Wall Street look)
│   ├── base.html                   # Layout, navy + gold palette, font + RTL setup
│   ├── index.html                  # /  (reorder by table)
│   ├── batch.html                  # /batch (combine many invoices)
│   ├── batch_results.html
│   ├── test.html                   # /test (single-invoice QA)
│   ├── results.html                # /results/<job>
│   ├── runs.html                   # /runs (history)
│   └── error.html
├── static/
│   └── cfo.jpg                     # The King CFO portrait (you place this here)
├── web_outputs/                    # Generated job folders from web runs
├── full_run_output/                # CLI run output (e.g. for the 101-page job)
└── samples/                        # Sample PDFs for testing
```

---

## Key Design Decisions (and why)

- **Claude vision instead of Tesseract**: Hebrew + handwriting + rotation are hard for Tesseract; vision LLMs read them natively. No traineddata, no Poppler, no OSD step.
- **Structured outputs via Pydantic** (`messages.parse`): no string-matching the response, no JSON parsing edge cases — schema validation is built in.
- **System prompt caching**: same prompt for every page → 5-minute ephemeral cache. First call writes; subsequent calls read at ~0.1× input price.
- **Rotation retry as a fallback, not the default path**: most pages don't need it. Triggering only on misreads keeps cost down.
- **Greedy match, not optimal**: simpler code, fast on 100s of rows, easy to debug. Can swap in Hungarian assignment if we ever hit thousands.
- **MM/YY date handling**: Israeli payment-order tables use billing periods (`04/26`); scanned receipts have full dates. The matcher compares **month + year**, not exact day.
- **Location-prefix stripping**: `בת ים-X` in the table vs. plain `X` on the receipt are still the same supplier. The matcher normalizes both sides before fuzzy comparison.
- **Hebrew-safe CSVs**: every CSV writer uses `utf-8-sig` so Excel opens them with Hebrew rendered correctly.

---

## Future improvements

- **Async / batched API calls**: currently one Claude call per page, sequentially. `asyncio.gather` (with `AsyncAnthropic`) could parallelize and cut wall time 5–10×.
- **Anthropic Batches API**: 50% cost reduction for large jobs, but adds 1-hour async turnaround. Worth it for >500-page batches.
- **Per-row "approve/edit" UI**: results page shows extracted fields next to the scan thumbnail; user can correct mistakes before exporting.
- **Globally-optimal matching** (`scipy.optimize.linear_sum_assignment`): only worth it at scale.
- **Multi-page invoices**: current code assumes 1 page = 1 invoice. Could cluster consecutive pages with the same supplier/amount.
