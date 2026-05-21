# Invoice Reorder — Specification & Business Logic

> **המלך CFO & Co. — Invoice Reconciliation System**

This document is the canonical spec for the invoice reorder pipeline. It covers the business problem, the domain rules (Hebrew, VAT, date formats), the API surface, the matching algorithm, and the validation rules an analyst should expect.

For a tour of the codebase and how each stage maps to a function, see [PROCESS.md](PROCESS.md). For the end-user "how do I run it" guide, see [README.md](README.md).

---

## 1. Business Problem

The accounting team receives two related PDF documents each month from a property-management operation in Bat Yam (`ביג מרכזי קניות`):

1. **`הוראת תשלום.pdf`** — a **payment order** ("transfer summary" / `ריכוז העברות`). One row per planned payment to a supplier. ~3 pages, ~57 rows, totaling ~₪1.15M with VAT. Generated digitally; Hebrew RTL.
2. **`all_invoices.pdf`** — the **scanned individual invoices** that back up the payment order. ~100+ pages, one invoice per page, arbitrary order. Mix of scanned paper receipts, generated PDFs, and phone photos (some upside-down). Hebrew + a few English suppliers (Rentokil, netvision).

The auditor's question is "**for every line in the payment order, where is the proving invoice?**" Today this is a manual job — printing, sorting, paper-clipping. We automate it.

### Desired outcome

Given the two PDFs above, the system produces:

| Output | What it is | Who uses it |
|---|---|---|
| `output_sorted.pdf` | A reordered version of `all_invoices.pdf` whose page sequence matches the table's row order. Pages without a table match appear at the end. | Auditor — pages line up with the payment order rows. |
| `match_report.csv` | One row per page of `output_sorted.pdf`. Carries the scan-side fields, the table-row-side fields, the VAT-adjusted comparison, the match score, and the status. | Analyst — exception list for any row that didn't match cleanly. |
| `all_invoices_extracted.csv` | One row per scanned page: supplier, amount, date, business ID. Independent of the table. | Accounting export, also feeds the matcher. |
| `table_extracted.csv` | One row per table line: supplier, pre-VAT amount, billing month, business ID (if any). | Accounting export, also feeds the matcher. |

The whole pipeline must work on Hebrew-language inputs.

---

## 2. Domain Rules (the things that aren't obvious from the code)

### 2.1 VAT (מע"מ)

**The single most important business rule.**

`הוראת תשלום` lists amounts **pre-VAT**. Scanned invoices show amounts **with VAT included**. So:

```
table_amount × (1 + VAT_RATE)  ≈  scanned_invoice_amount
```

The matcher MUST apply VAT to the table side before comparing. Without this adjustment, **zero rows match**.

| Constant | Value | Why |
|---|---|---|
| `DEFAULT_VAT_RATE` | `Decimal("0.17")` | Israeli standard VAT. The sample document shows 18% on its total line, but 17% is what the business rule in this organization uses; the matcher's ±1% / ±5% tolerance accommodates either. |

If the VAT rate changes in the future, update `DEFAULT_VAT_RATE` in [reorder_invoices.py](reorder_invoices.py). Negative table amounts (credits / זיכוי) have VAT applied the same way; the score function compares absolute values.

### 2.2 Hebrew text & RTL

- Both documents are in Hebrew. The web UI is also Hebrew, RTL (`<html lang="he" dir="rtl">`).
- Supplier names commonly carry a **location prefix**: `בת ים-X`. The matcher strips this before fuzzy-comparing against the scanned supplier name, which usually doesn't include the city. List of stripped prefixes lives in `SUPPLIER_LOCATION_PREFIXES`.
- Scanned receipt suppliers often have legal suffixes — `אלקטרה M&E` vs `אלקטרה בע"מ`. The matcher uses `thefuzz.token_set_ratio`, which is robust to subset / superset relationships.
- Mixed Hebrew/English suppliers (e.g. `Rentokil`, `netvision`) work natively — the fuzzy matcher is byte-blind.

### 2.3 Date formats

Three formats appear in the data:

| Source | Format | Example | How parsed |
|---|---|---|---|
| Table billing period | `MM/YY` | `03/26` | → first day of that month, 2000s century (`2026-03-01`) |
| Scanned invoice | `DD/MM/YY` or `DD.MM.YY` or `DD/MM/YYYY` | `11.7.23`, `14/05/2026` | day-first via `dateutil` |
| Cross-system CSV roundtrip | ISO `YYYY-MM-DD` | `2026-03-01` | direct via `date.fromisoformat()` — **must not** apply `dayfirst=True` |

The last point caused a real bug: re-loading an ISO date with `dateutil(dayfirst=True)` flips `2026-03-01` into `Jan 3, 2026`. `_norm_date()` now short-circuits to `date.fromisoformat()` whenever it sees `YYYY-MM-DD`.

### 2.4 Israeli business identifier

`עוסק מורשה / ח.פ / ע.מ` — an 8 or 9-digit business ID. When present on both the table and the scan, an exact match is worth full points (0.2 weight by default). The single `קוד הספק` printed in the document header of `הוראת תשלום` is NOT a per-row ID — the table rows don't have IDs; the matcher tolerates this.

### 2.5 Multi-page invoices

Currently out of scope. The system assumes **1 scanned page = 1 invoice**. Several real invoices that span 2+ pages are present in `all_invoices.pdf`; they're treated as separate single-page invoices and may match independently. This is a known limitation.

### 2.6 Rotation

Scanned invoices include phone photos and ad-hoc captures. Some land upside-down. Claude vision tolerates rotation natively — most rotated pages still extract correctly on the first try. For the small fraction that come back with mostly-null fields, the system retries automatically at 180° → 90° → 270° rotation, keeping whichever orientation yields the most non-null fields. See [reorder_invoices.py:_likely_misread()](reorder_invoices.py) and `_rotate_png_bytes()`.

---

## 3. Pipeline Specification

### 3.1 Extraction (per page, both sides)

Each PDF page is rendered to PNG at 200 DPI via PyMuPDF, then sent to **Claude Opus 4.7** with a structured-output schema (Pydantic).

**Scan side** — `EXTRACTION_SYSTEM_PROMPT` returns `{supplier, amount, date, id_number}`. All fields nullable; null means "couldn't read with confidence".

**Table side** — `TABLE_EXTRACTION_SYSTEM_PROMPT` returns a list of `{supplier, amount, date_raw, id_number}`. The `date_raw` is the literal cell text (no model-side conversion); Python applies MM/YY parsing in `_parse_table_date()`. This split is essential — letting the model convert dates caused `03/26` → `2026-01-03` (Jan 3) misreads.

The system prompt is cached (`cache_control: ephemeral`, 5-min TTL), so calls 2..N cost ~10% of call 1 for the prompt portion.

### 3.2 Matching algorithm

There are two matchers:

#### Strict 1:1 ([`match()`](reorder_invoices.py)) — **DEFAULT for table-vs-scans**

For each table row, score every still-unused scan page. Pick the highest-scoring scan above threshold.

Score = weighted sum, with `[supplier 0.4, amount 0.25, date 0.15, id 0.2]`. Each component is 0–100:

| Component | 100 points | 50 points | 0 points |
|---|---|---|---|
| `supplier` | `token_set_ratio` after stripping `בת ים-` location prefixes | (continuous) | low overlap |
| `amount` | `table×(1+VAT)` vs `scan` within ±1% | within ±5% | else |
| `date` | same year+month | adjacent month | else |
| `id` | exact match | — | else |

Threshold = 60. ≥75 → `matched`; 60–75 → `low_confidence`; below → `unmatched`. Each scan can be used by at most one row.

#### Many-to-one grouping ([`order_scans_by_table()`](reorder_invoices.py)) — fallback

For documents where the table is a summary and one table row sums many scans. Used by setting `use_grouped=True`.

- Hard floor on supplier match: `_score_supplier ≥ 60`. No supplier overlap → can't be assigned.
- Score = `0.5*supplier + 0.4*date + 0.1*id`. Amount weight = 0 (amounts don't roll up cleanly).
- Threshold = 50. Multiple scans can share one row.
- Within a row, scans sort by descending score then ascending source-page index for stability.

### 3.3 Output ordering

Output PDF page order = `(matched pages in table-row order) + (leftover scans in source order)`.

`match_report.csv` rows are in the SAME order as the output PDF — `row N of CSV ↔ page N of PDF`. The `output_page` column is the 1-indexed PDF position; `source_page` is the original 0-indexed position in `all_invoices.pdf`.

---

## 4. Validation Rules

What an analyst should expect to see, and what flags an exception:

| Check | Expected | Action if violated |
|---|---|---|
| Every table row has a matched scan | ≥85% match rate at score ≥ 60 (strict mode, VAT correct) | If the rate is much lower, suspect a wrong VAT rate or a table↔scans pair from different months. |
| Status distribution | mostly `matched`, a few `low_confidence`, a few `unmatched` | Spot-check every `low_confidence` row; review every `unmatched` row in case the scan exists but with degraded extraction. |
| `table_amount_with_vat` matches `scan_amount` | within ±1% | A larger gap means VAT rate is wrong, or table+scan are from different periods. |
| Date alignment | scan date in same month as `table_date` (which is the 1st of the billing month) | Adjacent-month is acceptable (invoice received in the next month for the prior period). Multi-month gap means likely wrong assignment. |
| Supplier alignment | table supplier (with `בת ים-` stripped) is a token-subset of the scan supplier | A wrong supplier flags the matcher took a low-quality fallback. Re-check against `order_scans_by_table` mode. |
| Total of `table_amount_pre_vat` × 1.17 | matches the sum of matched scans (±1% per row, ±0.5% in aggregate) | A larger aggregate gap means either VAT rate mis-spec or duplicated / missing scans. |

---

## 5. API Surface

### Python module ([reorder_invoices.py](reorder_invoices.py))

| Function | Purpose |
|---|---|
| `parse_table_pdf_with_claude(pdf, client=None)` | Extract table rows via Claude vision (use for RTL Hebrew tables). |
| `parse_table_pdf(pdf)` | Extract table rows via pdfplumber (use only for LTR / English / digitally-clean tables). |
| `extract_pages_with_claude(pdf, max_pages=None, retry_with_rotation=True)` | Extract scan-side fields via Claude vision. Auto-rotation retry on misreads. |
| `match(rows, pages)` | Strict 1:1 matching with VAT-aware amount comparison. |
| `order_scans_by_table(rows, pages, threshold=50)` | Many-to-one grouped ordering with supplier hard floor. |
| `retry_failed_pages_with_rotation(pdf, pages)` | Post-hoc rotation retry on a previously-extracted list of pages. |
| `load_extracted_csv(path)` | Load `all_invoices_extracted.csv` back into `OcrPage` objects. Uses ISO date parsing. |
| `write_sorted_pdf / write_report_csv / write_extracted_csv / write_table_csv` | Output writers. All CSV writers use `utf-8-sig` so Excel renders Hebrew correctly. |
| `combine_files_to_pdf(paths, out_pdf)` | Concatenate arbitrary PDFs + images into one PDF (used by `/batch`). |
| `batch_extract_and_combine(paths, out_dir, sort_by)` | Multi-file flow: extract each, sort by chosen field, combine into one PDF + CSV. |

### Flask web routes ([app.py](app.py))

| Route | Method | Purpose |
|---|---|---|
| `GET /` | `index` | `הוראת תשלום` + scans upload form (the main reconciliation flow). |
| `POST /reorder` | runs the pipeline; supports optional `max_pages` for partial runs |
| `GET /batch` | multi-file invoice combiner |
| `POST /batch` | extract every file, sort by `date/supplier/amount/id_number`, combine into one PDF |
| `GET /test` | single-invoice QA upload |
| `POST /test` | extract one file, return four fields + CSV |
| `POST /test.json` | same as `/test` but returns JSON (for scripting) |
| `GET /runs` | history of past jobs (both web + CLI runs) |
| `GET /results/<job>` | render the results page for any output folder (web_outputs or top-level) |
| `GET /download/<job>/<file>` | safe download of any output file |

### CLI ([reorder_invoices.py](reorder_invoices.py) `main`)

```
python reorder_invoices.py ^
  --table   "הוראת תשלום.pdf" ^
  --scanned "all_invoices.pdf" ^
  --out-dir "full_run_output" ^
  -v
```

Same pipeline as `POST /reorder`. Better for large batches (no browser idle).

---

## 6. Cost Model

Per-page Claude vision cost on Opus 4.7 at 200 DPI ≈ **$0.02**.

| Workload | Vision calls | Cost |
|---|---:|---:|
| 3-page table + 101-page scans | 104 | ~$2.10 |
| 3-page table + 50-page scans (typical month) | 53 | ~$1.05 |
| Single-invoice QA (`/test`) | 1 | ~$0.02 |
| Batch of 20 individual invoices (`/batch`) | 20 | ~$0.40 |

The system prompt is cached (5-min ephemeral), so prompts on calls 2..N cost ~10% of call 1.

To halve the cost, switch `CLAUDE_MODEL` in [reorder_invoices.py](reorder_invoices.py) to `claude-sonnet-4-6` (Hebrew extraction quality is still strong for printed receipts; handwriting tolerance drops).

---

## 7. Configuration Reference

All tunables live at the top of [reorder_invoices.py](reorder_invoices.py):

| Constant | Default | Purpose |
|---|---|---|
| `CLAUDE_MODEL` | `claude-opus-4-7` | Vision model. Change to `claude-sonnet-4-6` for lower cost. |
| `RENDER_DPI` | `200` | PNG render DPI. Bump to 300 for small / faint text; raises image tokens. |
| `DEFAULT_VAT_RATE` | `Decimal("0.17")` | Added to table amounts before comparison. |
| `SCORE_WEIGHTS` | `{supplier: 0.4, amount: 0.25, date: 0.15, id: 0.2}` | Weights for `match()`. |
| `MATCH_THRESHOLD` | `60.0` | Below → `unmatched`. |
| `HEB_COL_SUPPLIER / _AMOUNT / _DATE / _ID` | Hebrew header tokens | What `_detect_columns()` looks for in pdfplumber-extracted tables. Not used in vision path. |
| `SUPPLIER_LOCATION_PREFIXES` | `בת ים- / ת"א- / ירושלים-` etc | Stripped from supplier names before fuzzy match. |

---

## 8. Known Limitations

- **Multi-page invoices** — currently treated as separate single-page invoices. Cluster-adjacent pages with the same supplier+amount before matching to fix.
- **Globally-optimal assignment** — `match()` is greedy. At 100-200 invoices that's fine; at thousands, switch to `scipy.optimize.linear_sum_assignment`.
- **Async / batched API** — all calls are synchronous. `AsyncAnthropic + asyncio.gather` would parallelize 5–10×.
- **Anthropic Batches API** — 50% cost reduction for non-time-critical runs; adds 1-hour async turnaround. Worth wiring once volumes pass ~500 pages/month.
- **No per-row edit UI** — analyst can't correct an extracted field in the web app; must edit the CSV externally and re-run the matcher.

---

## 9. File Map

```
invoice_reorder/
├── reorder_invoices.py             # Core pipeline (extraction, matching, outputs)
├── app.py                          # Flask web app
├── reorder_invoices.ipynb          # Notebook for ad-hoc analysis
├── requirements.txt
├── README.md                       # Quick start
├── PROCESS.md                      # How it works (engineer-facing)
├── SPEC.md                         # This document (analyst + engineer)
├── templates/                      # Jinja2 templates (Hebrew RTL)
│   ├── base.html  index.html  batch.html  batch_results.html
│   ├── test.html  results.html  runs.html  error.html
├── static/
│   └── cfo.jpg                     # The King CFO portrait
├── web_outputs/<job>/              # Per-job outputs from web runs
├── full_run_output/                # Per-job output from CLI runs
└── samples/                        # Test PDFs
```
