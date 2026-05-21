# Invoice PDF Reorder

Reorder a scanned-invoices PDF so each page lines up with the matching row in an invoice-table PDF. Uses **Claude vision** (`claude-opus-4-7`) to read each scanned page directly — no Tesseract, no Poppler, no traineddata, no manual rotation.

## What it does

1. Parses the **table PDF** with `pdfplumber` — one row per invoice with supplier, amount, date, and business ID number (עוסק מורשה / ח.פ).
2. Renders each page of the **scanned PDF** to an image via PyMuPDF.
3. Sends each page image to **Claude Opus 4.7** with a Hebrew-aware extraction prompt — gets back structured `{supplier, amount, date, id_number}`. The system prompt is cached so per-page cost is mostly the image + ~100 output tokens.
4. Greedy fuzzy-matches each table row to a scanned page (supplier 40% / amount 25% / date 15% / id_number 20%).
5. Writes:
   - `output_sorted.pdf` — pages reordered to match the table; unmatched pages appended at the end.
   - `match_report.csv` — per-row matched page + score + status (`matched` / `low_confidence` / `unmatched`).
   - `extracted_invoices.csv` — flat table of `page, supplier, amount, date, id_number` per scanned page (also as a `pandas.DataFrame` via `pages_to_dataframe()` in the notebook).

Claude handles rotated phone photos and Hebrew handwriting natively — no orientation-detection step.

## Setup

### 1. Python deps

```
pip install -r requirements.txt
```

### 2. Anthropic API key

Set `ANTHROPIC_API_KEY` in your environment. PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Permanent (user-level):

```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."
```

(open a new shell for `setx` to take effect)

### Cost estimate

At ~200 DPI and Opus 4.7 pricing (Nov 2026: $5 / $25 per 1M tokens), each scanned page costs roughly **$0.015–0.025**. A typical 50-page batch runs about **$1**. To reduce cost, edit `CLAUDE_MODEL` in `reorder_invoices.py` to `"claude-sonnet-4-6"` (~40% cheaper) or `"claude-haiku-4-5"` (~80% cheaper) — vision quality is still good for clean printed receipts.

## Usage — CLI

```
python reorder_invoices.py ^
  --table   "C:\path\to\table.pdf" ^
  --scanned "C:\path\to\scanned_invoices.pdf" ^
  --out-dir "C:\path\to\output"
```

Outputs land in `--out-dir`:

- `output_sorted.pdf`
- `match_report.csv` (UTF-8 with BOM — opens correctly in Excel for Hebrew)
- `extracted_invoices.csv` — `page, supplier, amount, date, id_number` for every scanned page

## Usage — web app

**Step 1.** Open a PowerShell terminal in this folder and start the server:

```
python app.py
```

You should see a banner like:

```
============================================================
  Invoice Reorder web app
  Open in your browser:
    Reorder two PDFs:      http://127.0.0.1:5000/
    Single-invoice test:   http://127.0.0.1:5000/test
  Outputs land under:     C:\Users\...\invoice_reorder\web_outputs
  Stop the server:        Ctrl+C
============================================================
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

**Step 2.** Leave that terminal running. Open one of the URLs in your browser:

- **/** — Upload `table.pdf` + `scanned_invoices.pdf` → sorted PDF + report + extracted CSV as downloadable links.
- **/test** — Upload **one** invoice (PDF or image) → extracted fields as a one-row table, with a CSV saved to `web_outputs/<job>/`.

**Step 3.** When you're done, press `Ctrl+C` in the terminal to stop the server.

> **`This site can't be reached` / `connection refused`** = the server isn't running. Go back to Step 1.

## Usage — notebook

Open `reorder_invoices.ipynb` in Jupyter or VS Code. Drop your PDFs into `samples/` (`samples/table.pdf` and `samples/scanned_invoices.pdf`) and run the cells top to bottom. There's also a bottom cell that runs the single-invoice flow against `samples/single_invoice.pdf`.

## Tuning

All tunables live at the top of `reorder_invoices.py`:

| Constant | Default | Effect |
| --- | --- | --- |
| `CLAUDE_MODEL` | `claude-opus-4-7` | Switch to `claude-sonnet-4-6` or `claude-haiku-4-5` for cheaper extraction. |
| `RENDER_DPI` | `200` | Higher = better OCR on small text (e.g. 9-digit business IDs), but more image tokens. 300 DPI still fits Opus 4.7's 2576px long-edge limit on A4. |
| `SCORE_WEIGHTS` | `{supplier: 0.4, amount: 0.25, date: 0.15, id_number: 0.2}` | Re-weight if one field is consistently unreliable. |
| `MATCH_THRESHOLD` | `60.0` | Lower = more aggressive matching, more false positives. |

`status` is `matched` if score ≥ 75, `low_confidence` if 60–75, `unmatched` if no page scores ≥ 60.

## Limitations (v1)

- Assumes **1 page = 1 invoice**.
- Greedy match (good for ≤ a few hundred invoices). For thousands of pages, switch to the [Anthropic Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) for 50% cost reduction and use `scipy.optimize.linear_sum_assignment` for globally-optimal matching.
- Synchronous, one API call per page. For 100+ page batches, the async client + `asyncio.gather` would parallelize this 5–10×.
