# Invoice Reconciliation App — Setup Guide for a Friend

> Paste this entire file into an LLM (Claude, ChatGPT, etc.) and say:
> **"Help me set this up on my computer, step by step."**
> The LLM will have all the context it needs to guide you.

---

## What this app does

This is a web app that matches scanned invoices (a multi-page PDF) against a table of expected invoices (another PDF), using Claude AI vision. It reads each scanned page, extracts supplier name, amount, date, and VAT number, then matches them to the table and produces a sorted PDF + a CSV report.

The app is built with Python + Flask. You run it locally on your computer — no cloud server needed.

---

## What you need before starting

- **Windows 10 or 11** (the guide below is for Windows)
- **Python 3.10 or newer** — download from https://www.python.org/downloads/ (check "Add Python to PATH" during install)
- **Git** — download from https://git-scm.com/download/win
- **An Anthropic API key** — the person who shared this guide with you will give you one, or you can create one at https://console.anthropic.com

---

## Step 1 — Clone the repository

Open PowerShell (search "PowerShell" in the Start menu) and run:

```
git clone https://github.com/itailironne/invoice-reorder.git
cd invoice-reorder
```

---

## Step 2 — Install Python dependencies

Still in PowerShell, inside the `invoice-reorder` folder:

```
pip install -r requirements.txt
```

This downloads all the libraries the app needs. It may take a minute.

---

## Step 3 — Set your API key

Every time you open a new PowerShell window to run the app, paste this line (replace the placeholder with your real key):

```
$env:ANTHROPIC_API_KEY = "sk-ant-XXXXXXXXXXXXXXXX"
```

> The key is secret — don't share it or commit it to GitHub.

---

## Step 4 — Run the app

```
python app.py
```

You should see output like:
```
============================================================
  Invoice Reorder web app
  Open in your browser:    http://127.0.0.1:5000/
  Outputs land under:      ...\web_outputs
  Stop the server:         Ctrl+C
============================================================
```

Open your browser and go to **http://localhost:5000**

---

## Step 5 — Using the app

The main page ("מיון לפי טבלה") has two upload zones:

1. **קובץ הטבלה** — the invoice table PDF (list of expected invoices: supplier, amount, date, VAT number)
2. **קובץ החשבוניות הסרוקות** — the scanned invoices PDF (one invoice per page, in any order)

You can drag files onto the zones, or click to browse.

Then click **מיין וחלץ** (Sort & Extract).

A progress page will appear showing a progress bar and elapsed time. Do not close the window.

**How long does it take?**
About 5–10 seconds per page. A 30-page scanned PDF takes roughly 3–5 minutes.
A 100-page PDF takes roughly 8–17 minutes — be patient.

When it finishes, you are automatically redirected to the results page with:
- A downloadable sorted PDF
- A CSV match report
- A summary showing how many invoices matched, had low confidence, or were unmatched

---

## Outputs

All results are saved in the `web_outputs/` folder inside the project directory.
Each run gets its own timestamped subfolder (e.g. `web_outputs/20250523-143201-abc123/`).

---

## Stopping the server

Press **Ctrl+C** in the PowerShell window.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `python` not recognized | Reinstall Python and check "Add to PATH" |
| `pip` not recognized | Run `python -m pip install -r requirements.txt` instead |
| App starts but shows API key error | You forgot Step 3 — set `$env:ANTHROPIC_API_KEY` in the same PowerShell window |
| Progress page shows "שגיאה בעיבוד" (error) | Check the PowerShell window for the error message and share it with the LLM |
| Results look wrong / many unmatched | The table PDF and scanned PDF may be from different periods, or the VAT rate field needs adjusting |

---

## Project info (for the LLM)

- **Language:** Python 3.10+
- **Framework:** Flask
- **AI:** Anthropic Claude (vision API) — model claude-opus-4-5 or similar
- **Main files:**
  - `app.py` — Flask web server, routes, background job threading
  - `reorder_invoices.py` — core pipeline: PDF splitting, Claude extraction, matching logic
  - `templates/` — HTML templates (Hebrew RTL, monday.com style)
  - `requirements.txt` — Python dependencies
- **GitHub repo:** https://github.com/itailironne/invoice-reorder
- **Live demo (Render, free tier):** https://invoice-reorder.onrender.com — suitable for small runs (≤40 pages); larger runs should be done locally
