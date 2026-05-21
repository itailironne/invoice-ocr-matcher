"""Flask web app for the invoice reorder pipeline.

Routes:
  GET  /         — main page: upload table.pdf + scanned_invoices.pdf
  POST /reorder  — runs the full pipeline, returns a results page with download links
  GET  /test     — single-invoice extraction test page
  POST /test     — extracts fields from one uploaded file, returns DataFrame + CSV
  GET  /download/<job>/<filename> — serve generated outputs

Requires ANTHROPIC_API_KEY in the environment.

Run with:
    python app.py
"""
from __future__ import annotations

import csv
import logging
import os
import re
import secrets
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import (
    Flask, abort, redirect, render_template, request, send_from_directory, url_for, jsonify,
)

import reorder_invoices as ri

APP_ROOT = Path(__file__).resolve().parent
JOBS_DIR = APP_ROOT / "web_outputs"
JOBS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 50
ALLOWED_REORDER = {".pdf"}
ALLOWED_SINGLE = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("invoice_reorder.app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def _check_api_key() -> Optional[str]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is not set in the server's environment. Set it and restart the app."
    return None


def _new_job_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job = f"{stamp}-{secrets.token_hex(3)}"
    d = JOBS_DIR / job
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ext_ok(filename: str, allowed: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed


@app.route("/")
def index():
    return render_template("index.html", api_key_missing=_check_api_key() is not None)


@app.route("/reorder", methods=["POST"])
def reorder():
    err = _check_api_key()
    if err:
        return render_template("error.html", message=err), 500

    table_file = request.files.get("table_pdf")
    scanned_file = request.files.get("scanned_pdf")
    if not table_file or not scanned_file or not table_file.filename or not scanned_file.filename:
        return render_template("error.html", message="Please upload both PDF files."), 400
    if not _ext_ok(table_file.filename, ALLOWED_REORDER) or not _ext_ok(scanned_file.filename, ALLOWED_REORDER):
        return render_template("error.html", message="Both files must be PDFs."), 400

    max_pages_raw = (request.form.get("max_pages") or "").strip()
    max_pages: Optional[int] = None
    if max_pages_raw:
        try:
            max_pages = max(1, int(max_pages_raw))
        except ValueError:
            return render_template("error.html", message=f"max_pages must be a number, got: {max_pages_raw!r}"), 400

    # VAT rate override (e.g. user enters "18" or "0.18" → both work).
    vat_raw = (request.form.get("vat_rate") or "").strip()
    from decimal import Decimal as _Decimal
    vat_rate = ri.DEFAULT_VAT_RATE
    if vat_raw:
        try:
            v = float(vat_raw)
            if v > 1: v = v / 100  # accept "18" as 18% → 0.18
            vat_rate = _Decimal(str(v))
        except ValueError:
            return render_template("error.html", message=f"VAT rate must be a number, got: {vat_raw!r}"), 400

    job_dir = _new_job_dir()
    table_path = job_dir / "table.pdf"
    scanned_path = job_dir / "scanned.pdf"
    table_file.save(str(table_path))
    scanned_file.save(str(scanned_path))
    log.info("job %s: saved uploads, starting pipeline (max_pages=%s, vat=%s)", job_dir.name, max_pages, vat_rate)

    try:
        summary = ri.run(table_path, scanned_path, job_dir, max_pages=max_pages, vat_rate=vat_rate)
    except Exception as e:
        log.exception("job %s failed", job_dir.name)
        return render_template("error.html", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()), 500

    return render_template(
        "results.html",
        job=job_dir.name,
        summary=summary,
        out_pdf="output_sorted.pdf",
        out_csv="match_report.csv",
        extracted_csv="all_invoices_extracted.csv",
        table_csv="table_extracted.csv",
    )


@app.route("/test", methods=["GET", "POST"])
def test_page():
    if request.method == "GET":
        return render_template("test.html", api_key_missing=_check_api_key() is not None)

    err = _check_api_key()
    if err:
        return render_template("error.html", message=err), 500

    upload = request.files.get("invoice")
    if not upload or not upload.filename:
        return render_template("error.html", message="Please choose a file to upload."), 400
    if not _ext_ok(upload.filename, ALLOWED_SINGLE):
        return render_template("error.html", message=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_SINGLE))}"), 400

    job_dir = _new_job_dir()
    suffix = Path(upload.filename).suffix.lower()
    input_path = job_dir / f"input{suffix}"
    upload.save(str(input_path))
    log.info("job %s: single-invoice test on %s", job_dir.name, upload.filename)

    try:
        fields = ri.extract_single_file(input_path)
    except Exception as e:
        log.exception("job %s failed", job_dir.name)
        return render_template("error.html", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()), 500

    csv_path = job_dir / "single_invoice.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["supplier", "amount", "date", "id_number"])
        writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in fields.items()})

    return render_template(
        "test.html",
        api_key_missing=False,
        result_fields=fields,
        original_name=upload.filename,
        job=job_dir.name,
        csv_filename="single_invoice.csv",
    )


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", api_key_missing=_check_api_key() is not None)

    err = _check_api_key()
    if err:
        return render_template("error.html", message=err), 500

    uploads = request.files.getlist("invoices")
    uploads = [u for u in uploads if u and u.filename]
    if not uploads:
        return render_template("error.html", message="Please choose at least one file."), 400
    bad = [u.filename for u in uploads if not _ext_ok(u.filename, ALLOWED_SINGLE)]
    if bad:
        return render_template("error.html", message=f"Unsupported file type(s): {', '.join(bad)}"), 400

    sort_by = request.form.get("sort_by", "date")
    if sort_by not in {"date", "supplier", "amount", "id_number"}:
        sort_by = "date"

    job_dir = _new_job_dir()
    saved_paths = []
    for i, upload in enumerate(uploads):
        suffix = Path(upload.filename).suffix.lower()
        # Preserve the original filename stem for the table view, but namespace by index.
        safe_stem = "".join(c for c in Path(upload.filename).stem if c.isalnum() or c in "._- ")[:80] or "file"
        save_path = job_dir / f"{i:03d}_{safe_stem}{suffix}"
        upload.save(str(save_path))
        saved_paths.append(save_path)

    log.info("job %s: batch of %d files, sort_by=%s", job_dir.name, len(saved_paths), sort_by)

    try:
        result = ri.batch_extract_and_combine(saved_paths, job_dir, sort_by=sort_by)
    except Exception as e:
        log.exception("job %s failed", job_dir.name)
        return render_template("error.html", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()), 500

    return render_template(
        "batch_results.html",
        job=job_dir.name,
        rows=result["rows"],
        sort_by=result["sort_by"],
        count=result["count"],
        combined_pdf="combined_sorted.pdf",
        csv_filename="extracted_invoices.csv",
    )


@app.route("/test.json", methods=["POST"])
def test_json():
    """Same as /test but returns JSON — handy for scripting / curl."""
    err = _check_api_key()
    if err:
        return jsonify({"error": err}), 500
    upload = request.files.get("invoice")
    if not upload or not upload.filename:
        return jsonify({"error": "no file"}), 400
    if not _ext_ok(upload.filename, ALLOWED_SINGLE):
        return jsonify({"error": "unsupported file type"}), 400

    job_dir = _new_job_dir()
    suffix = Path(upload.filename).suffix.lower()
    input_path = job_dir / f"input{suffix}"
    upload.save(str(input_path))
    try:
        fields = ri.extract_single_file(input_path)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify(fields)


_SAFE_JOB_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _resolve_job_dir(job: str) -> Optional[Path]:
    """Find a job directory by name. Looks under web_outputs/ and the app root.
    Restricts to safe names (alphanumeric + _ + -) and only returns dirs that
    are physically contained within the project.
    """
    if not _SAFE_JOB_RE.match(job):
        return None
    candidates = [JOBS_DIR / job, APP_ROOT / job]
    app_root_real = APP_ROOT.resolve()
    for c in candidates:
        if c.is_dir():
            resolved = c.resolve()
            if str(resolved).startswith(str(app_root_real)):
                return resolved
    return None


def _pick_newest(job_dir: Path, candidates: list[str]) -> Optional[str]:
    """Pick the most recently modified existing file from a list of candidates."""
    existing = [(c, (job_dir / c).stat().st_mtime) for c in candidates if (job_dir / c).exists()]
    if not existing:
        return None
    existing.sort(key=lambda x: x[1], reverse=True)
    return existing[0][0]


@app.route("/results/<job>")
def view_results(job: str):
    """Render the results page for any output dir. Prefers the most recently
    modified variant of each output (e.g. match_report_v2.csv if it's newer
    than match_report.csv)."""
    job_dir = _resolve_job_dir(job)
    if not job_dir:
        return render_template("error.html", message=f"לא נמצאה תיקיית תוצאות בשם '{job}'."), 404

    report_name = _pick_newest(job_dir, ["match_report_v2.csv", "match_report.csv"])
    if not report_name:
        return render_template("error.html", message=f"לא נמצא match_report ב־'{job}'."), 404

    with open(job_dir / report_name, encoding="utf-8-sig", newline="") as f:
        report_rows = list(csv.DictReader(f))

    # Counts work for both the old schema (status per table row) and the new
    # schema (status per output page).
    matched = sum(1 for r in report_rows if r.get("status") == "matched")
    low_confidence = sum(1 for r in report_rows if r.get("status") == "low_confidence")
    unmatched = sum(1 for r in report_rows if r.get("status") in ("unmatched", "leftover"))

    extracted_csv_name = _pick_newest(job_dir, ["all_invoices_extracted.csv", "extracted_invoices.csv"])
    pages_count = 0
    if extracted_csv_name:
        with open(job_dir / extracted_csv_name, encoding="utf-8-sig", newline="") as f:
            pages_count = sum(1 for _ in csv.DictReader(f))

    table_csv_name = _pick_newest(job_dir, ["table_extracted.csv"])
    pdf_name = _pick_newest(job_dir, ["output_sorted_v2.pdf", "output_sorted.pdf"])

    summary = {
        "rows": len(report_rows),
        "pages": pages_count,
        "matched": matched,
        "low_confidence": low_confidence,
        "unmatched": unmatched,
    }
    return render_template(
        "results.html",
        job=job,
        summary=summary,
        out_pdf=pdf_name,
        out_csv=report_name,
        extracted_csv=extracted_csv_name,
        table_csv=table_csv_name,
        report_rows=report_rows[:50],
    )


@app.route("/runs")
def list_runs():
    """List all available job directories so the user can pick one to view."""
    runs = []
    for d in JOBS_DIR.iterdir() if JOBS_DIR.exists() else []:
        if d.is_dir() and (d / "match_report.csv").exists():
            runs.append((d.name, "web", d.stat().st_mtime))
    for d in APP_ROOT.iterdir():
        if d.is_dir() and d.name not in {"web_outputs", "templates", "static", "__pycache__", "samples", ".git"} \
                and (d / "match_report.csv").exists():
            runs.append((d.name, "cli", d.stat().st_mtime))
    runs.sort(key=lambda x: x[2], reverse=True)
    return render_template("runs.html", runs=runs)


@app.route("/download/<job>/<path:filename>")
def download(job: str, filename: str):
    """Serve a file from any job dir, with path-traversal protection."""
    job_dir = _resolve_job_dir(job)
    if not job_dir:
        abort(404)
    full = (job_dir / filename).resolve()
    if not str(full).startswith(str(job_dir)) or not full.is_file():
        abort(404)
    return send_from_directory(str(job_dir), filename, as_attachment=True)


@app.errorhandler(413)
def too_large(e):
    return render_template("error.html", message=f"Upload too large (limit {MAX_UPLOAD_MB} MB)."), 413


def main():
    """Local development entry point. In production (Render / any host), use
    a WSGI server like gunicorn that imports `app:app` directly — see
    render.yaml for the production command."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set. The app will start but every extraction will fail until you set it.", file=sys.stderr)
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    banner = (
        "\n"
        "============================================================\n"
        "  Invoice Reorder web app\n"
        f"  Open in your browser:    http://{host}:{port}/\n"
        f"  Outputs land under:      {JOBS_DIR}\n"
        "  Stop the server:         Ctrl+C\n"
        "============================================================\n"
    )
    print(banner)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
