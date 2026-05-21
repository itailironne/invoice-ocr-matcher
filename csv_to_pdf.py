"""Build a PDF whose page order matches the row order of a CSV.

Use case: you have `all_invoices_extracted.csv` and you've reordered its rows
(sorted by supplier, by date, manually dragged rows around in Excel). Run
this to regenerate `all_invoices.pdf` with its pages in that exact CSV order.

Default paths point at the existing full_run_output for convenience:
    csv:        full_run_output/all_invoices_extracted.csv
    source PDF: all_invoices.pdf
    output:     full_run_output/all_invoices_reordered.pdf

Override with command-line args:
    python csv_to_pdf.py --csv path.csv --source source.pdf --out out.pdf
"""
import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import reorder_invoices as ri

DEFAULT_CSV = "full_run_output/all_invoices_extracted.csv"
DEFAULT_SOURCE_PDF = "all_invoices.pdf"
DEFAULT_OUT_PDF = "full_run_output/all_invoices_reordered.pdf"

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"CSV with page numbers (default: {DEFAULT_CSV})")
    parser.add_argument("--source", default=DEFAULT_SOURCE_PDF, help=f"source PDF (default: {DEFAULT_SOURCE_PDF})")
    parser.add_argument("--out", default=DEFAULT_OUT_PDF, help=f"output PDF (default: {DEFAULT_OUT_PDF})")
    parser.add_argument("--page-column", default="page", help='CSV column name with page indices (default: "page")')
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    source_pdf = Path(args.source).resolve()
    out_pdf = Path(args.out).resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1
    if not source_pdf.exists():
        print(f"ERROR: source PDF not found: {source_pdf}", file=sys.stderr)
        return 1

    print(f"CSV:        {csv_path}")
    print(f"Source PDF: {source_pdf}")
    print(f"Output PDF: {out_pdf}")
    print()

    n = ri.write_pdf_from_csv_order(csv_path, source_pdf, out_pdf, page_column=args.page_column)
    print(f"\n✓ Wrote {n} pages to: {out_pdf}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
