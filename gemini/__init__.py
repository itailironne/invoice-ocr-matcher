"""Gemini-based alternative OCR pipeline for invoice_reorder.

Drop-in for reorder_invoices.run():

    from gemini import gemini_run
    gemini_run(table_pdf, scanned_pdf, out_dir, max_pages=N, vat_rate=Decimal("0.18"))

Requires GOOGLE_API_KEY in the environment.
"""
from gemini.pipeline import gemini_run
from gemini.pricing import GEMINI_DEFAULT_MODEL, GEMINI_PRICING_USD, GeminiUsageTotals

__all__ = ["gemini_run", "GEMINI_DEFAULT_MODEL", "GEMINI_PRICING_USD", "GeminiUsageTotals"]
