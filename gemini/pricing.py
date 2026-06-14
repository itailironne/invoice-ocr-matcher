"""Gemini model pricing constants and usage accumulator.

Mirrors MODEL_PRICING_USD / _UsageTotals from reorder_invoices.py but for
Google's Gemini API. Prices from ai.google.dev/gemini-api/docs/pricing.
Standard (sub-200K token) tier used; runs that exceed 200K context per call
will be slightly under-counted.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

GEMINI_DEFAULT_MODEL = "gemini-3.1-pro-preview"

# USD per token
GEMINI_PRICING_USD: dict[str, dict[str, Decimal]] = {
    "gemini-3.1-pro-preview":     {"input": Decimal("0.00000125"),  "output": Decimal("0.00001")},
    "gemini-3.5-flash":           {"input": Decimal("0.00000015"),  "output": Decimal("0.0000006")},
    "gemini-2.5-pro":             {"input": Decimal("0.00000125"),  "output": Decimal("0.00001")},
    "gemini-2.5-pro-preview":     {"input": Decimal("0.00000125"),  "output": Decimal("0.00001")},
    "gemini-2.5-flash":           {"input": Decimal("0.00000015"),  "output": Decimal("0.0000006")},
    "gemini-2.5-flash-preview":   {"input": Decimal("0.00000015"),  "output": Decimal("0.0000006")},
    "gemini-2.0-flash":           {"input": Decimal("0.0000001"),   "output": Decimal("0.0000004")},
    "gemini-2.0-flash-lite":      {"input": Decimal("0.000000075"), "output": Decimal("0.0000003")},
    "gemini-1.5-pro":             {"input": Decimal("0.00000125"),  "output": Decimal("0.000005")},
    "gemini-1.5-pro-latest":      {"input": Decimal("0.00000125"),  "output": Decimal("0.000005")},
    "gemini-1.5-flash":           {"input": Decimal("0.000000075"), "output": Decimal("0.0000003")},
    "gemini-1.5-flash-latest":    {"input": Decimal("0.000000075"), "output": Decimal("0.0000003")},
}


@dataclass
class GeminiUsageTotals:
    """Accumulates token usage across all Gemini calls in one run."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    # Shared lock so concurrent page workers can update totals safely.
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def add(self, usage_metadata, model: str) -> None:
        """Add a Gemini GenerateContentResponse.usage_metadata to this accumulator."""
        if usage_metadata is None:
            return
        self.calls += 1
        self.input_tokens  += getattr(usage_metadata, "prompt_token_count",     0) or 0
        self.output_tokens += getattr(usage_metadata, "candidates_token_count", 0) or 0
        if not self.model:
            self.model = model

    def add_http(self, usage_metadata: dict | None, model: str) -> None:
        """Add usage from a REST API usageMetadata dict (camelCase keys)."""
        if not usage_metadata:
            return
        with self._lock:
            self.calls += 1
            self.input_tokens  += usage_metadata.get("promptTokenCount",     0) or 0
            self.output_tokens += usage_metadata.get("candidatesTokenCount", 0) or 0
            if not self.model:
                self.model = model

    def cost_usd(self) -> Decimal:
        prices = GEMINI_PRICING_USD.get(self.model)
        if prices is None:
            return Decimal("0")
        return (
            Decimal(self.input_tokens)  * prices["input"]
            + Decimal(self.output_tokens) * prices["output"]
        )

    def reset(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model = ""
