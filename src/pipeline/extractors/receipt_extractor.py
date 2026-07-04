"""Rule-based extractor for pharmacy/clinic receipt documents.

Handles both Spanish (boleta, factura, ticket de venta) and English
(receipt, invoice) receipt formats common in Latin American pharmacies
and clinics.

The extracted dict is intentionally simple — receipts are financial
documents, not clinical ones.  The focus is on the pharmacy/store name,
transaction date, the list of items purchased (drug name + quantity +
unit price), and the total amount.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.pipeline.extractors.base import (
    extract_field,
    extract_date,
    extract_institution,
    resolve_field_flexible,
    resolve_date_flexible,
)


class ReceiptExtractor:
    """Extract structured fields from a pharmacy or clinic receipt.

    The extractor exposes an ``extract(text)`` method so it can be registered
    as the NER model inside a :class:`~src.pipeline.inference.ModelBundle`.
    """

    # ── Field alias lists ────────────────────────────────────────

    _PHARMACY_ALIASES = [
        # English
        "Pharmacy", "Drugstore", "Store", "Clinic", "Hospital", "Lab",
        # Spanish
        "Farmacia", "Drogueria", "Droguería", "Botica", "Clinica", "Clínica",
        "Laboratorio", "Establecimiento", "Razon Social", "Razón Social",
    ]

    _DATE_ALIASES = [
        # English
        "Date", "Transaction Date", "Issue Date", "Invoice Date", "Receipt Date",
        # Spanish
        "Fecha", "Fecha de Emision", "Fecha de Emisión", "Fecha de Venta",
        "Fecha de Factura", "Fecha Factura",
    ]

    _TOTAL_ALIASES = [
        # English
        "Total", "Grand Total", "Amount Due", "Total Amount",
        # Spanish
        "Total", "Total a Pagar", "Monto Total", "Importe Total",
        "Total Factura", "Total Boleta", "Total a pagar",
    ]

    _SUBTOTAL_ALIASES = [
        "Subtotal", "Sub-total",
        "Subtotal sin IVA", "Neto",
    ]

    _TAX_ALIASES = [
        "IVA", "Tax", "VAT", "GST",
        "IVA 19%", "Impuesto", "Taxes",
    ]

    _INVOICE_ALIASES = [
        # English
        "Invoice", "Invoice Number", "Receipt Number", "Transaction ID",
        # Spanish
        "Boleta", "Factura", "Ticket", "Folio", "Numero de Boleta",
        "Número de Boleta", "Numero Factura", "No. Factura",
    ]

    # ── Item line patterns ────────────────────────────────────────

    # Matches lines like:
    #   "Paracetamol 500mg x2         $3,500"
    #   "Amoxicillin 250mg  1  $12.00"
    #   "IBUPROFENO 400MG   3   6000"
    _ITEM_LINE_RE = re.compile(
        r"^(?P<name>[A-Za-záéíóúñÁÉÍÓÚÑ][A-Za-z0-9áéíóúñÁÉÍÓÚÑ .%/\-]+?)"
        r"\s{2,}"
        r"(?:x?\s*(?P<qty>\d+)\s+)?"
        r"(?:\$\s*)?(?P<price>[\d.,]+)\s*$",
        re.MULTILINE,
    )

    def extract(self, text: str) -> Dict[str, Any]:
        """Return a dictionary with receipt fields."""
        pharmacy = self._extract_pharmacy(text)
        date = resolve_date_flexible(text, self._DATE_ALIASES)
        invoice_number = resolve_field_flexible(text, self._INVOICE_ALIASES)
        subtotal = resolve_field_flexible(text, self._SUBTOTAL_ALIASES)
        tax = resolve_field_flexible(text, self._TAX_ALIASES)
        total = resolve_field_flexible(text, self._TOTAL_ALIASES)
        items = self._extract_items(text)

        return {
            "document_type": "receipt",
            "pharmacy": pharmacy,
            "date": date,
            "invoice_number": invoice_number,
            "items": items,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "raw_text": text,
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _extract_pharmacy(self, text: str) -> Optional[str]:
        """Try pharmacy-specific aliases first, then fall back to institution."""
        result = resolve_field_flexible(text, self._PHARMACY_ALIASES)
        if result:
            return result
        return extract_institution(text)

    def _extract_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract line items from the receipt body."""
        items: List[Dict[str, Any]] = []
        for m in self._ITEM_LINE_RE.finditer(text):
            name = m.group("name").strip()
            qty_raw = m.group("qty")
            price_raw = m.group("price")

            # Skip obvious header/footer lines
            lower = name.lower()
            if any(kw in lower for kw in ("total", "subtotal", "iva", "tax", "descuento", "discount")):
                continue

            item: Dict[str, Any] = {"name": name}
            if qty_raw:
                try:
                    item["quantity"] = int(qty_raw)
                except ValueError:
                    item["quantity"] = qty_raw
            if price_raw:
                item["unit_price"] = price_raw.replace(",", "").replace(".", "")

            items.append(item)

        return items
