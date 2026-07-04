"""LLM client for Ollama integration.

Provides a synchronous HTTP client that wraps Ollama's ``/api/generate``
endpoint.  The client handles connection management, timeouts, retries,
and JSON parsing.  It exposes two high-level methods:

- :meth:`OllamaClient.classify` — determine document type
- :meth:`OllamaClient.extract` — extract structured fields

If the LLM is disabled, unreachable, or returns invalid output, errors
are raised as :class:`~src.pipeline.exceptions.LLMError` subclasses so
the caller can fall back gracefully to rule-based results.

Usage
-----
    from src.pipeline.llm_client import OllamaClient

    client = OllamaClient()           # reads settings from config
    result = client.classify(raw_text) # {"document_type": ..., "confidence": ..., "language": ...}
    data = client.extract("prescription", raw_text)  # structured dict
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from src.config import settings
from src.pipeline.exceptions import (
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

__all__ = ["OllamaClient", "LLMClassifyResult"]


# ── Result types ─────────────────────────────────────────────────

@dataclass
class LLMClassifyResult:
    """Result from LLM document classification."""

    document_type: str
    confidence: float
    language: str


# ── Client ───────────────────────────────────────────────────────

class OllamaClient:
    """Synchronous client for the Ollama ``/api/generate`` endpoint.

    Parameters
    ----------
    endpoint : str, optional
        Base URL of the Ollama server.  Defaults to ``settings.llm_endpoint``.
    model : str, optional
        Model name.  Defaults to ``settings.llm_model``.
    timeout : int, optional
        Per-request timeout in seconds.  Defaults to ``settings.llm_timeout``.
    retries : int, optional
        Retry count on failure.  Defaults to ``settings.llm_retries``.
    keep_alive : str, optional
        Ollama keep-alive duration.  Defaults to ``settings.llm_keep_alive``.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        keep_alive: Optional[str] = None,
    ) -> None:
        self.endpoint = (endpoint or settings.llm_endpoint).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout
        self.retries = retries if retries is not None else settings.llm_retries
        self.keep_alive = keep_alive or settings.llm_keep_alive
        self._generate_url = f"{self.endpoint}/api/generate"

    # ── Low-level request ────────────────────────────────────────

    def _request(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the raw response text.

        Retries on transient failures up to ``self.retries`` times.

        Raises
        ------
        LLMConnectionError
            If the endpoint is unreachable after all retries.
        LLMTimeoutError
            If the request exceeds the configured timeout.
        LLMError
            On unexpected HTTP or transport errors.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1 + self.retries):
            if attempt > 0:
                wait = min(2 ** attempt, 10)
                logger.info(
                    "LLM retry %d/%d in %ds", attempt, self.retries, wait
                )
                time.sleep(wait)

            try:
                t0 = time.monotonic()
                resp = requests.post(
                    self._generate_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": self.keep_alive,
                    },
                    timeout=self.timeout,
                )
                elapsed = time.monotonic() - t0
                resp.raise_for_status()

                response_text = resp.json().get("response", "")
                logger.debug(
                    "LLM response in %.1fs (%d chars)",
                    elapsed,
                    len(response_text),
                )
                return response_text

            except requests.ConnectionError as exc:
                last_error = exc
                logger.warning(
                    "LLM connection failed (attempt %d/%d): %s",
                    attempt + 1,
                    1 + self.retries,
                    exc,
                )
            except requests.Timeout as exc:
                last_error = exc
                logger.warning("LLM request timed out after %ds", self.timeout)
            except requests.HTTPError as exc:
                last_error = exc
                logger.warning("LLM HTTP error: %s", exc)
            except Exception as exc:
                last_error = exc
                logger.warning("LLM unexpected error: %s", exc)

        # All retries exhausted
        if isinstance(last_error, requests.ConnectionError):
            raise LLMConnectionError(self.endpoint, str(last_error))
        if isinstance(last_error, requests.Timeout):
            raise LLMTimeoutError(self.timeout)
        raise LLMError(str(last_error))

    # ── JSON parsing ─────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences if present.

        Raises
        ------
        LLMResponseError
            If the response cannot be parsed as valid JSON.
        """
        cleaned = text.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"invalid JSON: {exc}. Raw: {text[:200]}"
            ) from exc

    # ── High-level API ───────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable.

        Returns
        -------
        bool
            True if the server responds, False otherwise.
        """
        try:
            resp = requests.get(f"{self.endpoint}/", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def classify(self, text: str) -> LLMClassifyResult:
        """Classify a document using the LLM.

        Parameters
        ----------
        text : str
            Raw OCR / extracted text from the document.

        Returns
        -------
        LLMClassifyResult
            Classification result with document_type, confidence, and language.

        Raises
        ------
        LLMConnectionError, LLMTimeoutError, LLMResponseError
        """
        # Truncate to configured context window size
        max_chars = settings.llm_context_chars
        truncated = text[:max_chars] if len(text) > max_chars else text
        prompt = f"classify: {truncated}"

        response_text = self._request(prompt)
        parsed = self._parse_json(response_text)

        doc_type = parsed.get("document_type", "unknown")
        confidence = float(parsed.get("confidence", 0.0))
        language = parsed.get("language", "unknown")

        logger.info(
            "LLM classify: type=%s confidence=%.2f lang=%s",
            doc_type,
            confidence,
            language,
        )
        return LLMClassifyResult(
            document_type=doc_type,
            confidence=confidence,
            language=language,
        )

    def extract(self, document_type: str, text: str) -> Dict[str, Any]:
        """Extract structured fields from a document using the LLM.

        Parameters
        ----------
        document_type : str
            Document type (e.g. "prescription", "result", "clinical_history").
        text : str
            Raw OCR / extracted text from the document.

        Returns
        -------
        dict
            Structured extraction matching the Pydantic schema for the type.

        Raises
        ------
        LLMConnectionError, LLMTimeoutError, LLMResponseError
        """
        max_chars = settings.llm_context_chars
        truncated = text[:max_chars] if len(text) > max_chars else text
        prompt = f"extract {document_type}: {truncated}"

        response_text = self._request(prompt)
        parsed = self._parse_json(response_text)

        field_count = len([v for v in parsed.values() if v is not None])
        logger.info(
            "LLM extract (%s): %d non-null fields", document_type, field_count
        )
        return parsed
