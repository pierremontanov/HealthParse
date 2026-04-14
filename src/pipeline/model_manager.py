"""Model persistence layer for NER, Classifier, and OCR configurations.

Provides :class:`ModelManager` – a unified save / load interface that bundles
classifier weights, OCR configuration, extractor registry metadata, and
training lineage into a single versioned artefact on disk.

The on-disk format is a JSON file with the following top-level keys:

* ``meta`` – version, timestamp, description, source info
* ``classifier_weights`` – keyword → weight tables per document type
* ``ocr_config`` – OCR settings captured at save time
* ``registered_types`` – list of document types the model supports
* ``training`` – optional training metadata (examples count, eval scores)

Usage
-----
    from src.pipeline.model_manager import ModelManager

    # Save current state
    manager = ModelManager()
    manager.save("models/v2.json", description="After retraining on 200 docs")

    # Load into a ready-to-use engine
    engine = ModelManager.load_engine("models/v2.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

__all__ = ["ModelMeta", "ModelManager"]

_FORMAT_VERSION = "1.0"


# ── Metadata container ──────────────────────────────────────────

@dataclass
class ModelMeta:
    """Read-only metadata for a saved model artefact."""

    version: str = ""
    format_version: str = _FORMAT_VERSION
    created_utc: str = ""
    description: str = ""
    registered_types: List[str] = field(default_factory=list)
    training_examples_count: int = 0
    eval_scores: Dict[str, float] = field(default_factory=dict)
    checksum: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "format_version": self.format_version,
            "created_utc": self.created_utc,
            "description": self.description,
            "registered_types": self.registered_types,
            "training_examples_count": self.training_examples_count,
            "eval_scores": self.eval_scores,
            "checksum": self.checksum,
        }


# ── Manager ─────────────────────────────────────────────────────

class ModelManager:
    """Unified model persistence for DocIQ.

    Captures the full inference state – classifier keyword weights, OCR
    configuration, and model registry metadata – so the pipeline can be
    restored to an identical state from a single JSON file.
    """

    def __init__(self) -> None:
        self._classifier_weights: Dict[str, Dict[str, float]] = {}
        self._ocr_config: Dict[str, Any] = {}
        self._registered_types: List[str] = []
        self._meta: Optional[ModelMeta] = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def meta(self) -> Optional[ModelMeta]:
        """Metadata of the last loaded artefact, or ``None``."""
        return self._meta

    @property
    def classifier_weights(self) -> Dict[str, Dict[str, float]]:
        return dict(self._classifier_weights)

    @property
    def ocr_config(self) -> Dict[str, Any]:
        return dict(self._ocr_config)

    @property
    def registered_types(self) -> List[str]:
        return list(self._registered_types)

    # ── Save ────────────────────────────────────────────────────

    def save(
        self,
        path: str | Path,
        *,
        version: str = "",
        description: str = "",
        training_examples_count: int = 0,
        eval_scores: Optional[Dict[str, float]] = None,
    ) -> ModelMeta:
        """Persist the current model state to *path*.

        Parameters
        ----------
        path : str or Path
            Destination JSON file.
        version : str, optional
            Arbitrary version tag (e.g. ``"v2.1"``).  Defaults to a
            timestamp-based string when empty.
        description : str, optional
            Human-readable description of this snapshot.
        training_examples_count : int, optional
            Number of training examples used (informational).
        eval_scores : dict, optional
            Per-type evaluation F1 scores.

        Returns
        -------
        ModelMeta
            Metadata of the artefact that was written.
        """
        self._capture_state()

        now = datetime.now(timezone.utc)
        if not version:
            version = now.strftime("v%Y%m%d_%H%M%S")

        payload: Dict[str, Any] = {
            "meta": {
                "version": version,
                "format_version": _FORMAT_VERSION,
                "created_utc": now.isoformat(),
                "description": description,
                "registered_types": self._registered_types,
                "training_examples_count": training_examples_count,
                "eval_scores": eval_scores or {},
            },
            "classifier_weights": self._classifier_weights,
            "ocr_config": self._ocr_config,
        }

        raw = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        checksum = hashlib.sha256(raw.encode()).hexdigest()[:16]
        payload["meta"]["checksum"] = checksum

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        meta = ModelMeta(
            version=version,
            format_version=_FORMAT_VERSION,
            created_utc=now.isoformat(),
            description=description,
            registered_types=list(self._registered_types),
            training_examples_count=training_examples_count,
            eval_scores=eval_scores or {},
            checksum=checksum,
        )
        self._meta = meta
        logger.info("Model saved to %s (version=%s, checksum=%s)", p, version, checksum)
        return meta

    # ── Load ────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> "ModelManager":
        """Load model state from a previously saved JSON artefact.

        Returns a :class:`ModelManager` with weights and OCR config
        populated.  Use :meth:`apply` to patch the live pipeline, or
        :meth:`load_engine` for a one-step load-and-build.

        Raises
        ------
        ModelLoadError
            If the file does not exist, cannot be parsed, or is missing
            required keys.
        """
        p = Path(path)
        if not p.exists():
            raise ModelLoadError(
                "ModelManager", f"Model file not found: {p}"
            )

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelLoadError(
                "ModelManager", f"Failed to parse model file {p}: {exc}"
            ) from exc

        if "classifier_weights" not in data:
            raise ModelLoadError(
                "ModelManager",
                f"Invalid model file {p}: missing 'classifier_weights' key",
            )

        mgr = cls()
        mgr._classifier_weights = data.get("classifier_weights", {})
        mgr._ocr_config = data.get("ocr_config", {})

        raw_meta = data.get("meta", {})
        mgr._registered_types = raw_meta.get("registered_types", list(mgr._classifier_weights.keys()))
        mgr._meta = ModelMeta(
            version=raw_meta.get("version", ""),
            format_version=raw_meta.get("format_version", "unknown"),
            created_utc=raw_meta.get("created_utc", ""),
            description=raw_meta.get("description", ""),
            registered_types=list(mgr._registered_types),
            training_examples_count=raw_meta.get("training_examples_count", 0),
            eval_scores=raw_meta.get("eval_scores", {}),
            checksum=raw_meta.get("checksum", ""),
        )

        logger.info(
            "Model loaded from %s (version=%s, types=%s)",
            p,
            mgr._meta.version,
            mgr._registered_types,
        )
        return mgr

    # ── Apply to live pipeline ─────────────────────────────────

    def apply(self) -> None:
        """Patch loaded classifier weights into the live pipeline.

        Updates the global ``DOCUMENT_TYPES`` keyword tables used by
        :class:`DocumentClassifier`, and optionally overrides OCR
        settings in :attr:`src.config.settings`.
        """
        if self._classifier_weights:
            from src.pipeline.extractors.document_classifier import DOCUMENT_TYPES

            for doc_type, kw_table in self._classifier_weights.items():
                if doc_type in DOCUMENT_TYPES:
                    DOCUMENT_TYPES[doc_type].update(kw_table)
                else:
                    DOCUMENT_TYPES[doc_type] = dict(kw_table)

            logger.info(
                "Classifier weights applied for %d type(s).",
                len(self._classifier_weights),
            )

        if self._ocr_config:
            try:
                from src.config import settings

                for key, value in self._ocr_config.items():
                    if hasattr(settings, key):
                        object.__setattr__(settings, key, value)
                logger.info("OCR config applied: %s", list(self._ocr_config.keys()))
            except Exception as exc:
                logger.warning("Could not apply OCR config: %s", exc)

    # ── One-step load → engine ─────────────────────────────────

    @classmethod
    def load_engine(cls, path: str | Path) -> "InferenceEngine":
        """Load a model and return a ready-to-use :class:`InferenceEngine`.

        This is the recommended production entry-point.  It:

        1. Loads the JSON artefact from *path*
        2. Patches the classifier keyword tables
        3. Optionally applies OCR settings
        4. Builds a default engine with rule-based extractors

        Returns
        -------
        InferenceEngine
            Engine pre-loaded with trained classifier weights.
        """
        mgr = cls.load(path)
        mgr.apply()

        from src.pipeline.inference import create_default_engine

        engine = create_default_engine()
        logger.info(
            "Inference engine created from model %s (version=%s)",
            path,
            mgr._meta.version if mgr._meta else "unknown",
        )
        return engine

    # ── Inspect ────────────────────────────────────────────────

    @classmethod
    def inspect(cls, path: str | Path) -> ModelMeta:
        """Read metadata from a model file without loading weights.

        Useful for listing available model versions.
        """
        mgr = cls.load(path)
        assert mgr._meta is not None
        return mgr._meta

    # ── Internal ───────────────────────────────────────────────

    def _capture_state(self) -> None:
        """Snapshot the current live pipeline state into this manager."""
        # Classifier weights
        try:
            from src.pipeline.extractors.document_classifier import DOCUMENT_TYPES

            self._classifier_weights = {
                dt: dict(kw) for dt, kw in DOCUMENT_TYPES.items()
            }
            self._registered_types = list(DOCUMENT_TYPES.keys())
        except ImportError:
            logger.warning("Could not import DOCUMENT_TYPES; classifier weights empty.")

        # OCR config
        try:
            from src.config import settings

            self._ocr_config = {
                "ocr_dpi": settings.ocr_dpi,
                "ocr_lang": settings.ocr_lang,
                "preprocessing_threshold": settings.preprocessing_threshold,
                "tesseract_cmd": settings.tesseract_cmd,
            }
        except Exception:
            logger.warning("Could not capture OCR config.")
