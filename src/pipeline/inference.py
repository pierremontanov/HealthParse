"""Inference engine for applying trained models over preprocessed documents.

This module exposes utilities to lazily load NER and classification models for
specific document types, run them over the preprocessed text, and validate the
structured payload using the existing Pydantic schemas.

It also provides a :func:`create_default_engine` factory that registers the
built-in rule-based extractors so the engine is ready to use out of the box.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from src.pipeline.preprocess import preprocess_text
from src.pipeline.validation.validator import (
    validate_clinical_history,
    validate_prescription,
    validate_result_schema,
)

logger = logging.getLogger(__name__)

@dataclass
class ModelBundle:
    """Container with the models required for a document type.

    Parameters
    ----------
    classifier : Any, optional
        Classification model used to generate global attributes. The model must
        expose either a ``predict`` method or be callable. The method should
        return a dictionary that matches, partially or entirely, the target
        schema.
    ner : Any, optional
        Named Entity Recognition model that extracts entities from the
        document. Similar to ``classifier`` it must expose either ``predict``,
        ``extract`` or be callable returning a dictionary.
    classifier_loader : Callable[[], Any], optional
        Lazy loader for the classification model. When provided the model will
        be loaded on first access and cached afterwards.
    ner_loader : Callable[[], Any], optional
        Lazy loader for the NER model.
    """

    classifier: Optional[Any] = None
    ner: Optional[Any] = None
    classifier_loader: Optional[Callable[[], Any]] = None
    ner_loader: Optional[Callable[[], Any]] = None

    def ensure_loaded(self) -> None:
        """Load models using the configured loaders when necessary."""
        if self.classifier is None and self.classifier_loader is not None:
            self.classifier = self.classifier_loader()
        if self.ner is None and self.ner_loader is not None:
            self.ner = self.ner_loader()


class ModelRegistry:
    """Registry responsible for storing the available model bundles."""

    def __init__(self, bundles: Optional[Dict[str, ModelBundle]] = None) -> None:
        self._bundles: Dict[str, ModelBundle] = bundles or {}

    def register(self, document_type: str, bundle: ModelBundle) -> None:
        """Register or override the model bundle for a given document type."""
        self._bundles[document_type] = bundle

    def get_bundle(self, document_type: str) -> ModelBundle:
        """Retrieve the model bundle associated with ``document_type``.

        Raises
        ------
        ValueError
            If there is no bundle registered for ``document_type``.
        """
        try:
            bundle = self._bundles[document_type]
        except KeyError as exc:
            raise ValueError(f"No models registered for document type '{document_type}'") from exc

        bundle.ensure_loaded()
        return bundle


@dataclass
class InferenceResult:
    """Structured result of the inference pipeline."""

    document_type: str
    raw_text: str
    preprocessed_text: str
    classifier_output: Dict[str, Any]
    ner_output: Dict[str, Any]
    combined_output: Dict[str, Any]
    validated_data: Optional[BaseModel]

    def as_dict(self) -> Dict[str, Any]:
        """Return the validated data as a dictionary when available."""
        if self.validated_data is not None:
            return self.validated_data.model_dump()
        return dict(self.combined_output)


class InferenceEngine:
    """Main interface that orchestrates preprocessing, inference and validation."""

    DEFAULT_VALIDATORS: Dict[str, Callable[[Dict[str, Any]], BaseModel]] = {
        "result": validate_result_schema,
        "prescription": validate_prescription,
        "clinical_history": validate_clinical_history,
    }

    def __init__(
        self,
        registry: ModelRegistry,
        validators: Optional[Dict[str, Callable[[Dict[str, Any]], BaseModel]]] = None,
        text_preprocessor: Callable[[str], str] = preprocess_text,
    ) -> None:
        self._registry = registry
        self._validators = validators or self.DEFAULT_VALIDATORS
        self._text_preprocessor = text_preprocessor

    def process_document(self, document_type: str, raw_text: str) -> InferenceResult:
        """Run the full inference pipeline for a document.

        Parameters
        ----------
        document_type : str
            Type of the input document. Used to select the appropriate models
            and validation schema.
        raw_text : str
            Text extracted from the document.
        """
        bundle = self._registry.get_bundle(document_type)
        preprocessed_text = self._text_preprocessor(raw_text)

        # Classifier works on normalised/lowercased text; NER extractors
        # need original casing to preserve proper nouns and dates.
        classifier_output = self._apply_model(bundle.classifier, preprocessed_text)
        ner_output = self._apply_model(bundle.ner, raw_text)
        combined_output = self._merge_outputs(classifier_output, ner_output)
        validated = self._validate(document_type, combined_output)

        return InferenceResult(
            document_type=document_type,
            raw_text=raw_text,
            preprocessed_text=preprocessed_text,
            classifier_output=classifier_output,
            ner_output=ner_output,
            combined_output=combined_output,
            validated_data=validated,
        )

    def _apply_model(self, model: Optional[Any], text: str) -> Dict[str, Any]:
        if model is None:
            return {}

        model_name = type(model).__name__
        try:
            if hasattr(model, "predict") and callable(model.predict):
                output = model.predict(text)
            elif hasattr(model, "extract") and callable(model.extract):
                output = model.extract(text)
            elif hasattr(model, "extract_entities") and callable(model.extract_entities):
                output = model.extract_entities(text)
            elif callable(model):
                output = model(text)
            else:
                raise TypeError(
                    f"Model {model_name} must be callable or expose "
                    "predict/extract/extract_entities methods"
                )
        except TypeError:
            raise
        except Exception as exc:
            logger.error("Model %s raised %s: %s", model_name, type(exc).__name__, exc)
            raise RuntimeError(
                f"Model {model_name} failed during inference: {exc}"
            ) from exc

        if output is None:
            return {}
        if isinstance(output, BaseModel):
            return output.model_dump()
        if not isinstance(output, dict):
            raise TypeError(
                f"Model {model_name} returned {type(output).__name__}; "
                "expected dict or Pydantic BaseModel"
            )
        return output

    @staticmethod
    def _merge_outputs(*outputs: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for output in outputs:
            if not output:
                continue
            merged.update(output)
        return merged

    def _validate(self, document_type: str, payload: Dict[str, Any]) -> Optional[BaseModel]:
        if not payload:
            logger.warning("Empty payload for document type '%s'; skipping validation.", document_type)
            return None
        validator = self._validators.get(document_type)
        if validator is None:
            logger.warning("No validator registered for document type '%s'.", document_type)
            return None
        # Remove engine metadata that isn't part of the target schema.
        validation_payload = {k: v for k, v in payload.items() if k != "document_type"}
        try:
            return validator(validation_payload)
        except Exception as exc:
            logger.error(
                "Validation failed for document type '%s': %s", document_type, exc
            )
            raise

    # ── Introspection helpers ─────────────────────────────────────

    @property
    def registered_types(self) -> List[str]:
        """Return the document types that have models registered."""
        return list(self._registry._bundles.keys())

    def classify(self, raw_text: str) -> Optional[str]:
        """Classify *raw_text* using the first available classifier.

        Iterates over registered bundles and returns the first non-None
        classification result.  Returns ``None`` if no classifier matches.
        """
        for doc_type in self.registered_types:
            bundle = self._registry.get_bundle(doc_type)
            if bundle.classifier is None:
                continue
            result = self._apply_model(bundle.classifier, raw_text)
            doc_type_detected = result.get("document_type")
            if doc_type_detected is not None:
                return doc_type_detected
        return None


# ═══════════════════════════════════════════════════════════════════
# Factory: default engine with built-in rule-based extractors
# ═══════════════════════════════════════════════════════════════════

def create_default_engine() -> InferenceEngine:
    """Build an :class:`InferenceEngine` pre-loaded with rule-based extractors.

    This is the recommended way to instantiate the engine for v1.0.  The
    returned engine can process ``prescription``, ``result``, and
    ``clinical_history`` document types out of the box.

    Returns
    -------
    InferenceEngine
        Ready-to-use engine instance.
    """
    from src.pipeline.extractors import (
        ClinicalHistoryExtractor,
        LabResultExtractor,
        PrescriptionExtractor,
    )
    from src.pipeline.extractors.document_classifier import DocumentClassifier

    classifier = DocumentClassifier()

    registry = ModelRegistry({
        "prescription": ModelBundle(
            classifier=classifier,
            ner=PrescriptionExtractor(),
        ),
        "result": ModelBundle(
            classifier=classifier,
            ner=LabResultExtractor(),
        ),
        "clinical_history": ModelBundle(
            classifier=classifier,
            ner=ClinicalHistoryExtractor(),
        ),
    })

    return InferenceEngine(registry=registry)
