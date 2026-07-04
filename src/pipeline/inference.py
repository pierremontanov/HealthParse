"""Inference engine for applying trained models over preprocessed documents.

This module exposes utilities to lazily load NER and classification models for
specific document types, run them over the preprocessed text, and validate the
structured payload using the existing Pydantic schemas.

When NER output contains a flat ``"entities"`` list (as produced by ML-based
NER models), the engine automatically applies the :class:`RelationMapper` to
wire entities into structured relations before validation.

It also provides a :func:`create_default_engine` factory that registers the
built-in rule-based extractors so the engine is ready to use out of the box.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel

from src.pipeline.preprocess import preprocess_text
from src.pipeline.relation_configs import RELATION_CONFIG_REGISTRY, get_relation_config
from src.pipeline.relation_mapper import RelationMapper, RelationMappingResult
from src.pipeline.validation.validator import (
    validate_clinical_history,
    validate_prescription,
    validate_result_schema,
)

logger = logging.getLogger(__name__)

# Fields that should never be overwritten by LLM output
_LLM_SKIP_FIELDS = {"raw_text", "document_type"}


def _is_empty(value: Any) -> bool:
    """Check if a value is considered empty for merge purposes."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list) and not value:
        return True
    return False

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
    relation_mapping: Optional[RelationMappingResult] = None
    llm_output: Optional[Dict[str, Any]] = None
    llm_fields_used: Optional[List[str]] = None

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
        proximity_window: Optional[int] = None,
    ) -> None:
        self._registry = registry
        self._validators = validators or self.DEFAULT_VALIDATORS
        self._text_preprocessor = text_preprocessor
        self._proximity_window = proximity_window
        # Cached LLM client — created once on first use, reused for all documents.
        self._llm_client: Optional[Any] = None
        self._llm_available: Optional[bool] = None  # None = not yet checked

    def process_document(self, document_type: str, raw_text: str) -> InferenceResult:
        """Run the full inference pipeline for a document.

        Parameters
        ----------
        document_type : str
            Type of the input document. Used to select the appropriate models
            and validation schema.
        raw_text : str
            Text extracted from the document.

        The pipeline steps are:

        1. **Preprocessing** -- normalise the raw text.
        2. **Classification** -- extract global attributes.
        3. **NER** -- extract entities (structured dict *or* flat entity list).
        4. **Relation mapping** -- if NER output contains a flat ``"entities"``
           list, run :class:`RelationMapper` to wire entities into structured
           relations that match the validation schema.
        5. **Merge** -- combine classifier, NER, and relation outputs.
        6. **Validation** -- validate the merged dict against the Pydantic schema.
        7. **LLM enhancement** -- optionally call the LLM to fill gaps.
        """
        bundle = self._registry.get_bundle(document_type)
        preprocessed_text = self._text_preprocessor(raw_text)

        # Classifier works on normalised/lowercased text; NER extractors
        # need original casing to preserve proper nouns and dates.
        classifier_output = self._apply_model(bundle.classifier, preprocessed_text)
        ner_output = self._apply_model(bundle.ner, raw_text)

        # Relation mapping: wire flat entity lists into structured relations.
        relation_result = self._apply_relation_mapping(document_type, ner_output)

        combined_output = self._merge_outputs(classifier_output, ner_output)

        # If relation mapping produced results, merge them (relations take
        # precedence over raw NER fields for structured data).
        if relation_result is not None:
            relation_dict = self._relations_to_dict(relation_result)
            combined_output.update(relation_dict)

        validated = self._validate(document_type, combined_output)

        # -- LLM enhancement (opt-in) --
        llm_output = None
        llm_fields_used = None

        try:
            from src.config import settings as _cfg

            if _cfg.llm_enabled:
                llm_output, llm_fields_used = self._enhance_with_llm(
                    document_type, raw_text, combined_output,
                )
                if llm_fields_used:
                    # Re-validate with the enhanced output
                    validated = self._validate(document_type, combined_output)
        except Exception as exc:
            # Graceful fallback: LLM failure never blocks the pipeline
            logger.warning("LLM enhancement failed, using rule-based results: %s", exc)

        return InferenceResult(
            document_type=document_type,
            raw_text=raw_text,
            preprocessed_text=preprocessed_text,
            classifier_output=classifier_output,
            ner_output=ner_output,
            combined_output=combined_output,
            validated_data=validated,
            relation_mapping=relation_result,
            llm_output=llm_output,
            llm_fields_used=llm_fields_used,
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
            from src.pipeline.exceptions import ModelExecutionError
            logger.error("Model %s raised %s: %s", model_name, type(exc).__name__, exc)
            raise ModelExecutionError(model_name, str(exc)) from exc

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

    # -- LLM enhancement ---------------------------------------------------

    def _enhance_with_llm(
        self,
        document_type: str,
        raw_text: str,
        combined_output: Dict[str, Any],
    ) -> tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
        """Call the LLM and merge its output into rule-based results.

        The "middle" approach: both rule-based and LLM run on every
        document.  For each field, the system picks the best value:

        - If rule-based has a non-empty value, keep it (deterministic wins).
        - If rule-based is None/empty but LLM has a value, use the LLM's.
        - Special case for ``items`` (prescriptions): if LLM has more items
          or richer sub-fields, prefer the LLM list.

        Parameters
        ----------
        document_type : str
            Classified document type.
        raw_text : str
            Full text from extraction (for the LLM prompt).
        combined_output : dict
            Current rule-based extraction output (mutated in place).

        Returns
        -------
        tuple[dict | None, list[str] | None]
            The raw LLM output and list of field names where LLM values
            were used, or ``(None, None)`` on failure.
        """
        from src.pipeline.llm_client import OllamaClient
        from src.pipeline.metrics import Timer, get_collector

        collector = get_collector()

        # Reuse the cached client; create it once on first LLM call.
        if self._llm_client is None:
            self._llm_client = OllamaClient()

        # Re-check availability only when previously unavailable (lazy re-check).
        if self._llm_available is None or self._llm_available is False:
            self._llm_available = self._llm_client.is_available()

        if not self._llm_available:
            logger.info("LLM server not available, skipping enhancement.")
            return None, None

        client = self._llm_client

        with Timer("llm_extract") as t:
            llm_data = client.extract(document_type, raw_text)
        collector.record("llm_extract", t.elapsed_ms)

        if not llm_data:
            return None, None

        fields_used = self._merge_llm_fields(combined_output, llm_data)

        if fields_used:
            logger.info(
                "LLM enhanced %d field(s): %s",
                len(fields_used),
                ", ".join(fields_used),
            )
        else:
            logger.debug("LLM produced no additional fields beyond rule-based.")

        return llm_data, fields_used

    @staticmethod
    def _merge_llm_fields(
        rule_based: Dict[str, Any],
        llm_data: Dict[str, Any],
    ) -> List[str]:
        """Merge LLM output into rule-based results, filling gaps.

        Rule-based values always take precedence when they are non-empty.
        The LLM fills in fields that are ``None``, empty strings, or
        missing entirely.

        For the ``items`` list (prescriptions), the LLM list replaces the
        rule-based list only if it contains more items or if each item
        has richer sub-fields (more non-null values).

        Parameters
        ----------
        rule_based : dict
            Mutated in place with LLM values where appropriate.
        llm_data : dict
            Raw LLM extraction output.

        Returns
        -------
        list[str]
            Field names where LLM values were used.
        """
        fields_used: List[str] = []

        for key, llm_value in llm_data.items():
            if key in _LLM_SKIP_FIELDS:
                continue

            # Only merge into fields that the rule-based extractor
            # already created — adding unknown keys would break
            # Pydantic validation (extra="forbid" schemas).
            if key not in rule_based:
                continue

            rule_value = rule_based.get(key)

            # Special handling for prescription items list
            if key == "items" and isinstance(llm_value, list):
                rule_items = rule_based.get("items")
                if not isinstance(rule_items, list) or not rule_items:
                    # Rule-based found nothing, use LLM
                    rule_based["items"] = llm_value
                    fields_used.append("items")
                elif len(llm_value) > len(rule_items):
                    # LLM found more items
                    rule_based["items"] = llm_value
                    fields_used.append("items")
                else:
                    # Same count -- check if LLM items are richer
                    def _richness(items: list) -> int:
                        return sum(
                            1
                            for item in items
                            if isinstance(item, dict)
                            for v in item.values()
                            if v is not None
                        )

                    if _richness(llm_value) > _richness(rule_items):
                        rule_based["items"] = llm_value
                        fields_used.append("items")
                continue

            # Standard field: LLM fills gaps only
            if _is_empty(rule_value) and not _is_empty(llm_value):
                rule_based[key] = llm_value
                fields_used.append(key)

        return fields_used

    # -- Relation mapping --------------------------------------------------

    def _apply_relation_mapping(
        self, document_type: str, ner_output: Dict[str, Any]
    ) -> Optional[RelationMappingResult]:
        """Run :class:`RelationMapper` when NER output has a flat entity list.

        If ``ner_output`` contains an ``"entities"`` key whose value is a list
        of entity dicts, the mapper converts them into structured relations
        using the domain config for *document_type*.

        Returns ``None`` when the NER output is already structured (i.e. the
        rule-based extractors) or when no relation config exists.
        """
        entities = ner_output.get("entities")
        if not isinstance(entities, list) or not entities:
            return None

        try:
            config = get_relation_config(document_type)
        except KeyError:
            logger.debug("No relation config for '%s'; skipping entity wiring.", document_type)
            return None

        mapper = RelationMapper(
            config,
            proximity_window=self._proximity_window,
            keep_metadata=False,
        )
        result = mapper.map_relations(entities)
        logger.info(
            "Relation mapping for '%s': %d relations, %d orphans",
            document_type, len(result.relations), len(result.orphans),
        )
        return result

    @staticmethod
    def _relations_to_dict(result: RelationMappingResult) -> Dict[str, Any]:
        """Flatten a :class:`RelationMappingResult` into a schema-friendly dict.

        Converts the list of relation dicts into a structure that can be
        merged with the combined output before validation.  For example, a
        list of MEDICATION relations becomes an ``"items"`` list for the
        prescription schema.
        """
        output: Dict[str, Any] = {}
        if not result.relations:
            return output

        # Group relations by their anchor label.
        from collections import defaultdict
        by_anchor: Dict[str, list] = defaultdict(list)
        for rel in result.relations:
            # First key is always the anchor label.
            anchor_label = next(iter(rel))
            by_anchor[anchor_label].append(rel)

        # Expose grouped relations under a predictable key.
        output["_relations"] = dict(by_anchor)
        output["_orphans"] = [
            {"label": o.get("label", ""), "text": o.get("text", "")}
            for o in result.orphans
        ]
        return output

    # -- Introspection helpers ---------------------------------------------

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


# ==================================================================
# Factory: default engine with built-in rule-based extractors
# ==================================================================

def create_default_engine(
    model_path: str | None = None,
) -> InferenceEngine:
    """Build an :class:`InferenceEngine` pre-loaded with rule-based extractors.

    This is the recommended way to instantiate the engine for v1.0.  The
    returned engine can process ``prescription``, ``result``, and
    ``clinical_history`` document types out of the box.

    Parameters
    ----------
    model_path : str, optional
        Path to a trained model JSON artefact.  When provided, classifier
        weights and OCR settings are loaded from the file before building
        the engine.  Falls back to ``settings.model_path`` when ``None``.

    Returns
    -------
    InferenceEngine
        Ready-to-use engine instance.
    """
    # Apply trained model weights when a path is configured.
    effective_path = model_path
    if effective_path is None:
        try:
            from src.config import settings as _cfg
            effective_path = _cfg.model_path
        except Exception:
            pass

    if effective_path is not None:
        from src.pipeline.model_manager import ModelManager

        mgr = ModelManager.load(effective_path)
        mgr.apply()
        logger.info("Loaded trained model from %s", effective_path)

    from src.pipeline.extractors import (
        ClinicalHistoryExtractor,
        LabResultExtractor,
        PrescriptionExtractor,
        ReceiptExtractor,
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
        "receipt": ModelBundle(
            classifier=classifier,
            ner=ReceiptExtractor(),
        ),
    })

    return InferenceEngine(registry=registry)
