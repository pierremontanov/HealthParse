"""DocIQ processing pipeline.

This package contains the full document-processing pipeline: text extraction
(PDF + OCR), language detection, classification, NER, validation, export, and
FHIR mapping.

Commonly used imports::

    from src.pipeline import (
        DocIQEngine, EngineResult,
        InferenceEngine, create_default_engine,
        ModelManager,
        OutputCollector,
        process_folder,
    )
"""

from src.pipeline.core_engine import DocIQEngine, EngineResult
from src.pipeline.inference import (
    InferenceEngine,
    InferenceResult,
    ModelBundle,
    ModelRegistry,
    create_default_engine,
)
from src.pipeline.model_manager import ModelManager, ModelMeta
from src.pipeline.output_collector import OutputCollector
from src.pipeline.relation_configs import (
    CLINICAL_HISTORY_RELATIONS,
    PRESCRIPTION_RELATIONS,
    RELATION_CONFIG_REGISTRY,
    RESULT_RELATIONS,
    get_relation_config,
)
from src.pipeline.relation_mapper import RelationMapper, RelationMappingResult, connect_entities

# NOTE: We deliberately avoid ``from src.pipeline.process_folder import
# process_folder`` here.  Importing a name identical to a sub-module shadows
# the module in the package namespace, which breaks ``unittest.mock.patch``
# targeting ``src.pipeline.process_folder.<attr>``.  Users can still do:
#     from src.pipeline.process_folder import process_folder, DocumentResult
import src.pipeline.process_folder  # noqa: F401 – ensure submodule is importable

__all__ = [
    # Core engine
    "DocIQEngine",
    "EngineResult",
    # Inference
    "InferenceEngine",
    "InferenceResult",
    "ModelBundle",
    "ModelRegistry",
    "create_default_engine",
    # Model persistence
    "ModelManager",
    "ModelMeta",
    # Batch processing
    "OutputCollector",
    # Relation mapping
    "RelationMapper",
    "RelationMappingResult",
    "connect_entities",
    "get_relation_config",
    "PRESCRIPTION_RELATIONS",
    "RESULT_RELATIONS",
    "CLINICAL_HISTORY_RELATIONS",
    "RELATION_CONFIG_REGISTRY",
]
