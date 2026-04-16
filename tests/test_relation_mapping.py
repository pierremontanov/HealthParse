"""Tests for #1 Relation Mapping – Entity Wiring.

Covers:
  • RelationMapper core behaviour (anchor–dependent linking, proximity, orphans)
  • Domain-specific relation configs (prescription, result, clinical history)
  • InferenceEngine integration (auto entity wiring for flat NER output)
  • connect_entities convenience wrapper
  • Edge cases (empty input, unknown labels, all orphans)
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.pipeline.relation_configs import (
    CLINICAL_HISTORY_RELATIONS,
    PRESCRIPTION_RELATIONS,
    RELATION_CONFIG_REGISTRY,
    RESULT_RELATIONS,
    get_relation_config,
)
from src.pipeline.relation_mapper import (
    RelationMapper,
    RelationMappingResult,
    connect_entities,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _ent(label: str, text: str, start: int, end: int | None = None) -> Dict[str, Any]:
    """Build a minimal entity dict."""
    return {"label": label, "text": text, "start": start, "end": end or start + len(text)}


# ═══════════════════════════════════════════════════════════════════
# RelationMapper – core mechanics
# ═══════════════════════════════════════════════════════════════════


class TestRelationMapperCore:
    """Test anchor–dependent linking, proximity window, and aggregation."""

    def test_single_anchor_no_dependents(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        result = mapper.map_relations([_ent("MEDICATION", "Amoxicillin", 0)])
        assert len(result.relations) == 1
        assert result.relations[0]["MEDICATION"] == "Amoxicillin"
        assert result.orphans == []

    def test_anchor_with_one_dependent(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        entities = [
            _ent("MEDICATION", "Ibuprofen", 0),
            _ent("DOSAGE", "400mg", 12),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 1
        rel = result.relations[0]
        assert rel["MEDICATION"] == "Ibuprofen"
        assert rel["DOSAGE"] == "400mg"

    def test_anchor_with_multiple_dependents(self):
        mapper = RelationMapper(
            {"MEDICATION": ["DOSAGE", "FREQUENCY", "ROUTE"]}, keep_metadata=False
        )
        entities = [
            _ent("MEDICATION", "Metformin", 0),
            _ent("DOSAGE", "850mg", 12),
            _ent("FREQUENCY", "twice daily", 20),
            _ent("ROUTE", "oral", 35),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 1
        rel = result.relations[0]
        assert rel["MEDICATION"] == "Metformin"
        assert rel["DOSAGE"] == "850mg"
        assert rel["FREQUENCY"] == "twice daily"
        assert rel["ROUTE"] == "oral"

    def test_multiple_anchors(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        entities = [
            _ent("MEDICATION", "Drug A", 0),
            _ent("DOSAGE", "10mg", 10),
            _ent("MEDICATION", "Drug B", 30),
            _ent("DOSAGE", "20mg", 40),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 2
        assert result.relations[0]["MEDICATION"] == "Drug A"
        assert result.relations[0]["DOSAGE"] == "10mg"
        assert result.relations[1]["MEDICATION"] == "Drug B"
        assert result.relations[1]["DOSAGE"] == "20mg"

    def test_orphan_entities(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        entities = [
            _ent("UNKNOWN_LABEL", "some text", 0),
            _ent("MEDICATION", "Aspirin", 20),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 1
        assert len(result.orphans) == 1
        assert result.orphans[0]["label"] == "UNKNOWN_LABEL"

    def test_dependent_before_any_anchor_becomes_orphan(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        entities = [
            _ent("DOSAGE", "500mg", 0),
            _ent("MEDICATION", "Aspirin", 20),
        ]
        result = mapper.map_relations(entities)
        # The dosage has no preceding anchor → orphan
        assert len(result.orphans) == 1
        assert result.orphans[0]["text"] == "500mg"

    def test_proximity_window_filters_distant_dependents(self):
        mapper = RelationMapper(
            {"MEDICATION": ["DOSAGE"]}, proximity_window=15, keep_metadata=False
        )
        entities = [
            _ent("MEDICATION", "Drug A", 0, 6),
            _ent("DOSAGE", "10mg", 100),  # Too far
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 1
        assert "DOSAGE" not in result.relations[0]
        assert len(result.orphans) == 1

    def test_proximity_window_allows_nearby_dependents(self):
        mapper = RelationMapper(
            {"MEDICATION": ["DOSAGE"]}, proximity_window=20, keep_metadata=False
        )
        entities = [
            _ent("MEDICATION", "Drug A", 0, 6),
            _ent("DOSAGE", "10mg", 10),
        ]
        result = mapper.map_relations(entities)
        assert result.relations[0].get("DOSAGE") == "10mg"

    def test_empty_entities(self):
        mapper = RelationMapper({"A": ["B"]}, keep_metadata=False)
        result = mapper.map_relations([])
        assert result.relations == []
        assert result.orphans == []

    def test_all_orphans(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        entities = [
            _ent("SOMETHING", "x", 0),
            _ent("OTHER", "y", 10),
        ]
        result = mapper.map_relations(entities)
        assert result.relations == []
        assert len(result.orphans) == 2


# ═══════════════════════════════════════════════════════════════════
# RelationMapper – aggregation modes
# ═══════════════════════════════════════════════════════════════════


class TestAggregation:
    """Test different aggregation strategies."""

    def _entities_with_duplicate_dep(self):
        return [
            _ent("MEDICATION", "Aspirin", 0),
            _ent("DOSAGE", "100mg", 10),
            _ent("DOSAGE", "200mg", 20),
        ]

    def test_auto_aggregator_single_value(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, aggregator="auto", keep_metadata=False)
        result = mapper.map_relations([
            _ent("MEDICATION", "X", 0),
            _ent("DOSAGE", "5mg", 10),
        ])
        assert result.relations[0]["DOSAGE"] == "5mg"  # single → string

    def test_auto_aggregator_multiple_values(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, aggregator="auto", keep_metadata=False)
        result = mapper.map_relations(self._entities_with_duplicate_dep())
        assert result.relations[0]["DOSAGE"] == ["100mg", "200mg"]

    def test_list_aggregator(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, aggregator="list", keep_metadata=False)
        result = mapper.map_relations([
            _ent("MEDICATION", "X", 0),
            _ent("DOSAGE", "5mg", 10),
        ])
        assert result.relations[0]["DOSAGE"] == ["5mg"]  # always a list

    def test_concat_aggregator(self):
        mapper = RelationMapper(
            {"MEDICATION": ["DOSAGE"]}, aggregator="concat", join_token=", ", keep_metadata=False
        )
        result = mapper.map_relations(self._entities_with_duplicate_dep())
        assert result.relations[0]["DOSAGE"] == "100mg, 200mg"

    def test_custom_aggregator(self):
        def upper_join(label, values):
            return " | ".join(v.upper() for v in values)

        mapper = RelationMapper(
            {"MEDICATION": ["DOSAGE"]}, aggregator=upper_join, keep_metadata=False
        )
        result = mapper.map_relations(self._entities_with_duplicate_dep())
        assert result.relations[0]["DOSAGE"] == "100MG | 200MG"

    def test_invalid_aggregator_raises(self):
        with pytest.raises(ValueError, match="Invalid aggregator"):
            RelationMapper({"A": ["B"]}, aggregator="invalid")


# ═══════════════════════════════════════════════════════════════════
# RelationMapper – entity normalisation
# ═══════════════════════════════════════════════════════════════════


class TestEntityNormalisation:
    """Test that different entity formats are handled."""

    def test_huggingface_format(self):
        """HuggingFace NER pipelines use 'entity_group' and 'word'."""
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        entities = [
            {"entity_group": "MEDICATION", "word": "Aspirin", "start": 0, "end": 7},
            {"entity_group": "DOSAGE", "word": "100mg", "start": 10, "end": 15},
        ]
        result = mapper.map_relations(entities)
        assert result.relations[0]["MEDICATION"] == "Aspirin"
        assert result.relations[0]["DOSAGE"] == "100mg"

    def test_custom_format_with_type_key(self):
        mapper = RelationMapper({"MED": ["DOSE"]}, keep_metadata=False)
        entities = [
            {"type": "MED", "word": "Ibuprofen", "start_char": 0, "end_char": 9},
        ]
        # 'type' maps to label, 'word' maps to text
        result = mapper.map_relations(entities)
        assert result.relations[0]["MED"] == "Ibuprofen"

    def test_entity_with_score(self):
        mapper = RelationMapper({"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        entities = [
            {"label": "MEDICATION", "text": "Aspirin", "start": 0, "end": 7, "score": 0.95},
        ]
        result = mapper.map_relations(entities)
        assert result.relations[0]["MEDICATION"] == "Aspirin"


# ═══════════════════════════════════════════════════════════════════
# RelationMapper – metadata
# ═══════════════════════════════════════════════════════════════════


class TestMetadata:
    """Test the optional _metadata tracking."""

    def test_metadata_included_by_default(self):
        mapper = RelationMapper({"A": ["B"]})
        result = mapper.map_relations([_ent("A", "anchor", 0)])
        assert "_metadata" in result.relations[0]

    def test_metadata_excluded_when_disabled(self):
        mapper = RelationMapper({"A": ["B"]}, keep_metadata=False)
        result = mapper.map_relations([_ent("A", "anchor", 0)])
        assert "_metadata" not in result.relations[0]


# ═══════════════════════════════════════════════════════════════════
# RelationMappingResult
# ═══════════════════════════════════════════════════════════════════


class TestRelationMappingResult:
    """Test the result container."""

    def test_as_dict(self):
        r = RelationMappingResult(
            relations=[{"MEDICATION": "Aspirin", "DOSAGE": "100mg"}],
            orphans=[{"label": "UNKNOWN", "text": "xyz"}],
        )
        d = r.as_dict()
        assert "relations" in d
        assert "orphans" in d
        assert len(d["relations"]) == 1
        assert len(d["orphans"]) == 1

    def test_empty_result(self):
        r = RelationMappingResult(relations=[], orphans=[])
        d = r.as_dict()
        assert d == {"relations": [], "orphans": []}


# ═══════════════════════════════════════════════════════════════════
# connect_entities convenience wrapper
# ═══════════════════════════════════════════════════════════════════


class TestConnectEntities:
    """Test the module-level convenience function."""

    def test_basic_usage(self):
        entities = [
            _ent("MEDICATION", "Aspirin", 0),
            _ent("DOSAGE", "100mg", 10),
        ]
        result = connect_entities(entities, {"MEDICATION": ["DOSAGE"]}, keep_metadata=False)
        assert isinstance(result, RelationMappingResult)
        assert result.relations[0]["DOSAGE"] == "100mg"

    def test_with_proximity_window(self):
        entities = [
            _ent("MEDICATION", "Aspirin", 0, 7),
            _ent("DOSAGE", "100mg", 500),  # far away
        ]
        result = connect_entities(
            entities, {"MEDICATION": ["DOSAGE"]}, proximity_window=10, keep_metadata=False
        )
        assert "DOSAGE" not in result.relations[0]
        assert len(result.orphans) == 1


# ═══════════════════════════════════════════════════════════════════
# Relation configs – domain-specific
# ═══════════════════════════════════════════════════════════════════


class TestRelationConfigs:
    """Validate the three domain-specific relation configurations."""

    def test_prescription_config_has_medication_anchor(self):
        assert "MEDICATION" in PRESCRIPTION_RELATIONS
        deps = PRESCRIPTION_RELATIONS["MEDICATION"]
        assert "DOSAGE" in deps
        assert "FREQUENCY" in deps
        assert "ROUTE" in deps
        assert "DURATION" in deps

    def test_prescription_config_has_radiology_anchor(self):
        assert "RADIOLOGY" in PRESCRIPTION_RELATIONS
        assert "MODALITY" in PRESCRIPTION_RELATIONS["RADIOLOGY"]
        assert "BODY_PART" in PRESCRIPTION_RELATIONS["RADIOLOGY"]

    def test_prescription_config_has_lab_test_anchor(self):
        assert "LAB_TEST" in PRESCRIPTION_RELATIONS
        assert "TEST_TYPE" in PRESCRIPTION_RELATIONS["LAB_TEST"]

    def test_prescription_config_has_specialist_anchor(self):
        assert "SPECIALIST" in PRESCRIPTION_RELATIONS
        assert "SPECIALTY" in PRESCRIPTION_RELATIONS["SPECIALIST"]

    def test_prescription_config_has_therapy_anchor(self):
        assert "THERAPY" in PRESCRIPTION_RELATIONS
        assert "THERAPY_TYPE" in PRESCRIPTION_RELATIONS["THERAPY"]

    def test_result_config_has_test_name_anchor(self):
        assert "TEST_NAME" in RESULT_RELATIONS
        deps = RESULT_RELATIONS["TEST_NAME"]
        assert "TEST_VALUE" in deps
        assert "REFERENCE_RANGE" in deps
        assert "UNIT" in deps

    def test_result_config_has_exam_type_anchor(self):
        assert "EXAM_TYPE" in RESULT_RELATIONS
        assert "FINDINGS" in RESULT_RELATIONS["EXAM_TYPE"]
        assert "IMPRESSION" in RESULT_RELATIONS["EXAM_TYPE"]

    def test_clinical_history_config_has_diagnosis_anchor(self):
        assert "DIAGNOSIS" in CLINICAL_HISTORY_RELATIONS
        assert "TREATMENT" in CLINICAL_HISTORY_RELATIONS["DIAGNOSIS"]
        assert "STATUS" in CLINICAL_HISTORY_RELATIONS["DIAGNOSIS"]

    def test_clinical_history_config_has_medication_anchor(self):
        assert "MEDICATION" in CLINICAL_HISTORY_RELATIONS
        assert "INDICATION" in CLINICAL_HISTORY_RELATIONS["MEDICATION"]

    def test_clinical_history_config_has_complaint_anchor(self):
        assert "COMPLAINT" in CLINICAL_HISTORY_RELATIONS
        assert "ASSESSMENT" in CLINICAL_HISTORY_RELATIONS["COMPLAINT"]

    def test_registry_has_all_three_types(self):
        assert set(RELATION_CONFIG_REGISTRY.keys()) == {
            "prescription", "result", "clinical_history"
        }

    def test_get_relation_config_valid(self):
        config = get_relation_config("prescription")
        assert config is PRESCRIPTION_RELATIONS

    def test_get_relation_config_invalid_raises(self):
        with pytest.raises(KeyError, match="No relation config"):
            get_relation_config("nonexistent")


# ═══════════════════════════════════════════════════════════════════
# Domain configs work with RelationMapper
# ═══════════════════════════════════════════════════════════════════


class TestDomainConfigIntegration:
    """End-to-end: domain configs + RelationMapper produce correct output."""

    def test_prescription_medication_wiring(self):
        mapper = RelationMapper(PRESCRIPTION_RELATIONS, keep_metadata=False)
        entities = [
            _ent("MEDICATION", "Amoxicillin", 0),
            _ent("DOSAGE", "500mg", 15),
            _ent("FREQUENCY", "every 8 hours", 22),
            _ent("ROUTE", "oral", 38),
            _ent("DURATION", "7 days", 44),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 1
        rel = result.relations[0]
        assert rel["MEDICATION"] == "Amoxicillin"
        assert rel["DOSAGE"] == "500mg"
        assert rel["FREQUENCY"] == "every 8 hours"
        assert rel["ROUTE"] == "oral"
        assert rel["DURATION"] == "7 days"

    def test_prescription_mixed_item_types(self):
        mapper = RelationMapper(PRESCRIPTION_RELATIONS, keep_metadata=False)
        entities = [
            _ent("MEDICATION", "Ibuprofen", 0),
            _ent("DOSAGE", "400mg", 12),
            _ent("RADIOLOGY", "Chest X-Ray", 30),
            _ent("MODALITY", "X-Ray", 45),
            _ent("BODY_PART", "Chest", 52),
            _ent("SPECIALIST", "Cardiology consult", 70),
            _ent("SPECIALTY", "Cardiology", 90),
            _ent("REASON", "Hypertension", 105),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 3

        med = result.relations[0]
        assert med["MEDICATION"] == "Ibuprofen"

        rad = result.relations[1]
        assert rad["RADIOLOGY"] == "Chest X-Ray"
        assert rad["MODALITY"] == "X-Ray"

        spec = result.relations[2]
        assert spec["SPECIALIST"] == "Cardiology consult"
        assert spec["REASON"] == "Hypertension"

    def test_result_test_wiring(self):
        mapper = RelationMapper(RESULT_RELATIONS, keep_metadata=False)
        entities = [
            _ent("TEST_NAME", "Glucose", 0),
            _ent("TEST_VALUE", "95.5", 10),
            _ent("REFERENCE_RANGE", "70-100", 16),
            _ent("UNIT", "mg/dL", 25),
            _ent("TEST_NAME", "HbA1c", 40),
            _ent("TEST_VALUE", "5.4", 48),
            _ent("REFERENCE_RANGE", "4.0-5.6", 53),
            _ent("UNIT", "%", 62),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 2
        assert result.relations[0]["TEST_NAME"] == "Glucose"
        assert result.relations[0]["TEST_VALUE"] == "95.5"
        assert result.relations[1]["TEST_NAME"] == "HbA1c"
        assert result.relations[1]["UNIT"] == "%"

    def test_clinical_history_wiring(self):
        mapper = RelationMapper(CLINICAL_HISTORY_RELATIONS, keep_metadata=False)
        entities = [
            _ent("DIAGNOSIS", "Type 2 Diabetes", 0),
            _ent("DATE", "2020-01-15", 20),
            _ent("TREATMENT", "Metformin 850mg", 35),
            _ent("STATUS", "controlled", 55),
            _ent("MEDICATION", "Metformin", 80),
            _ent("DOSAGE", "850mg", 92),
            _ent("INDICATION", "Diabetes", 100),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 2

        diag = result.relations[0]
        assert diag["DIAGNOSIS"] == "Type 2 Diabetes"
        assert diag["TREATMENT"] == "Metformin 850mg"
        assert diag["STATUS"] == "controlled"

        med = result.relations[1]
        assert med["MEDICATION"] == "Metformin"
        assert med["INDICATION"] == "Diabetes"


# ═══════════════════════════════════════════════════════════════════
# InferenceEngine integration
# ═══════════════════════════════════════════════════════════════════


class TestInferenceEngineRelationWiring:
    """Test that InferenceEngine auto-wires entities when NER returns flat list."""

    def _make_engine_with_entity_ner(
        self, doc_type: str, entities: List[Dict[str, Any]]
    ):
        """Build an engine whose NER model returns a flat entity list."""
        from src.pipeline.inference import (
            InferenceEngine,
            ModelBundle,
            ModelRegistry,
        )

        # NER model that returns {"entities": [...]}
        # Use spec=[] so MagicMock doesn't expose predict/extract/etc.
        ner_model = MagicMock(spec=[])
        ner_model.return_value = {"entities": entities}

        # Classifier: also use spec=[] + make callable
        classifier = MagicMock(spec=[])
        classifier.return_value = {"document_type": doc_type}

        registry = ModelRegistry({
            doc_type: ModelBundle(classifier=classifier, ner=ner_model),
        })

        # Skip validation to focus on relation mapping
        engine = InferenceEngine(
            registry=registry,
            validators={doc_type: lambda payload: None},
        )
        return engine

    def test_entities_trigger_relation_mapping(self):
        entities = [
            _ent("MEDICATION", "Aspirin", 0),
            _ent("DOSAGE", "100mg", 10),
            _ent("FREQUENCY", "daily", 18),
        ]
        engine = self._make_engine_with_entity_ner("prescription", entities)
        result = engine.process_document("prescription", "Aspirin 100mg daily")

        assert result.relation_mapping is not None
        assert len(result.relation_mapping.relations) == 1
        rel = result.relation_mapping.relations[0]
        assert rel["MEDICATION"] == "Aspirin"
        assert rel["DOSAGE"] == "100mg"

    def test_structured_ner_skips_relation_mapping(self):
        """Rule-based extractors return dicts without 'entities' key → no mapping."""
        from src.pipeline.inference import (
            InferenceEngine,
            ModelBundle,
            ModelRegistry,
        )

        ner_model = MagicMock(spec=[])
        ner_model.return_value = {
            "patient_name": "John",
            "items": [{"type": "medicine", "name": "Aspirin", "dosage": "100mg"}],
        }

        classifier = MagicMock(spec=[])
        classifier.return_value = {"document_type": "prescription"}

        registry = ModelRegistry({
            "prescription": ModelBundle(classifier=classifier, ner=ner_model),
        })
        engine = InferenceEngine(
            registry=registry,
            validators={"prescription": lambda p: None},
        )

        result = engine.process_document("prescription", "some text")
        assert result.relation_mapping is None

    def test_relation_data_in_combined_output(self):
        entities = [
            _ent("TEST_NAME", "Glucose", 0),
            _ent("TEST_VALUE", "95", 10),
        ]
        engine = self._make_engine_with_entity_ner("result", entities)
        result = engine.process_document("result", "Glucose 95")

        assert "_relations" in result.combined_output
        assert "TEST_NAME" in result.combined_output["_relations"]

    def test_orphans_in_combined_output(self):
        entities = [
            _ent("UNKNOWN", "mystery", 0),
            _ent("TEST_NAME", "Glucose", 20),
        ]
        engine = self._make_engine_with_entity_ner("result", entities)
        result = engine.process_document("result", "mystery... Glucose")

        assert "_orphans" in result.combined_output
        assert any(o["label"] == "UNKNOWN" for o in result.combined_output["_orphans"])

    def test_proximity_window_parameter(self):
        from src.pipeline.inference import (
            InferenceEngine,
            ModelBundle,
            ModelRegistry,
        )

        entities = [
            _ent("MEDICATION", "Aspirin", 0, 7),
            _ent("DOSAGE", "100mg", 500),  # Very far
        ]

        ner_model = MagicMock(spec=[])
        ner_model.return_value = {"entities": entities}
        classifier = MagicMock(spec=[])
        classifier.return_value = {"document_type": "prescription"}

        registry = ModelRegistry({
            "prescription": ModelBundle(classifier=classifier, ner=ner_model),
        })
        engine = InferenceEngine(
            registry=registry,
            validators={"prescription": lambda p: None},
            proximity_window=20,
        )

        result = engine.process_document("prescription", "Aspirin ... 100mg")
        assert result.relation_mapping is not None
        # DOSAGE too far → orphan
        assert len(result.relation_mapping.orphans) == 1

    def test_empty_entity_list_skips_mapping(self):
        from src.pipeline.inference import (
            InferenceEngine,
            ModelBundle,
            ModelRegistry,
        )

        ner_model = MagicMock(spec=[])
        ner_model.return_value = {"entities": []}
        classifier = MagicMock(spec=[])
        classifier.return_value = {"document_type": "prescription"}

        registry = ModelRegistry({
            "prescription": ModelBundle(classifier=classifier, ner=ner_model),
        })
        engine = InferenceEngine(
            registry=registry,
            validators={"prescription": lambda p: None},
        )

        result = engine.process_document("prescription", "empty")
        assert result.relation_mapping is None


# ═══════════════════════════════════════════════════════════════════
# Pipeline __init__.py exports
# ═══════════════════════════════════════════════════════════════════


class TestExports:
    """Verify relation mapping is accessible via pipeline package."""

    def test_relation_mapper_importable(self):
        from src.pipeline import RelationMapper
        assert RelationMapper is not None

    def test_relation_mapping_result_importable(self):
        from src.pipeline import RelationMappingResult
        assert RelationMappingResult is not None

    def test_connect_entities_importable(self):
        from src.pipeline import connect_entities
        assert callable(connect_entities)

    def test_config_registry_importable(self):
        from src.pipeline import RELATION_CONFIG_REGISTRY
        assert "prescription" in RELATION_CONFIG_REGISTRY

    def test_get_relation_config_importable(self):
        from src.pipeline import get_relation_config
        assert callable(get_relation_config)

    def test_domain_configs_importable(self):
        from src.pipeline import (
            PRESCRIPTION_RELATIONS,
            RESULT_RELATIONS,
            CLINICAL_HISTORY_RELATIONS,
        )
        assert "MEDICATION" in PRESCRIPTION_RELATIONS
        assert "TEST_NAME" in RESULT_RELATIONS
        assert "DIAGNOSIS" in CLINICAL_HISTORY_RELATIONS


# ═══════════════════════════════════════════════════════════════════
# Edge cases & validation
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_empty_config_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            RelationMapper({})

    def test_entity_with_no_label(self):
        mapper = RelationMapper({"A": ["B"]}, keep_metadata=False)
        result = mapper.map_relations([{"text": "no label", "start": 0, "end": 8}])
        assert len(result.orphans) == 1

    def test_shared_dependent_across_anchors(self):
        """NOTES is a dependent of both MEDICATION and RADIOLOGY."""
        mapper = RelationMapper(PRESCRIPTION_RELATIONS, keep_metadata=False)
        entities = [
            _ent("MEDICATION", "Aspirin", 0),
            _ent("NOTES", "take with food", 10),
            _ent("RADIOLOGY", "Chest X-Ray", 30),
            _ent("NOTES", "compare with prior", 45),
        ]
        result = mapper.map_relations(entities)
        assert len(result.relations) == 2
        assert result.relations[0]["NOTES"] == "take with food"
        assert result.relations[1]["NOTES"] == "compare with prior"

    def test_multiple_values_same_dependent(self):
        """Two DOSAGE entities for same medication → auto-aggregated to list."""
        mapper = RelationMapper(PRESCRIPTION_RELATIONS, keep_metadata=False)
        entities = [
            _ent("MEDICATION", "Prednisone", 0),
            _ent("DOSAGE", "40mg", 12),
            _ent("DOSAGE", "20mg", 18),
        ]
        result = mapper.map_relations(entities)
        assert result.relations[0]["DOSAGE"] == ["40mg", "20mg"]
