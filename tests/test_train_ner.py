"""Tests for src.pipeline.train_ner – NER training, eval, and persistence (#17)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.train_ner import (
    EvalReport,
    FieldScore,
    NERTrainer,
    TrainingExample,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures – annotated examples
# ═══════════════════════════════════════════════════════════════════

PRESCRIPTION_TEXT = """\
Patient Name: Maria Garcia
Patient ID: 45012
Date of Birth: 1985-03-15
Date of Prescription: 2025-06-10
Doctor: Dr. Carlos Rodriguez
Clinic: Central Medical Center

Prescription:
Amoxicillin 500mg, 3 times daily for 7 days.
"""

LAB_RESULT_TEXT = """\
Patient Name: Anthony Harper
Patient ID: 32373
Date of Birth: 1965-05-12
Exam Date: 2025-01-12
Clinic: Foster-Bailey

Test Results:
- Glucose Test: 95.50 (Ref: 70-100)
- Hemoglobin: 14.20 (Ref: 12-17)

Summary: Normal results.
"""

CLINICAL_HISTORY_TEXT = """\
Patient Name: Shelby Brown
Patient ID: 13098
Date of Birth: 1931-09-29
Clinic: Garrett-Wagner

Annotations:
- 2025-02-22: Persistent headache reported.
- 2025-01-06: Initial consultation for migraines.
"""


@pytest.fixture
def examples() -> list[TrainingExample]:
    return [
        TrainingExample(
            text=PRESCRIPTION_TEXT,
            document_type="prescription",
            annotations={
                "patient_name": "Maria Garcia",
                "doctor_name": "Dr. Carlos Rodriguez",
                "institution": "Central Medical Center",
            },
        ),
        TrainingExample(
            text=LAB_RESULT_TEXT,
            document_type="result",
            annotations={
                "patient_name": "Anthony Harper",
                "institution": "Foster-Bailey",
            },
        ),
        TrainingExample(
            text=CLINICAL_HISTORY_TEXT,
            document_type="clinical_history",
            annotations={
                "patient_name": "Shelby Brown",
                "institution": "Garrett-Wagner",
                "consultation_date": "2025-02-22",
            },
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
# TrainingExample
# ═══════════════════════════════════════════════════════════════════

class TestTrainingExample:
    def test_round_trip(self):
        ex = TrainingExample(
            text="hello",
            document_type="result",
            annotations={"patient_name": "Test"},
            source="unit_test",
        )
        d = ex.to_dict()
        restored = TrainingExample.from_dict(d)
        assert restored.text == "hello"
        assert restored.annotations["patient_name"] == "Test"
        assert restored.source == "unit_test"

    def test_from_dict_missing_source(self):
        ex = TrainingExample.from_dict({
            "text": "x",
            "document_type": "result",
            "annotations": {},
        })
        assert ex.source is None


# ═══════════════════════════════════════════════════════════════════
# FieldScore
# ═══════════════════════════════════════════════════════════════════

class TestFieldScore:
    def test_perfect_score(self):
        fs = FieldScore("name", true_positives=10, false_positives=0, false_negatives=0)
        assert fs.precision == 1.0
        assert fs.recall == 1.0
        assert fs.f1 == 1.0

    def test_zero_score(self):
        fs = FieldScore("name", true_positives=0, false_positives=0, false_negatives=0)
        assert fs.precision == 0.0
        assert fs.recall == 0.0
        assert fs.f1 == 0.0

    def test_partial_score(self):
        fs = FieldScore("name", true_positives=5, false_positives=5, false_negatives=0)
        assert fs.precision == 0.5
        assert fs.recall == 1.0
        assert 0.6 < fs.f1 < 0.7


# ═══════════════════════════════════════════════════════════════════
# EvalReport
# ═══════════════════════════════════════════════════════════════════

class TestEvalReport:
    def test_macro_f1(self):
        report = EvalReport(
            document_type="all",
            total_examples=2,
            classification_accuracy=1.0,
            field_scores={
                "a": FieldScore("a", true_positives=10),
                "b": FieldScore("b", true_positives=0, false_negatives=10),
            },
        )
        # a.f1 = 1.0, b.f1 = 0.0 → macro = 0.5
        assert report.macro_f1 == 0.5

    def test_summary_is_string(self):
        report = EvalReport(
            document_type="result",
            total_examples=1,
            classification_accuracy=1.0,
        )
        text = report.summary()
        assert "result" in text
        assert "Examples: 1" in text


# ═══════════════════════════════════════════════════════════════════
# NERTrainer – evaluate
# ═══════════════════════════════════════════════════════════════════

class TestNERTrainerEvaluate:
    def test_evaluate_all(self, examples):
        trainer = NERTrainer()
        report = trainer.evaluate(examples)
        assert report.total_examples == 3
        assert report.classification_accuracy > 0

    def test_evaluate_by_type(self, examples):
        trainer = NERTrainer()
        report = trainer.evaluate(examples, document_type="prescription")
        assert report.total_examples == 1
        assert report.document_type == "prescription"

    def test_patient_name_scores_high(self, examples):
        trainer = NERTrainer()
        report = trainer.evaluate(examples)
        fs = report.field_scores.get("patient_name")
        assert fs is not None
        assert fs.f1 > 0.8  # rule-based should nail patient_name

    def test_empty_examples(self):
        trainer = NERTrainer()
        report = trainer.evaluate([])
        assert report.total_examples == 0
        assert report.classification_accuracy == 0.0

    def test_classification_accuracy_all_correct(self, examples):
        trainer = NERTrainer()
        report = trainer.evaluate(examples)
        assert report.classification_accuracy == 1.0

    def test_field_scores_keys_match_annotations(self, examples):
        trainer = NERTrainer()
        report = trainer.evaluate(examples)
        # Every annotated field should appear in scores
        for ex in examples:
            for field_name in ex.annotations:
                assert field_name in report.field_scores


class TestNERTrainerValuesMatch:
    def test_string_case_insensitive(self):
        assert NERTrainer._values_match("Hello", "hello")

    def test_string_strips_whitespace(self):
        assert NERTrainer._values_match("  Test  ", "Test")

    def test_none_matches_none(self):
        assert NERTrainer._values_match(None, None)

    def test_none_does_not_match_string(self):
        assert not NERTrainer._values_match(None, "x")

    def test_list_match(self):
        assert NERTrainer._values_match(["a", "b"], ["A", "B"])

    def test_list_length_mismatch(self):
        assert not NERTrainer._values_match(["a"], ["a", "b"])


# ═══════════════════════════════════════════════════════════════════
# NERTrainer – retrain classifier
# ═══════════════════════════════════════════════════════════════════

class TestNERTrainerRetrain:
    def test_retrain_returns_accuracies(self, examples):
        trainer = NERTrainer()
        accuracies = trainer.retrain_classifier(examples, epochs=3)
        assert isinstance(accuracies, dict)
        assert len(accuracies) >= 3
        for acc in accuracies.values():
            assert 0.0 <= acc <= 1.0

    def test_retrain_improves_or_maintains_accuracy(self, examples):
        trainer = NERTrainer()
        # Pre-training accuracy
        pre_report = trainer.evaluate(examples)
        # Retrain
        trainer.retrain_classifier(examples, epochs=5)
        # Post-training accuracy
        post_report = trainer.evaluate(examples)
        assert post_report.classification_accuracy >= pre_report.classification_accuracy


# ═══════════════════════════════════════════════════════════════════
# NERTrainer – persistence
# ═══════════════════════════════════════════════════════════════════

class TestNERTrainerPersistence:
    def test_save_and_load(self, tmp_path, examples):
        trainer = NERTrainer()
        trainer.retrain_classifier(examples, epochs=2)
        save_path = tmp_path / "model.json"
        trainer.save(save_path)

        assert save_path.exists()
        data = json.loads(save_path.read_text())
        assert "classifier_weights" in data
        assert "registered_types" in data

        loaded = NERTrainer.load(save_path)
        assert loaded._classifier_weights is not None

    def test_save_creates_parent_dirs(self, tmp_path):
        trainer = NERTrainer()
        deep_path = tmp_path / "a" / "b" / "model.json"
        trainer.save(deep_path)
        assert deep_path.exists()

    def test_load_patches_classifier(self, tmp_path, examples):
        """Loaded weights should affect classification."""
        trainer = NERTrainer()
        trainer.retrain_classifier(examples, epochs=3)
        path = tmp_path / "model.json"
        trainer.save(path)

        # Load and verify classification still works
        loaded = NERTrainer.load(path)
        report = loaded.evaluate(examples)
        assert report.classification_accuracy > 0


# ═══════════════════════════════════════════════════════════════════
# NERTrainer – example I/O
# ═══════════════════════════════════════════════════════════════════

class TestExampleIO:
    def test_save_and_load_examples(self, tmp_path, examples):
        path = tmp_path / "examples.json"
        NERTrainer.save_examples(examples, path)

        loaded = NERTrainer.load_examples(path)
        assert len(loaded) == len(examples)
        assert loaded[0].document_type == examples[0].document_type
        assert loaded[0].annotations == examples[0].annotations

    def test_saved_json_is_valid(self, tmp_path, examples):
        path = tmp_path / "examples.json"
        NERTrainer.save_examples(examples, path)
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 3


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

class TestTrainNERCLI:
    def test_cli_evaluate(self, tmp_path, examples):
        from src.pipeline.train_ner import main as cli_main

        data_path = tmp_path / "data.json"
        NERTrainer.save_examples(examples, data_path)

        code = cli_main(["--data", str(data_path)])
        assert code == 0

    def test_cli_retrain_and_save(self, tmp_path, examples):
        from src.pipeline.train_ner import main as cli_main

        data_path = tmp_path / "data.json"
        save_path = tmp_path / "model.json"
        NERTrainer.save_examples(examples, data_path)

        code = cli_main([
            "--data", str(data_path),
            "--retrain",
            "--epochs", "2",
            "--save", str(save_path),
        ])
        assert code == 0
        assert save_path.exists()

    def test_cli_load_and_evaluate(self, tmp_path, examples):
        from src.pipeline.train_ner import main as cli_main

        data_path = tmp_path / "data.json"
        model_path = tmp_path / "model.json"
        NERTrainer.save_examples(examples, data_path)

        # First train and save
        trainer = NERTrainer()
        trainer.retrain_classifier(examples, epochs=2)
        trainer.save(model_path)

        # Then load and evaluate
        code = cli_main(["--data", str(data_path), "--load", str(model_path)])
        assert code == 0

    def test_cli_filter_by_type(self, tmp_path, examples):
        from src.pipeline.train_ner import main as cli_main

        data_path = tmp_path / "data.json"
        NERTrainer.save_examples(examples, data_path)

        code = cli_main(["--data", str(data_path), "--type", "result"])
        assert code == 0
