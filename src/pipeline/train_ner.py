"""NER training, evaluation, and model persistence (#17).

This module provides tooling to:

1. **Evaluate** the existing rule-based extractors against annotated examples,
   computing per-field precision, recall, and F1.
2. **Retrain** the keyword-based classifier by tuning keyword weights from
   labelled data.
3. **Persist** and **load** trained artefacts (classifier weights, extractor
   configs) as JSON so they survive restarts.

Design
------
The current NER pipeline is rule-based (regex extractors + keyword classifier).
This module does *not* replace them with a neural model — instead it wraps
them in a train/eval/persist loop so performance can be measured, regressions
caught, and weights optimised from real data.

Usage
-----
    from src.pipeline.train_ner import NERTrainer, TrainingExample

    examples = NERTrainer.load_examples("data/annotated.json")
    trainer = NERTrainer()
    report = trainer.evaluate(examples)
    print(report.summary())

    trainer.retrain_classifier(examples)
    trainer.save("models/ner_v1.json")
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from src.pipeline.extractors.document_classifier import DocumentClassifier, DOCUMENT_TYPES
from src.pipeline.inference import InferenceEngine, create_default_engine

logger = logging.getLogger(__name__)

DocType = Literal["prescription", "result", "clinical_history"]


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrainingExample:
    """A single annotated document for training / evaluation.

    Parameters
    ----------
    text : str
        Raw document text (as it would arrive from extraction).
    document_type : str
        Ground-truth document type label.
    annotations : dict
        Ground-truth field→value mapping that the NER extractor should
        produce.  Only fields present here are scored — extra fields
        returned by the extractor are ignored (not penalised).
    """

    text: str
    document_type: str
    annotations: Dict[str, Any]
    source: Optional[str] = None  # optional provenance tag

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingExample":
        return cls(
            text=d["text"],
            document_type=d["document_type"],
            annotations=d["annotations"],
            source=d.get("source"),
        )


@dataclass
class FieldScore:
    """Precision / recall / F1 for a single field across all examples."""

    field_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalReport:
    """Aggregated evaluation report for an extractor."""

    document_type: str
    total_examples: int
    classification_accuracy: float
    field_scores: Dict[str, FieldScore] = field(default_factory=dict)

    @property
    def macro_f1(self) -> float:
        """Macro-averaged F1 across all scored fields."""
        scores = [fs.f1 for fs in self.field_scores.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            f"=== {self.document_type} ===",
            f"Examples: {self.total_examples}",
            f"Classification accuracy: {self.classification_accuracy:.1%}",
            f"{'Field':<25} {'Prec':>6} {'Rec':>6} {'F1':>6}",
            "-" * 45,
        ]
        for name, fs in sorted(self.field_scores.items()):
            lines.append(
                f"{name:<25} {fs.precision:>6.1%} {fs.recall:>6.1%} {fs.f1:>6.1%}"
            )
        lines.append(f"{'Macro F1':<25} {'':>6} {'':>6} {self.macro_f1:>6.1%}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class NERTrainer:
    """Train, evaluate, and persist the NER pipeline.

    Parameters
    ----------
    engine : InferenceEngine, optional
        Pre-built engine.  When *None* the default rule-based engine is
        used.
    """

    def __init__(self, engine: Optional[InferenceEngine] = None) -> None:
        self._engine = engine or create_default_engine()
        self._classifier_weights: Optional[Dict[str, Dict[str, float]]] = None

    # ── I/O ──────────────────────────────────────────────────────

    @staticmethod
    def load_examples(path: str | Path) -> List[TrainingExample]:
        """Load annotated examples from a JSON file.

        Expected format::

            [
              {
                "text": "Patient Name: ...",
                "document_type": "prescription",
                "annotations": {"patient_name": "Maria Garcia", ...}
              },
              ...
            ]
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [TrainingExample.from_dict(d) for d in raw]

    @staticmethod
    def save_examples(examples: Sequence[TrainingExample], path: str | Path) -> None:
        """Persist training examples as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in examples], f, indent=2, ensure_ascii=False)

    # ── Evaluation ───────────────────────────────────────────────

    def evaluate(
        self,
        examples: Sequence[TrainingExample],
        document_type: Optional[str] = None,
    ) -> EvalReport:
        """Evaluate the current pipeline against annotated examples.

        Parameters
        ----------
        examples :
            Annotated training / test examples.
        document_type :
            If given, only evaluate examples of this type.  Otherwise
            evaluates all examples whose ``document_type`` matches a
            registered extractor.

        Returns
        -------
        EvalReport
            Per-field precision, recall, F1, and classification accuracy.
        """
        subset = [
            e for e in examples
            if document_type is None or e.document_type == document_type
        ]
        if not subset:
            return EvalReport(
                document_type=document_type or "all",
                total_examples=0,
                classification_accuracy=0.0,
            )

        correct_classifications = 0
        field_scores: Dict[str, FieldScore] = defaultdict(lambda: FieldScore(field_name=""))

        for ex in subset:
            # Classification check
            predicted_type = self._engine.classify(ex.text)
            if predicted_type == ex.document_type:
                correct_classifications += 1

            # NER extraction
            try:
                ir = self._engine.process_document(ex.document_type, ex.text)
                predicted = ir.as_dict()
            except Exception as exc:
                logger.warning("Extraction failed for example: %s", exc)
                predicted = {}

            # Score each annotated field
            for field_name, expected_value in ex.annotations.items():
                if field_name not in field_scores:
                    field_scores[field_name] = FieldScore(field_name=field_name)

                fs = field_scores[field_name]
                pred_value = predicted.get(field_name)

                match = self._values_match(expected_value, pred_value)
                if expected_value is not None and match:
                    fs.true_positives += 1
                elif expected_value is not None and not match:
                    fs.false_negatives += 1
                    if pred_value is not None:
                        fs.false_positives += 1
                elif expected_value is None and pred_value is not None:
                    fs.false_positives += 1

        accuracy = correct_classifications / len(subset) if subset else 0.0
        return EvalReport(
            document_type=document_type or "all",
            total_examples=len(subset),
            classification_accuracy=accuracy,
            field_scores=dict(field_scores),
        )

    @staticmethod
    def _values_match(expected: Any, predicted: Any) -> bool:
        """Flexible comparison: string equality after strip/lower, or deep equal."""
        if expected is None and predicted is None:
            return True
        if expected is None or predicted is None:
            return False
        if isinstance(expected, str) and isinstance(predicted, str):
            return expected.strip().lower() == predicted.strip().lower()
        if isinstance(expected, list) and isinstance(predicted, list):
            if len(expected) != len(predicted):
                return False
            return all(
                NERTrainer._values_match(e, p)
                for e, p in zip(expected, predicted)
            )
        return expected == predicted

    # ── Classifier retraining ────────────────────────────────────

    def retrain_classifier(
        self,
        examples: Sequence[TrainingExample],
        learning_rate: float = 0.5,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """Retrain keyword weights from labelled classification examples.

        Uses a simple frequency-based reweighting: keywords that appear
        more often in their target class get boosted, those that appear
        in wrong classes get penalised.

        Parameters
        ----------
        examples :
            Annotated examples with ``document_type`` labels.
        learning_rate :
            Step size for weight adjustments (0–1).
        epochs :
            Number of passes over the data.

        Returns
        -------
        dict
            Mapping of ``{doc_type: accuracy}`` after retraining.
        """
        # Start from current weights
        weights: Dict[str, Dict[str, float]] = {}
        for doc_type, kw_table in DOCUMENT_TYPES.items():
            weights[doc_type] = dict(kw_table)

        # Collect keyword frequencies per true class
        for _epoch in range(epochs):
            adjustments: Dict[str, Dict[str, float]] = {
                dt: defaultdict(float) for dt in weights
            }

            for ex in examples:
                text_lower = ex.text.lower()
                true_type = ex.document_type
                if true_type not in weights:
                    continue

                # For each keyword in the true class, boost if present
                for kw in weights[true_type]:
                    if kw in text_lower:
                        adjustments[true_type][kw] += learning_rate

                # For keywords in other classes, penalise if they triggered
                for other_type in weights:
                    if other_type == true_type:
                        continue
                    for kw in weights[other_type]:
                        if kw in text_lower:
                            adjustments[other_type][kw] -= learning_rate * 0.3

            # Apply adjustments
            for doc_type in weights:
                for kw in weights[doc_type]:
                    delta = adjustments[doc_type].get(kw, 0.0)
                    weights[doc_type][kw] = max(0.1, weights[doc_type][kw] + delta)

        self._classifier_weights = weights

        # Patch the live classifier
        for doc_type, kw_table in weights.items():
            DOCUMENT_TYPES[doc_type].update(kw_table)

        # Measure post-training accuracy
        accuracies = self._measure_classification_accuracy(examples)
        logger.info("Classifier retrained over %d epochs. Accuracies: %s", epochs, accuracies)
        return accuracies

    def _measure_classification_accuracy(
        self,
        examples: Sequence[TrainingExample],
    ) -> Dict[str, float]:
        """Return per-type classification accuracy."""
        counts: Dict[str, int] = defaultdict(int)
        correct: Dict[str, int] = defaultdict(int)

        classifier = DocumentClassifier()
        for ex in examples:
            counts[ex.document_type] += 1
            if classifier.classify(ex.text) == ex.document_type:
                correct[ex.document_type] += 1

        return {
            dt: correct[dt] / counts[dt] if counts[dt] else 0.0
            for dt in counts
        }

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist trained artefacts (classifier weights) to JSON."""
        payload: Dict[str, Any] = {
            "classifier_weights": self._classifier_weights or {
                dt: dict(kw) for dt, kw in DOCUMENT_TYPES.items()
            },
            "registered_types": self._engine.registered_types,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("NER artefacts saved to %s", p)

    @classmethod
    def load(cls, path: str | Path) -> "NERTrainer":
        """Load a trainer with previously saved classifier weights.

        The loaded weights are patched into the global ``DOCUMENT_TYPES``
        keyword tables so subsequent classifier instances use them.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        weights = payload.get("classifier_weights", {})
        for doc_type, kw_table in weights.items():
            if doc_type in DOCUMENT_TYPES:
                DOCUMENT_TYPES[doc_type].update(kw_table)

        trainer = cls()
        trainer._classifier_weights = weights
        logger.info("NER artefacts loaded from %s", path)
        return trainer


# ═══════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    """Standalone CLI for training / evaluating the NER pipeline.

    Usage::

        python -m src.pipeline.train_ner --data data/annotated.json
        python -m src.pipeline.train_ner --data data/annotated.json --retrain --save models/v1.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="DocIQ NER trainer / evaluator")
    parser.add_argument("--data", required=True, help="Path to annotated examples JSON.")
    parser.add_argument("--type", default=None, help="Filter to a single document type.")
    parser.add_argument("--retrain", action="store_true", help="Retrain classifier weights.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default 10).")
    parser.add_argument("--save", default=None, help="Path to save trained artefacts.")
    parser.add_argument("--load", default=None, help="Path to load pre-trained artefacts.")
    args = parser.parse_args(argv)

    from src.logging_config import setup_logging
    setup_logging(level="INFO")

    # Load or create trainer
    if args.load:
        trainer = NERTrainer.load(args.load)
    else:
        trainer = NERTrainer()

    examples = NERTrainer.load_examples(args.data)
    logger.info("Loaded %d examples from %s", len(examples), args.data)

    # Evaluate
    report = trainer.evaluate(examples, document_type=args.type)
    print(report.summary())

    # Optionally retrain
    if args.retrain:
        accuracies = trainer.retrain_classifier(examples, epochs=args.epochs)
        print(f"\nPost-training accuracies: {accuracies}")

        # Re-evaluate after training
        report = trainer.evaluate(examples, document_type=args.type)
        print(f"\n{report.summary()}")

    # Save
    if args.save:
        trainer.save(args.save)
        print(f"\nArtefacts saved to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
