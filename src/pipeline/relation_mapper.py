"""Utilities for wiring together entities extracted by a NER model.

The real project uses a large language model to identify individual
entities (patient names, medications, dosages, etc.) inside medical
documents.  The raw NER output is a flat list of spans, but the rest of the
pipeline expects structured relationships – for instance a medication entity
linked to its corresponding dosage and frequency.  The :mod:`relation_mapper`
module provides a lightweight, dependency free implementation that can be
used in tests to simulate this behaviour.

The entry point of the module is :class:`RelationMapper`.  It receives a
relation configuration describing which labels should be considered
"anchors" (typically main entities such as a medication name) and which
labels are dependent on them (dosage, frequency, duration, ...).  The mapper
then walks through the list of entities, associates each dependent with the
nearest compatible anchor, and finally returns a list of relations.  The
module is intentionally flexible and works with the slightly different
dictionary shapes produced by the HuggingFace pipelines or custom NER
models.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

Entity = Mapping[str, Any]
NormalizedEntity = Dict[str, Any]
Relation = Dict[str, Any]


@dataclass
class RelationMappingResult:
    """Container returned by :class:`RelationMapper`.

    Attributes
    ----------
    relations:
        A list of dictionaries describing the mapped relations.  Each
        dictionary contains the anchor label/text pair plus any dependent
        labels that were linked to it.
    orphans:
        Entities that could not be linked to any anchor according to the
        provided configuration.  Keeping track of them is useful in the real
        project to spot configuration mistakes during debugging, while tests
        can easily ignore the field if they do not need it.
    """

    relations: List[Relation]
    orphans: List[NormalizedEntity]

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the mapping result."""

        return {"relations": self.relations, "orphans": self.orphans}


class RelationMapper:
    """Map raw NER entities into structured relations.

    Parameters
    ----------
    relation_config:
        Mapping of anchor labels to the dependent labels that should be
        associated with them.  For example::

            {
                "MEDICATION": ["DOSAGE", "FREQUENCY", "DURATION"],
                "TEST_NAME": ["TEST_VALUE", "UNIT"],
            }

        In the configuration above, ``MEDICATION`` and ``TEST_NAME`` are the
        anchor labels.
    proximity_window:
        Maximum character distance allowed between an anchor and a dependent
        entity.  Use ``None`` (the default) to disable the constraint.
    aggregator:
        Strategy used to collapse multiple dependent values into a single
        output.  Accepted values are ``"auto"`` (default) to return a single
        string when only one value exists and a list otherwise, ``"list"`` to
        *always* return a list, ``"concat"`` to join values using
        ``join_token`` or a custom callable accepting ``(label, values)``.
    join_token:
        Token used when the ``"concat"`` aggregator is selected.
    keep_metadata:
        If ``True`` the mapper adds an ``"_metadata"`` section in every
        relation containing the normalised entities that were linked during
        the mapping step.  The metadata is useful for debugging but can be
        disabled to obtain a minimal output.
    """

    def __init__(
        self,
        relation_config: Mapping[str, Sequence[str]],
        *,
        proximity_window: Optional[int] = None,
        aggregator: str | Callable[[str, Sequence[str]], Any] = "auto",
        join_token: str = " ",
        keep_metadata: bool = True,
    ) -> None:
        if not relation_config:
            raise ValueError("relation_config cannot be empty")

        self._relation_config: Dict[str, Tuple[str, ...]] = {
            anchor: tuple(dependents)
            for anchor, dependents in relation_config.items()
        }
        self._anchors = tuple(self._relation_config.keys())
        self._dependent_to_anchor: Dict[str, Tuple[str, ...]] = defaultdict(tuple)
        for anchor, dependents in self._relation_config.items():
            for dep in dependents:
                existing = self._dependent_to_anchor.get(dep)
                if existing:
                    self._dependent_to_anchor[dep] = existing + (anchor,)
                else:
                    self._dependent_to_anchor[dep] = (anchor,)

        self._proximity_window = proximity_window
        self._keep_metadata = keep_metadata
        self._aggregator = self._build_aggregator(aggregator, join_token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def map_relations(self, entities: Sequence[Entity]) -> RelationMappingResult:
        """Connect entities according to the configured relations."""

        normalised_entities = [self._normalise_entity(entity) for entity in entities]
        normalised_entities.sort(key=lambda item: item["start"])

        relations: List[MutableMapping[str, Any]] = []
        orphans: List[NormalizedEntity] = []

        for entity in normalised_entities:
            label = entity["label"]

            if not label:
                orphans.append(entity)
                continue

            if label in self._anchors:
                relations.append(self._start_relation(entity))
                continue

            anchor_options = self._dependent_to_anchor.get(label, ())
            if not anchor_options:
                orphans.append(entity)
                continue

            relation = self._find_best_relation(relations, entity, anchor_options)
            if relation is None:
                orphans.append(entity)
                continue

            self._attach_entity(relation, entity)

        formatted_relations = [self._finalise_relation(relation) for relation in relations]

        return RelationMappingResult(relations=formatted_relations, orphans=orphans)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _build_aggregator(
        self,
        aggregator: str | Callable[[str, Sequence[str]], Any],
        join_token: str,
    ) -> Callable[[str, Sequence[str]], Any]:
        if callable(aggregator):
            return aggregator

        if aggregator not in {"auto", "list", "concat"}:
            raise ValueError("Invalid aggregator value")

        if aggregator == "list":
            return lambda label, values: list(values)

        if aggregator == "concat":
            return lambda label, values: join_token.join(values)

        # "auto" behaviour
        def _auto(label: str, values: Sequence[str]) -> Any:
            if not values:
                return None
            if len(values) == 1:
                return values[0]
            return list(values)

        return _auto

    def _normalise_entity(self, entity: Entity) -> NormalizedEntity:
        """Convert a raw entity into a predictable dictionary structure."""

        label = self._first_present(entity, ("label", "entity", "entity_group", "type"))
        text = self._first_present(entity, ("text", "word", "value"), default="", allow_empty=True)
        start = self._first_present(entity, ("start", "start_char", "begin"), default=0)
        end = self._first_present(entity, ("end", "end_char", "finish"), default=start)

        normalised = {
            "label": label,
            "text": text,
            "start": int(start),
            "end": int(end),
            "score": entity.get("score"),
            "source": entity,
        }

        return normalised

    @staticmethod
    def _first_present(
        entity: Mapping[str, Any],
        keys: Sequence[str],
        *,
        default: Any | None = None,
        allow_empty: bool = False,
    ) -> Any:
        for key in keys:
            if key not in entity:
                continue
            value = entity[key]
            if value is None:
                continue
            if not allow_empty and value == "":
                continue
            return value
        return default

    def _start_relation(self, entity: NormalizedEntity) -> MutableMapping[str, Any]:
        relation: MutableMapping[str, Any] = defaultdict(list)
        relation["_anchor_label"] = entity["label"]
        relation["_anchor_entity"] = entity
        relation[entity["label"]] = str(entity["text"])
        relation["_dependents"] = defaultdict(list)

        if self._keep_metadata:
            relation.setdefault("_metadata", {"anchor": entity, "dependents": []})

        return relation

    def _find_best_relation(
        self,
        relations: Sequence[MutableMapping[str, Any]],
        entity: NormalizedEntity,
        anchor_options: Sequence[str],
    ) -> Optional[MutableMapping[str, Any]]:
        best_relation: Optional[MutableMapping[str, Any]] = None
        best_distance: Optional[int] = None

        for relation in reversed(relations):
            anchor_label = relation.get("_anchor_label")
            if anchor_label not in anchor_options:
                continue

            anchor_entity = relation.get("_anchor_entity", {})
            anchor_end = anchor_entity.get("end", 0)
            distance = abs(entity["start"] - anchor_end)

            if self._proximity_window is not None and distance > self._proximity_window:
                continue

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_relation = relation

        return best_relation

    def _attach_entity(self, relation: MutableMapping[str, Any], entity: NormalizedEntity) -> None:
        label = entity["label"]
        relation["_dependents"][label].append(str(entity["text"]))

        if self._keep_metadata:
            metadata = relation.setdefault("_metadata", {"anchor": relation.get("_anchor_entity"), "dependents": []})
            metadata.setdefault("dependents", []).append(entity)

    def _finalise_relation(self, relation: MutableMapping[str, Any]) -> Relation:
        result: Relation = {relation["_anchor_label"]: relation[relation["_anchor_label"]]}

        for label, values in relation.get("_dependents", {}).items():
            aggregated = self._aggregator(label, values)
            if aggregated is None:
                continue
            result[label] = aggregated

        if self._keep_metadata:
            metadata = relation.get("_metadata")
            if metadata:
                result["_metadata"] = metadata

        return result


def connect_entities(
    entities: Sequence[Entity],
    relation_config: Mapping[str, Sequence[str]],
    *,
    proximity_window: Optional[int] = None,
    aggregator: str | Callable[[str, Sequence[str]], Any] = "auto",
    join_token: str = " ",
    keep_metadata: bool = True,
) -> RelationMappingResult:
    """Convenience wrapper around :class:`RelationMapper`.

    The helper mirrors the constructor of :class:`RelationMapper` and returns
    the same :class:`RelationMappingResult` structure.  It is handy when
    callers do not need to keep a mapper instance around.
    """

    mapper = RelationMapper(
        relation_config,
        proximity_window=proximity_window,
        aggregator=aggregator,
        join_token=join_token,
        keep_metadata=keep_metadata,
    )
    return mapper.map_relations(entities)


__all__ = ["RelationMapper", "RelationMappingResult", "connect_entities"]

