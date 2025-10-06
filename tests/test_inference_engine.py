import pytest

from src.pipeline.inference import InferenceEngine, InferenceResult, ModelBundle, ModelRegistry
from src.pipeline.validation.schemas import ResultSchema
from src.pipeline.validation.prescription_schema import Prescription


class DummyClassifier:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        return dict(self.payload)


class DummyNER:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def extract(self, text):
        self.calls.append(text)
        return dict(self.payload)


@pytest.fixture
def inference_engine():
    result_classifier = DummyClassifier(
        {
            "patient_name": "John Doe",
            "exam_type": "Chest X-Ray",
            "exam_date": "2024-01-01",
            "professional": "Dr. Smith",
            "institution": "General Hospital",
        }
    )
    result_ner = DummyNER({"findings": "No acute cardiopulmonary disease."})

    prescription_classifier = DummyClassifier(
        {
            "patient_name": "Jane Roe",
            "patient_id": "ABC123",
            "date": "2024-02-02",
            "doctor_name": "Dr. House",
            "institution": "General Hospital",
        }
    )
    prescription_ner = DummyNER(
        {
            "items": [
                {
                    "type": "medicine",
                    "name": "Ibuprofen",
                    "dosage": "200mg",
                    "frequency": "twice a day",
                }
            ]
        }
    )

    registry = ModelRegistry(
        {
            "result": ModelBundle(classifier=result_classifier, ner=result_ner),
            "prescription": ModelBundle(
                classifier=prescription_classifier, ner=prescription_ner
            ),
        }
    )

    engine = InferenceEngine(registry)
    return {
        "engine": engine,
        "result_classifier": result_classifier,
        "result_ner": result_ner,
        "prescription_classifier": prescription_classifier,
        "prescription_ner": prescription_ner,
    }


def test_process_result_document_returns_validated_schema(inference_engine):
    raw_text = "   PATIENT: JOHN DOE\nFindings: Normal study   "
    engine = inference_engine["engine"]
    result = engine.process_document("result", raw_text)

    assert isinstance(result, InferenceResult)
    assert result.preprocessed_text == "patient: john doe findings: normal study"
    assert isinstance(result.validated_data, ResultSchema)

    expected = {
        "patient_name": "John Doe",
        "exam_type": "Chest X-Ray",
        "exam_date": "2024-01-01",
        "professional": "Dr. Smith",
        "institution": "General Hospital",
        "findings": "No acute cardiopulmonary disease.",
    }
    data = result.as_dict()
    for key, value in expected.items():
        assert data[key] == value

    dummy_classifier = inference_engine["result_classifier"]
    dummy_ner = inference_engine["result_ner"]
    assert dummy_classifier.calls == [result.preprocessed_text]
    assert dummy_ner.calls == [result.preprocessed_text]


def test_process_prescription_document_uses_prescription_validator(inference_engine):
    raw_text = "Jane Roe prescribed Ibuprofen"
    engine = inference_engine["engine"]
    result = engine.process_document("prescription", raw_text)

    assert isinstance(result.validated_data, Prescription)
    data = result.as_dict()
    assert data["doctor_name"] == "Dr. House"
    assert data["items"][0]["name"] == "Ibuprofen"


def test_unknown_document_type_raises_error():
    engine = InferenceEngine(ModelRegistry())
    with pytest.raises(ValueError):
        engine.process_document("unknown", "text")
