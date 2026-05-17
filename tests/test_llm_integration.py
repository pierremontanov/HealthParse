"""Tests for LLM client, merge logic, and pipeline integration.

All tests mock the Ollama HTTP endpoint so they run without a real
LLM server.  The test suite covers:

- OllamaClient: JSON parsing, retries, error handling
- Merge logic: field-by-field merging, items list handling
- Pipeline integration: LLM enabled/disabled, fallback on failure
- Exception hierarchy: LLMError subclasses
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.pipeline.inference import InferenceEngine, _is_empty, _LLM_SKIP_FIELDS
from src.pipeline.llm_client import OllamaClient, LLMClassifyResult
from src.pipeline.exceptions import (
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
)


# ═══════════════════════════════════════════════════════════════════
# _is_empty helper
# ═══════════════════════════════════════════════════════════════════


class TestIsEmpty:
    def test_none_is_empty(self):
        assert _is_empty(None) is True

    def test_empty_string_is_empty(self):
        assert _is_empty("") is True

    def test_whitespace_string_is_empty(self):
        assert _is_empty("   ") is True

    def test_empty_list_is_empty(self):
        assert _is_empty([]) is True

    def test_nonempty_string_is_not_empty(self):
        assert _is_empty("hello") is False

    def test_zero_is_not_empty(self):
        assert _is_empty(0) is False

    def test_nonempty_list_is_not_empty(self):
        assert _is_empty(["a"]) is False

    def test_false_is_not_empty(self):
        assert _is_empty(False) is False


# ═══════════════════════════════════════════════════════════════════
# OllamaClient — JSON parsing
# ═══════════════════════════════════════════════════════════════════


class TestJsonParsing:
    def test_plain_json(self):
        result = OllamaClient._parse_json('{"type": "prescription"}')
        assert result == {"type": "prescription"}

    def test_json_with_whitespace(self):
        result = OllamaClient._parse_json('  \n {"type": "result"} \n ')
        assert result == {"type": "result"}

    def test_json_with_markdown_fences(self):
        text = '```json\n{"type": "clinical_history"}\n```'
        result = OllamaClient._parse_json(text)
        assert result == {"type": "clinical_history"}

    def test_json_with_plain_fences(self):
        text = '```\n{"type": "result"}\n```'
        result = OllamaClient._parse_json(text)
        assert result == {"type": "result"}

    def test_invalid_json_raises_response_error(self):
        with pytest.raises(LLMResponseError):
            OllamaClient._parse_json("not json at all")

    def test_empty_string_raises_response_error(self):
        with pytest.raises(LLMResponseError):
            OllamaClient._parse_json("")


# ═══════════════════════════════════════════════════════════════════
# OllamaClient — classify and extract
# ═══════════════════════════════════════════════════════════════════


class TestOllamaClientClassify:
    @patch("src.pipeline.llm_client.requests.post")
    def test_classify_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "response": '{"document_type": "prescription", "confidence": 0.95, "language": "es"}'
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        client = OllamaClient(endpoint="http://fake:11434")
        result = client.classify("Paciente: Juan Garcia")

        assert isinstance(result, LLMClassifyResult)
        assert result.document_type == "prescription"
        assert result.confidence == 0.95
        assert result.language == "es"

    @patch("src.pipeline.llm_client.requests.post")
    def test_classify_invalid_json(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "I cannot classify this document."},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        client = OllamaClient(endpoint="http://fake:11434", retries=0)
        with pytest.raises(LLMResponseError):
            client.classify("some text")


class TestOllamaClientExtract:
    @patch("src.pipeline.llm_client.requests.post")
    def test_extract_success(self, mock_post):
        llm_response = {
            "patient_name": "Maria Garcia",
            "patient_id": "12345678",
            "age": 45,
            "sex": "F",
            "assessment": "Gastritis cronica",
            "doctor_name": "Dr. Lopez",
        }
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": json.dumps(llm_response)},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        client = OllamaClient(endpoint="http://fake:11434")
        result = client.extract("clinical_history", "some text")

        assert result["patient_name"] == "Maria Garcia"
        assert result["age"] == 45

    @patch("src.pipeline.llm_client.requests.post")
    def test_extract_prescription_with_items(self, mock_post):
        llm_response = {
            "patient_name": "Carlos Ruiz",
            "items": [
                {"type": "medicine", "name": "Amoxicilina", "dosage": "500mg"},
                {"type": "lab_test", "name": "Hemograma"},
            ],
        }
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": json.dumps(llm_response)},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        client = OllamaClient(endpoint="http://fake:11434")
        result = client.extract("prescription", "some text")

        assert len(result["items"]) == 2
        assert result["items"][0]["name"] == "Amoxicilina"


# ═══════════════════════════════════════════════════════════════════
# OllamaClient — error handling
# ═══════════════════════════════════════════════════════════════════


class TestOllamaClientErrors:
    @patch("src.pipeline.llm_client.requests.post")
    def test_connection_refused(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        client = OllamaClient(endpoint="http://fake:11434", retries=0)
        with pytest.raises(LLMConnectionError) as exc_info:
            client.classify("text")
        assert "fake:11434" in str(exc_info.value)

    @patch("src.pipeline.llm_client.requests.post")
    def test_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout("timed out")

        client = OllamaClient(endpoint="http://fake:11434", timeout=5, retries=0)
        with pytest.raises(LLMTimeoutError) as exc_info:
            client.classify("text")
        assert "5s" in str(exc_info.value)

    @patch("src.pipeline.llm_client.requests.post")
    def test_http_500_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        client = OllamaClient(endpoint="http://fake:11434", retries=0)
        with pytest.raises(LLMError):
            client.classify("text")

    @patch("src.pipeline.llm_client.requests.post")
    @patch("src.pipeline.llm_client.time.sleep")
    def test_retry_then_succeed(self, mock_sleep, mock_post):
        """First call fails, second succeeds."""
        fail_response = MagicMock()
        fail_response.raise_for_status.side_effect = requests.HTTPError("503")

        success_response = MagicMock(
            status_code=200,
            json=lambda: {
                "response": '{"document_type": "result", "confidence": 0.9, "language": "en"}'
            },
        )
        success_response.raise_for_status = MagicMock()

        mock_post.side_effect = [fail_response, success_response]

        client = OllamaClient(endpoint="http://fake:11434", retries=1)
        result = client.classify("text")

        assert result.document_type == "result"
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.pipeline.llm_client.requests.get")
    def test_is_available_true(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        client = OllamaClient(endpoint="http://fake:11434")
        assert client.is_available() is True

    @patch("src.pipeline.llm_client.requests.get")
    def test_is_available_false(self, mock_get):
        mock_get.side_effect = requests.ConnectionError()
        client = OllamaClient(endpoint="http://fake:11434")
        assert client.is_available() is False


# ═══════════════════════════════════════════════════════════════════
# OllamaClient — text truncation
# ═══════════════════════════════════════════════════════════════════


class TestTextTruncation:
    @patch("src.pipeline.llm_client.requests.post")
    def test_long_text_truncated_to_3000(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "response": '{"document_type": "result", "confidence": 0.5, "language": "es"}'
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        client = OllamaClient(endpoint="http://fake:11434")
        long_text = "A" * 5000
        client.classify(long_text)

        # Check the prompt sent to Ollama was truncated
        call_args = mock_post.call_args
        prompt = call_args[1]["json"]["prompt"] if "json" in call_args[1] else call_args[0][0]
        # "classify: " prefix + 3000 chars
        assert len(prompt) <= 3000 + 20


# ═══════════════════════════════════════════════════════════════════
# Merge logic — _merge_llm_fields
# ═══════════════════════════════════════════════════════════════════


class TestMergeLlmFields:
    def test_llm_fills_none_fields(self):
        rule = {"patient_name": "Juan", "patient_id": None, "age": None}
        llm = {"patient_name": "Juan Carlos", "patient_id": "123456", "age": 45}

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert rule["patient_name"] == "Juan"  # rule-based wins
        assert rule["patient_id"] == "123456"  # LLM fills
        assert rule["age"] == 45  # LLM fills
        assert "patient_name" not in fields
        assert "patient_id" in fields
        assert "age" in fields

    def test_llm_fills_empty_string(self):
        rule = {"institution": "", "doctor_name": "   "}
        llm = {"institution": "Clinica X", "doctor_name": "Dr. Lopez"}

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert rule["institution"] == "Clinica X"
        assert rule["doctor_name"] == "Dr. Lopez"

    def test_skip_fields_never_overwritten(self):
        rule = {"raw_text": "original", "document_type": "result"}
        llm = {"raw_text": "replaced", "document_type": "prescription"}

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert rule["raw_text"] == "original"
        assert rule["document_type"] == "result"
        assert fields == []

    def test_rule_based_value_preserved(self):
        rule = {"patient_name": "Juan", "age": 30}
        llm = {"patient_name": "Juan Carlos", "age": 45}

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert rule["patient_name"] == "Juan"
        assert rule["age"] == 30
        assert fields == []

    def test_llm_skips_unknown_field(self):
        """Fields not in rule-based output are ignored to avoid schema violations."""
        rule = {"patient_name": "Juan"}
        llm = {"patient_name": "Juan", "institution": "Hospital ABC"}

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert "institution" not in rule  # not added
        assert fields == []

    def test_items_empty_rule_uses_llm(self):
        rule = {"items": []}
        llm = {"items": [{"type": "medicine", "name": "Amoxicilina"}]}

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert len(rule["items"]) == 1
        assert "items" in fields

    def test_items_none_rule_uses_llm(self):
        rule = {"items": None}
        llm = {"items": [{"type": "medicine", "name": "Amoxicilina"}]}

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert rule["items"] is not None
        assert "items" in fields

    def test_items_llm_has_more(self):
        rule = {"items": [{"type": "medicine", "name": "Drug A"}]}
        llm = {
            "items": [
                {"type": "medicine", "name": "Drug A"},
                {"type": "lab_test", "name": "CBC"},
            ]
        }

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert len(rule["items"]) == 2
        assert "items" in fields

    def test_items_same_count_llm_richer(self):
        rule = {"items": [{"type": "medicine", "name": "Drug A", "dosage": None}]}
        llm = {
            "items": [
                {
                    "type": "medicine",
                    "name": "Drug A",
                    "dosage": "500mg",
                    "frequency": "daily",
                }
            ]
        }

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert rule["items"][0]["dosage"] == "500mg"
        assert "items" in fields

    def test_items_same_count_rule_richer_stays(self):
        rule = {
            "items": [
                {
                    "type": "medicine",
                    "name": "Drug A",
                    "dosage": "500mg",
                    "frequency": "daily",
                    "route": "oral",
                }
            ]
        }
        llm = {"items": [{"type": "medicine", "name": "Drug A"}]}

        fields = InferenceEngine._merge_llm_fields(rule, llm)

        assert rule["items"][0]["dosage"] == "500mg"  # kept rule-based
        assert "items" not in fields


# ═══════════════════════════════════════════════════════════════════
# Pipeline integration — LLM disabled
# ═══════════════════════════════════════════════════════════════════


class TestPipelineLlmDisabled:
    """When llm_enabled=False (default), the pipeline should not call the LLM."""

    def test_llm_disabled_skips_enhancement(self):
        from src.pipeline.inference import create_default_engine

        engine = create_default_engine()

        text = "Paciente: Maria Garcia\nDiagnostico: Gastritis\nMedico: Dr. Lopez"
        result = engine.process_document("clinical_history", text)

        # Default config has llm_enabled=False
        assert result.llm_output is None
        assert result.llm_fields_used is None
        assert result.validated_data is not None


# ═══════════════════════════════════════════════════════════════════
# Pipeline integration — LLM enabled with mock
# ═══════════════════════════════════════════════════════════════════


class TestPipelineLlmEnabled:
    def _make_engine(self):
        from src.pipeline.inference import create_default_engine
        return create_default_engine()

    @patch("src.pipeline.llm_client.OllamaClient")
    def test_llm_fills_gaps(self, MockClient):
        from src.config import settings as _real_settings

        instance = MockClient.return_value
        instance.is_available.return_value = True
        instance.extract.return_value = {
            "patient_name": "Maria Garcia",
            "patient_id": "99887766",
            "age": 50,
            "sex": "F",
            "consultation_date": "2024-03-15",
            "chief_complaint": "Dolor abdominal",
            "assessment": "Gastritis cronica",
            "plan": "Control en 15 dias",
            "doctor_name": "Dr. Lopez",
            "institution": "Clinica San Jose",
        }

        engine = self._make_engine()
        text = "Paciente: Maria Garcia\nDiagnostico: Gastritis cronica\nMedico: Dr. Lopez"

        with patch.object(_real_settings, "llm_enabled", True):
            result = engine.process_document("clinical_history", text)

        assert result.llm_output is not None
        instance.extract.assert_called_once()

    @patch("src.pipeline.llm_client.OllamaClient")
    def test_llm_unavailable_falls_back(self, MockClient):
        from src.config import settings as _real_settings

        instance = MockClient.return_value
        instance.is_available.return_value = False

        engine = self._make_engine()
        text = "Paciente: Maria Garcia\nDiagnostico: Gastritis\nMedico: Dr. Lopez"

        with patch.object(_real_settings, "llm_enabled", True):
            result = engine.process_document("clinical_history", text)

        assert result.llm_output is None
        assert result.llm_fields_used is None
        assert result.validated_data is not None

    @patch("src.pipeline.llm_client.OllamaClient")
    def test_llm_exception_falls_back(self, MockClient):
        from src.config import settings as _real_settings

        instance = MockClient.return_value
        instance.is_available.return_value = True
        instance.extract.side_effect = LLMConnectionError("http://fake:11434")

        engine = self._make_engine()
        text = "Paciente: Maria Garcia\nDiagnostico: Gastritis\nMedico: Dr. Lopez"

        with patch.object(_real_settings, "llm_enabled", True):
            result = engine.process_document("clinical_history", text)

        assert result.validated_data is not None


# ═══════════════════════════════════════════════════════════════════
# Exception hierarchy
# ═══════════════════════════════════════════════════════════════════


class TestLlmExceptions:
    def test_llm_error_is_dociq_error(self):
        from src.pipeline.exceptions import DocIQError

        assert issubclass(LLMError, DocIQError)

    def test_connection_error_is_llm_error(self):
        assert issubclass(LLMConnectionError, LLMError)

    def test_timeout_error_is_llm_error(self):
        assert issubclass(LLMTimeoutError, LLMError)

    def test_response_error_is_llm_error(self):
        assert issubclass(LLMResponseError, LLMError)

    def test_connection_error_message(self):
        err = LLMConnectionError("http://localhost:11434", "refused")
        assert "localhost:11434" in str(err)

    def test_timeout_error_message(self):
        err = LLMTimeoutError(120)
        assert "120s" in str(err)

    def test_response_error_message(self):
        err = LLMResponseError("bad json")
        assert "bad json" in str(err)
