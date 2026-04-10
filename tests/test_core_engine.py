"""Tests for the DocIQ core engine and CLI."""

import json
import os
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.pipeline.core_engine import DocIQEngine, EngineResult
from src.pipeline.exceptions import (
    ClassificationError,
    DocIQError,
    DocumentExtractionError,
    NERExtractionError,
    OCRError,
    UnsupportedFileError,
    ValidationError,
)
from src.cli import main as cli_main, _build_parser, _load_config


# ── Shared mocks ──────────────────────────────────────────────────

@pytest.fixture
def _mock_extraction():
    """Patch low-level extraction and detection for unit tests."""
    with (
        patch("src.pipeline.process_folder.detect_pdf_language") as m_pdf_lang,
        patch("src.pipeline.process_folder.is_pdf_text_based") as m_text,
        patch("src.pipeline.process_folder.extract_text_directly") as m_direct,
        patch("src.pipeline.process_folder.extract_text_from_pdf_ocr") as m_ocr,
        patch("src.pipeline.process_folder.extract_text_from_image") as m_img,
        patch("src.pipeline.process_folder.detect_language") as m_lang,
    ):
        m_pdf_lang.return_value = SimpleNamespace(language="en", text_sample="sample")
        m_text.return_value = True
        m_direct.return_value = "PDF extracted text"
        m_ocr.return_value = "OCR text"
        m_img.return_value = "Image text"
        m_lang.return_value = "en"
        yield


@pytest.fixture
def fake_folder(tmp_path):
    for f in ["doc.pdf", "img.png", "notes.txt"]:
        (tmp_path / f).write_text("fake")
    return tmp_path


@pytest.fixture
def fake_pdf(tmp_path):
    p = tmp_path / "test.pdf"
    p.write_text("fake pdf")
    return p


# ═══════════════════════════════════════════════════════════════════
# Exception hierarchy
# ═══════════════════════════════════════════════════════════════════

class TestExceptions:
    def test_all_inherit_from_dociq_error(self):
        for exc_cls in (
            DocumentExtractionError,
            ClassificationError,
            NERExtractionError,
            ValidationError,
            OCRError,
            UnsupportedFileError,
        ):
            assert issubclass(exc_cls, DocIQError)

    def test_document_extraction_error_attrs(self):
        e = DocumentExtractionError("f.pdf", "corrupt header")
        assert e.filename == "f.pdf"
        assert "corrupt header" in str(e)

    def test_unsupported_file_error_attrs(self):
        e = UnsupportedFileError("data.xlsx", ".xlsx")
        assert e.extension == ".xlsx"
        assert "Unsupported" in str(e)

    def test_classification_error_default_reason(self):
        e = ClassificationError("x.pdf")
        assert "unrecognised" in str(e)

    def test_ner_extraction_error_attrs(self):
        e = NERExtractionError("y.pdf", "prescription", "regex failed")
        assert e.document_type == "prescription"

    def test_validation_error_attrs(self):
        e = ValidationError("z.pdf", "result", "missing field")
        assert e.document_type == "result"


# ═══════════════════════════════════════════════════════════════════
# EngineResult
# ═══════════════════════════════════════════════════════════════════

class TestEngineResult:
    def test_ok_and_errors(self):
        items = [
            {"file": "a.pdf", "status": "ok"},
            {"file": "b.pdf", "status": "extraction_error"},
            {"file": "c.pdf", "status": "ok"},
        ]
        er = EngineResult(items)
        assert len(er.ok) == 2
        assert len(er.errors) == 1
        assert er.count == 3

    def test_summary(self):
        items = [
            {"file": "a.pdf", "status": "ok"},
            {"file": "b.pdf", "status": "ok"},
            {"file": "c.pdf", "status": "inference_error"},
        ]
        er = EngineResult(items)
        assert er.summary() == {"ok": 2, "inference_error": 1}

    def test_iteration(self):
        items = [{"file": "x.pdf", "status": "ok"}]
        er = EngineResult(items)
        assert list(er) == items
        assert er[0] == items[0]
        assert len(er) == 1

    def test_repr(self):
        er = EngineResult([{"file": "a.pdf", "status": "ok"}])
        assert "ok=1" in repr(er)


# ═══════════════════════════════════════════════════════════════════
# DocIQEngine – process_file
# ═══════════════════════════════════════════════════════════════════

class TestDocIQEngineProcessFile:
    def test_raises_on_missing_file(self):
        engine = DocIQEngine(run_inference=False)
        with pytest.raises(FileNotFoundError):
            engine.process_file("/nonexistent/file.pdf")

    def test_raises_on_unsupported_extension(self, tmp_path):
        txt = tmp_path / "data.txt"
        txt.write_text("hello")
        engine = DocIQEngine(run_inference=False)
        with pytest.raises(UnsupportedFileError):
            engine.process_file(str(txt))

    def test_processes_pdf_without_inference(self, fake_pdf, _mock_extraction):
        engine = DocIQEngine(run_inference=False)
        result = engine.process_file(str(fake_pdf))
        assert result["status"] == "ok"
        assert result["file"] == "test.pdf"
        assert result["document_type"] is None
        assert result["text"] == "PDF extracted text"

    def test_processes_pdf_with_inference(self, fake_pdf, _mock_extraction):
        mock_ie = MagicMock()
        mock_ie.classify.return_value = "prescription"
        mock_ir = MagicMock()
        mock_ir.as_dict.return_value = {"patient_name": "Test"}
        mock_ir.validated_data = True
        mock_ie.process_document.return_value = mock_ir

        engine = DocIQEngine(inference_engine=mock_ie, run_inference=True)
        result = engine.process_file(str(fake_pdf))
        assert result["document_type"] == "prescription"
        assert result["validated"] is True
        assert result["extracted_data"] == {"patient_name": "Test"}

    def test_inference_error_is_captured(self, fake_pdf, _mock_extraction):
        mock_ie = MagicMock()
        mock_ie.classify.side_effect = RuntimeError("boom")

        engine = DocIQEngine(inference_engine=mock_ie, run_inference=True)
        result = engine.process_file(str(fake_pdf))
        assert result["status"] == "inference_error"
        assert "boom" in result["error"]

    def test_processes_image(self, tmp_path, _mock_extraction):
        img = tmp_path / "scan.png"
        img.write_text("fake image")
        engine = DocIQEngine(run_inference=False)
        result = engine.process_file(str(img))
        assert result["method"] == "image"
        assert result["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════
# DocIQEngine – process_batch
# ═══════════════════════════════════════════════════════════════════

class TestDocIQEngineBatch:
    def test_returns_engine_result(self, fake_folder, _mock_extraction):
        engine = DocIQEngine(run_inference=False)
        result = engine.process_batch(str(fake_folder))
        assert isinstance(result, EngineResult)
        assert result.count == 2  # doc.pdf + img.png (notes.txt skipped)

    def test_batch_delegates_to_process_folder(self, fake_folder, _mock_extraction):
        engine = DocIQEngine(run_inference=False)
        result = engine.process_batch(str(fake_folder))
        filenames = {r["file"] for r in result}
        assert "doc.pdf" in filenames
        assert "img.png" in filenames
        assert "notes.txt" not in filenames


# ═══════════════════════════════════════════════════════════════════
# DocIQEngine – export
# ═══════════════════════════════════════════════════════════════════

class TestDocIQEngineExport:
    def test_export_csv(self, tmp_path):
        items = [
            {"file": "a.pdf", "status": "ok", "text": "hello", "extracted_data": None},
        ]
        path = DocIQEngine.export(items, output_dir=str(tmp_path), fmt="csv")
        assert os.path.exists(path)
        content = open(path).read()
        assert "a.pdf" in content
        assert "hello" in content

    def test_export_json(self, tmp_path):
        items = [
            {"file": "a.pdf", "status": "ok", "text": "hello"},
            {"file": "b.png", "status": "ok", "text": "world"},
        ]
        path = DocIQEngine.export(items, output_dir=str(tmp_path), fmt="json")
        assert os.path.isdir(path)
        files = os.listdir(path)
        assert "a.json" in files
        assert "b.json" in files
        with open(os.path.join(path, "a.json")) as f:
            data = json.load(f)
        assert data["text"] == "hello"

    def test_export_csv_with_extracted_data(self, tmp_path):
        items = [
            {
                "file": "a.pdf",
                "status": "ok",
                "text": "x",
                "extracted_data": {"patient_name": "Test"},
            },
        ]
        path = DocIQEngine.export(items, output_dir=str(tmp_path), fmt="csv")
        content = open(path).read()
        assert "Test" in content

    def test_export_fhir(self, tmp_path):
        items = [
            {
                "file": "result_1.pdf",
                "status": "ok",
                "document_type": "result",
                "extracted_data": {
                    "patient_name": "Test Patient",
                    "exam_type": "CBC",
                    "exam_date": "2024-01-01",
                    "findings": "Normal",
                    "professional": "Dr. Test",
                    "institution": "Test Lab",
                },
            },
        ]
        path = DocIQEngine.export(items, output_dir=str(tmp_path), fmt="fhir")
        assert os.path.isdir(path)
        fhir_file = os.path.join(path, "result_1_fhir.json")
        assert os.path.exists(fhir_file)
        with open(fhir_file) as f:
            data = json.load(f)
        assert data["resourceType"] == "DiagnosticReport"
        assert data["subject"]["name"] == "Test Patient"

    def test_export_fhir_skips_unclassified(self, tmp_path):
        items = [
            {"file": "x.pdf", "status": "ok", "document_type": None, "extracted_data": None},
        ]
        path = DocIQEngine.export(items, output_dir=str(tmp_path), fmt="fhir")
        assert os.path.isdir(path)
        assert len(os.listdir(path)) == 0

    def test_export_invalid_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            DocIQEngine.export([], output_dir=str(tmp_path), fmt="xml")

    def test_export_empty_csv(self, tmp_path):
        path = DocIQEngine.export([], output_dir=str(tmp_path), fmt="csv")
        assert os.path.exists(path)

    def test_export_creates_output_dir(self, tmp_path):
        out = str(tmp_path / "nested" / "dir")
        DocIQEngine.export(
            [{"file": "a.pdf", "status": "ok"}],
            output_dir=out,
            fmt="json",
        )
        assert os.path.isdir(out)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

class TestCLI:
    def test_parser_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["--input", "data/generated"])
        assert args.input == "data/generated"
        assert args.config is None

    def test_parser_all_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "-i", "file.pdf",
            "-o", "out",
            "-f", "csv",
            "--no-inference",
            "--max-workers", "4",
            "--log-level", "DEBUG",
            "--config", "dociq.yaml",
        ])
        assert args.input == "file.pdf"
        assert args.output_dir == "out"
        assert args.format == "csv"
        assert args.no_inference is True
        assert args.max_workers == 4
        assert args.log_level == "DEBUG"
        assert args.config == "dociq.yaml"

    def test_parser_accepts_fhir_format(self):
        parser = _build_parser()
        args = parser.parse_args(["--input", "x", "--format", "fhir"])
        assert args.format == "fhir"

    def test_cli_returns_1_for_missing_input(self):
        code = cli_main(["--input", "/nonexistent/path"])
        assert code == 1

    def test_cli_returns_1_for_no_input_at_all(self):
        code = cli_main([])
        assert code == 1

    def test_cli_processes_folder(self, fake_folder, tmp_path, _mock_extraction):
        code = cli_main([
            "--input", str(fake_folder),
            "--output-dir", str(tmp_path / "out"),
            "--format", "csv",
            "--no-inference",
        ])
        assert code == 0
        assert (tmp_path / "out" / "dociq_results.csv").exists()

    def test_cli_processes_single_file(self, fake_pdf, tmp_path, _mock_extraction):
        code = cli_main([
            "--input", str(fake_pdf),
            "--output-dir", str(tmp_path / "out"),
            "--format", "json",
            "--no-inference",
        ])
        assert code == 0

    def test_cli_returns_1_for_bad_config(self):
        code = cli_main(["--config", "/nonexistent/config.yaml"])
        assert code == 1


# ═══════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════

class TestConfigLoading:
    def test_load_simple_yaml(self, tmp_path):
        cfg = tmp_path / "dociq.yaml"
        cfg.write_text(
            "input: data/generated\n"
            "output_dir: output\n"
            "format: fhir\n"
            "log_level: DEBUG\n"
            "no_inference: true\n"
            "max_workers: 4\n"
        )
        data = _load_config(str(cfg))
        assert data["input"] == "data/generated"
        assert data["output_dir"] == "output"
        assert data["format"] == "fhir"
        assert data["log_level"] == "DEBUG"
        assert data["no_inference"] is True
        assert data["max_workers"] == 4

    def test_config_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_config("/nonexistent/config.yaml")

    def test_config_comments_and_blanks_ignored(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            "# Comment line\n"
            "\n"
            "input: data/test\n"
            "# Another comment\n"
        )
        data = _load_config(str(cfg))
        assert data["input"] == "data/test"
        assert len(data) == 1

    def test_cli_uses_config_values(self, fake_folder, tmp_path, _mock_extraction):
        cfg = tmp_path / "dociq.yaml"
        cfg.write_text(
            f"input: {fake_folder}\n"
            f"output_dir: {tmp_path / 'out'}\n"
            "format: csv\n"
            "no_inference: true\n"
        )
        code = cli_main(["--config", str(cfg)])
        assert code == 0
        assert (tmp_path / "out" / "dociq_results.csv").exists()

    def test_cli_flags_override_config(self, fake_folder, tmp_path, _mock_extraction):
        cfg = tmp_path / "dociq.yaml"
        cfg.write_text(
            f"input: {fake_folder}\n"
            "format: csv\n"
            "no_inference: true\n"
        )
        # --format json overrides config's csv
        code = cli_main([
            "--config", str(cfg),
            "--output-dir", str(tmp_path / "out"),
            "--format", "json",
        ])
        assert code == 0
        assert (tmp_path / "out" / "dociq_results").is_dir()
