"""Tests for src.pipeline.exceptions – custom exception hierarchy."""
import pytest

from src.pipeline.exceptions import (
    ClassificationError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigurationError,
    DocIQError,
    DocumentExtractionError,
    ExportError,
    FHIRMappingError,
    ImageLoadError,
    ModelError,
    ModelExecutionError,
    ModelLoadError,
    NERExtractionError,
    OCRError,
    PDFExtractionError,
    PDFOpenError,
    PageTimeoutError,
    SchemaValidationError,
    TesseractError,
    UnsupportedFileError,
    ValidationError,
)


# ── Hierarchy checks ────────────────────────────────────────────

class TestExceptionHierarchy:
    """Every custom exception should be a subclass of DocIQError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            ConfigFileNotFoundError,
            ConfigParseError,
            DocumentExtractionError,
            PDFOpenError,
            PDFExtractionError,
            PageTimeoutError,
            OCRError,
            ImageLoadError,
            TesseractError,
            ClassificationError,
            NERExtractionError,
            ModelError,
            ModelLoadError,
            ModelExecutionError,
            SchemaValidationError,
            ExportError,
            FHIRMappingError,
            UnsupportedFileError,
        ],
    )
    def test_inherits_from_dociq_error(self, exc_cls):
        assert issubclass(exc_cls, DocIQError)

    def test_pdf_open_is_extraction_error(self):
        assert issubclass(PDFOpenError, DocumentExtractionError)

    def test_page_timeout_is_extraction_error(self):
        assert issubclass(PageTimeoutError, DocumentExtractionError)

    def test_image_load_is_ocr_error(self):
        assert issubclass(ImageLoadError, OCRError)

    def test_tesseract_is_ocr_error(self):
        assert issubclass(TesseractError, OCRError)

    def test_model_load_is_model_error(self):
        assert issubclass(ModelLoadError, ModelError)

    def test_model_execution_is_model_error(self):
        assert issubclass(ModelExecutionError, ModelError)

    def test_fhir_mapping_is_export_error(self):
        assert issubclass(FHIRMappingError, ExportError)

    def test_config_file_not_found_is_configuration_error(self):
        assert issubclass(ConfigFileNotFoundError, ConfigurationError)

    def test_config_parse_is_configuration_error(self):
        assert issubclass(ConfigParseError, ConfigurationError)

    def test_validation_error_alias(self):
        assert ValidationError is SchemaValidationError


# ── Context attributes ──────────────────────────────────────────

class TestExceptionContext:
    def test_document_extraction_error_attrs(self):
        exc = DocumentExtractionError("test.pdf", "corrupt")
        assert exc.filename == "test.pdf"
        assert exc.reason == "corrupt"
        assert "test.pdf" in str(exc)

    def test_pdf_open_error_attrs(self):
        exc = PDFOpenError("bad.pdf", "encrypted")
        assert exc.filename == "bad.pdf"
        assert "encrypted" in str(exc)

    def test_pdf_extraction_error_page(self):
        exc = PDFExtractionError("doc.pdf", 3, "fitz error")
        assert exc.page == 3
        assert "page 3" in str(exc)

    def test_page_timeout_error_attrs(self):
        exc = PageTimeoutError("slow.pdf", 5, 300)
        assert exc.page == 5
        assert exc.timeout_seconds == 300
        assert "300s" in str(exc)

    def test_ocr_error_attrs(self):
        exc = OCRError("scan.png", "tesseract crashed")
        assert exc.filename == "scan.png"
        assert exc.reason == "tesseract crashed"

    def test_image_load_error(self):
        exc = ImageLoadError("bad.jpg")
        assert exc.filename == "bad.jpg"
        assert "could not load" in str(exc)

    def test_classification_error_default_reason(self):
        exc = ClassificationError("doc.pdf")
        assert "unrecognised" in str(exc)

    def test_ner_extraction_error_attrs(self):
        exc = NERExtractionError("doc.pdf", "prescription", "model crash")
        assert exc.document_type == "prescription"
        assert exc.filename == "doc.pdf"

    def test_model_execution_error_attrs(self):
        exc = ModelExecutionError("BertNER", "CUDA OOM")
        assert exc.model_name == "BertNER"
        assert "CUDA OOM" in str(exc)

    def test_schema_validation_error_attrs(self):
        exc = SchemaValidationError(
            "doc.pdf", "result", "missing patient_name",
            pydantic_error=ValueError("test"),
        )
        assert exc.filename == "doc.pdf"
        assert exc.document_type == "result"
        assert exc.pydantic_error is not None

    def test_export_error_attrs(self):
        exc = ExportError("csv", "write failed")
        assert exc.fmt == "csv"
        assert exc.reason == "write failed"

    def test_fhir_mapping_error_attrs(self):
        exc = FHIRMappingError("doc.pdf", "unknown type")
        assert exc.filename == "doc.pdf"
        assert "fhir" in str(exc).lower()

    def test_unsupported_file_error_attrs(self):
        exc = UnsupportedFileError("file.doc", ".doc")
        assert exc.filename == "file.doc"
        assert exc.extension == ".doc"

    def test_configuration_error_with_path(self):
        exc = ConfigurationError("bad syntax", path="/etc/dociq.yaml")
        assert exc.path == "/etc/dociq.yaml"
        assert "bad syntax" in str(exc)

    def test_config_file_not_found(self):
        exc = ConfigFileNotFoundError("/missing.yaml")
        assert exc.path == "/missing.yaml"

    def test_config_parse_error(self):
        exc = ConfigParseError("/bad.yaml", "invalid YAML at line 5")
        assert "invalid YAML" in str(exc)


# ── Catching by base class ──────────────────────────────────────

class TestBaseCatching:
    def test_catch_all_with_dociq_error(self):
        """All custom exceptions can be caught with except DocIQError."""
        exceptions = [
            PDFOpenError("x.pdf"),
            OCRError("x.png", "fail"),
            ClassificationError("x.pdf"),
            ModelExecutionError("M", "crash"),
            SchemaValidationError("x", "t", "r"),
            ExportError("json", "disk full"),
            UnsupportedFileError("x.doc", ".doc"),
            ConfigFileNotFoundError("/x.yaml"),
        ]
        for exc in exceptions:
            with pytest.raises(DocIQError):
                raise exc

    def test_catch_extraction_family(self):
        """All extraction errors catchable via DocumentExtractionError."""
        for exc in [
            PDFOpenError("x.pdf"),
            PDFExtractionError("x.pdf", 1, "fail"),
            PageTimeoutError("x.pdf", 1, 60),
        ]:
            with pytest.raises(DocumentExtractionError):
                raise exc


# ── Integration: modules raise custom exceptions ────────────────

class TestModuleIntegration:
    def test_pdf_type_detector_raises_pdf_open_error(self):
        from src.pipeline.pdf_type_detector import is_pdf_text_based
        with pytest.raises(PDFOpenError):
            is_pdf_text_based("/nonexistent/corrupt.pdf")

    def test_config_raises_config_file_not_found(self):
        from src.config import get_settings
        with pytest.raises(ConfigFileNotFoundError):
            get_settings(config_path="/nonexistent/config.yaml")

    def test_fhir_mapper_raises_fhir_mapping_error(self):
        from src.pipeline.fhir_mapper import map_to_fhir_loose
        with pytest.raises(FHIRMappingError):
            map_to_fhir_loose({"not": "a model"})

    def test_export_results_raises_export_error(self):
        from src.pipeline.output_formatter import export_results
        with pytest.raises(ExportError):
            export_results([], output_dir="/tmp", fmt="xml")
