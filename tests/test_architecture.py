"""Tests for modular architecture improvements (#16).

Validates:
  • Schema imports go through validation/__init__.py (no direct submodule coupling)
  • Field alias resolvers produce identical results to inline chains
  • Consistent snake_case schema file naming
  • Module-level imports (no unnecessary late imports)
  • Extractor __init__.py exports field alias resolvers
"""
from __future__ import annotations

import ast
import importlib
import os
import re
from pathlib import Path
from typing import List

import pytest

# ═══════════════════════════════════════════════════════════════════
# Schema file naming consistency
# ═══════════════════════════════════════════════════════════════════


class TestSchemaFileNaming:
    """All schema files under validation/ must use snake_case naming."""

    VALIDATION_DIR = Path(__file__).resolve().parent.parent / "src" / "pipeline" / "validation"

    def test_no_pascal_case_schema_files(self):
        """Schema modules must be snake_case (no CamelCase filenames)."""
        for f in self.VALIDATION_DIR.glob("*.py"):
            if f.name.startswith("_"):
                continue
            assert f.name == f.name.lower(), (
                f"Schema file {f.name} uses non-snake_case naming"
            )

    def test_clinical_history_schema_exists(self):
        assert (self.VALIDATION_DIR / "clinical_history_schema.py").exists()

    def test_old_pascal_case_file_removed(self):
        assert not (self.VALIDATION_DIR / "ClinicalHistorySchema.py").exists()


# ═══════════════════════════════════════════════════════════════════
# Schema import coupling
# ═══════════════════════════════════════════════════════════════════


class TestSchemaImportCoupling:
    """Consumers must import schemas from the validation package, not submodules."""

    # Files that should NOT import directly from validation submodules
    CONSUMER_FILES = [
        Path(__file__).resolve().parent.parent / "src" / "pipeline" / "output_formatter.py",
        Path(__file__).resolve().parent.parent / "src" / "api" / "app.py",
    ]

    # Only __init__.py, validator.py, and fhir_mapper.py may use direct submodule imports
    # (they are part of the validation/mapping layer itself)

    @pytest.mark.parametrize("filepath", CONSUMER_FILES, ids=lambda p: p.name)
    def test_no_direct_submodule_imports(self, filepath):
        """Consumer files should import from validation package, not submodules."""
        if not filepath.exists():
            pytest.skip(f"{filepath.name} does not exist")

        source = filepath.read_text()
        # Should NOT have: from src.pipeline.validation.schemas import ...
        # Should NOT have: from src.pipeline.validation.prescription_schema import ...
        # Should NOT have: from src.pipeline.validation.clinical_history_schema import ...
        direct_imports = re.findall(
            r"from src\.pipeline\.validation\.(schemas|prescription_schema|clinical_history_schema)\s+import",
            source,
        )
        # Allow imports inside try/except blocks or conditional blocks for FHIR mapping
        # but the top-level imports should go through the package
        top_level_direct = []
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if re.match(
                    r"src\.pipeline\.validation\.(schemas|prescription_schema|clinical_history_schema)",
                    node.module,
                ):
                    top_level_direct.append(node.module)

        assert not top_level_direct, (
            f"{filepath.name} has direct submodule imports at top level: {top_level_direct}. "
            "Use 'from src.pipeline.validation import ...' instead."
        )


# ═══════════════════════════════════════════════════════════════════
# Field alias resolvers
# ═══════════════════════════════════════════════════════════════════


from src.pipeline.extractors.field_aliases import (
    resolve_assessment,
    resolve_chief_complaint,
    resolve_doctor,
    resolve_exam_date,
    resolve_institution,
    resolve_medications,
    resolve_physical_exam,
    resolve_plan,
    resolve_prescription_date,
)


class TestFieldAliasResolvers:
    """Verify shared field-alias resolvers return correct values."""

    def test_resolve_institution_clinic(self):
        text = "Clinic: Central Hospital\nOther: stuff"
        assert resolve_institution(text) == "Central Hospital"

    def test_resolve_institution_fallback(self):
        text = "Institution: General Medical\nOther: stuff"
        assert resolve_institution(text) == "General Medical"

    def test_resolve_institution_none(self):
        assert resolve_institution("no match here") is None

    def test_resolve_doctor_primary(self):
        text = "Doctor: Dr. Smith"
        assert resolve_doctor(text) == "Dr. Smith"

    def test_resolve_doctor_physician_fallback(self):
        text = "Physician: Dr. Jones"
        assert resolve_doctor(text) == "Dr. Jones"

    def test_resolve_doctor_professional_fallback(self):
        text = "Professional: Dr. Lee"
        assert resolve_doctor(text) == "Dr. Lee"

    def test_resolve_exam_date_primary(self):
        text = "Exam Date: 2025-01-15"
        assert resolve_exam_date(text) == "2025-01-15"

    def test_resolve_exam_date_fallback(self):
        text = "Date: 2025-03-20"
        assert resolve_exam_date(text) == "2025-03-20"

    def test_resolve_prescription_date(self):
        text = "Date of Prescription: 2025-06-10"
        assert resolve_prescription_date(text) == "2025-06-10"

    def test_resolve_prescription_date_fallback(self):
        text = "Date: 2025-06-10"
        assert resolve_prescription_date(text) == "2025-06-10"

    def test_resolve_assessment_primary(self):
        text = "Assessment: Hypertension stage 2"
        assert resolve_assessment(text) == "Hypertension stage 2"

    def test_resolve_assessment_fallback(self):
        text = "Diagnosis: Type 2 diabetes"
        assert resolve_assessment(text) == "Type 2 diabetes"

    def test_resolve_plan_primary(self):
        text = "Plan: Continue current medication"
        assert resolve_plan(text) == "Continue current medication"

    def test_resolve_plan_fallback(self):
        text = "Treatment Plan: Start physiotherapy"
        assert resolve_plan(text) == "Start physiotherapy"

    def test_resolve_physical_exam(self):
        text = "Physical Exam: Normal findings"
        assert resolve_physical_exam(text) == "Normal findings"

    def test_resolve_physical_exam_fallback(self):
        text = "Examination: Lungs clear"
        assert resolve_physical_exam(text) == "Lungs clear"

    def test_resolve_chief_complaint(self):
        text = "Chief Complaint: Chest pain"
        assert resolve_chief_complaint(text) == "Chest pain"

    def test_resolve_chief_complaint_fallback(self):
        text = "Reason for Visit: Annual checkup"
        assert resolve_chief_complaint(text) == "Annual checkup"

    def test_resolve_medications(self):
        text = "Current Medications: Aspirin, Metformin"
        assert resolve_medications(text) == "Aspirin, Metformin"

    def test_resolve_medications_fallback(self):
        text = "Medications: Lisinopril"
        assert resolve_medications(text) == "Lisinopril"

    def test_priority_order_institution(self):
        """When both Clinic and Institution present, Clinic wins."""
        text = "Clinic: First Choice\nInstitution: Second Choice"
        assert resolve_institution(text) == "First Choice"

    def test_priority_order_doctor(self):
        """When both Doctor and Physician present, Doctor wins."""
        text = "Doctor: Dr. Alpha\nPhysician: Dr. Beta"
        assert resolve_doctor(text) == "Dr. Alpha"


# ═══════════════════════════════════════════════════════════════════
# Extractor consistency after refactor
# ═══════════════════════════════════════════════════════════════════


PRESCRIPTION_TEXT = """Patient Name: Maria Garcia
Patient ID: 45012
Date of Birth: 1985-03-15
Date of Prescription: 2025-06-10
Doctor: Dr. Carlos Rodriguez
Clinic: Central Medical Center

Prescription:
Amoxicillin 500mg, 3 times daily for 7 days.
Ibuprofen 400mg as needed for pain.
"""

LAB_RESULT_TEXT = """Patient Name: Anthony Harper
Patient ID: 32373
Date of Birth: 1965-05-12
Exam Date: 2025-01-12
Clinic: Foster-Bailey

Test Results:
- Glucose Test: 95.50 (Ref: 70-100)
- Cholesterol Test: 210.30 (Ref: 125-200)
- Hemoglobin: 14.20 (Ref: 12-17)

Summary: Cholesterol slightly elevated. Recommend dietary changes.
"""

CLINICAL_HISTORY_TEXT = """Patient Name: Shelby Brown
Patient ID: 13098
Date of Birth: 1931-09-29
Clinic: Garrett-Wagner

Annotations:
- 2025-02-22: Persistent headache reported. Prescribed analgesics.
- 2025-02-19: Follow-up visit. Blood pressure stable at 130/85.
- 2025-01-06: Initial consultation for recurring migraines and dizziness.
"""


class TestExtractorConsistencyAfterRefactor:
    """Ensure extractors still produce the same output after field_aliases refactor."""

    def test_prescription_extractor_output(self):
        from src.pipeline.extractors import PrescriptionExtractor

        result = PrescriptionExtractor().extract(PRESCRIPTION_TEXT)
        assert result["patient_name"] == "Maria Garcia"
        assert result["patient_id"] == "45012"
        assert result["date"] == "2025-06-10"
        assert result["doctor_name"] == "Dr. Carlos Rodriguez"
        assert result["institution"] == "Central Medical Center"
        assert isinstance(result["items"], list)
        assert len(result["items"]) >= 1

    def test_lab_result_extractor_output(self):
        from src.pipeline.extractors import LabResultExtractor

        result = LabResultExtractor().extract(LAB_RESULT_TEXT)
        assert result["patient_name"] == "Anthony Harper"
        assert result["patient_id"] == "32373"
        assert result["exam_date"] == "2025-01-12"
        assert result["institution"] == "Foster-Bailey"
        assert "Glucose Test" in result["findings"]
        assert result["professional"] == "Not specified"

    def test_clinical_history_extractor_output(self):
        from src.pipeline.extractors import ClinicalHistoryExtractor

        result = ClinicalHistoryExtractor().extract(CLINICAL_HISTORY_TEXT)
        assert result["patient_name"] == "Shelby Brown"
        assert result["patient_id"] == "13098"
        assert result["date_of_birth"] == "1931-09-29"
        assert result["institution"] == "Garrett-Wagner"
        assert result["consultation_date"] == "2025-02-22"
        assert "headache" in result["chief_complaint"].lower()

    def test_prescription_with_institution_alias(self):
        """Prescription extractor resolves Institution when Clinic is absent."""
        from src.pipeline.extractors import PrescriptionExtractor

        text = PRESCRIPTION_TEXT.replace("Clinic:", "Institution:")
        result = PrescriptionExtractor().extract(text)
        assert result["institution"] == "Central Medical Center"

    def test_lab_result_with_physician_alias(self):
        """Lab result extractor resolves Physician alias for doctor."""
        from src.pipeline.extractors import LabResultExtractor

        text = LAB_RESULT_TEXT + "\nPhysician: Dr. House"
        result = LabResultExtractor().extract(text)
        assert result["professional"] == "Dr. House"


# ═══════════════════════════════════════════════════════════════════
# Module-level imports in preprocess.py
# ═══════════════════════════════════════════════════════════════════


class TestPreprocessImports:
    """Verify detect_language is imported at module level."""

    def test_no_late_import_of_detect_language(self):
        preprocess_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "pipeline"
            / "preprocess.py"
        )
        source = preprocess_path.read_text()
        tree = ast.parse(source)

        late_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.ImportFrom) and child.module:
                        if "language" in child.module:
                            late_imports.append(
                                f"{child.module} in {node.name}()"
                            )

        assert not late_imports, (
            f"detect_language still imported inside function body: {late_imports}"
        )

    def test_module_level_import_present(self):
        preprocess_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "pipeline"
            / "preprocess.py"
        )
        source = preprocess_path.read_text()
        tree = ast.parse(source)

        top_level_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "language" in node.module:
                    names = [alias.name for alias in node.names]
                    top_level_imports.extend(names)

        assert "detect_language" in top_level_imports


# ═══════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════


class TestExtractorExports:
    """Verify field alias resolvers are exported from extractors package."""

    def test_field_alias_resolvers_exported(self):
        from src.pipeline import extractors

        resolver_names = [
            "resolve_institution",
            "resolve_doctor",
            "resolve_exam_date",
            "resolve_prescription_date",
            "resolve_assessment",
            "resolve_plan",
            "resolve_physical_exam",
            "resolve_chief_complaint",
            "resolve_medications",
        ]
        for name in resolver_names:
            assert hasattr(extractors, name), f"{name} not exported from extractors"

    def test_validation_package_exports_all_schemas(self):
        from src.pipeline import validation

        assert hasattr(validation, "ResultSchema")
        assert hasattr(validation, "Prescription")
        assert hasattr(validation, "ClinicalHistorySchema")
        assert hasattr(validation, "SCHEMA_REGISTRY")
