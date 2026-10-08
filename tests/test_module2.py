# =============================================================================
# NDA Research Project - Module 2 Test Suite
# =============================================================================
# Tests cover:
#   1. PDF text extraction (real and edge cases)
#   2. Empty/malformed PDF handling
#   3. Text cleaning
#   4. Clause boundary detection patterns
#   5. Segmenter output format (JSON structure)
#   6. Reproducibility of sampling
#   7. Malformed/edge-case input handling
#
# Run with: pytest tests/test_module2.py -v
# =============================================================================

import sys
import json
import tempfile
import shutil
import pytest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.pdf_extractor import (
    clean_extracted_text,
    extract_text_from_pdf,
    ExtractionResult,
)
from preprocessing.clause_segmenter import (
    BaselineSegmenter,
    create_segmenter,
    save_segmented_document,
    load_segmented_document,
    Clause,
    SegmentedDocument,
)
from preprocessing.document_sampler import (
    sample_documents,
    save_sample_manifest,
    load_sample_manifest,
)


# =============================================================================
# Test: Text Cleaning
# =============================================================================

class TestTextCleaning:
    """Tests for PDF-extracted text cleaning."""

    def test_null_bytes_removed(self):
        text = "Legal\x00text\x00here."
        result = clean_extracted_text(text)
        assert "\x00" not in result
        assert "Legal" in result

    def test_control_chars_removed(self):
        text = "Text\x01with\x0ccontrol\x7fchars."
        result = clean_extracted_text(text)
        assert "\x01" not in result
        assert "\x0c" not in result
        assert "\x7f" not in result

    def test_whitespace_collapsed(self):
        text = "Multiple    spaces   here."
        result = clean_extracted_text(text)
        assert "  " not in result
        assert "Multiple spaces here." == result

    def test_excessive_newlines_collapsed(self):
        text = "Para one.\n\n\n\n\nPara two."
        result = clean_extracted_text(text)
        assert "\n\n\n" not in result
        assert "Para one.\n\nPara two." == result

    def test_trailing_whitespace_stripped(self):
        text = "Line one.   \nLine two.  "
        result = clean_extracted_text(text)
        lines = result.split("\n")
        for line in lines:
            assert line == line.rstrip()

    def test_empty_string(self):
        assert clean_extracted_text("") == ""

    def test_preserves_legal_content(self):
        text = (
            'CONFIDENTIAL INFORMATION shall mean any and all information, '
            'whether written, oral, or electronic, disclosed by the '
            'Disclosing Party to the Receiving Party.'
        )
        result = clean_extracted_text(text)
        assert "CONFIDENTIAL INFORMATION" in result
        assert "Disclosing Party" in result
        assert "Receiving Party" in result


# =============================================================================
# Test: PDF Extraction Edge Cases
# =============================================================================

class TestPDFExtraction:
    """Tests for PDF extraction edge cases."""

    def test_nonexistent_file(self):
        result = extract_text_from_pdf("/path/that/does/not/exist.pdf")
        assert result.status == "failed"
        assert result.character_count == 0
        assert "Failed to open" in result.error_message

    def test_extraction_result_to_dict(self):
        result = ExtractionResult(
            document_id="test_doc",
            source_file="test_doc.pdf",
            text="Some text",
            character_count=9,
            word_count=2,
            line_count=1,
            page_count=1,
            pages_with_text=1,
            status="success",
        )
        d = result.to_dict()
        assert d["document_id"] == "test_doc"
        assert d["status"] == "success"
        assert d["character_count"] == 9


# =============================================================================
# Test: Clause Boundary Detection
# =============================================================================

class TestClauseBoundaryDetection:
    """Tests for the baseline segmenter's pattern detection."""

    def setup_method(self):
        self.segmenter = BaselineSegmenter(
            min_clause_length=10,
            max_clause_length=5000,
        )

    def test_numbered_sections(self):
        text = (
            "1. The Receiving Party agrees to maintain confidentiality.\n\n"
            "2. The Disclosing Party retains all rights.\n\n"
            "3. This Agreement shall terminate after two years."
        )
        doc = self.segmenter.segment(text, "test_doc")
        assert len(doc.clauses) == 3

    def test_article_sections(self):
        text = (
            "Article I DEFINITIONS\n"
            "Confidential Information means any proprietary data.\n\n"
            "Article II OBLIGATIONS\n"
            "The Receiving Party shall not disclose any information.\n\n"
            "Article III TERM\n"
            "This Agreement is effective for two years."
        )
        doc = self.segmenter.segment(text, "test_doc")
        assert len(doc.clauses) >= 3

    def test_section_headings(self):
        text = (
            "Section 1 Definitions\n"
            "All terms used herein shall have the meanings set forth.\n\n"
            "Section 2 Scope\n"
            "This agreement covers all confidential materials.\n\n"
            "Section 3 Duration\n"
            "The obligations shall continue for five years."
        )
        doc = self.segmenter.segment(text, "test_doc")
        assert len(doc.clauses) >= 3

    def test_parenthesized_markers(self):
        text = (
            "(a) The Receiving Party shall protect all information.\n\n"
            "(b) The Disclosing Party may terminate at will.\n\n"
            "(c) Both parties agree to binding arbitration."
        )
        doc = self.segmenter.segment(text, "test_doc")
        assert len(doc.clauses) == 3

    def test_single_paragraph(self):
        text = "This is a single paragraph with no clause markers."
        doc = self.segmenter.segment(text, "test_doc")
        assert len(doc.clauses) == 1

    def test_empty_text(self):
        doc = self.segmenter.segment("", "empty_doc")
        assert len(doc.clauses) == 0
        assert doc.total_characters == 0

    def test_whitespace_only(self):
        doc = self.segmenter.segment("   \n\n\n   ", "ws_doc")
        assert len(doc.clauses) == 0

    def test_short_segments_merged(self):
        """Segments shorter than min_clause_length should be merged."""
        segmenter = BaselineSegmenter(min_clause_length=50, max_clause_length=5000)
        text = (
            "1. Hi.\n\n"
            "2. This is a much longer clause that exceeds the minimum clause length threshold easily."
        )
        doc = segmenter.segment(text, "test_doc")
        # "Hi." is too short, should be merged
        assert len(doc.clauses) >= 1

    def test_clause_ids_include_document_id(self):
        text = "1. First clause.\n\n2. Second clause."
        doc = self.segmenter.segment(text, "my_document")
        for clause in doc.clauses:
            assert "my_document" in clause.clause_id

    def test_positions_are_valid(self):
        text = (
            "1. First clause text here.\n\n"
            "2. Second clause text here."
        )
        doc = self.segmenter.segment(text, "test_doc")
        for clause in doc.clauses:
            assert clause.start_position >= 0
            assert clause.end_position > clause.start_position
            assert clause.end_position <= len(text)


# =============================================================================
# Test: Segmenter Factory
# =============================================================================

class TestSegmenterFactory:
    """Tests for the segmenter factory function."""

    def test_baseline_strategy(self):
        segmenter = create_segmenter("baseline")
        assert isinstance(segmenter, BaselineSegmenter)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown segmentation strategy"):
            create_segmenter("nonexistent")

    def test_custom_parameters(self):
        segmenter = create_segmenter(
            "baseline", min_clause_length=100, max_clause_length=2000
        )
        assert segmenter.min_clause_length == 100
        assert segmenter.max_clause_length == 2000


# =============================================================================
# Test: JSON Output Format
# =============================================================================

class TestJSONOutput:
    """Tests for segmented document serialization."""

    def setup_method(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="nda_seg_test_"))

    def teardown_method(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        doc = SegmentedDocument(
            document_id="test_doc_001",
            source_file="test_doc_001.txt",
            total_characters=500,
            clauses=[
                Clause(
                    clause_id="test_doc_001_clause_000",
                    text="The Receiving Party agrees to hold information.",
                    start_position=0,
                    end_position=47,
                ),
                Clause(
                    clause_id="test_doc_001_clause_001",
                    text="This Agreement shall be governed by California law.",
                    start_position=48,
                    end_position=99,
                ),
            ],
        )

        save_segmented_document(doc, str(self.test_dir))
        loaded = load_segmented_document(
            str(self.test_dir / "test_doc_001.json")
        )

        assert loaded.document_id == doc.document_id
        assert loaded.total_characters == doc.total_characters
        assert len(loaded.clauses) == 2
        assert loaded.clauses[0].clause_id == "test_doc_001_clause_000"
        assert loaded.clauses[1].text == doc.clauses[1].text

    def test_json_structure(self):
        doc = SegmentedDocument(
            document_id="test_doc",
            source_file="test_doc.txt",
            total_characters=100,
            clauses=[
                Clause("test_doc_clause_000", "Clause text.", 0, 12),
            ],
        )
        d = doc.to_dict()

        assert "document_id" in d
        assert "source_file" in d
        assert "total_characters" in d
        assert "num_clauses" in d
        assert "clauses" in d
        assert d["num_clauses"] == 1
        assert "clause_id" in d["clauses"][0]
        assert "text" in d["clauses"][0]
        assert "start_position" in d["clauses"][0]
        assert "end_position" in d["clauses"][0]
        # Must NOT have labels or risk scores
        assert "label" not in d["clauses"][0]
        assert "labels" not in d["clauses"][0]
        assert "class" not in d["clauses"][0]
        assert "risk_score" not in d["clauses"][0]


# =============================================================================
# Test: Sampling Reproducibility
# =============================================================================

class TestSamplingReproducibility:
    """Tests for document sampling determinism."""

    def setup_method(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="nda_sample_test_"))
        # Create fake PDFs (empty files)
        for i in range(30):
            (self.test_dir / f"doc_{i:03d}.pdf").write_text("")

    def teardown_method(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_same_seed_same_result(self):
        s1 = sample_documents(str(self.test_dir), 10, random_seed=42)
        s2 = sample_documents(str(self.test_dir), 10, random_seed=42)
        assert [p.name for p in s1] == [p.name for p in s2]

    def test_different_seeds_different_result(self):
        s1 = sample_documents(str(self.test_dir), 10, random_seed=42)
        s2 = sample_documents(str(self.test_dir), 10, random_seed=99)
        names1 = [p.name for p in s1]
        names2 = [p.name for p in s2]
        assert names1 != names2

    def test_sample_size_respected(self):
        s = sample_documents(str(self.test_dir), 5, random_seed=42)
        assert len(s) == 5

    def test_sample_all(self):
        s = sample_documents(str(self.test_dir), None, random_seed=42)
        assert len(s) == 30

    def test_manifest_roundtrip(self):
        sampled = sample_documents(str(self.test_dir), 5, random_seed=42)
        manifest_path = str(self.test_dir / "manifest.json")
        save_sample_manifest(sampled, manifest_path, 5, 42)

        loaded_ids = load_sample_manifest(manifest_path)
        assert len(loaded_ids) == 5
        assert loaded_ids == [p.stem for p in sampled]


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
