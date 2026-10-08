# =============================================================================
# NDA Research Project - Test Suite for Module 1
# =============================================================================
# Tests cover:
#   1. Dataset loading from files
#   2. Clause parsing with correct delimiters
#   3. Multi-label parsing
#   4. Malformed input handling (graceful degradation)
#   5. Text preprocessing
#   6. Reproducible splitting
#
# Run with:  pytest tests/ -v
# =============================================================================

import os
import sys
import json
import tempfile
import shutil
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.dataset_loader import (
    ClauseRecord,
    parse_clauses_from_text,
    load_single_document,
    load_dataset,
)
from preprocessing.text_preprocessor import (
    normalize_whitespace,
    normalize_unicode,
    clean_encoding_artifacts,
    preprocess_clause_text,
    preprocess_labels,
    preprocess_clauses,
)
from preprocessing.dataset_splitter import split_by_documents
from preprocessing.dataset_statistics import compute_statistics


# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_SINGLE_LABEL = """
[INIT_CLAUSE]
The Receiving Party agrees to hold and maintain the Confidential Information
in strict confidence for the sole benefit of the Disclosing Party.
[INIT_CLASSE]Confidentiality[END_CLASSE]
[END_CLAUSE]
"""

SAMPLE_MULTI_LABEL = """
[INIT_CLAUSE]
The Receiving Party shall not disclose any Confidential Information
to third parties without prior written consent.
[INIT_CLASSE]Confidentiality, Non-Disclosure[END_CLASSE]
[END_CLAUSE]
"""

SAMPLE_MULTIPLE_CLAUSES = """
[INIT_CLAUSE]
This Agreement shall be governed by the laws of the State of California.
[INIT_CLASSE]Governing Law[END_CLASSE]
[END_CLAUSE]

[INIT_CLAUSE]
Either party may terminate this Agreement upon thirty days written notice.
[INIT_CLASSE]Termination[END_CLASSE]
[END_CLAUSE]

[INIT_CLAUSE]
The obligations under this NDA shall survive for a period of five years.
[INIT_CLASSE]Duration, Survival[END_CLASSE]
[END_CLAUSE]
"""

SAMPLE_MALFORMED_NO_LABELS = """
[INIT_CLAUSE]
This is a clause with no labels section.
[END_CLAUSE]
"""

SAMPLE_MALFORMED_EMPTY_LABELS = """
[INIT_CLAUSE]
This clause has empty labels.
[INIT_CLASSE][END_CLASSE]
[END_CLAUSE]
"""

SAMPLE_MALFORMED_EMPTY_TEXT = """
[INIT_CLAUSE]
[INIT_CLASSE]SomeLabel[END_CLASSE]
[END_CLAUSE]
"""

SAMPLE_MALFORMED_NO_CLAUSES = "This file has no clause delimiters at all."

SAMPLE_MIXED = """
Some header text that is not a clause.

[INIT_CLAUSE]
Valid clause number one with proper annotations.
[INIT_CLASSE]Obligations[END_CLASSE]
[END_CLAUSE]

Random text between clauses.

[INIT_CLAUSE]
Another valid clause about indemnification.
[INIT_CLASSE]Indemnification, Liability[END_CLASSE]
[END_CLAUSE]

Trailing text.
"""


# =============================================================================
# Test: Clause Parsing
# =============================================================================

class TestClauseParsing:
    """Tests for the core clause parsing logic."""
    
    def test_single_label_clause(self):
        """Verify that a clause with one label is parsed correctly."""
        clauses, warnings = parse_clauses_from_text(SAMPLE_SINGLE_LABEL, "test_doc")
        assert len(clauses) == 1
        assert clauses[0].labels == ["Confidentiality"]
        assert "Receiving Party" in clauses[0].text
        assert clauses[0].doc_id == "test_doc"
        assert clauses[0].clause_id == 0
        assert len(warnings) == 0
    
    def test_multi_label_clause(self):
        """Verify that comma-separated labels are parsed into a list."""
        clauses, warnings = parse_clauses_from_text(SAMPLE_MULTI_LABEL, "test_doc")
        assert len(clauses) == 1
        assert len(clauses[0].labels) == 2
        assert "Confidentiality" in clauses[0].labels
        assert "Non-Disclosure" in clauses[0].labels
        assert len(warnings) == 0
    
    def test_multiple_clauses(self):
        """Verify that multiple clause blocks in one file are all parsed."""
        clauses, warnings = parse_clauses_from_text(SAMPLE_MULTIPLE_CLAUSES, "test_doc")
        assert len(clauses) == 3
        assert clauses[0].clause_id == 0
        assert clauses[1].clause_id == 1
        assert clauses[2].clause_id == 2
        assert clauses[0].labels == ["Governing Law"]
        assert clauses[1].labels == ["Termination"]
        assert "Duration" in clauses[2].labels
        assert "Survival" in clauses[2].labels
    
    def test_clause_record_to_dict(self):
        """Verify ClauseRecord serialization."""
        record = ClauseRecord("doc1", 0, "test text", ["label1", "label2"])
        d = record.to_dict()
        assert d["global_id"] == "doc1_clause_0"
        assert d["doc_id"] == "doc1"
        assert d["clause_id"] == 0
        assert d["text"] == "test text"
        assert d["labels"] == ["label1", "label2"]
        assert d["num_labels"] == 2
    
    def test_mixed_content_parses_only_clauses(self):
        """Verify that non-clause text is ignored."""
        clauses, warnings = parse_clauses_from_text(SAMPLE_MIXED, "mixed_doc")
        assert len(clauses) == 2
        assert clauses[0].labels == ["Obligations"]
        assert "Indemnification" in clauses[1].labels


# =============================================================================
# Test: Malformed Input Handling
# =============================================================================

class TestMalformedInput:
    """Tests for graceful handling of malformed input."""
    
    def test_no_labels_section(self):
        """Clause without label section should be skipped with a warning."""
        clauses, warnings = parse_clauses_from_text(
            SAMPLE_MALFORMED_NO_LABELS, "bad_doc"
        )
        assert len(clauses) == 0
        assert len(warnings) > 0
        assert any("No class labels" in w for w in warnings)
    
    def test_empty_labels(self):
        """Clause with empty label section should be skipped."""
        clauses, warnings = parse_clauses_from_text(
            SAMPLE_MALFORMED_EMPTY_LABELS, "bad_doc"
        )
        assert len(clauses) == 0
        assert len(warnings) > 0
    
    def test_empty_clause_text(self):
        """Clause with no text (only labels) should be skipped."""
        clauses, warnings = parse_clauses_from_text(
            SAMPLE_MALFORMED_EMPTY_TEXT, "bad_doc"
        )
        assert len(clauses) == 0
        assert len(warnings) > 0
    
    def test_no_clause_delimiters(self):
        """File with no clause markers should produce a warning."""
        clauses, warnings = parse_clauses_from_text(
            SAMPLE_MALFORMED_NO_CLAUSES, "bad_doc"
        )
        assert len(clauses) == 0
        assert len(warnings) > 0
        assert any("No clause blocks" in w for w in warnings)
    
    def test_empty_string(self):
        """Empty string input should produce a warning."""
        clauses, warnings = parse_clauses_from_text("", "empty_doc")
        assert len(clauses) == 0
        assert len(warnings) > 0


# =============================================================================
# Test: File-based Loading
# =============================================================================

class TestFileLoading:
    """Tests for loading NDA files from disk."""
    
    def setup_method(self):
        """Create a temporary directory with test files."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="nda_test_"))
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _write_file(self, name: str, content: str) -> Path:
        filepath = self.test_dir / name
        filepath.write_text(content, encoding="utf-8")
        return filepath
    
    def test_load_single_document(self):
        """Load a single NDA file and verify parsing."""
        filepath = self._write_file("nda_001.txt", SAMPLE_MULTIPLE_CLAUSES)
        clauses, warnings = load_single_document(filepath)
        assert len(clauses) == 3
        assert all(c.doc_id == "nda_001" for c in clauses)
    
    def test_load_dataset_directory(self):
        """Load all files from a directory."""
        self._write_file("nda_001.txt", SAMPLE_SINGLE_LABEL)
        self._write_file("nda_002.txt", SAMPLE_MULTI_LABEL)
        self._write_file("nda_003.txt", SAMPLE_MULTIPLE_CLAUSES)
        
        clauses, warnings = load_dataset(str(self.test_dir))
        # 1 + 1 + 3 = 5 clauses total
        assert len(clauses) == 5
        doc_ids = set(c.doc_id for c in clauses)
        assert doc_ids == {"nda_001", "nda_002", "nda_003"}
    
    def test_load_nonexistent_directory(self):
        """Loading from non-existent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/this/path/does/not/exist")
    
    def test_load_empty_directory(self):
        """Loading from empty directory should raise ValueError."""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No files"):
            load_dataset(str(empty_dir))
    
    def test_load_ignores_non_txt_files(self):
        """Non-.txt files should be ignored."""
        self._write_file("nda_001.txt", SAMPLE_SINGLE_LABEL)
        self._write_file("readme.md", "# Not an NDA file")
        self._write_file("data.csv", "col1,col2")
        
        clauses, _ = load_dataset(str(self.test_dir))
        assert len(clauses) == 1


# =============================================================================
# Test: Text Preprocessing
# =============================================================================

class TestTextPreprocessing:
    """Tests for the text preprocessing pipeline."""
    
    def test_whitespace_normalization(self):
        """Multiple spaces and tabs should be collapsed."""
        text = "This   has    multiple   spaces\tand\ttabs."
        result = normalize_whitespace(text)
        assert "  " not in result
        assert "\t" not in result
    
    def test_preserves_single_newlines(self):
        """Single newlines should be preserved."""
        text = "Line one.\nLine two."
        result = normalize_whitespace(text)
        assert "\n" in result
    
    def test_collapses_excessive_newlines(self):
        """Three or more newlines should be collapsed to two."""
        text = "Paragraph one.\n\n\n\n\nParagraph two."
        result = normalize_whitespace(text)
        assert "\n\n\n" not in result
    
    def test_unicode_normalization(self):
        """Unicode should be normalized to NFC form."""
        # e with combining acute accent vs precomposed é
        text_nfd = "caf\u0065\u0301"  # e + combining acute
        result = normalize_unicode(text_nfd)
        assert result == "caf\u00e9"  # precomposed é
    
    def test_encoding_artifacts_cleaned(self):
        """Control characters should be removed."""
        text = "Clean\x00text\x0cwith\x01artifacts."
        result = clean_encoding_artifacts(text)
        assert "\x00" not in result
        assert "\x0c" not in result
        assert "\x01" not in result
        assert "Cleantext" in result
    
    def test_full_preprocessing_pipeline(self):
        """Full pipeline should clean text without destroying content."""
        text = "  Legal   clause\x00  with\t\t extra   spaces.  "
        result = preprocess_clause_text(text)
        assert result == "Legal clause with extra spaces."
    
    def test_label_preprocessing(self):
        """Labels should be lowercased, stripped, and deduplicated."""
        labels = ["  Confidentiality ", "Non-Disclosure", "confidentiality"]
        result = preprocess_labels(labels)
        assert result == ["confidentiality", "non-disclosure"]
    
    def test_preprocess_clauses_immutability(self):
        """Preprocessing should not modify original ClauseRecord objects."""
        original = ClauseRecord("doc1", 0, "  Original text  ", ["Label"])
        original_text = original.text
        processed = preprocess_clauses([original])
        assert original.text == original_text  # Original unchanged
        assert processed[0].text == "Original text"  # Processed is cleaned


# =============================================================================
# Test: Dataset Splitting
# =============================================================================

class TestDatasetSplitting:
    """Tests for the train/val/test splitting logic."""
    
    def _create_test_clauses(self, num_docs=10, clauses_per_doc=5):
        """Helper to create a set of test clauses from multiple documents."""
        clauses = []
        for d in range(num_docs):
            for c in range(clauses_per_doc):
                clauses.append(
                    ClauseRecord(
                        doc_id=f"doc_{d:03d}",
                        clause_id=c,
                        text=f"Clause {c} from document {d}.",
                        labels=["label_a"],
                    )
                )
        return clauses
    
    def test_split_produces_three_sets(self):
        """Splitting should produce exactly three non-empty sets."""
        clauses = self._create_test_clauses(num_docs=10)
        train, val, test = split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=42)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
    
    def test_split_no_overlap(self):
        """No document should appear in more than one split."""
        clauses = self._create_test_clauses(num_docs=20)
        train, val, test = split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=42)
        
        train_docs = set(c.doc_id for c in train)
        val_docs = set(c.doc_id for c in val)
        test_docs = set(c.doc_id for c in test)
        
        assert train_docs.isdisjoint(val_docs), "Train and val overlap!"
        assert train_docs.isdisjoint(test_docs), "Train and test overlap!"
        assert val_docs.isdisjoint(test_docs), "Val and test overlap!"
    
    def test_split_covers_all_clauses(self):
        """All clauses should be present across the three splits."""
        clauses = self._create_test_clauses(num_docs=10)
        train, val, test = split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=42)
        
        total = len(train) + len(val) + len(test)
        assert total == len(clauses)
    
    def test_split_reproducibility(self):
        """Same seed should produce identical splits."""
        clauses = self._create_test_clauses(num_docs=20)
        
        train1, val1, test1 = split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=42)
        train2, val2, test2 = split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=42)
        
        train_ids_1 = [c.global_id for c in train1]
        train_ids_2 = [c.global_id for c in train2]
        assert train_ids_1 == train_ids_2
        
        val_ids_1 = [c.global_id for c in val1]
        val_ids_2 = [c.global_id for c in val2]
        assert val_ids_1 == val_ids_2
    
    def test_different_seeds_produce_different_splits(self):
        """Different seeds should (very likely) produce different splits."""
        clauses = self._create_test_clauses(num_docs=20)
        
        train1, _, _ = split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=42)
        train2, _, _ = split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=99)
        
        train_docs_1 = set(c.doc_id for c in train1)
        train_docs_2 = set(c.doc_id for c in train2)
        # With 20 docs, different seeds should give different splits
        assert train_docs_1 != train_docs_2
    
    def test_invalid_ratios(self):
        """Ratios that don't sum to 1.0 should raise ValueError."""
        clauses = self._create_test_clauses(num_docs=10)
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_by_documents(clauses, 0.5, 0.3, 0.3, random_seed=42)
    
    def test_too_few_documents(self):
        """Fewer than 3 documents should raise ValueError."""
        clauses = self._create_test_clauses(num_docs=2)
        with pytest.raises(ValueError, match="at least 3"):
            split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=42)
    
    def test_document_level_integrity(self):
        """All clauses from one document must be in the same split."""
        clauses = self._create_test_clauses(num_docs=15, clauses_per_doc=8)
        train, val, test = split_by_documents(clauses, 0.7, 0.15, 0.15, random_seed=42)
        
        # For each split, check that if a doc_id appears, ALL its clauses are there
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            doc_ids_in_split = set(c.doc_id for c in split_data)
            for doc_id in doc_ids_in_split:
                expected_clause_ids = set(range(8))
                actual_clause_ids = set(
                    c.clause_id for c in split_data if c.doc_id == doc_id
                )
                assert actual_clause_ids == expected_clause_ids, (
                    f"Document {doc_id} in {split_name} is missing clauses: "
                    f"expected {expected_clause_ids}, got {actual_clause_ids}"
                )


# =============================================================================
# Test: Statistics
# =============================================================================

class TestStatistics:
    """Tests for dataset statistics computation."""
    
    def test_basic_statistics(self):
        """Verify that statistics are computed correctly for a small dataset."""
        clauses = [
            ClauseRecord("doc1", 0, "Short clause.", ["label_a"]),
            ClauseRecord("doc1", 1, "A longer clause with more words.", ["label_a", "label_b"]),
            ClauseRecord("doc2", 0, "Another document clause.", ["label_c"]),
        ]
        stats = compute_statistics(clauses)
        
        assert stats["documents"]["total_documents"] == 2
        assert stats["clauses"]["total_clauses"] == 3
        assert stats["clauses"]["single_label_clauses"] == 2
        assert stats["clauses"]["multi_label_clauses"] == 1
        assert stats["labels"]["num_unique_labels"] == 3
    
    def test_empty_clauses(self):
        """Statistics for empty list should return error."""
        stats = compute_statistics([])
        assert "error" in stats


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
