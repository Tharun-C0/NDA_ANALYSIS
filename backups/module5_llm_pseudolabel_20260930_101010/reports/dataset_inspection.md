# Kleister-NDA Dataset Inspection Report

> **Generated**: 2026-09-29
> **Inspector**: Module 1 — Dataset Acquisition & Inspection
> **Purpose**: Determine the structure and format of the Kleister-NDA dataset and compare it against the reference paper's description.

---

## 1. Repository Source

| Field | Value |
|-------|-------|
| **Repository** | [applicaai/kleister-nda](https://github.com/applicaai/kleister-nda) |
| **Git URL** | `https://github.com/applicaai/kleister-nda.git` |
| **Commit** | `c2c7bf069b919bfb618b268fda2a2c079c0db316` |
| **Commit Date** | 2021-08-18 |
| **Data Source** | [SEC EDGAR Database](https://www.sec.gov/edgar.shtml) |

---

## 2. Downloaded Repository Path

```
NDA/external_data/kleister-nda/
```

The repository was cloned into `external_data/` (separate from `data/raw/`) to preserve the original dataset structure without modification.

---

## 3. Directory Structure

```
kleister-nda/
├── .git/
├── .gitignore                  (11 bytes)
├── README.md                   (7,494 bytes)
├── config.txt                  (462 bytes — GEval metric config)
├── in-header.tsv               (64 bytes — TSV column header)
├── train/
│   ├── in.tsv.xz               (977,016 bytes — compressed input data)
│   ├── expected.tsv            (24,515 bytes — ground truth labels)
│   └── expected-original.tsv   (24,560 bytes — labels with original casing)
├── dev-0/
│   ├── in.tsv.xz               (314,624 bytes — compressed input data)
│   └── expected.tsv            (8,181 bytes — ground truth labels)
├── test-A/
│   └── in.tsv.xz               (695,484 bytes — compressed input, NO labels)
└── documents/
    └── *.pdf                   (726 PDF files)
```

---

## 4. Files Found

### Root-Level Files

| File | Size | Description |
|------|------|-------------|
| `README.md` | 7,494 B | Full dataset documentation |
| `config.txt` | 462 B | GEval evaluation metric configuration |
| `in-header.tsv` | 64 B | Column names: `filename`, `keys`, `text_djvu`, `text_tesseract`, `text_textract`, `text_best` |
| `.gitignore` | 11 B | Ignores `*~` and `*.pyc` |

### Split Directories

| Directory | `in.tsv.xz` | `expected.tsv` | `expected-original.tsv` |
|-----------|-------------|----------------|------------------------|
| `train/` | Yes (977 KB) | Yes (254 lines) | Yes (254 lines) |
| `dev-0/` | Yes (315 KB) | Yes (83 lines) | No |
| `test-A/` | Yes (695 KB) | No (hidden) | No |

### Documents Directory

- **726 PDF files** in `documents/`
- Filenames are MD5 hashes of binary content (e.g., `00782839aac5f3edc5ddeaf9642d454b.pdf`)
- File sizes range from ~5 KB to ~409 KB

---

## 5. File Formats

| Format | Location | Description |
|--------|----------|-------------|
| **PDF** | `documents/` | Original NDA documents (rendered from HTML via Puppeteer) |
| **TSV.XZ** | `train/`, `dev-0/`, `test-A/` | XZ-compressed TSV files containing document text extracted via multiple OCR methods |
| **TSV** | `train/`, `dev-0/` | Tab-separated ground truth label files |
| **TSV** | root | Column header file (`in-header.tsv`) |

### Input TSV Columns (`in.tsv.xz`)

| Column | Description |
|--------|-------------|
| `filename` | PDF filename (MD5 hash + `.pdf`) |
| `keys` | Space-separated list of keys to extract for this document |
| `text_djvu` | Text extracted by pdf2djvu/djvu2hocr |
| `text_tesseract` | Text extracted by Tesseract OCR |
| `text_textract` | Text extracted by textract |
| `text_best` | Best of pdf2djvu/tesseract (heuristic selection) |

### Expected TSV Format (`expected.tsv`)

Each line contains space-separated `key=value` pairs for one document:

```
effective_date=2001-04-18 jurisdiction=Oregon party=Eric_Dean_Sprunk party=Nike_Inc.
effective_date=2017-02-10 jurisdiction=California party=Kite_Pharma_Inc. party=Gilead_Sciences_Inc. term=7_years
```

---

## 6. Document Counts

| Split | Documents (from `expected.tsv` line count) | Source |
|-------|-------------------------------------------|--------|
| **Train** | 254 | Directly counted from file |
| **Dev-0** | 83 | Directly counted from file |
| **Test-A** | 203 | Stated in README (labels hidden) |
| **Total** | 540 | Sum of above |
| **PDF files in `documents/`** | 726 | Directly counted |

> **Note**: The README states: "train=254, dev-0=83, test-A=203" items. The total of 540 documents does not equal 726 PDFs. This likely means some PDFs are shared across splits or there are additional unused documents. The `in.tsv.xz` files reference specific PDFs per split.

---

## 7. Annotation / Label Format

### Keys (Attributes) Extracted

The dataset defines **4 key types** to extract from each document:

| Key | Description | Format |
|-----|-------------|--------|
| `effective_date` | Date the contract is legally binding | `YYYY-MM-DD` |
| `jurisdiction` | State or country of jurisdiction | Text with underscores |
| `party` | Contracting parties (may appear multiple times) | Text with underscores |
| `term` | Contract duration | `{number}_{units}` (e.g., `2_years`) |

### Label Representation

- Labels are **document-level key-value pairs**, NOT clause-level annotations.
- Each line in `expected.tsv` corresponds to one document.
- Multiple values for the same key (e.g., multiple `party=` entries) appear as separate pairs on the same line.
- Not all keys appear for every document (some are "decoy" keys with no value).
- The `expected-original.tsv` preserves original casing; `expected.tsv` is lowercased.

### Example

```
effective_date=2017-02-10 jurisdiction=California party=Kite_Pharma_Inc. party=Gilead_Sciences_Inc. term=7_years
```

---

## 8. Clause-Level Information Available

> **IMPORTANT: No clause-level annotations exist in this repository.**

The Kleister-NDA dataset provides:
- Full document text (OCR-extracted, in `in.tsv.xz`)
- Document-level key-value extraction labels (in `expected.tsv`)
- Original PDF documents (in `documents/`)
- **No clause segmentation**
- **No clause boundaries**
- **No clause-level class labels**
- **No `[INIT_CLAUSE]`/`[END_CLAUSE]` annotation format**
- **No 14-category clause classification labels**

---

## 9. Comparison with the Reference Paper

The reference paper describes:

| Paper Claims | Kleister-NDA Repository | Match? |
|--------------|------------------------|--------|
| 322 annotated NDA documents | 540 documents (254 train + 83 dev + 203 test) | NO — Different count |
| 3,714 clauses | No clause-level data | NO — Not present |
| 14 clause categories | 4 document-level keys (`effective_date`, `jurisdiction`, `party`, `term`) | NO — Different task |
| Multi-label clause classification | Document-level key-value extraction | NO — Different task type |
| TXT files with `[INIT_CLAUSE]` format | PDF documents + OCR-extracted TSV | NO — Different format |
| Clause text + class labels | Full document text + key-value pairs | NO — Different granularity |

> **CRITICAL FINDING: The Kleister-NDA repository does NOT contain the 3,714-clause annotated dataset described in the reference paper.** The repository provides a document-level information extraction benchmark, not a clause-level classification dataset.

---

## 10. Transformation Required

### What the paper's dataset requires (but is NOT available here):

1. **Clause segmentation**: The paper's authors performed manual or LLM-based segmentation to divide NDA documents into individual clauses. This annotation layer is **not part of the Kleister-NDA repository**.

2. **Clause classification labels**: The 14 clause categories and the `[INIT_CLAUSE]/[END_CLAUSE]` + `[INIT_CLASSE]/[END_CLASSE]` annotation format appear to be **created by the paper's authors as a derived dataset**, not part of the original Kleister-NDA benchmark.

3. **Document filtering**: The paper uses 322 documents vs. the repository's 540. The paper likely used a subset (possibly only train + dev-0 = 337, or another filtered subset).

### Possible paths forward:

| Option | Description | Feasibility |
|--------|-------------|-------------|
| **A. Contact the paper's authors** | Request their annotated 3,714-clause dataset directly | Best option for exact reproduction |
| **B. Use the raw Kleister-NDA PDFs** | Build our own clause segmentation and classification pipeline from the original PDFs | Requires significant annotation effort |
| **C. Use the OCR text** | Extract text from `in.tsv.xz` and build a segmentation pipeline | Text available but no clause boundaries |
| **D. Use the document-level task** | Work with the Kleister-NDA task as-is (key-value extraction) | Different task than the paper |

> **WARNING: Do NOT proceed with dataset conversion until a decision is made on which approach to take.** The gap between the repository's data and the paper's described dataset is fundamental — it cannot be bridged by simple format conversion.

---

## Summary

The Kleister-NDA GitHub repository is a **document-level information extraction** benchmark. It contains 540 NDA documents as PDFs with OCR-extracted text and 4 types of key-value labels. It does **not** contain the **clause-level segmentation and 14-category classification annotations** described in the reference paper. The paper's 3,714-clause dataset appears to be a **derived/annotated dataset created by the paper's authors** on top of (a subset of) the Kleister-NDA documents.
