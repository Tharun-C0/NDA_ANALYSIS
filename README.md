# NDA Research Project

**A Two-Stage Architecture for NDA Analysis: LLM-based Segmentation and Transformer-based Clause Classification**

This project reproduces and extends the baseline methodology from the reference paper, with the goal of improving NDA clause classification using different models and techniques.

---

## Project Structure

```
NDA-Research-Project/
├── data/
│   ├── raw/              # Place raw NDA .txt files here
│   ├── processed/        # Cleaned/parsed clause data (auto-generated)
│   └── splits/           # Train/val/test splits (auto-generated)
├── models/               # Saved model checkpoints (future)
├── preprocessing/        # Dataset loading, cleaning, splitting
│   ├── dataset_loader.py
│   ├── text_preprocessor.py
│   ├── dataset_statistics.py
│   ├── dataset_splitter.py
│   └── prepare_dataset.py    # ← Main entry point for Module 1
├── training/             # Model training scripts (future)
├── evaluation/           # Evaluation and metrics (future)
├── experiments/          # Experiment tracking (future)
├── reports/              # Statistics, logs, CSV reports
├── configs/
│   ├── config.yaml       # Central configuration
│   └── config_loader.py  # Configuration utility
├── app/                  # GUI / API (future)
├── notebooks/            # Exploratory analysis (future)
├── tests/                # Unit and integration tests
│   └── test_module1.py
├── requirements.txt
└── README.md
```

---

## Module 1: Dataset Preparation

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1: Place Your Dataset

Place your annotated NDA `.txt` files in the `data/raw/` directory.

Each file should represent one NDA document and use the following annotation format:

```
[INIT_CLAUSE]
The Receiving Party agrees to hold and maintain the Confidential
Information in strict confidence for the sole benefit of the
Disclosing Party.
[INIT_CLASSE]Confidentiality[END_CLASSE]
[END_CLAUSE]

[INIT_CLAUSE]
The Receiving Party shall not disclose any Confidential Information
to third parties without prior written consent.
[INIT_CLASSE]Confidentiality, Non-Disclosure[END_CLASSE]
[END_CLAUSE]
```

**Format details:**
- `[INIT_CLAUSE]` / `[END_CLAUSE]` — delimit clause text
- `[INIT_CLASSE]` / `[END_CLASSE]` — delimit class labels (inside the clause block)
- Multiple labels are comma-separated
- One file = one NDA document (filename used as document ID)

### Step 2: Run the Pipeline

From the project root directory:

```bash
python preprocessing/prepare_dataset.py
```

Or with a custom config:

```bash
python preprocessing/prepare_dataset.py --config path/to/config.yaml
```

### Step 3: Review Outputs

After running, you will find:

| Output | Location |
|--------|----------|
| Processed clauses | `data/processed/all_clauses.json` |
| Training split | `data/splits/train.json` |
| Validation split | `data/splits/val.json` |
| Test split | `data/splits/test.json` |
| Split metadata | `data/splits/split_info.json` |
| Dataset statistics | `reports/dataset_statistics.json` |
| Class frequencies | `reports/class_frequencies.csv` |
| Processing log | `reports/logs/prepare_dataset.log` |

The console will also print a formatted summary of dataset statistics and class distribution.

### Step 4: Run Tests

```bash
pytest tests/ -v
```

---

## Configuration

All parameters are centralized in `configs/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `random_seed` | `42` | Seed for reproducible splitting |
| `dataset.raw_dir` | `data/raw` | Input directory for raw NDA files |
| `sampling.sample_size` | `20` | Number of PDFs to process (null = all) |
| `splitting.train_ratio` | `0.70` | Proportion of documents for training |
| `segmentation.strategy` | `baseline` | Segmentation method: `baseline` or future `llm` |
| `segmentation.min_clause_length` | `30` | Minimum characters for a clause |
| `extraction.min_characters` | `50` | Minimum chars for successful extraction |
| `output.save_format` | `json` | Output format: `json` or `csv` |

---

## Module 2: NDA Text Extraction & Clause Segmentation

### Overview

Module 2 extracts text from NDA PDF documents and segments them into candidate clauses using a rule-based baseline approach. The segmenter architecture is modular, designed for future replacement with LLM-based methods.

> **IMPORTANT**: The automatically segmented clauses are preliminary outputs and are **NOT considered ground-truth annotations**.

### Input

- **Source**: PDF documents from `external_data/kleister-nda/documents/`
- **Sampling**: Configurable number of documents via `configs/config.yaml` → `sampling.sample_size`
- The original external dataset is **never modified**

### Running Module 2

```bash
python preprocessing/run_segmentation.py
```

### Pipeline Steps

1. **Sampling** — Select N documents reproducibly (fixed seed)
2. **PDF Extraction** — Extract text from PDFs using PyMuPDF
3. **Quality Check** — Validate extraction (char count, word count, status)
4. **Clause Segmentation** — Detect clause boundaries via legal formatting patterns
5. **Save Clauses** — Structured JSON with clause text and character positions
6. **Review Queue** — CSV file for human verification workflow
7. **Statistics** — Compute and save segmentation metrics
8. **Visualization** — Bar chart of clauses per document

### Baseline Segmentation

The baseline segmenter detects boundaries using:
- Numbered sections (`1.`, `1.1`, `1.1.1`)
- Article/Section headings (`Article I`, `Section 2`)
- Legal headings (ALL CAPS lines)
- Parenthesized markers (`(a)`, `(i)`, `(1)`)
- Paragraph boundaries

Short segments are merged; long segments are split at paragraph breaks.

### Output

| Output | Location |
|--------|----------|
| Extracted text files | `data/processed/text/` |
| Segmented clause JSON | `data/segmentation/` |
| Sample manifest | `data/segmentation/sample_manifest.json` |
| Human review queue | `data/segmentation/review_queue.csv` |
| Extraction report | `reports/text_extraction_report.csv` |
| Segmentation statistics | `reports/segmentation_statistics.csv` |
| Clause count chart | `reports/figures/clauses_per_document.png` |

### Human Verification

The review queue CSV contains all extracted clauses with `human_verified=false`. This file is designed for manual review before any clauses are treated as ground truth.

### Modular Segmenter Architecture

The segmenter uses an abstract base class (`BaseSegmenter`) that can be extended:

```python
class LLMSegmenter(BaseSegmenter):
    def segment(self, text, document_id):
        # Use Llama, Qwen, Mistral, or Gemma
        ...
```

Future LLM-based segmenters can be plugged in by changing `segmentation.strategy` in the config.

### Limitations

1. The baseline segmenter uses heuristic rules — it does not understand legal semantics
2. Some clauses may be over-segmented or under-segmented
3. No clause labels or risk scores are assigned
4. Results require human review before use in training

---

## Research Design Decisions

1. **Document-level splitting**: All clauses from a single NDA stay in the same split to prevent data leakage.
2. **Conservative preprocessing**: Only formatting artifacts are cleaned. No paraphrasing, lowercasing of text, or removal of legal content.
3. **Label normalization**: Labels are lowercased and deduplicated for consistency across annotations.
4. **Deterministic pipeline**: Fixed random seed + sorted file discovery ensures identical results on every run.
5. **Graceful error handling**: Malformed annotations are skipped with warnings, not silent failures.
6. **Modular segmentation**: Abstract base class enables drop-in replacement of baseline with LLM-based segmenters.

---

## External Dataset: Kleister-NDA

### Storage Location

The original Kleister-NDA repository is stored at:

```
NDA/external_data/kleister-nda/
```

This is **intentionally separate** from `data/raw/` for the following reasons:

1. **Preservation**: The original repository must remain completely untouched. No files should be modified, renamed, or deleted.
2. **Format mismatch**: The Kleister-NDA repository provides a document-level key-value extraction task (PDFs + `expected.tsv` with `effective_date`, `jurisdiction`, `party`, `term` labels). The reference paper describes a clause-level classification task (3,714 clauses with 14 categories). These are fundamentally different.
3. **Separation of concerns**: `data/raw/` is reserved for data in the project's expected format (`[INIT_CLAUSE]`/`[END_CLAUSE]` annotated TXT files). The external repository uses a completely different format.
4. **Auditability**: Keeping the original data separate makes it clear what was downloaded vs. what was derived/transformed.

### Key Finding

> **The Kleister-NDA repository does NOT contain the clause-level annotations described in the reference paper.** See `reports/dataset_inspection.md` for the full analysis.

The repository contains 540 NDA documents (254 train + 83 dev + 203 test) with document-level labels, while the paper describes 322 documents with 3,714 clause-level annotations across 14 categories. The paper's dataset appears to be a derived annotation layer created by the paper's authors.

### Dataset Manifest

Metadata about the downloaded dataset is recorded in:

```
data/dataset_manifest.json
```

### Next Steps

Before proceeding to dataset conversion:
1. Review the inspection report at `reports/dataset_inspection.md`
2. Decide whether to contact the paper's authors for their annotated dataset, or build clause segmentation from the raw PDFs/OCR text
3. Only after this decision should data be placed into `data/raw/`

---

## Reference Paper

> *A Two-Stage Architecture for NDA Analysis: LLM-based Segmentation and Transformer-based Clause Classification*
>
> Dataset: Kleister-NDA (322 documents, 3,714 clauses, 14 categories)

---

## Status

- [x] Module 1: Project Foundation & Dataset Preparation
- [x] Module 1b: Dataset Acquisition & Inspection
- [x] Module 2: NDA Text Extraction & Clause Segmentation
- [ ] Module 3: Clause Labeling & Classification
- [ ] Module 4: Evaluation & Analysis
- [ ] Module 5: Application & Deployment
