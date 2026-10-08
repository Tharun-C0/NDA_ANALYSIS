# Module 9: Document-Disjoint Benchmark Construction Analysis & Readiness Report

## Executive Summary

- **Current Benchmark Status**: **BLOCKED** ("benchmark dataset not yet ready")
- **Infrastructure Status**: **READY** ("benchmark-ready infrastructure" established)
- **Primary Blocker**: Severe category imbalance across represented documents. With only **7 represented documents**, any document-disjoint 3-way partition (Train / Validation / Test) fails to provide adequate category coverage across all three splits simultaneously.
- **Key Empirical Finding**: Category coverage cannot be assumed simply by reaching a target document count (e.g., 15 documents). Document diversity and internal category richness per document are the deciding factors.

---

## 1. Current Dataset Size & Provenance Audit

- **Total CSV Rows**: 254
- **Valid Pseudo-Labeled Clauses**: 203 (80.0%)
- **API Error Rows**: 49 (19.3%) - Excluded from dataset
- **API Quota Exhausted Rows**: 2 (0.78%) - Excluded from dataset
- **Label Provenance**: 100% of valid records bear explicit provenance metadata (`label_source = "llm_pseudo_label"`, model ID `gemini-2.5-flash`, confidence tier, and pairwise model agreement score).

### Confidence Tier Breakdown
- **HIGH_CONFIDENCE**: 195 clauses (96.06%)
- **MEDIUM_CONFIDENCE**: 4 clauses (1.97%)
- **LOW_CONFIDENCE**: 0 clauses (0.00%)
- **DISAGREEMENT**: 4 clauses (1.97%)

---

## 2. Current Document Diversity

- **Represented Documents**: 7 distinct NDA documents in valid dataset
  - `0859334b` (50 clauses, 13 categories present)
  - `0b59dfc4` (33 clauses, 10 categories present)
  - `2268c5d1` (25 clauses, 1 category present - *Additional Information*)
  - `247166e0` (20 clauses, 1 category present - *Additional Information*)
  - `30647c0b` (43 clauses, 4 categories present)
  - `32668bcf` (16 clauses, 6 categories present)
  - `35a40a59` (16 clauses, 6 categories present)
- **Unseen Documents in Queue**: 13 documents (465 clauses remaining)
- **Total Corpus Size**: 20 documents (668 total clauses across corpus)

---

## 3. Document / Category Matrix Summary

| Document ID | Valid Clauses | Categories Present | Dominant Categories | Single-Category Doc? |
| :--- | :---: | :---: | :--- | :---: |
| `0859334b` | 50 | **13 / 14** | Definition of Confidential Information (11), Confidentiality Obligations (9), Exceptions to Confidentiality (8) | No |
| `0b59dfc4` | 33 | **10 / 14** | Definition of Confidential Information (10), Confidentiality Obligations (6), Purpose of Disclosure (4) | No |
| `2268c5d1` | 25 | **1 / 14** | Additional Information (25) | **YES** |
| `247166e0` | 20 | **1 / 14** | Additional Information (20) | **YES** |
| `30647c0b` | 43 | **4 / 14** | Definition of Confidential Information (19), Purpose of Disclosure (16), Exceptions to Confidentiality (7) | No |
| `32668bcf` | 16 | **6 / 14** | Definition of Confidential Information (5), Return or Destruction of Information (5), Confidentiality Obligations (3) | No |
| `35a40a59` | 16 | **6 / 14** | Term of Agreement (4), Definition of Confidential Information (4), Confidentiality Obligations (3) | No |

### Document Categorization Breakdown
- **Single-Category Documents (1 category)**: 2 documents (`2268c5d1`, `247166e0`)
- **Moderate Diversity (3-5 categories)**: 1 document (`30647c0b`)
- **High Diversity (>= 5 categories)**: 4 documents (`0859334b`, `0b59dfc4`, `32668bcf`, `35a40a59`)
- **Maximum Diversity Document**: `0859334b` (13 out of 14 NDA categories)

---

## 4. Candidate Split Strategies Evaluation

Four deterministic document-disjoint split strategies were evaluated on the 7 valid documents:

### Strategy A: Size-Based Ratio (70% Train / 15% Val / 15% Test by clause count)
- **Train (5 docs - `0859334b`, `30647c0b`, `0b59dfc4`, `2268c5d1`, `247166e0`)**: 184 clauses | 14/14 categories present | Missing 0 categories
- **Val (1 doc - `32668bcf`)**: 16 clauses | 6/14 categories present | Missing 8 categories
- **Test (1 doc - `35a40a59`)**: 3 clauses (filtered) / 16 total | 2/14 categories present | **Missing 12 categories**
- **Assessment**: **UNSUITABLE**. Test split missing 12 of 14 categories.

### Strategy B: Standard ML Ratio (60% Train / 20% Val / 20% Test by clause count)
- **Train (4 docs - `0859334b`, `30647c0b`, `0b59dfc4`, `2268c5d1`)**: 164 clauses | 14/14 categories present | Missing 0 categories
- **Val (2 docs - `247166e0`, `32668bcf`)**: 36 clauses | 6/14 categories present | Missing 8 categories
- **Test (1 doc - `35a40a59`)**: 3 clauses (filtered) / 16 total | 2/14 categories present | **Missing 12 categories**
- **Assessment**: **UNSUITABLE**. Test split missing 12 of 14 categories.

### Strategy C: Greedy Category Maximization (Doc `0859334b` assigned to Test)
- **Train (4 docs - `2268c5d1`, `247166e0`, `32668bcf`, `35a40a59`)**: 104 clauses | 4/14 categories present | **Missing 10 categories**
- **Val (2 docs - `0b59dfc4`, `30647c0b`)**: 49 clauses | 12/14 categories present | Missing 2 categories
- **Test (1 doc - `0859334b`)**: 50 clauses | 13/14 categories present | Missing 1 category
- **Assessment**: **UNSUITABLE**. Training split is stripped of 10 categories, destroying model training viability.

### Strategy D: Train Category Focus (Doc `0859334b` assigned to Train)
- **Train (1 doc - `0859334b`)**: 50 clauses | 13/14 categories present | Missing 1 category (*No Assignment of Rights*)
- **Val (2 docs - `0b59dfc4`, `30647c0b`)**: 89 clauses | 11/14 categories present | Missing 3 categories
- **Test (4 docs - `2268c5d1`, `247166e0`, `32668bcf`, `35a40a59`)**: 64 clauses | 7/14 categories present | **Missing 7 categories**
- **Assessment**: **UNSUITABLE**. Test split missing 7 of 14 categories.

---

## 5. Missing-Category & Coverage Analysis

The fundamental cause of split failure is **extreme category concentration**:
1. **Document `0859334b`** acts as a single-point-of-failure for category coverage. Placing it in Test leaves Train starved of 10 categories; placing it in Train leaves Test missing 7 to 12 categories.
2. **Documents `2268c5d1` and `247166e0`** contribute 45 clauses total, but 100% of their clauses belong to a single category (*Additional Information*). They provide no multi-label or multi-category diversity.
3. **Minority Categories Concentration**: Categories like *Remedies for Breach* (3 clauses), *No Assignment of Rights* (2 clauses), *Governing Law & Jurisdiction* (5 clauses), and *Non-Solicitation* (6 clauses) exist in only 1 or 2 documents across the current 7-doc set.

---

## 6. Document Leakage & Provenance Integrity

- **Document Overlap Count across Train / Val / Test**: **0** (100% strict document-disjoint partitioning enforced across all strategies).
- **Clause Overlap**: **0** (No clause-level random splitting allowed).
- **Data Provenance**: Every record maintains full traceability back to source document, clause ID, LLM prompt version, and model confidence scores.

---

## 7. Pseudo-Label & Human Verification Limitations

1. **Non-Ground-Truth Disclaimer**: LLM pseudo-labels are **not human ground truth**. They represent automated predictions generated by `gemini-2.5-flash` ensemble validation and are subject to potential systematic model biases.
2. **Human Verification Status**: A human verification sample of 30 clauses (`data/annotations/human_verification_sample.csv`) has been generated following stratified sampling rules across confidence tiers and categories. Human verification is currently pending.
3. **Evaluation Protocol Constraint**: Final benchmark metrics (Macro F1, Precision, Recall) reported in published work MUST be evaluated against human-verified ground-truth test labels, not raw LLM pseudo-labels.

---

## 8. Empirical Minimum Document Requirement Analysis

Using the metadata of the **13 remaining unseen documents** in the queue:
- **Queue Clause Distribution**: `9a5cb310` (74), `586c367e` (63), `f28c4f3d` (60), `d4566b17` (45), `d714d261` (43), `e52e4a13` (43), `64303e5a` (23), `4fd432d8` (22), `53c8f90c` (22), `b82a10c4` (18), `3504e06a` (16), `c58882f7` (16), `5180f107` (10). Total = 465 clauses.
- **Empirical Rule**: A static target of 15 documents does **not** guarantee benchmark readiness if new documents also suffer from single-category concentration.
- **Minimum Observed Document Count Required**: Based on empirical category density, a minimum of **12 to 15 well-distributed, multi-category documents** (with at least 3-5 categories per doc) is required to achieve at least 10+ categories represented in Test and Validation while maintaining >=12 categories in Train.

---

## 9. Recommended Dataset Expansion Action

1. **Await Gemini API Quota Reset**: Resume queue runner once API quota resets (~17.68h window).
2. **Prioritize Unseen High-Density Documents**: Run `scripts/resume_runner.py` prioritizing unseen documents `9a5cb310`, `586c367e`, and `f28c4f3d` (representing 197 clauses across 3 new documents).
3. **Re-evaluate Benchmark Readiness**: Automatically re-run `scripts/analyze_document_splits.py` after pseudo-labeling 5 additional documents to test if 12 represented documents break the bottleneck.
