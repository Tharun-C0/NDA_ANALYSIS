# Module 11: Benchmark Stopping Criteria & Readiness Audit

## Executive Summary

This document establishes the empirical stopping criteria required to declare the NDA multi-label clause classification benchmark **READY** for model training and research publication.

---

## 1. Evaluation of Candidate Stopping Criteria Against Empirical Data

Seven candidate stopping criteria were evaluated against the current 11-document empirical matrix:

| Candidate Criterion | Current Empirical Value | Evaluation Status | Empirical Findings & Vulnerability |
| :--- | :---: | :---: | :--- |
| **1. Total Represented Documents $\ge 12$** | 11 documents | **INSUFFICIENT** | Having 11 documents still leaves *Liability for Damages* present in only 2 docs. 12 docs alone cannot guarantee coverage if the 12th doc is single-category. |
| **2. Total Represented Documents $\ge 15$** | 11 documents | **INSUFFICIENT ALONE** | Document count alone is a proxy metric. 15 documents with heavy single-category bias can still fail 3-way split coverage. |
| **3. Every Category Present in $\ge 3$ Documents** | 13 / 14 categories | **FAIL** | *Liability for Damages* is present in only 2 docs (`0859334b`, `5180f107`). In a 3-way split, at least one split is guaranteed zero support. |
| **4. Every Category Present in $\ge 4$ Documents** | 9 / 14 categories | **FAIL** | 5 categories (*Liability for Damages*, *Competition Rights*, *Intellectual Property*, *Governing Law*, *Term & Termination*) have $<4$ docs. |
| **5. No Category Confined to Single Document** | **14 / 14 categories** | **PASS** | Passed (lowest category support is 2 docs for *Liability for Damages*). However, 2 docs are still insufficient for a 3-way split. |
| **6. Train Split Contains $\ge 12 / 14$ Categories** | 13 / 14 categories | **PASS** | Currently satisfied under Strategy C and D. |
| **7. Test Split Contains $\ge 10 / 14$ Categories** | 7 to 13 / 14 categories | **CONDITIONAL** | Satisfied under Strategy C/D, but at the cost of stripping categories from Train. |

---

## 2. Definitive Multi-Condition Empirical Stopping Criterion

To prevent false readiness claims, a **Composite 5-Point Empirical Rule** is mandated. All 5 conditions must be satisfied simultaneously:

$$\text{BENCHMARK READY} \iff \bigwedge_{k=1}^{5} C_k = \text{TRUE}$$

### Condition $C_1$: Minimum Document Diversity
- **Requirement**: Total represented documents in valid dataset $N_{\text{docs}} \ge 14$.

### Condition $C_2$: Minimum Category Redundancy
- **Requirement**: Every one of the 14 NDA categories MUST occur in at least **$\ge 3$ distinct documents**.
- *Rationale*: Guarantees that at least 1 document containing the category can be allocated to Train, 1 to Validation, and 1 to Test.

### Condition $C_3$: Minority Category Redundancy
- **Requirement**: Severe bottleneck categories (*Liability for Damages*, *Competition Rights*, *Intellectual Property*) MUST occur in at least **$\ge 3$ distinct documents**.

### Condition $C_4$: Balanced Split Coverage
- **Requirement**: There exists at least one deterministic document-disjoint split where:
  - $\text{Train Category Count} \ge 13 / 14$
  - $\text{Val Category Count} \ge 10 / 14$
  - $\text{Test Category Count} \ge 10 / 14$

### Condition $C_5$: Zero Document & Clause Leakage
- **Requirement**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

---

## 3. Current Benchmark Readiness Audit

- $C_1$ (Docs $\ge 14$): **FAIL** (11 docs)
- $C_2$ (All cats in $\ge 3$ docs): **FAIL** (*Liability for Damages* in 2 docs)
- $C_3$ (Minority cats in $\ge 3$ docs): **FAIL** (*Liability for Damages* in 2 docs)
- $C_4$ (Balanced split coverage $\ge 10$ cats each): **FAIL** (Strategy A/B missing 7 cats in Test; Strategy C missing 3 cats in Train)
- $C_5$ (Zero leakage): **PASS**

### Overall Verdict: **BLOCKED**
