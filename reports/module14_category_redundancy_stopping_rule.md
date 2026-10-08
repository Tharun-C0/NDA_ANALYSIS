# Module 14: Category Redundancy Stopping Rule

## Executive Summary

This document defines the official empirical stopping rule governing active learning dataset expansion. 

Rather than relying on arbitrary static document counts (e.g. "15 documents"), dataset expansion stops when and only when **enough document-level category redundancy exists to construct a defensible 3-way document-disjoint benchmark**.

---

## 1. Empirical Stopping Rule Definition

Active learning pseudo-labeling SHALL CEASE and final model training SHALL BE AUTHORIZED if and only if all five conditions below are satisfied simultaneously on the valid pseudo-label dataset:

$$\text{BENCHMARK READY} \iff (C_1 \land C_2 \land C_3 \land C_4 \land C_5) = \text{TRUE}$$

### Condition $C_1$: Universal Category Redundancy
- Every one of the 14 NDA categories MUST occur in at least **$\ge 3$ distinct represented documents**.
- *Mathematical Necessity*: In a 3-way split ($\text{Train}, \text{Val}, \text{Test}$), a category present in fewer than 3 documents is mathematically guaranteed to have zero support in at least one split.

### Condition $C_2$: Minority Category Protection
- Underrepresented bottleneck categories (*Liability for Damages*, *Competition Rights*, *Intellectual Property*, *Governing Law and Jurisdiction*) MUST occur in at least **$\ge 3$ distinct represented documents**.

### Condition $C_3$: Disjoint Split Coverage Balance
- There exists at least one deterministic document-disjoint partition where:
  - $\text{Train Split Categories} \ge 13 / 14$
  - $\text{Val Split Categories} \ge 10 / 14$
  - $\text{Test Split Categories} \ge 10 / 14$

### Condition $C_4$: Document Disjointness & Zero Leakage
- $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- Zero duplicate clause IDs across splits.

### Condition $C_5$: Human Verification Sample Availability
- A stratified human verification sample covering all categories and confidence tiers is prepared and ready for gold evaluation.

---

## 2. Current Empirical Evaluation

| Stopping Rule Condition | Required Target | Current Value (N=11 Docs) | Status |
| :--- | :--- | :---: | :---: |
| $C_1$: Universal Category Redundancy | Every cat in $\ge 3$ docs | 13 / 14 categories | **FAIL** (*Liability* in 2 docs) |
| $C_2$: Minority Category Protection | Bottleneck cats in $\ge 3$ docs | 3 / 4 categories | **FAIL** (*Liability* in 2 docs) |
| $C_3$: Disjoint Split Coverage | Train $\ge 13$, Val $\ge 10$, Test $\ge 10$ | Train 13, Val 13, Test 14 | **PASS** |
| $C_4$: Document Disjointness | 0 document overlap | 0 overlap | **PASS** |
| $C_5$: Human Verification Sample | Stratified sample available | Sample prepared (30 clauses) | **PASS** |

### Verdict: **BLOCKED (EXPANSION REQUIRED)**
Active learning pseudo-labeling must continue until *Liability for Damages* achieves $\ge 3$-document redundancy.
