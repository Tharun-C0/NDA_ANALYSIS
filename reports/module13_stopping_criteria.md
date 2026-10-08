# Module 13: Targeted Labeling Stopping Criteria

## Executive Summary

This document establishes the empirical stopping criteria governing when targeted active learning pseudo-labeling should cease and model training may proceed.

The stopping decision is derived **EXCLUSIVELY from the actual document $\times$ category matrix** and split simulation results.

---

## 1. Candidate Criteria vs. Empirical Evaluation

| Stopping Criterion | Target Condition | Current Empirical Value | Evaluation Status |
| :--- | :--- | :---: | :---: |
| **Document Diversity Target** | Total Represented Documents $N_{\text{docs}} \ge 14$ | 11 documents | **FAIL** |
| **Minimum Category Redundancy** | Every category present in $\ge 3$ distinct documents | 13 / 14 categories | **FAIL** (*Liability* in 2 docs) |
| **Minority Redundancy** | *Liability for Damages* present in $\ge 3$ distinct documents | 2 documents | **FAIL** |
| **Train Split Category Coverage** | $\ge 13 / 14$ categories present in Train | 13 / 14 categories | **PASS** |
| **Val Split Category Coverage** | $\ge 10 / 14$ categories present in Validation | 12 / 14 categories | **PASS** |
| **Test Split Category Coverage** | $\ge 10 / 14$ categories present in Test | 14 / 14 categories | **PASS** |
| **Leakage Isolation** | Zero document overlap across splits | 0 overlap | **PASS** |

---

## 2. Definitive Empirical Stopping Rule

Targeted pseudo-labeling shall STOP and model training shall be AUTHORIZED if and only if the following **3-Point Stopping Rule** is satisfied:

1. **Rule 1 (Category Redundancy)**: Every one of the 14 NDA categories MUST occur in at least **$\ge 3$ distinct documents** across the valid dataset.
2. **Rule 2 (Disjoint Split Balance)**: At least one document-disjoint split exists where:
   - $\text{Train Categories} \ge 13 / 14$
   - $\text{Val Categories} \ge 10 / 14$
   - $\text{Test Categories} \ge 10 / 14$
   - Zero categories have zero support in both Val AND Test simultaneously.
3. **Rule 3 (Document Diversity)**: At least 14 distinct documents are represented in the valid dataset.

---

## 3. Current Decision Verdict

**DECISION: CONTINUE TARGETED LABELING**

- **Reason**: Rule 1 is violated (*Liability for Damages* present in only 2 documents). Rule 3 is violated (11 documents represented).
- **Stopping Trigger**: Execute Stage 1 labeling (45 clauses across 9 unseen documents). Re-evaluate stopping criteria after Stage 1 completion.
