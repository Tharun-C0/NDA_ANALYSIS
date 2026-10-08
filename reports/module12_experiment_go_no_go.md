# Module 12: Classifier Experiment Go / No-Go Decision Report

## Executive Summary

This report establishes the scientific Go / No-Go decision for all 9 core transformer classifier experiments in the NDA research matrix.

- **Overall Execution Verdict**: **NO-GO FOR ALL EXPERIMENTS**
- **Reasoning**: Training multi-epoch models on a dataset where the evaluation benchmark is blocked by document/category concentration (*Liability for Damages* present in only 2 documents) would produce scientifically invalid, un-publishable results.

---

## 1. Experiment Matrix Go / No-Go Audit

| Experiment ID | Model Architecture | Loss Function | Scientific Execution Decision | Missing Prerequisite / Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **EXP-01** | `saibo/legal-roberta-base` | `bce` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |
| **EXP-02** | `saibo/legal-roberta-base` | `focal` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |
| **EXP-03** | `saibo/legal-roberta-base` | `class_weighted_bce` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |
| **EXP-04** | `nlpaueb/legal-bert-base-uncased` | `bce` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |
| **EXP-05** | `nlpaueb/legal-bert-base-uncased` | `focal` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |
| **EXP-06** | `nlpaueb/legal-bert-base-uncased` | `class_weighted_bce` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |
| **EXP-07** | `microsoft/deberta-v3-base` | `bce` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |
| **EXP-08** | `microsoft/deberta-v3-base` | `focal` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |
| **EXP-09** | `microsoft/deberta-v3-base` | `class_weighted_bce` | **NO-GO** | Benchmark expansion (+3-5 docs) and 3-way category redundancy. |

---

## 2. Scientific Prerequisites for "GO" Authorization

Training will be authorized only when ALL four prerequisites are fulfilled:
1. **Prerequisite 1**: Total represented documents $N_{\text{docs}} \ge 14$ (currently 11).
2. **Prerequisite 2**: *Liability for Damages* present in $\ge 4$ distinct documents (currently 2).
3. **Prerequisite 3**: At least one document-disjoint split achieves $\text{Train} \ge 13$, $\text{Val} \ge 10$, $\text{Test} \ge 10$ categories without category starvation.
4. **Prerequisite 4**: Human verification of the test set gold annotations is completed.
