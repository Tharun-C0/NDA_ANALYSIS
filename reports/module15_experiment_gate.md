# Module 15: Classifier Experiment Gate & Decision Matrix

## Executive Summary

This report establishes the official scientific Gate Decision for all 9 core experiments in the NDA research execution matrix.

- **Overall Gate Status**: **ALL EXPERIMENTS BLOCKED**
- **Policy Compliance**: Scientific standards forbid executing multi-epoch model training until dataset expansion resolves the 2-document category redundancy bottleneck for *Liability for Damages*.

---

## 1. Official Experiment Gate Decision Matrix

| Experiment ID | Model Architecture | Loss Function | Gate Decision | Blocking Reason / Prerequisite |
| :--- | :--- | :--- | :---: | :--- |
| **EXP-01** | `saibo/legal-roberta-base` | `bce` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |
| **EXP-02** | `saibo/legal-roberta-base` | `focal` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |
| **EXP-03** | `saibo/legal-roberta-base` | `class_weighted_bce` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |
| **EXP-04** | `nlpaueb/legal-bert-base-uncased` | `bce` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |
| **EXP-05** | `nlpaueb/legal-bert-base-uncased` | `focal` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |
| **EXP-06** | `nlpaueb/legal-bert-base-uncased` | `class_weighted_bce` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |
| **EXP-07** | `microsoft/deberta-v3-base` | `bce` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |
| **EXP-08** | `microsoft/deberta-v3-base` | `focal` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |
| **EXP-09** | `microsoft/deberta-v3-base` | `class_weighted_bce` | **BLOCKED** | Dataset expansion required (*Liability* in 2 docs). |

---

## 2. Gate Unlocking Criteria

The experiment gate will be unlocked and set to **READY** if and only if Stage 1 queue execution completes and `scripts/check_benchmark_readiness.py` returns `OVERALL BENCHMARK READINESS STATUS: READY`.
