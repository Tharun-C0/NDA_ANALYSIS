# Module 15: Benchmark Validation & Split Feasibility Audit

## Executive Summary

This report evaluates candidate document-disjoint split configurations under current empirical facts and projected future dataset states.

---

## 1. Current Empirical Benchmark Validation (N=11 Docs, 237 Clauses)

Evaluating candidate splits on the current 237 valid clauses:

- **Train Split**: 109 clauses across 3 documents (`0859334b`, `0a42e159`, `266929af`) — 13/14 categories present (Missing *Employees*).
- **Val Split**: 49 clauses across 2 documents (`0b59dfc4`, `293f5937`) — 12/14 categories present (Missing *Liability for Damages*, *Term and Termination*).
- **Test Split**: 79 clauses across 6 documents (`2268c5d1`, `247166e0`, `3504e06a`, `4fd432d8`, `5180f107`, `53c8f90c`) — 14/14 categories present.

### Integrity Checks
- **Document Overlap**: **0** (100% strict document-disjoint partitioning).
- **Clause Overlap**: **0** (No clause-level random splitting).
- **Duplicate Clause IDs**: **0**

### Benchmark Verdict
**CURRENT BENCHMARK STATUS: BLOCKED**

*Reason*: *Liability for Damages* is present in only 2 documents dataset-wide (`0859334b`, `5180f107`), making 3-way disjoint presence across Train, Validation, and Test impossible.

---

## 2. Simulated Future Benchmark State Validation (N=20 Docs, ~282 Clauses)

Projected split feasibility after Stage 1 queue execution:

- **Projected Train Split**: 10 docs, ~175 clauses — **14/14 categories present**
- **Projected Val Split**: 5 docs, ~55 clauses — **14/14 categories present**
- **Projected Test Split**: 5 docs, ~52 clauses — **14/14 categories present**

### Post-Expansion Integrity Protocol
1. Re-run `scripts/validate_experiment_integrity.py` to confirm 0 document overlap.
2. Verify *Liability for Damages* occurs in $\ge 3$ distinct documents.
3. Confirm Train, Val, and Test splits each contain $\ge 12$ categories simultaneously.

---

## 3. Benchmark Status Summary

The benchmark remains **BLOCKED** on current data, and will be converted to **READY** only after Stage 1 API queue execution fulfills all 5 stopping criteria conditions.
