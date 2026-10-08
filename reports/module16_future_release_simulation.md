# Module 16: Future Benchmark Release Simulation

## Executive Summary

This report simulates projected benchmark status changes across three future dataset expansion scenarios using the Module 14 prioritized queue (`data/annotations/module14_final_targeted_queue.csv`).

---

## 1. facts vs. Hypothetical Release Scenarios

> [!IMPORTANT]
> - **CURRENT FACTS**: Verified empirical status of 237 valid pseudo-labeled clauses across 11 represented documents.
> - **FUTURE SCENARIOS**: Simulated projections based on processing un-completed queue documents. All future scenarios report document-level redundancy projections without fabricating or assigning clause-level labels.

---

## 2. Release Scenario Simulation Matrix

| Release Scenario | Processing Target | Represented Documents ($N_{\text{docs}}$) | Projected Valid Clauses | Projected Category Redundancy | Benchmark Status Verdict |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Current Baseline State** | None (Current dataset) | **11 docs** | **237 clauses** | *Liability for Damages* in only 2 docs; missing from Val | **BLOCKED** |
| **Scenario 1: Stage 1 Diversity Sampling** | Process 45 clauses (5 per doc across 9 unseen docs) | **20 docs** | **~282 clauses** | Projected $\ge 4$ docs for *Liability*, $\ge 5$ docs for *Competition*, *IP*, *Gov Law* | **HYPOTHETICALLY READY** |
| **Scenario 2: High-Signal Targeted Expansion** | Process top 100 queue clauses matching target signals | **20 docs** | **~337 clauses** | Projected $\ge 5$ docs for all bottleneck categories | **HYPOTHETICALLY READY** |
| **Scenario 3: Full Queue Execution** | Process all 423 pending queue clauses | **20 docs** | **~660 clauses** | Maximum corpus coverage across all 20 Kleister-NDA documents | **HYPOTHETICALLY READY** |

---

## 3. Projected Benchmark Split Improvement

### Current Baseline (Current Facts)
- Train: 5 docs, 154 clauses | 13/14 categories (*Employees* missing)
- Val: 3 docs, 55 clauses | 13/14 categories (*Liability for Damages* missing)
- Test: 3 docs, 28 clauses | 14/14 categories
- **Status**: **BLOCKED** (*Liability for Damages* in 2 docs total)

### Projected Scenario 1 (Hypothetical Post-Expansion)
- Projected Train: 10 docs, ~175 clauses | **14/14 categories present**
- Projected Val: 5 docs, ~55 clauses | **14/14 categories present**
- Projected Test: 5 docs, ~52 clauses | **14/14 categories present**
- **Status**: **HYPOTHETICALLY READY**

---

## 4. Verification Milestone

When Gemini API quota resets, executing `python scripts/module15_active_learning_runner.py --max-clauses 45` will convert Scenario 1 from a hypothetical simulation into confirmed empirical fact.
