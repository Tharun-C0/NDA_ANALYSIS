# Error Analysis Protocol

> **Status**: Protocol established. Diagnostics to be executed post-experimentation.
> Strictly distinguishes **OBSERVED PATTERNS** from **POSSIBLE EXPLANATIONS**.

---

## 1. Objectives

The goal of error analysis is to systematically characterize model failure modes across categories, label confidence tiers, and multi-label clause structures.

No causal claims will be asserted without direct empirical backing (e.g., error rate statistics, confusion analysis, or attention map verification).

---

## 2. Core Error Categories to Document

Every false prediction on the test set will be cataloged under one or more of the following diagnostic dimensions:

### 2.1 False Positives (FP)
- **Definition**: Model predicts label c = 1, ground-truth pseudo-label is c = 0.
- **Observed Metric**: FP count and FP rate per category.
- **Diagnostics**: Identify over-sensitive trigger phrases or keyword reliance (e.g. any mention of legal obligations triggering Governing Law).

### 2.2 False Negatives (FN)
- **Definition**: Model predicts label c = 0, ground-truth pseudo-label is c = 1.
- **Observed Metric**: FN count and FN rate per category.
- **Diagnostics**: Evaluate whether long context, non-standard phrasing, or minority representation caused missed detections.

### 2.3 Minority Category Performance
- **Target Categories**: Categories with lowest test support (e.g. Employees, Competition Rights).
- **Diagnostics**: Compare FN rate on minority vs majority categories under BCE vs Focal Loss.

### 2.4 Multi-Label Overlap & Confusion
- **Definition**: Clauses possessing >= 2 positive categories.
- **Diagnostics**: Measure whether multi-label clauses exhibit higher error rates than single-label clauses. Construct co-occurrence error matrix.

### 2.5 Additional Information Overprediction
- **Target Category**: C14 (Additional Information / Boilerplate).
- **Diagnostics**: Measure frequency of model assigning C14 concurrently with legal obligations vs assigning C14 as a fallback for low-confidence clauses.

### 2.6 Disagreement & Low-Confidence Pseudo-Label Analysis
- **Target Subsets**: Clauses labeled under DISAGREEMENT or LOW_CONFIDENCE by the Gemini ensemble.
- **Diagnostics**: Evaluate error rate on high-confidence test clauses vs low-confidence test clauses to quantify noise propagation.

---

## 3. Strict Separation of Observation vs Explanation

To maintain scientific integrity in reporting error analysis:

| Rule | Requirement |
| :--- | :--- |
| **OBSERVED PATTERN** | Quantified metric or empirical count observed directly in test results (e.g. Model EXP-01 produced 14 FP for C14). |
| **POSSIBLE EXPLANATION** | Hypothesized root cause explicitly framed as speculative (e.g. Possible explanation: C14 prompt guidance in LLM ensemble had high recall bias). |

**Rule**: Never assert a POSSIBLE EXPLANATION as factual causal truth without controlled ablation evidence.

---

## 4. Analytical Workflow

`
[Frozen Test Set Evaluation]
           |
           v
[Extract All FP and FN Cases]
           |
           v
[Aggregate Error Counts by Category & Confidence]
           |
           v
[Record OBSERVED PATTERNS in Table 7]
           |
           v
[Formulate & Qualify POSSIBLE EXPLANATIONS]
`

---
*Module 7 - Research Analysis and Experimental Framework*
*Created: 2026-10-03*
