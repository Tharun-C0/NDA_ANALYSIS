# LLM-Assisted Pseudo-Labeling Methodology

## Research Context & Objectives
This research experiment builds an automated, multi-model LLM pseudo-labeling pipeline to generate initial multi-label predictions across 699 NDA clauses for 14 standard NDA clause categories. 

**IMPORTANT RESEARCH DISCLAIMER**:
All labels produced by this pipeline are explicitly flagged with `label_source = llm_pseudo_label`. They are **NOT** expert human ground-truth annotations and must never be treated as such. They serve as pre-annotations and pseudo-labeled experimental datasets to compare model performance under varying data regime conditions.

---

## 1. Multi-Model Ensemble Architecture
To avoid reliance or bias from any single LLM, we employ a 3-model ensemble using available Gemini models:
- **Model A**: `gemini-3.6-flash`
- **Model B**: `gemini-3.5-flash`
- **Model C**: `gemini-3.1-flash-lite`

Each model independently receives the clause text, document ID, and clause ID, and evaluates the clause against the strict 14-category schema.

---

## 2. Standard 14-Category Schema
The classification task is multi-label. Each model must select zero or more categories strictly from:
1. `Party Identification`
2. `Purpose`
3. `NDA Type`
4. `Definition of Confidential Information`
5. `Confidentiality Obligations`
6. `Authorized Disclosure`
7. `Non-Confidential Information`
8. `Liability for Damages`
9. `Competition Rights`
10. `Term and Termination`
11. `Intellectual Property`
12. `Employees`
13. `Governing Law and Jurisdiction`
14. `Additional Information`

Free-form category names or non-schema inventions are strictly forbidden. If no specific category is applicable, models return `["Additional Information"]`.

---

## 3. Consensus & Agreement Formulas

### A. Jaccard Similarity between Model Pairs
For any two model label sets $L_i$ and $L_j$:
$$J(L_i, L_j) = \frac{|L_i \cap L_j|}{|L_i \cup L_j|}$$
*Special case*: If $L_i = \emptyset$ and $L_j = \emptyset$, $J(L_i, L_j) = 1.0$.

### B. Agreement Score
The total agreement score is the mean pairwise Jaccard similarity across all 3 model pairs $(A, B)$, $(B, C)$, and $(A, C)$:
$$\text{agreement\_score} = \frac{J(L_A, L_B) + J(L_B, L_C) + J(L_A, L_C)}{3}$$

### C. Average Confidence Score
Each model outputs a self-reported confidence $\text{conf}_i \in [0.0, 1.0]$:
$$\text{average\_confidence} = \frac{\text{conf}_A + \text{conf}_B + \text{conf}_C}{3}$$

### D. Label Disagreement Flag
$$\text{label\_disagreement} = \neg (L_A = L_B = L_C)$$

### E. Final Pseudo-Label Selection Logic (Majority Voting)
1. For each category $c$ in the 14 categories, count votes across Model A, Model B, and Model C.
2. A category $c$ is included in `final_pseudo_labels` if $\text{votes}(c) \ge 2$ (majority vote).
3. If no category receives $\ge 2$ votes (complete 3-way conflict), `final_pseudo_labels` takes the label set of the model with the highest self-reported confidence.

---

## 4. Pseudo-Label Quality Stratification
Each classified clause is categorized into one of four quality tiers:

| Quality Bucket | Criteria | Description |
|---|---|---|
| `HIGH_CONFIDENCE` | `agreement_score >= 0.8` AND `average_confidence >= 0.8` | Strong inter-model agreement and high model confidence. |
| `MEDIUM_CONFIDENCE` | `agreement_score >= 0.5` AND `average_confidence >= 0.6` (and not High) | Moderate agreement and acceptable confidence. |
| `DISAGREEMENT` | `agreement_score < 0.5` | Substantial conflict/disagreement between model predictions. |
| `LOW_CONFIDENCE` | All other cases (`average_confidence < 0.6`) | Weak model confidence despite possible agreement. |

---

## 5. Downstream Research Workflow
The generated pseudo-labels feed directly into:
1. `human_review_priority.csv`: Uncertainty-ranked queue for priority human review.
2. `human_verification_sample.csv`: Stratified small random sample (30-50 clauses) for manual precision/recall estimation.
3. `train_pseudolabel_baseline.py`: Baseline classifier training supporting experimental flags `--use-high-confidence-only` and `--use-high-medium-confidence`.
