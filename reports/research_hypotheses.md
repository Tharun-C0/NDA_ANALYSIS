# Research Hypotheses

> Status: HYPOTHESES ONLY. None have been confirmed or refuted.
> Wording is deliberately neutral. No directional superiority is claimed.

---

## H1 - Transformer Architecture (RQ1)

**H1:** Transformer model architecture will be associated with differences in
multi-label NDA clause classification performance as measured by Macro F1.

### Rationale
Legal-domain pre-training (Legal-RoBERTa, Legal-BERT) may expose models to
vocabulary more representative of NDA text than general-domain pre-training
(DeBERTa-v3). Whether this materialises under fine-tuning on the 14-category task
is an open empirical question.

### Decision Criterion
Macro F1 difference greater than 2 percentage points between best and worst model
under the BCE baseline condition, considered alongside per-category performance.

---

## H2 - Loss Function Under Class Imbalance (RQ2)

**H2:** Loss functions designed to address class imbalance (Focal Loss,
Class-Weighted BCE) will produce different per-category performance for
minority categories compared with standard BCE loss.

### Rationale
Minority categories receive fewer gradient updates under BCE. Focal Loss and
class-weighted BCE re-weight this gradient signal. Whether this produces
measurably different minority-category F1 is an empirical question.

### Decision Criterion
Per-category F1 for the five lowest-support categories, averaged across
architectures, under Focal Loss vs BCE baseline.

---

## H3 - Pseudo-Label Confidence Effect (RQ3)

**H3:** Training using only high-confidence pseudo-labels (HIGH_CONFIDENCE +
MEDIUM_CONFIDENCE) may produce different downstream classification behaviour
compared with training using all valid pseudo-labels including DISAGREEMENT
and LOW_CONFIDENCE records.

### Rationale
DISAGREEMENT and LOW_CONFIDENCE records represent ensemble conflicts or
uncertainty. Including them increases training set size at the cost of potential
label noise. Whether this tradeoff helps or harms is an empirical question that
depends on the final dataset size after full pseudo-labeling.

### Decision Criterion
Macro F1 difference between conditions greater than 1 percentage point across
the majority of model-loss combinations.

---

## H4 - Minority Category Consistency (RQ4)

**H4:** Classification performance will vary across the 14 NDA categories,
with categories having lower positive support in the test split likely
exhibiting greater variability and lower F1 scores than categories with
higher support.

### Rationale
Models are exposed to fewer positive examples for rare categories during
training, which typically makes decision boundary learning harder. Whether
this pattern holds for the specific 14-category NDA taxonomy is empirical.

### Decision Criterion
Pearson or Spearman correlation between per-category support and per-category F1
across all 14 categories, computed over the test split.

---

## Summary Table

| Hypothesis | Paired RQ | Claimed as True? |
| :--- | :--- | :---: |
| H1 - Architecture differences | RQ1 | No |
| H2 - Loss minority effect | RQ2 | No |
| H3 - Pseudo-label confidence | RQ3 | No |
| H4 - Support correlation | RQ4 | No |

---
*Module 7 - Research Analysis and Experimental Framework*
*Created: 2026-10-03*
