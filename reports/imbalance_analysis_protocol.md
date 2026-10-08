# Class Imbalance Analysis Protocol

> **Status**: Protocol defined. Final statistics pending completion of full pseudo-labeling dataset.
> All numerical values here are template definitions or marked TBD.

---

## 1. Overview of Class Imbalance in NDA Classification

NDA multi-label clause classification suffers from significant category imbalance:
- **Majority Categories**: Party Identification, Purpose, Definition of Confidential Information, Confidentiality Obligations, Additional Information.
- **Minority Categories**: Employees, Competition Rights, Intellectual Property, Liability for Damages, Term and Termination.

Because models trained with standard loss functions under severe imbalance tend to collapse predictions toward high-frequency categories, this protocol defines standard measurement and weighting procedures.

---

## 2. Quantitative Metrics Required Per Category

For each of the 14 NDA categories, the following imbalance metrics must be computed once the final dataset is frozen:

| Metric | Definition / Formula | Purpose |
| :--- | :--- | :--- |
| **Positive Support (N_+)** | Count of clauses containing category c | Measures absolute positive sample count |
| **Negative Support (N_-)** | Count of clauses NOT containing category c | Measures absolute negative sample count |
| **Prevalence (P_c)** | N_+ / (N_+ + N_-) | Proportion of positive instances |
| **Imbalance Ratio (IR_c)** | N_- / N_+ | Ratio of negative to positive instances |
| **Pos-Weight (w_c)** | N_- / N_+ (or inverse frequency variant) | Raw weight used in Class-Weighted BCE |

---

## 3. Class Weight Calculation Protocol (Strict Leakage Prevention)

To prevent data leakage during model training:

1. **Training Set Only**: Class weights w_c MUST be calculated **EXCLUSIVELY** from the training split of the final document-disjoint dataset.
2. **Validation/Test Leakage Prohibited**: Under no circumstances shall positive/negative counts from the validation or test splits be included in w_c computation.
3. **Formula**:
   w_c = N_{train,-,c} / N_{train,+,c}
   Where N_{train,-,c} is the number of negative instances for category c in the training split, and N_{train,+,c} is the number of positive instances for category c in the training split.
4. **Clipping / Normalization (TBD)**: If extreme imbalance (IR_c > 50) causes numerical instability, weight clipping max limit (e.g. w_{max} = 10.0) will be evaluated on validation split only.

---

## 4. Current Data State (203 Valid Pseudo-Labels)

> **Note**: The current dataset contains only 203 valid pseudo-labels across 7 documents and is **TEMPORARY**.
> Final statistics will be published in 	able_1_dataset_statistics.csv after Phase 2 dataset construction.

| Category | Category Name | Status |
| :---: | :--- | :--- |
| C1 | Party Identification | TBD - PENDING FINAL DATASET |
| C2 | Purpose | TBD - PENDING FINAL DATASET |
| C3 | NDA Type | TBD - PENDING FINAL DATASET |
| C4 | Definition of Confidential Information | TBD - PENDING FINAL DATASET |
| C5 | Confidentiality Obligations | TBD - PENDING FINAL DATASET |
| C6 | Authorized Disclosure | TBD - PENDING FINAL DATASET |
| C7 | Non-Confidential Information | TBD - PENDING FINAL DATASET |
| C8 | Liability for Damages | TBD - PENDING FINAL DATASET |
| C9 | Competition Rights | TBD - PENDING FINAL DATASET |
| C10 | Term and Termination | TBD - PENDING FINAL DATASET |
| C11 | Intellectual Property | TBD - PENDING FINAL DATASET |
| C12 | Employees | TBD - PENDING FINAL DATASET |
| C13 | Governing Law and Jurisdiction | TBD - PENDING FINAL DATASET |
| C14 | Additional Information | TBD - PENDING FINAL DATASET |

---
*Module 7 - Research Analysis and Experimental Framework*
*Created: 2026-10-03*
