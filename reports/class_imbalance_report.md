# Module 6 — Category Distribution & Class Imbalance Analysis

> [!NOTE]
> **Dataset Note**: All label statistics are calculated over **203 usable LLM pseudo-labeled clauses** derived from 7 NDA documents.

- **Total Usable Clauses**: 203
- **Total Unique Documents**: 7
- **Total Category Assignments**: 229 (Avg: 1.13 labels/clause)

## 1. 14-Category Frequency & Distribution Table

| Category Name | Positive Clauses | Percentage | Document Frequency | Pos:Neg Ratio | Imbalance Tier |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Additional Information** | 157 | 77.34% | 7 / 7 | 1:0.3 | Majority (>=15) |
| **Party Identification** | 15 | 7.39% | 3 / 7 | 1:12.5 | Majority (>=15) |
| **Confidentiality Obligations** | 12 | 5.91% | 3 / 7 | 1:15.9 | Minority (<15) |
| **Authorized Disclosure** | 12 | 5.91% | 3 / 7 | 1:15.9 | Minority (<15) |
| **Purpose** | 6 | 2.96% | 3 / 7 | 1:32.8 | Minority (<15) |
| **Governing Law and Jurisdiction** | 5 | 2.46% | 2 / 7 | 1:39.6 | Minority (<15) |
| **NDA Type** | 4 | 1.97% | 3 / 7 | 1:49.8 | Minority (<15) |
| **Definition of Confidential Information** | 3 | 1.48% | 2 / 7 | 1:66.7 | Minority (<15) |
| **Non-Confidential Information** | 3 | 1.48% | 2 / 7 | 1:66.7 | Minority (<15) |
| **Liability for Damages** | 3 | 1.48% | 1 / 7 | 1:66.7 | Minority (<15) |
| **Intellectual Property** | 3 | 1.48% | 2 / 7 | 1:66.7 | Minority (<15) |
| **Employees** | 3 | 1.48% | 2 / 7 | 1:66.7 | Minority (<15) |
| **Competition Rights** | 2 | 0.99% | 2 / 7 | 1:100.5 | Minority (<15) |
| **Term and Termination** | 1 | 0.49% | 1 / 7 | 1:202.0 | Minority (<15) |

## 2. Minority Category Identification
- **Majority Categories (>= 15 clauses)**: Additional Information, Party Identification
- **Minority Categories (< 15 clauses)**: Confidentiality Obligations, Authorized Disclosure, Purpose, Governing Law and Jurisdiction, NDA Type, Definition of Confidential Information, Non-Confidential Information, Liability for Damages, Intellectual Property, Employees, Competition Rights, Term and Termination

> [!IMPORTANT]
> **Methodology Rule**: Minority categories are preserved in the multi-label taxonomy without removal or premature oversampling during dataset preparation. Class-weighted loss functions (e.g. Focal Loss, Weighted BCE) will address imbalance during training.