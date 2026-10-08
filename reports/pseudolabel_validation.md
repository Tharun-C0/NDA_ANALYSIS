# LLM Pseudo-Label Dataset Validation Report

**Validation Status**: ✅ PASSED
**Total Records Validated**: 393

| Check ID | Check Name | Status | Details |
|---|---|---|---|
| 01 | File Non-Empty | PASS | Found 393 pseudo-labeled rows. |
| 02 | No Duplicate Clause IDs | PASS | Duplicates found: 0 |
| 03 | Required Columns Present | PASS | Missing columns: None |
| 04 | Correct Label Source | PASS | Non-'llm_pseudo_label' rows: 0 |
| 05 | Valid JSON Format in Label Fields | PASS | Invalid JSON rows: 0 |
| 06 | No Invented Categories | PASS | Invented categories found: 0 |
| 07 | Confidence Values in Range [0,1] | PASS | Out-of-range confidence rows: 0 |
| 08 | Agreement Score in Range [0,1] | PASS | Out-of-range agreement rows: 0 |
| 09 | Valid Quality Buckets | PASS | Invalid quality bucket rows: 0 |
| 10 | No Missing Clause Text | PASS | Empty text rows: 0 |