# Module 6 — Classification Split Leakage & Integrity Validation Report

**Validation Status**: ✅ PASSED (0 Leakage Found)
**Total Valid Clauses**: 203

| Check ID | Check Name | Status | Details |
|---|---|---|---|
| 01 | Non-Empty Splits | PASS | Train: 109, Val: 49, Test: 45 |
| 02 | No Duplicate Clause IDs | PASS | Duplicates found in all_valid: 0 |
| 03 | Document-Disjoint Splits (Zero Leakage) | PASS | Overlap count: 0 (Train-Val: 0, Train-Test: 0, Val-Test: 0) |
| 04 | No Missing Text or Document IDs | PASS | Empty text: 0, Empty doc: 0 |
| 05 | Zero API_ERROR / Quota Rows | PASS | Errored/Quota rows found: 0 |
| 06 | Valid 14 Category Vocabulary | PASS | Invalid categories or JSON errors: 0 |
| 07 | 14 Multi-Hot Binary Columns Present | PASS | Missing columns: None |
| 08 | Total Clause Count Integrity | PASS | Train+Val+Test (203) == All Valid (203) |

## Document Split Breakdown
- **Train**: 109 clauses across 3 documents (['0a42e159', '266929af', '0859334b'])
- **Validation**: 49 clauses across 2 documents (['293f5937', '0b59dfc4'])
- **Test**: 45 clauses across 2 documents (['247166e0', '2268c5d1'])