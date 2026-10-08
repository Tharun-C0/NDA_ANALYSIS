# Module 13: Future Gemini Labeling Plan

## Executive Summary

This plan defines the targeted, diversity-first queue execution strategy for when the Gemini API quota resets (~17.68h retry window).

Rather than selecting arbitrary clauses to inflate dataset size, this plan targets specific unseen documents containing strong textual signals for underrepresented categories (*Liability for Damages*, *Competition Rights*, *Intellectual Property*, *Governing Law and Jurisdiction*).

---

## 1. Targeted Document Priority & Sampling Plan

| Rank | Document ID (Prefix) | Pending Clauses | Target Categories Investigated | Initial Sampling Batch | Total Candidate Clauses | Primary Rationale |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **1** | `d714d261` | 43 | Liability, Competition, IP, Governing Law | **5 clauses** | 43 | Matches **all 4 target categories** (14 IP, 11 Gov Law, 9 Competition, 6 Liability signals). |
| **2** | `9a5cb310` | 74 | Liability, Competition, IP, Governing Law | **5 clauses** | 74 | Matches **all 4 target categories** (14 Competition, 9 IP, 6 Liability, 4 Gov Law signals). |
| **3** | `586c367e` | 58 | Liability, Competition, IP, Governing Law | **5 clauses** | 58 | Matches **all 4 target categories** (19 IP, 6 Liability, 4 Competition, 4 Gov Law signals). |
| **4** | `f28c4f3d` | 60 | Liability, Competition, IP, Governing Law | **5 clauses** | 60 | Matches **all 4 target categories** (12 IP, 6 Liability, 6 Competition, 6 Gov Law signals). |
| **5** | `e52e4a13` | 43 | Liability, Competition, IP, Governing Law | **5 clauses** | 43 | Matches **all 4 target categories** (13 Competition, 4 Gov Law, 3 Liability, 2 IP signals). |
| **6** | `d4566b17` | 45 | Liability, Competition, IP, Governing Law | **5 clauses** | 45 | High *Liability for Damages* signal density (7 Liability clauses). |
| **7** | `b82a10c4` | 18 | Liability, Competition, IP, Governing Law | **5 clauses** | 18 | High *Competition Rights* signal density (6 Competition clauses). |
| **8** | `c58882f7` | 16 | Liability, Competition, IP, Governing Law | **5 clauses** | 16 | Matches all 4 target categories in a compact document. |
| **9** | `64303e5a` | 23 | Liability, Competition, Governing Law | **5 clauses** | 23 | Supplemental coverage for Liability and Competition. |

---

## 2. Multi-Stage Execution Strategy

### Stage 1: Round-Robin Initial Diversity Sampling (45 Clauses)
- **Batch Size**: 5 clauses $\times$ 9 unseen documents = **45 clauses total**.
- **Execution Command**:
  ```powershell
  .\.venv\Scripts\python.exe scripts\resume_runner.py --max-clauses 45
  ```
- **Objective**: Establish initial valid pseudo-labels for all 9 unseen documents, expanding represented documents from **11 to 20**.

### Stage 2: Post-Batch Category Audit & Re-Evaluation
- Re-run `scripts/_generate_module12_matrix.py` and `scripts/analyze_document_splits.py`.
- Evaluate whether *Liability for Damages* reaches $\ge 3$ represented documents.

### Stage 3: Targeted Redundancy Completion (Optional Stage 2)
- If specific categories remain confined to $<3$ documents, execute a targeted 20-clause follow-up batch from `data/annotations/module13_targeted_clause_queue.csv` focused specifically on clauses with matching `suspected_category_signal`.
