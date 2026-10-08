# Module 3A: Human Verification Instructions

## Purpose
The purpose of this review is strictly to verify the **segmentation** boundaries of preliminary NDA clauses. 

This establishes the ground truth for clause segmentation before any machine learning models are trained.

**IMPORTANT:** Do NOT assign any of the 14 NDA categories (e.g., Governing Law, Definitions, etc.) during this step. We are ONLY checking if the boundaries are correct and if the text represents substantive contractual content.

## What is a Valid Legal Clause?
A valid clause or contractual content segment generally contains meaningful agreement text.

**Examples:**
- "The Receiving Party shall maintain the confidentiality of all Confidential Information."
- "This Agreement shall remain in effect for a period of two years."
- "THIS NON-DISCLOSURE AGREEMENT is entered into by and between..."

These should be preserved and treated as legal content.

## What Should NOT be a Clause?
Examples of structural/non-clause content include:
- Isolated page numbers
- Repeated page headers/footers
- SEC metadata (e.g., "EX-10.3")
- Document identifiers
- Isolated company names or isolated titles
- Punctuation-only fragments or obvious OCR fragments
- Empty segments

These can be marked `REMOVE_NON_LEGAL`. **However, do NOT remove something merely because it is short if it contains valid legal text.**

## Important Title Rule
- A title such as `"NON-DISCLOSURE AGREEMENT"` may be a section heading rather than a legal clause.
- But `"THIS NON-DISCLOSURE AGREEMENT is entered into by and between..."` contains substantive contractual content and should normally be preserved.
- Similarly, `"CONFIDENTIALITY"` may be a heading, but `"The Receiving Party shall maintain all Confidential Information..."` is legal content.

## Action Rules

### KEEP
Use `KEEP` when the segmentation is reasonable and contains meaningful legal content. The original clause text remains unchanged.

### MERGE_WITH_NEXT
Use `MERGE_WITH_NEXT` when the current segment is clearly an incomplete fragment whose meaning continues into the next segment.
*Example:* 
Segment 1: "shall maintain the"
Segment 2: "confidentiality of all Confidential Information."
-> Mark Segment 1 as `MERGE_WITH_NEXT`.

### MERGE_WITH_PREVIOUS
Use `MERGE_WITH_PREVIOUS` when the current segment is clearly a continuation of the previous segment.
*Note: Do NOT merge simply because two clauses are conceptually related. Only merge if they are structurally part of the same paragraph/clause that was incorrectly split.*

### SPLIT
Use `SPLIT` when one segment contains multiple clearly separate contractual clauses that should be separate segments.
*Example:* A single segment contains both "1. The Receiving Party shall..." and "2. The Receiving Party must...", and formatting clearly indicates two independent clauses.
*Note: Do not split merely because a paragraph is long.*

### REMOVE_NON_LEGAL
Use `REMOVE_NON_LEGAL` for non-clause content like page numbers, SEC headers, or OCR garbage.

### REVIEW
Use `REVIEW` when the segmentation cannot be confidently determined. Do not guess. Human uncertainty should be recorded instead of forcing an incorrect decision.
