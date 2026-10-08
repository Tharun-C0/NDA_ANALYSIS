# NDA Clause Annotation Guidelines
# Module 5 — Category Annotation Framework

## Overview

These guidelines describe how to assign one or more of the 14 categories to each clause
in the NDA research corpus. They are intended for human annotators working with
`data/annotations/annotation_queue.csv`.

### Ground rules

- Every annotation must be created by a qualified human annotator.
- Do NOT use an LLM or automated tool to assign these category labels.
- Do NOT treat Module 3A structural review actions (KEEP / REMOVE_NON_LEGAL /
  MERGE_WITH_NEXT / MERGE_WITH_PREVIOUS / SPLIT / REVIEW) as category labels.
  Those actions describe segmentation decisions, not legal clause types.
- A clause may receive **more than one category** (multi-label). Use a JSON array:
  `["Category A", "Category B"]`
- A clause that fits no category should receive `[]` and the annotator should note why.
- These guidelines reflect the taxonomy used in the research paper. They are NOT
  official legal standards and should not be represented as such.
- Disagreements between annotators should be adjudicated and the resolution recorded
  in `annotation_notes`.
- Label field `label_verified` must be set to `true` only after a second reviewer
  has confirmed the assignment.

---

## The 14 Categories

---

### 1. Party Identification

**Definition**  
Text that identifies the parties entering into the NDA: their names, legal entity types
(corporation, LLC, individual, etc.), jurisdictions of incorporation, and the short-form
designations or aliases used throughout the agreement (e.g., "hereinafter referred to
as the 'Disclosing Party'").

**Include**
- Full legal names of all parties.
- Jurisdiction and entity-type declarations ("a Delaware corporation").
- Role labels assigned to each party ("Disclosing Party", "Receiving Party",
  "Company", "Recipient").
- Effective date of the agreement when embedded inside the party-identification
  sentence.
- Representative signatories named in the introductory paragraph.

**Do NOT include**
- Signature blocks or execution pages (these are non-legal structural text, already
  excluded by REMOVE_NON_LEGAL).
- Address or contact details appearing in notice provisions (those belong under
  category 14 Additional Information or are standalone notice clauses).
- References to a party in mid-agreement operational clauses; only the primary
  identification paragraph qualifies.

**NDA-style example**  
> "This Non-Disclosure Agreement ('Agreement') is entered into as of January 1, 2024,
> by and between Acme Corp., a Delaware corporation ('Disclosing Party'), and Beta
> Inc., a California corporation ('Receiving Party')."

**Boundary cases**
- If the introductory paragraph also states the purpose of the agreement, apply
  **both** Party Identification and Purpose.
- A "WHEREAS" recital that re-introduces parties may qualify if it provides the
  primary legal identification not found elsewhere.

**Multi-label?** Yes — may co-occur with **Purpose**, **NDA Type**.

---

### 2. Purpose

**Definition**  
Text that describes why the parties are sharing confidential information: the
commercial, technical, or strategic objective that motivates the agreement.

**Include**
- Stated business purpose ("in connection with a potential business transaction").
- Recital clauses describing the contemplated relationship or project.
- Scope-of-use statements restricting the receiving party to using information only
  for the stated purpose.

**Do NOT include**
- Definitions of what constitutes Confidential Information (→ category 4).
- General confidentiality obligations (→ category 5).
- A stated purpose embedded only inside a definition clause without independent
  purpose language.

**NDA-style example**  
> "The parties wish to explore a potential acquisition of Target Co. and, in connection
> therewith, may disclose certain Confidential Information to each other solely for the
> purpose of evaluating such transaction."

**Boundary cases**
- A "WHEREAS" clause that states both the business purpose and identifies the parties
  should receive both Purpose and Party Identification.
- A permitted-use restriction inside a confidentiality obligation clause may receive
  both Purpose and Confidentiality Obligations.

**Multi-label?** Yes — commonly co-occurs with **Party Identification**, **NDA Type**,
**Confidentiality Obligations**.

---

### 3. NDA Type

**Definition**  
Text that explicitly classifies or characterises the agreement as mutual/bilateral or
unilateral/one-way, or that designates the directional flow of disclosure (who discloses
to whom).

**Include**
- Explicit "mutual" or "one-way" / "unilateral" declarations.
- Text establishing which party is the Disclosing Party and which is the Receiving Party
  when framed as a structural classification rather than simple identification.
- Title text **only if** it is part of a substantive clause retained in the dataset (not
  a standalone heading already removed).

**Do NOT include**
- A bare agreement title that was already removed as REMOVE_NON_LEGAL.
- Role labels within Party Identification sentences that only name parties without
  classifying the agreement type.

**NDA-style example**  
> "This Agreement is a mutual non-disclosure agreement under which each party may
> act as both a Disclosing Party and a Receiving Party with respect to Confidential
> Information."

**Boundary cases**
- When the classification appears only in the title and the title was retained, apply
  NDA Type to that retained clause.
- A clause that says "each party may disclose" implies mutual NDA and may qualify.

**Multi-label?** Yes — commonly co-occurs with **Party Identification**, **Purpose**.

---

### 4. Definition of Confidential Information

**Definition**  
Text that defines what information is treated as "Confidential Information" under the
agreement, including inclusion criteria, exclusion criteria, and marking requirements.

**Include**
- The core definition clause (e.g., "'Confidential Information' means…").
- Any sub-clauses that list types of information included in the definition.
- Marking or designation requirements (e.g., "information that is marked CONFIDENTIAL
  in writing").
- Oral disclosure procedures (e.g., "orally disclosed information that is confirmed
  in writing within 30 days").

**Do NOT include**
- Exceptions to confidentiality (→ category 7 Non-Confidential Information).
- The obligations themselves (→ category 5 Confidentiality Obligations).
- Definitions of other terms (e.g., "Affiliate", "Representative") unless those
  definitions directly expand or limit the scope of Confidential Information.

**NDA-style example**  
> "'Confidential Information' means all non-public information, whether written, oral,
> or in any other form, disclosed by one party to the other that is designated as
> confidential or that reasonably should be understood to be confidential given the
> nature of the information and the circumstances of disclosure."

**Boundary cases**
- A clause that both defines Confidential Information and states what is excluded
  should receive both **Definition of Confidential Information** and
  **Non-Confidential Information**.
- A clause that defines Confidential Information and immediately states obligations
  may receive both this category and **Confidentiality Obligations**.

**Multi-label?** Yes — frequently co-occurs with **Non-Confidential Information**,
**Confidentiality Obligations**.

---

### 5. Confidentiality Obligations

**Definition**  
Text imposing affirmative duties on the receiving party to protect, restrict access to,
or maintain the secrecy of Confidential Information.

**Include**
- Core non-disclosure obligations ("shall not disclose", "shall keep confidential").
- Use restrictions ("shall use solely for the Purpose").
- Access restrictions ("shall limit access to those Representatives who have a need
  to know").
- Duty-of-care standards ("shall use at least the same degree of care…").
- Obligations that apply on termination (return/destruction duties are often here
  or in Term and Termination—apply both if combined).

**Do NOT include**
- Permitted disclosures (→ category 6 Authorized Disclosure).
- Exceptions to the obligation (→ category 7 Non-Confidential Information).
- Clauses solely about remedies for breach (→ category 8 Liability for Damages).

**NDA-style example**  
> "Each Receiving Party shall: (a) hold all Confidential Information in strict
> confidence; (b) not disclose any Confidential Information to any third party without
> the prior written consent of the Disclosing Party; and (c) use the Confidential
> Information solely for the Purpose."

**Boundary cases**
- A combined clause stating obligations and exceptions should receive both
  **Confidentiality Obligations** and **Non-Confidential Information**.
- Return/destruction obligations may overlap with **Term and Termination**.

**Multi-label?** Yes — commonly co-occurs with **Authorized Disclosure**,
**Non-Confidential Information**, **Term and Termination**.

---

### 6. Authorized Disclosure

**Definition**  
Text that permits or excuses disclosure of Confidential Information under specified
conditions, including disclosure required by law, to professional advisors, or to
employees with a need to know.

**Include**
- Carve-outs for legally compelled disclosure (court order, regulatory requirement).
- Permitted disclosure to Representatives, Affiliates, advisors, or employees.
- Consent-based disclosure provisions.
- Notice requirements that must be fulfilled before a permitted disclosure is made.

**Do NOT include**
- The definition of what is excluded from the definition of Confidential Information
  altogether (→ category 7 Non-Confidential Information).
- General confidentiality obligations (→ category 5).

**NDA-style example**  
> "The Receiving Party may disclose Confidential Information to its employees,
> agents, and professional advisors who (a) need to know such information for the
> Purpose and (b) are bound by confidentiality obligations no less restrictive than
> those set forth herein."

**Boundary cases**
- A clause that both describes an obligation and carves out a permitted disclosure
  should receive **Confidentiality Obligations** and **Authorized Disclosure**.
- Compelled-disclosure clauses that require prior notice to the Disclosing Party
  may also touch on **Governing Law and Jurisdiction** if they specify the
  applicable legal process.

**Multi-label?** Yes — co-occurs with **Confidentiality Obligations**.

---

### 7. Non-Confidential Information

**Definition**  
Text that excludes certain categories of information from the definition of Confidential
Information or from the confidentiality obligations — i.e., information that was already
known, becomes public, is independently developed, or is received from a third party
without restriction.

**Include**
- Standard exclusions: (a) publicly available information; (b) information already
  known to recipient; (c) information independently developed without use of
  Confidential Information; (d) information received from a third party free of
  restriction.
- Any additional negotiated exclusions.

**Do NOT include**
- Permitted disclosures that still involve Confidential Information (→ category 6
  Authorized Disclosure).
- The core definition of what IS Confidential Information (→ category 4).

**NDA-style example**  
> "The obligations of confidentiality shall not apply to information that: (a) is or
> becomes publicly available through no fault of the Receiving Party; (b) was already
> known to the Receiving Party prior to disclosure; (c) is independently developed by
> the Receiving Party; or (d) is received from a third party who is not bound by any
> confidentiality obligation."

**Boundary cases**
- A clause that defines Confidential Information and immediately enumerates exclusions
  should receive both **Definition of Confidential Information** and
  **Non-Confidential Information**.

**Multi-label?** Yes — co-occurs with **Definition of Confidential Information**,
**Confidentiality Obligations**.

---

### 8. Liability for Damages

**Definition**  
Text that addresses remedies, damages, indemnification, limitations of liability, or
disclaimers of warranties in the context of breach or unauthorised disclosure.

**Include**
- Statements that breach may cause irreparable harm entitling the Disclosing Party
  to seek injunctive relief.
- Limitation-of-liability or disclaimer-of-warranty clauses.
- Indemnification obligations for breach of the NDA.
- Explicit damage or remedy provisions.

**Do NOT include**
- General obligations not framed as remedies (→ category 5).
- Governing law clauses that determine which law governs the remedy (→ category 13).

**NDA-style example**  
> "Each party acknowledges that any breach of this Agreement may cause irreparable
> injury for which monetary damages would be an inadequate remedy, and each party
> agrees that the non-breaching party shall be entitled to seek equitable relief,
> including injunction and specific performance, in addition to all other remedies
> available at law or in equity."

**Boundary cases**
- A limitation-of-liability clause that is part of a broader termination provision
  may receive both **Liability for Damages** and **Term and Termination**.

**Multi-label?** Yes — may co-occur with **Term and Termination**,
**Governing Law and Jurisdiction**.

---

### 9. Competition Rights

**Definition**  
Text that addresses non-compete, non-solicitation, or similar competitive-restriction
obligations arising from or related to the NDA.

**Include**
- Explicit non-compete covenants.
- Non-solicitation of clients or business partners.
- Restrictions on using Confidential Information to compete.
- Stand-still agreements (agreement not to acquire shares during a specified period).

**Do NOT include**
- Standard "use only for the Purpose" language (→ category 5 Confidentiality
  Obligations) unless specifically framed as a competition restriction.
- Employee non-solicitation (→ category 12 Employees).

**NDA-style example**  
> "During the Term and for a period of twelve (12) months thereafter, the Receiving
> Party shall not, without the prior written consent of the Disclosing Party, engage
> in any business activity that directly competes with the Disclosing Party's
> principal line of business, to the extent such activity would require the use of
> Confidential Information."

**Boundary cases**
- A combined non-compete and non-solicitation of employees clause should receive
  both **Competition Rights** and **Employees**.

**Multi-label?** Yes — may co-occur with **Employees**, **Term and Termination**.

---

### 10. Term and Termination

**Definition**  
Text that establishes the duration of the agreement, conditions for termination, and
post-termination obligations (such as return or destruction of Confidential Information).

**Include**
- Effective date and expiry/duration clauses.
- Termination triggers (with or without cause, upon written notice).
- Survival clauses specifying which obligations continue after termination.
- Return or destruction of Confidential Information requirements on termination.
- Obligations to certify destruction.

**Do NOT include**
- Core confidentiality obligations that apply during the term (→ category 5).
- Governing law applicable to dispute resolution at termination (→ category 13).

**NDA-style example**  
> "This Agreement shall remain in effect for a period of three (3) years from the
> Effective Date unless earlier terminated by either party upon thirty (30) days'
> written notice. Upon termination, the Receiving Party shall promptly return or
> destroy all Confidential Information and certify such destruction in writing."

**Boundary cases**
- A survival clause that specifies which specific obligations continue may apply to
  multiple categories (e.g., **Confidentiality Obligations** + **Term and Termination**).
- Return/destruction obligations are sometimes contained within a broader
  Confidentiality Obligations clause; in that case apply both.

**Multi-label?** Yes — co-occurs with **Confidentiality Obligations**,
**Competition Rights**, **Liability for Damages**.

---

### 11. Intellectual Property

**Definition**  
Text that addresses ownership, assignment, licensing, or non-assignment of intellectual
property rights arising from or disclosed in connection with the NDA.

**Include**
- Statements that no IP rights are transferred or licensed by disclosure.
- Ownership disclaimers ("nothing in this Agreement shall be construed as granting
  any license or right…").
- Provisions addressing work product or inventions developed using Confidential
  Information.
- IP assignment restrictions.

**Do NOT include**
- General confidentiality obligations about protecting trade secrets (→ category 5).
- Definitions of Confidential Information that mention trade secrets incidentally
  (→ category 4).

**NDA-style example**  
> "Nothing in this Agreement shall be construed as granting, by implication, estoppel,
> or otherwise, any license or right in or to any Confidential Information, patent,
> copyright, trademark, or other intellectual property right of the Disclosing Party."

**Boundary cases**
- A clause that addresses both IP ownership and the purpose of the disclosure may
  receive both **Intellectual Property** and **Purpose**.

**Multi-label?** Yes — may co-occur with **Definition of Confidential Information**,
**Purpose**.

---

### 12. Employees

**Definition**  
Text that specifically addresses non-solicitation of employees, restrictions on hiring
the other party's staff, or obligations relating to the receiving party's own employees
who have access to Confidential Information.

**Include**
- Non-solicitation of employees or contractors of the Disclosing Party.
- Restrictions on hiring employees who were involved in the disclosed information.
- Requirements to bind employees to confidentiality obligations no less restrictive
  than the agreement.
- Reference to Representatives (when "Representatives" is defined to include
  employees and the clause imposes obligations specifically on them).

**Do NOT include**
- Non-solicitation of customers or business partners (→ category 9 Competition Rights).
- General permitted-disclosure clauses that mention employees incidentally (→ category 6).

**NDA-style example**  
> "During the Term and for a period of one (1) year thereafter, neither party shall
> directly solicit for employment any employee of the other party who was involved in
> the evaluation of the proposed transaction."

**Boundary cases**
- A combined non-compete and non-solicitation-of-employees clause receives both
  **Competition Rights** and **Employees**.

**Multi-label?** Yes — co-occurs with **Competition Rights**, **Term and Termination**.

---

### 13. Governing Law and Jurisdiction

**Definition**  
Text that specifies the legal system or jurisdiction whose laws govern the agreement,
or that establishes where disputes must be resolved (courts, arbitration, venue).

**Include**
- Choice-of-law clauses.
- Exclusive or non-exclusive jurisdiction clauses.
- Venue selection clauses.
- Arbitration clauses specifying the seat or rules.
- Service-of-process provisions.

**Do NOT include**
- Remedies provisions (→ category 8 Liability for Damages), unless the clause is
  purely about choice of forum for those remedies.
- General miscellaneous provisions without a law or jurisdiction element (→ category 14).

**NDA-style example**  
> "This Agreement shall be governed by and construed in accordance with the laws of
> the State of Delaware, without regard to its conflict-of-laws principles. Each party
> hereby consents to the exclusive jurisdiction of the state and federal courts located
> in Delaware for the resolution of any dispute arising under this Agreement."

**Boundary cases**
- An arbitration clause that also disclaims jury trial may overlap with
  **Liability for Damages**.

**Multi-label?** Yes — may co-occur with **Liability for Damages**.

---

### 14. Additional Information

**Definition**  
Text containing general miscellaneous or "boilerplate" contractual provisions that do
not clearly fit any of the above 13 categories, or that provide supplementary
procedural, administrative, or interpretive rules for the agreement.

**Include**
- Entire-agreement / integration clauses.
- Amendment and waiver provisions.
- Severability clauses.
- Notice provisions (addresses, delivery methods).
- Counterparts clauses.
- No-waiver provisions.
- Headings-not-binding clauses.
- Force majeure (if present).
- Any provision the annotator determines does not fit categories 1–13.

**Do NOT include**
- Clauses that clearly fit one of categories 1–13; do not use Additional Information
  as a catch-all for clauses that could reasonably fit a named category.

**NDA-style example**  
> "This Agreement constitutes the entire agreement between the parties with respect
> to its subject matter and supersedes all prior discussions, representations, or
> agreements. This Agreement may not be amended except by a written instrument signed
> by both parties."

**Boundary cases**
- A notice clause may overlap with **Party Identification** if it primarily specifies
  the parties' contact details used throughout the agreement.
- A severability clause that specifically addresses the survival of confidentiality
  obligations may also receive **Term and Termination**.

**Multi-label?** Yes — may co-occur with **Term and Termination**,
**Party Identification**.

---

## Multi-Label Annotation Instructions

### When to apply multiple labels

Apply multiple category labels when a single clause contains substantive content
belonging to more than one category. Do NOT split clauses artificially; annotate the
clause as it exists in the dataset.

### Format

`category_labels` must always be a valid JSON array of exact category name strings:

```
["Confidentiality Obligations","Authorized Disclosure"]
```

For a single category:

```
["Term and Termination"]
```

For no applicable category:

```
[]
```

### Order of labels

The order within the JSON array does not carry semantic meaning. Use alphabetical or
most-specific-first ordering as a convention.

### Common multi-label combinations

| Combination | Rationale |
|---|---|
| Party Identification + Purpose | Introductory paragraph states parties and reason for disclosure |
| Party Identification + NDA Type | Introduction explicitly classifies agreement as mutual/unilateral |
| Definition of Confidential Information + Non-Confidential Information | Combined definition-and-exclusion clause |
| Confidentiality Obligations + Authorized Disclosure | Core obligation with carve-outs embedded |
| Confidentiality Obligations + Term and Termination | Return/destroy obligation in termination clause |
| Competition Rights + Employees | Combined non-compete and non-solicitation of staff |
| Liability for Damages + Governing Law and Jurisdiction | Remedies clause with forum selection |
| Term and Termination + Additional Information | Termination with severability language |

---

## Annotation Status Values

| Value | Meaning |
|---|---|
| `PENDING` | Clause has not yet been reviewed |
| `ANNOTATED` | Labels assigned, not yet verified by a second reviewer |
| `VERIFIED` | Second reviewer has confirmed the labels |
| `DISPUTED` | Reviewers disagree; awaiting adjudication |
| `SKIPPED` | Annotator chose to skip (provide reason in annotation_notes) |
| `UNCLASSIFIABLE` | No applicable category found; `category_labels` = `[]` |

---

## Provenance Fields

Every completed annotation row must have:

| Field | Required value |
|---|---|
| `label_source` | Always `"human_annotation"` for manual labels |
| `label_verified` | `false` until second reviewer confirms; then `true` |
| `label_reviewer` | Full name or unique identifier of the annotator |
| `annotation_notes` | Reasoning, boundary-case notes, or adjudication outcome |

---

*End of guidelines — Module 5, NDA Research Project.*
