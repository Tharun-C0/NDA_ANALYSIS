"""Module 5B — NDA Clause Annotation Interface (Active Learning Edition).

Flask application for manually assigning 14 category labels to NDA clauses.
Supports two modes:
    --mode seed   : shows only the 100-clause initial seed set
    --mode batch  : shows the next_active_learning_batch.csv uncertain clauses
    --mode all    : shows the full 699-clause queue (default)

Annotations are saved to TWO places atomically:
    1. data/annotations/annotation_queue.csv  (in-place update, preserves all 699 rows)
    2. data/annotations/annotations.csv       (append-only research database)

No automatic labelling.  No Gemini calls.  Human annotator is always the final authority.

Usage:
    python app/annotation_app.py [--mode seed|batch|all] [--port PORT]

Then open http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse, csv, io, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from flask import Flask, jsonify, redirect, render_template_string, request, url_for
except ImportError:
    sys.exit("Flask is required. Install with: .venv\\Scripts\\pip install flask")

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "Party Identification", "Purpose", "NDA Type",
    "Definition of Confidential Information", "Confidentiality Obligations",
    "Authorized Disclosure", "Non-Confidential Information",
    "Liability for Damages", "Competition Rights", "Term and Termination",
    "Intellectual Property", "Employees",
    "Governing Law and Jurisdiction", "Additional Information",
]

ANNOTATION_STATUSES = ["PENDING", "ANNOTATED", "DISPUTED", "SKIPPED"]

QUEUE_COLUMNS = [
    "document_id", "clause_id", "clause_text",
    "category_labels", "annotation_status",
    "label_verified", "label_source", "label_reviewer", "annotation_notes",
    "source", "review_status", "segment_type",
]

DB_COLUMNS = [
    "clause_id", "document_id", "clause_text",
    "category_labels", "reviewer", "notes",
    "annotation_status", "verified", "created_at", "updated_at",
    "annotation_round", "selection_source",
]

MODE_LABELS = {
    "seed":  "INITIAL SEED ANNOTATION",
    "batch": "ACTIVE LEARNING — UNCERTAIN CLAUSES",
    "all":   "FULL QUEUE ANNOTATION",
}

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _save_queue(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUEUE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _append_db(path: Path, row: dict) -> None:
    """Upsert by clause_id: update if exists, else append."""
    existing = _load_csv(path)
    updated = False
    for i, r in enumerate(existing):
        if r.get("clause_id") == row.get("clause_id"):
            existing[i] = row
            updated = True
            break
    if not updated:
        existing.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DB_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(queue_path: Path, mode: str = "all",
               seed_path: Path | None = None,
               batch_path: Path | None = None,
               db_path: Path | None = None) -> Flask:

    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    # ── determine which clause IDs to show based on mode ──────────────────
    def _get_active_ids() -> list[str]:
        if mode == "seed" and seed_path and seed_path.exists():
            rows = _load_csv(seed_path)
            return [r["clause_id"] for r in rows]
        if mode == "batch" and batch_path and batch_path.exists():
            rows = _load_csv(batch_path)
            if rows:
                return [r["clause_id"] for r in rows]
        # fallback: all
        rows = _load_csv(queue_path)
        return [r["clause_id"] for r in rows]

    def _state():
        all_queue = _load_csv(queue_path)
        by_id = {r["clause_id"]: r for r in all_queue}
        active_ids = _get_active_ids()
        active_rows = [by_id[cid] for cid in active_ids if cid in by_id]
        total = len(active_rows)
        done = sum(
            1 for r in active_rows
            if r.get("annotation_status", "PENDING") not in ("PENDING", "")
            and json.loads(r.get("category_labels") or "[]")
        )
        return active_rows, total, done, all_queue

    def _first_pending(rows: list[dict]) -> int:
        for i, r in enumerate(rows):
            if r.get("annotation_status", "PENDING") in ("PENDING", ""):
                return i
        return 0

    # ── routes ──────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        active_rows, total, done, _ = _state()
        if total == 0:
            return render_template_string(EMPTY_TPL, mode=mode)
        return redirect(url_for("annotate", idx=_first_pending(active_rows)))

    @app.route("/annotate/<int:idx>", methods=["GET"])
    def annotate(idx: int):
        active_rows, total, done, _ = _state()
        if total == 0:
            return render_template_string(EMPTY_TPL, mode=mode)
        idx = max(0, min(idx, total - 1))
        row = active_rows[idx]
        try:
            selected = json.loads(row.get("category_labels") or "[]")
            if not isinstance(selected, list):
                selected = []
        except (json.JSONDecodeError, TypeError):
            selected = []
        pct = round(done / total * 100) if total else 0
        mode_label = MODE_LABELS.get(mode, "ANNOTATION")
        # batch: show model suggestion if available
        predicted = []
        if mode == "batch" and batch_path and batch_path.exists():
            batch_rows = {r["clause_id"]: r for r in _load_csv(batch_path)}
            br = batch_rows.get(row["clause_id"], {})
            try:
                predicted = json.loads(br.get("predicted_labels") or "[]")
            except (json.JSONDecodeError, TypeError):
                predicted = []
        return render_template_string(
            ANNOTATE_TPL,
            row=row, idx=idx, total=total, done=done, pct=pct,
            categories=CATEGORIES, statuses=ANNOTATION_STATUSES,
            selected=selected, predicted=predicted,
            prev_idx=max(0, idx - 1), next_idx=min(total - 1, idx + 1),
            mode=mode, mode_label=mode_label,
        )

    @app.route("/save/<int:idx>", methods=["POST"])
    def save(idx: int):
        active_rows, total, _, all_queue = _state()
        if idx < 0 or idx >= total:
            return jsonify({"error": "index out of range"}), 400

        form = request.form
        labels = [cat for i, cat in enumerate(CATEGORIES) if form.get(f"cat_{i}")]
        labels_json = json.dumps(labels, ensure_ascii=False)

        reviewer = form.get("label_reviewer", "").strip()
        notes    = form.get("annotation_notes", "").strip()
        status   = form.get("annotation_status", "PENDING").strip()
        verified = "true" if form.get("label_verified") == "on" and reviewer else "false"

        # Determine annotation_round from mode
        annotation_round = "0" if mode in ("seed", "all") else form.get("annotation_round", "1")
        selection_source = {"seed": "active_learning_seed",
                            "batch": "uncertainty_sampling",
                            "all": "full_queue"}.get(mode, "full_queue")

        now_iso = datetime.now(timezone.utc).isoformat()
        target_cid = active_rows[idx]["clause_id"]

        # ── Update queue CSV ─────────────────────────────────────────────
        queue_by_id = {r["clause_id"]: r for r in all_queue}
        if target_cid in queue_by_id:
            queue_by_id[target_cid].update({
                "category_labels":   labels_json,
                "annotation_status": status,
                "label_verified":    verified,
                "label_source":      "human_annotation" if labels else "",
                "label_reviewer":    reviewer,
                "annotation_notes":  notes,
            })
        _save_queue(queue_path, list(queue_by_id.values()))

        # ── Upsert annotation DB ─────────────────────────────────────────
        if db_path:
            src_row = queue_by_id.get(target_cid, active_rows[idx])
            existing_db = {r["clause_id"]: r for r in _load_csv(db_path)}
            prev = existing_db.get(target_cid, {})
            db_row = {
                "clause_id":        target_cid,
                "document_id":      src_row.get("document_id", ""),
                "clause_text":      src_row.get("clause_text", ""),
                "category_labels":  labels_json,
                "reviewer":         reviewer,
                "notes":            notes,
                "annotation_status": status,
                "verified":         verified,
                "created_at":       prev.get("created_at", now_iso),
                "updated_at":       now_iso,
                "annotation_round": annotation_round,
                "selection_source": selection_source,
            }
            _append_db(db_path, db_row)

        return redirect(url_for("annotate", idx=min(total - 1, idx + 1)))

    @app.route("/jump", methods=["POST"])
    def jump():
        try:
            idx = max(0, int(request.form.get("idx", 1)) - 1)
        except (ValueError, TypeError):
            idx = 0
        return redirect(url_for("annotate", idx=idx))

    @app.route("/api/progress")
    def api_progress():
        active_rows, total, done, _ = _state()
        status_counts: dict[str, int] = {}
        for r in active_rows:
            s = r.get("annotation_status", "PENDING") or "PENDING"
            status_counts[s] = status_counts.get(s, 0) + 1
        return jsonify({"mode": mode, "total": total, "done": done,
                        "pending": total - done, "status_counts": status_counts})

    return app


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

EMPTY_TPL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Queue Empty</title>
<style>body{font-family:system-ui,sans-serif;display:flex;justify-content:center;
align-items:center;height:100vh;background:#0f172a;color:#e2e8f0;margin:0}
.box{text-align:center;padding:2rem;border:1px solid #334155;border-radius:1rem}
h1{color:#38bdf8}</style></head>
<body><div class="box">
<h1>Queue Empty</h1>
<p>Mode: <strong>{{ mode }}</strong></p>
<p>No clauses found for this mode. Check that the relevant CSV file exists.</p>
<p>Start with: <code>python app/annotation_app.py --mode seed</code></p>
</div></body></html>"""


ANNOTATE_TPL = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NDA Annotation [{{ mode_label }}] — {{ idx + 1 }}/{{ total }}</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0f172a; --surface: #1e293b; --surface2: #273348; --border: #334155;
  --accent: #38bdf8; --accent2: #818cf8; --green: #34d399; --yellow: #fbbf24;
  --red: #f87171; --text: #e2e8f0; --muted: #94a3b8; --mono: 'Consolas','Menlo',monospace;
}
body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
       line-height:1.6; min-height:100vh; }
.shell { display:grid; grid-template-columns:270px 1fr; grid-template-rows:auto 1fr auto; min-height:100vh; }
header { grid-column:1/-1; background:linear-gradient(90deg,#1e293b,#0f2440);
         border-bottom:1px solid var(--border); padding:.6rem 1.25rem;
         display:flex; align-items:center; gap:1rem; }
header h1 { font-size:1rem; font-weight:700; color:var(--accent); white-space:nowrap; }
.mode-badge { font-size:.7rem; background:#1e3a5f; color:var(--accent);
              border:1px solid #2563eb44; border-radius:.3rem; padding:.15rem .5rem; white-space:nowrap; }
.progress-bar-wrap { flex:1; background:var(--border); border-radius:99px; height:7px; overflow:hidden; }
.progress-bar-fill { height:100%; background:linear-gradient(90deg,var(--accent),var(--accent2)); transition:width .4s; }
.progress-label { font-size:.8rem; color:var(--muted); white-space:nowrap; }
.jump-form { display:flex; gap:.4rem; align-items:center; }
.jump-form input { width:4rem; background:var(--surface2); border:1px solid var(--border);
                   color:var(--text); border-radius:.4rem; padding:.25rem .5rem; font-size:.8rem; text-align:center; }
.jump-form button { background:var(--surface2); border:1px solid var(--border); color:var(--text);
                    border-radius:.4rem; padding:.25rem .6rem; cursor:pointer; font-size:.8rem; }
aside { background:var(--surface); border-right:1px solid var(--border); padding:1rem; overflow-y:auto; }
aside h2 { font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:.6rem; }
.meta-row { margin-bottom:.5rem; }
.meta-label { font-size:.65rem; color:var(--muted); text-transform:uppercase; }
.meta-value { font-family:var(--mono); font-size:.7rem; color:var(--accent); word-break:break-all;
              background:var(--surface2); padding:.15rem .35rem; border-radius:.25rem;
              display:inline-block; margin-top:.15rem; max-width:100%; }
.badge { display:inline-block; font-size:.65rem; padding:.1rem .45rem; border-radius:99px; font-weight:600; margin-top:.15rem; }
.badge-pending { background:#422006; color:var(--yellow); }
.badge-annotated { background:#052e16; color:var(--green); }
.badge-disputed { background:#3b0a0a; color:var(--red); }
.badge-skipped { background:#1e293b; color:var(--muted); }
.label-chip { display:inline-block; background:#1e3a5f; color:var(--accent);
              border:1px solid #2563eb44; border-radius:.25rem; padding:.1rem .35rem;
              margin:.1rem; font-size:.65rem; }
.pred-chip { display:inline-block; background:#1a2e1a; color:#86efac;
             border:1px solid #166534; border-radius:.25rem; padding:.1rem .35rem;
             margin:.1rem; font-size:.65rem; }
main { overflow-y:auto; padding:1.25rem; display:flex; flex-direction:column; gap:1rem; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:.75rem; padding:1.1rem; }
.card-title { font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
              margin-bottom:.65rem; display:flex; align-items:center; gap:.4rem; }
.card-title::after { content:''; flex:1; height:1px; background:var(--border); }
.clause-text { font-family:var(--mono); font-size:.85rem; white-space:pre-wrap; word-break:break-word;
               color:#cbd5e1; background:var(--surface2); border:1px solid var(--border);
               border-radius:.5rem; padding:.9rem; max-height:320px; overflow-y:auto; line-height:1.7; }
.cat-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:.45rem; }
.cat-item { display:flex; align-items:center; gap:.55rem; background:var(--surface2);
            border:1px solid var(--border); border-radius:.45rem; padding:.5rem .7rem;
            cursor:pointer; transition:border-color .15s,background .15s; user-select:none; }
.cat-item:hover { border-color:var(--accent); background:#1a2e42; }
.cat-item input[type="checkbox"] { display:none; }
.cat-item.checked { border-color:var(--accent); background:#0c2d48; }
.cat-item.checked .cat-box { background:var(--accent); border-color:var(--accent); }
.cat-box { width:15px; height:15px; border:2px solid var(--border); border-radius:3px;
           flex-shrink:0; display:flex; align-items:center; justify-content:center; }
.cat-box svg { display:none; }
.cat-item.checked .cat-box svg { display:block; }
.cat-name { font-size:.82rem; color:var(--text); }
.cat-item.suggested { border-color:#166534; background:#0d2015; }
.cat-item.suggested .cat-name::after { content:' (suggested)'; color:#86efac; font-size:.7rem; }
.field-row { display:flex; gap:.9rem; flex-wrap:wrap; }
.field-group { display:flex; flex-direction:column; gap:.3rem; flex:1; min-width:170px; }
.field-label { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
input[type="text"], select, textarea { background:var(--surface2); border:1px solid var(--border);
  color:var(--text); border-radius:.45rem; padding:.45rem .7rem; font-size:.88rem;
  font-family:inherit; transition:border-color .15s; outline:none; width:100%; }
input[type="text"]:focus, select:focus, textarea:focus { border-color:var(--accent); }
textarea { resize:vertical; min-height:75px; }
select option { background:var(--surface2); }
.verify-row { display:flex; align-items:center; gap:.65rem; background:var(--surface2);
              border:1px solid var(--border); border-radius:.45rem; padding:.55rem .8rem;
              cursor:pointer; user-select:none; width:fit-content; }
.verify-row input[type="checkbox"] { accent-color:var(--green); width:15px; height:15px; cursor:pointer; }
.verify-label { font-size:.85rem; color:var(--text); }
.verify-hint { font-size:.7rem; color:var(--muted); }
footer { grid-column:1/-1; background:var(--surface); border-top:1px solid var(--border);
         padding:.65rem 1.25rem; display:flex; gap:.65rem; align-items:center; justify-content:flex-end; }
.btn { padding:.45rem 1.1rem; border-radius:.45rem; font-size:.88rem; font-weight:600; cursor:pointer;
       border:1px solid transparent; transition:opacity .15s; text-decoration:none;
       display:inline-flex; align-items:center; gap:.35rem; }
.btn:hover { opacity:.87; }
.btn-ghost { background:transparent; border-color:var(--border); color:var(--muted); }
.btn-save  { background:var(--accent); color:#0f172a; }
.btn-skip  { background:transparent; border-color:var(--border); color:var(--yellow); }
.btn-prev  { background:var(--surface2); border-color:var(--border); color:var(--text); }
.model-hint { background:#1a2e1a; border:1px solid #166534; border-radius:.5rem;
              padding:.55rem .8rem; font-size:.78rem; color:#86efac; margin-bottom:.5rem; }
</style>
</head>
<body>
<div class="shell">

<header>
  <h1>NDA Annotation</h1>
  <span class="mode-badge">{{ mode_label }}</span>
  <div class="progress-bar-wrap">
    <div class="progress-bar-fill" style="width:{{ pct }}%"></div>
  </div>
  <span class="progress-label">{{ done }}/{{ total }} ({{ pct }}%)</span>
  <form class="jump-form" method="post" action="/jump">
    <input type="number" name="idx" min="1" max="{{ total }}" placeholder="{{ idx+1 }}">
    <button type="submit">Go</button>
  </form>
</header>

<aside>
  <h2>Clause Info</h2>
  <div class="meta-row"><div class="meta-label">Position</div>
    <div class="meta-value">{{ idx+1 }} / {{ total }}</div></div>
  <div class="meta-row"><div class="meta-label">Document ID</div>
    <div class="meta-value" title="{{ row.document_id }}">{{ row.document_id[:16] }}…</div></div>
  <div class="meta-row"><div class="meta-label">Clause ID</div>
    <div class="meta-value">{{ row.clause_id }}</div></div>
  <div class="meta-row"><div class="meta-label">Segment Type</div>
    <div class="meta-value">{{ row.segment_type or '—' }}</div></div>
  <div class="meta-row"><div class="meta-label">Review Status</div>
    <div class="meta-value">{{ row.review_status }}</div></div>
  <hr style="border-color:var(--border);margin:.65rem 0">
  <h2>Current Labels</h2>
  {% set cur_status = row.annotation_status or 'PENDING' %}
  <span class="badge badge-{{ cur_status | lower }}">{{ cur_status }}</span>
  <div style="margin-top:.4rem">
    {% if selected %}{% for lbl in selected %}<span class="label-chip">{{ lbl }}</span>{% endfor %}
    {% else %}<span style="color:var(--muted);font-size:.75rem">None yet</span>{% endif %}
  </div>
  {% if predicted %}
  <hr style="border-color:var(--border);margin:.65rem 0">
  <h2>Model Suggestion (NOT ground truth)</h2>
  <div style="margin-top:.3rem">
    {% for lbl in predicted %}<span class="pred-chip">{{ lbl }}</span>{% endfor %}
  </div>
  <div style="font-size:.65rem;color:var(--muted);margin-top:.3rem">
    Review independently. Do not copy blindly.
  </div>
  {% endif %}
  <hr style="border-color:var(--border);margin:.65rem 0">
  <h2>Navigate</h2>
  <a class="btn btn-prev" style="width:100%;justify-content:center;margin-bottom:.35rem"
     href="/annotate/{{ prev_idx }}">← Prev</a>
  <a class="btn btn-prev" style="width:100%;justify-content:center"
     href="/annotate/{{ next_idx }}">Next →</a>
</aside>

<main>
  <div class="card">
    <div class="card-title">Clause Text</div>
    <pre class="clause-text">{{ row.clause_text }}</pre>
  </div>

  <form method="post" action="/save/{{ idx }}" id="ann-form">

    {% if predicted %}
    <div class="model-hint">
      Model suggestion (uncertainty sampling): {{ predicted | join(', ') }}<br>
      <strong>This is a hint only. You must judge independently.</strong>
    </div>
    {% endif %}

    <div class="card">
      <div class="card-title">Category Labels — select all that apply</div>
      <div class="cat-grid">
        {% for cat in categories %}
        {% set ci = loop.index0 %}
        {% set is_checked = cat in selected %}
        {% set is_suggested = cat in predicted %}
        <label class="cat-item {% if is_checked %}checked{% elif is_suggested %}suggested{% endif %}"
               id="lbl_{{ ci }}">
          <input type="checkbox" name="cat_{{ ci }}" value="{{ cat }}"
                 {% if is_checked %}checked{% endif %}
                 onchange="toggleCheck(this,{{ ci }})">
          <span class="cat-box">
            <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
              <path d="M1 4l3 3 5-6" stroke="#0f172a" stroke-width="2"
                    stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </span>
          <span class="cat-name">{{ cat }}</span>
        </label>
        {% endfor %}
      </div>
    </div>

    <div class="card">
      <div class="card-title">Annotation Details</div>
      <div class="field-row" style="margin-bottom:.9rem">
        <div class="field-group">
          <label class="field-label" for="ann-status">Status</label>
          <select name="annotation_status" id="ann-status">
            {% for s in statuses %}
            <option value="{{ s }}" {% if (row.annotation_status or 'PENDING')==s %}selected{% endif %}>{{ s }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="field-group">
          <label class="field-label" for="ann-rev">Reviewer Name *</label>
          <input type="text" name="label_reviewer" id="ann-rev"
                 placeholder="Your name or ID" value="{{ row.label_reviewer or '' }}" autocomplete="name">
        </div>
      </div>
      <div class="field-group" style="margin-bottom:.9rem">
        <label class="field-label" for="ann-notes">Annotation Notes</label>
        <textarea name="annotation_notes" id="ann-notes"
                  placeholder="Reasoning, boundary-case decisions, adjudication…">{{ row.annotation_notes or '' }}</textarea>
      </div>
      <label class="verify-row">
        <input type="checkbox" name="label_verified" id="ann-verify"
               {% if row.label_verified=='true' %}checked{% endif %}>
        <span>
          <div class="verify-label">Mark as Verified</div>
          <div class="verify-hint">Only check after a second reviewer has confirmed the labels.</div>
        </span>
      </label>
    </div>
  </form>
</main>

<footer>
  <a class="btn btn-ghost" href="/api/progress" target="_blank">Progress API</a>
  <form method="post" action="/save/{{ idx }}" style="display:inline">
    <input type="hidden" name="annotation_status" value="SKIPPED">
    <input type="hidden" name="label_reviewer" value="{{ row.label_reviewer or '' }}">
    <input type="hidden" name="annotation_notes" value="{{ row.annotation_notes or 'Skipped.' }}">
    <button class="btn btn-skip" type="submit">Skip →</button>
  </form>
  <button class="btn btn-save" type="submit" form="ann-form">Save &amp; Next →</button>
</footer>

</div>
<script>
function toggleCheck(cb, idx) {
  document.getElementById('lbl_'+idx).classList.toggle('checked', cb.checked);
}
document.addEventListener('keydown', e => {
  if ((e.ctrlKey||e.metaKey) && e.key==='Enter') document.getElementById('ann-form').submit();
});
</script>
</body></html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["seed", "batch", "all"], default="seed",
                        help="Annotation mode: seed (100-clause pilot), batch (uncertain clauses), all (full queue)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--queue",
                        default=str(ROOT / "data/annotations/annotation_queue.csv"))
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    seed_path  = ROOT / "data/annotations/active_learning_seed.csv"
    batch_path = ROOT / "data/annotations/next_active_learning_batch.csv"
    db_path    = ROOT / "data/annotations/annotations.csv"

    if not queue_path.exists():
        sys.exit(f"Queue not found: {queue_path}")

    app = create_app(queue_path, mode=args.mode,
                     seed_path=seed_path, batch_path=batch_path, db_path=db_path)

    print(f"\n  NDA Annotation Interface — Module 5B")
    print(f"  Mode  : {MODE_LABELS.get(args.mode, args.mode)}")
    print(f"  Queue : {queue_path}")
    print(f"  DB    : {db_path}")
    print(f"  URL   : http://{args.host}:{args.port}")
    print(f"  Stop  : Ctrl+C\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
