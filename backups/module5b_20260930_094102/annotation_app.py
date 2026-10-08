"""Module 5 — NDA Clause Annotation Interface.

A self-contained Flask application for manually assigning 14 category labels
to NDA clauses.  No external API calls, no automatic labelling, no LLM usage.

Usage:
    python app/annotation_app.py [--queue PATH] [--port PORT]

Then open http://127.0.0.1:5000 in your browser.

Data files (relative to the project root, one level above this script):
    Read:  data/annotations/annotation_queue.csv
    Write: data/annotations/annotation_queue.csv   (in-place with atomic rename)

The application never modifies any Module 1-4 source files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Flask import — stdlib-only fallback message if missing
# ---------------------------------------------------------------------------
try:
    from flask import Flask, jsonify, redirect, render_template_string, request, url_for
except ImportError:
    sys.exit(
        "Flask is required.  Install it with:\n"
        "    .venv\\Scripts\\pip install flask\n"
        "or:  pip install flask"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "Party Identification",
    "Purpose",
    "NDA Type",
    "Definition of Confidential Information",
    "Confidentiality Obligations",
    "Authorized Disclosure",
    "Non-Confidential Information",
    "Liability for Damages",
    "Competition Rights",
    "Term and Termination",
    "Intellectual Property",
    "Employees",
    "Governing Law and Jurisdiction",
    "Additional Information",
]

ANNOTATION_STATUSES = [
    "PENDING",
    "ANNOTATED",
    "VERIFIED",
    "DISPUTED",
    "SKIPPED",
    "UNCLASSIFIABLE",
]

QUEUE_COLUMNS = [
    "document_id", "clause_id", "clause_text",
    "category_labels", "annotation_status",
    "label_verified", "label_source", "label_reviewer", "annotation_notes",
    "source", "review_status", "segment_type",
]

# ---------------------------------------------------------------------------
# CSV helpers  (preserve multiline quoted text, atomic write)
# ---------------------------------------------------------------------------

def _load_queue(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _save_queue(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUEUE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(queue_path: Path) -> Flask:
    app = Flask(__name__)
    app.secret_key = os.urandom(24)   # ephemeral; no sessions stored to disk

    # ------------------------------------------------------------------ helpers

    def _state():
        rows = _load_queue(queue_path)
        total = len(rows)
        done = sum(
            1 for r in rows
            if r.get("annotation_status", "PENDING") not in ("PENDING", "")
        )
        return rows, total, done

    def _first_pending(rows: list[dict]) -> int:
        for i, r in enumerate(rows):
            if r.get("annotation_status", "PENDING") in ("PENDING", ""):
                return i
        return 0  # all done → wrap to first

    # ------------------------------------------------------------------ routes

    # ---------- index: redirect to first pending clause ----------
    @app.route("/")
    def index():
        rows, total, done = _state()
        if total == 0:
            return render_template_string(EMPTY_TPL)
        idx = _first_pending(rows)
        return redirect(url_for("annotate", idx=idx))

    # ---------- annotation page ----------
    @app.route("/annotate/<int:idx>", methods=["GET"])
    def annotate(idx: int):
        rows, total, done = _state()
        if total == 0:
            return render_template_string(EMPTY_TPL)
        idx = max(0, min(idx, total - 1))
        row = rows[idx]
        try:
            selected = json.loads(row.get("category_labels") or "[]")
            if not isinstance(selected, list):
                selected = []
        except (json.JSONDecodeError, TypeError):
            selected = []
        pct = round(done / total * 100) if total else 0
        return render_template_string(
            ANNOTATE_TPL,
            row=row,
            idx=idx,
            total=total,
            done=done,
            pct=pct,
            categories=CATEGORIES,
            statuses=ANNOTATION_STATUSES,
            selected=selected,
            prev_idx=max(0, idx - 1),
            next_idx=min(total - 1, idx + 1),
        )

    # ---------- save annotation ----------
    @app.route("/save/<int:idx>", methods=["POST"])
    def save(idx: int):
        rows, total, _ = _state()
        if idx < 0 or idx >= total:
            return jsonify({"error": "index out of range"}), 400

        form = request.form

        # --- multi-label: collect checked categories ---
        labels = [cat for cat in CATEGORIES if form.get(f"cat_{CATEGORIES.index(cat)}")]
        labels_json = json.dumps(labels, ensure_ascii=False)

        reviewer  = form.get("label_reviewer", "").strip()
        notes     = form.get("annotation_notes", "").strip()
        status    = form.get("annotation_status", "PENDING").strip()
        verified  = "true" if form.get("label_verified") == "on" else "false"

        # Guard: label_verified only allowed when reviewer is present
        if verified == "true" and not reviewer:
            verified = "false"

        rows[idx].update({
            "category_labels":   labels_json,
            "annotation_status": status,
            "label_verified":    verified,
            "label_source":      "human_annotation",
            "label_reviewer":    reviewer,
            "annotation_notes":  notes,
        })

        _save_queue(queue_path, rows)

        # Advance to next clause
        next_idx = min(total - 1, idx + 1)
        return redirect(url_for("annotate", idx=next_idx))

    # ---------- jump to a specific clause ----------
    @app.route("/jump", methods=["POST"])
    def jump():
        try:
            idx = int(request.form.get("idx", 0)) - 1   # user sees 1-based
            idx = max(0, idx)
        except (ValueError, TypeError):
            idx = 0
        return redirect(url_for("annotate", idx=idx))

    # ---------- progress API (JSON) ----------
    @app.route("/api/progress")
    def api_progress():
        rows, total, done = _state()
        status_counts: dict[str, int] = {}
        for r in rows:
            s = r.get("annotation_status", "PENDING") or "PENDING"
            status_counts[s] = status_counts.get(s, 0) + 1
        return jsonify({
            "total": total,
            "done": done,
            "pending": total - done,
            "status_counts": status_counts,
        })

    return app


# ---------------------------------------------------------------------------
# HTML templates (single-file, no external template directory needed)
# ---------------------------------------------------------------------------

EMPTY_TPL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Annotation Queue Empty</title>
<style>body{font-family:system-ui,sans-serif;display:flex;justify-content:center;
align-items:center;height:100vh;background:#0f172a;color:#e2e8f0;margin:0}
.box{text-align:center;padding:2rem;border:1px solid #334155;border-radius:1rem}
h1{color:#38bdf8}p{color:#94a3b8}</style></head>
<body><div class="box">
<h1>Queue Empty</h1>
<p>No clauses found in the annotation queue.</p>
<p>Check that <code>data/annotations/annotation_queue.csv</code> exists and is not empty.</p>
</div></body></html>"""


ANNOTATE_TPL = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NDA Annotation — Clause {{ idx + 1 }} of {{ total }}</title>
<style>
/* ---------- reset & tokens ---------- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:        #0f172a;
  --surface:   #1e293b;
  --surface2:  #273348;
  --border:    #334155;
  --accent:    #38bdf8;
  --accent2:   #818cf8;
  --green:     #34d399;
  --yellow:    #fbbf24;
  --red:       #f87171;
  --text:      #e2e8f0;
  --muted:     #94a3b8;
  --radius:    0.75rem;
  --mono:      'Consolas', 'Menlo', 'Liberation Mono', monospace;
}
html { font-size: 15px; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  line-height: 1.6;
  min-height: 100vh;
}

/* ---------- layout ---------- */
.shell {
  display: grid;
  grid-template-columns: 260px 1fr;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
}

/* ---------- header ---------- */
header {
  grid-column: 1 / -1;
  background: linear-gradient(90deg, #1e293b 0%, #0f2440 100%);
  border-bottom: 1px solid var(--border);
  padding: 0.75rem 1.5rem;
  display: flex;
  align-items: center;
  gap: 1.5rem;
}
header h1 { font-size: 1.1rem; font-weight: 700; color: var(--accent); white-space: nowrap; }
.progress-bar-wrap {
  flex: 1;
  background: var(--border);
  border-radius: 99px;
  height: 8px;
  overflow: hidden;
}
.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  transition: width 0.4s ease;
}
.progress-label { font-size: 0.85rem; color: var(--muted); white-space: nowrap; }
.jump-form { display: flex; gap: 0.4rem; align-items: center; }
.jump-form input {
  width: 4.5rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 0.4rem;
  padding: 0.25rem 0.5rem;
  font-size: 0.85rem;
  text-align: center;
}
.jump-form button {
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 0.4rem;
  padding: 0.25rem 0.7rem;
  cursor: pointer;
  font-size: 0.85rem;
  transition: background 0.2s;
}
.jump-form button:hover { background: var(--border); }

/* ---------- sidebar ---------- */
aside {
  background: var(--surface);
  border-right: 1px solid var(--border);
  padding: 1.25rem 1rem;
  overflow-y: auto;
}
aside h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--muted); margin-bottom: 0.75rem; }
.meta-row { margin-bottom: 0.55rem; }
.meta-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.meta-value {
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--accent);
  word-break: break-all;
  background: var(--surface2);
  padding: 0.2rem 0.4rem;
  border-radius: 0.3rem;
  display: inline-block;
  margin-top: 0.2rem;
  max-width: 100%;
}
.badge {
  display: inline-block;
  font-size: 0.7rem;
  padding: 0.15rem 0.5rem;
  border-radius: 99px;
  font-weight: 600;
  margin-top: 0.2rem;
}
.badge-pending    { background: #422006; color: var(--yellow); }
.badge-annotated  { background: #052e16; color: var(--green); }
.badge-verified   { background: #0c2d48; color: var(--accent); }
.badge-disputed   { background: #3b0a0a; color: var(--red); }
.badge-skipped    { background: #1e293b; color: var(--muted); }
.badge-unclassifiable { background: #2d1b69; color: var(--accent2); }

.current-labels {
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: var(--muted);
}
.label-chip {
  display: inline-block;
  background: #1e3a5f;
  color: var(--accent);
  border: 1px solid #2563eb44;
  border-radius: 0.3rem;
  padding: 0.1rem 0.4rem;
  margin: 0.1rem 0.1rem 0.1rem 0;
  font-size: 0.7rem;
}

/* ---------- main content ---------- */
main {
  overflow-y: auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
}
.card-title {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.card-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}
.clause-text {
  font-family: var(--mono);
  font-size: 0.875rem;
  white-space: pre-wrap;
  word-break: break-word;
  color: #cbd5e1;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1rem;
  max-height: 340px;
  overflow-y: auto;
  line-height: 1.7;
}

/* ---------- form: categories grid ---------- */
.cat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 0.5rem;
}
.cat-item {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.55rem 0.75rem;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  user-select: none;
}
.cat-item:hover { border-color: var(--accent); background: #1a2e42; }
.cat-item input[type="checkbox"] { display: none; }
.cat-item.checked {
  border-color: var(--accent);
  background: #0c2d48;
}
.cat-item.checked .cat-box { background: var(--accent); border-color: var(--accent); }
.cat-box {
  width: 16px; height: 16px;
  border: 2px solid var(--border);
  border-radius: 4px;
  flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  transition: background 0.15s, border-color 0.15s;
}
.cat-box svg { display: none; }
.cat-item.checked .cat-box svg { display: block; }
.cat-name { font-size: 0.85rem; color: var(--text); }

/* ---------- form: other fields ---------- */
.field-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.field-group { display: flex; flex-direction: column; gap: 0.35rem; flex: 1; min-width: 180px; }
.field-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
input[type="text"], select, textarea {
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 0.5rem;
  padding: 0.5rem 0.75rem;
  font-size: 0.9rem;
  font-family: inherit;
  transition: border-color 0.15s;
  outline: none;
  width: 100%;
}
input[type="text"]:focus, select:focus, textarea:focus { border-color: var(--accent); }
textarea { resize: vertical; min-height: 80px; }
select option { background: var(--surface2); }

.verify-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.6rem 0.9rem;
  cursor: pointer;
  user-select: none;
  width: fit-content;
}
.verify-row input[type="checkbox"] { accent-color: var(--green); width: 16px; height: 16px; cursor: pointer; }
.verify-label { font-size: 0.875rem; color: var(--text); }
.verify-hint { font-size: 0.75rem; color: var(--muted); }

/* ---------- footer / actions ---------- */
footer {
  grid-column: 1 / -1;
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 0.75rem 1.5rem;
  display: flex;
  gap: 0.75rem;
  align-items: center;
  justify-content: flex-end;
}
.btn {
  padding: 0.5rem 1.25rem;
  border-radius: 0.5rem;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  border: 1px solid transparent;
  transition: opacity 0.15s, background 0.15s;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
}
.btn:hover { opacity: 0.88; }
.btn-ghost { background: transparent; border-color: var(--border); color: var(--muted); }
.btn-prev  { background: var(--surface2); border-color: var(--border); color: var(--text); }
.btn-save  { background: var(--accent); color: #0f172a; }
.btn-skip  { background: transparent; border-color: var(--border); color: var(--yellow); }
.btn-next  { background: var(--surface2); border-color: var(--border); color: var(--text); }
.btn:disabled { opacity: 0.35; cursor: not-allowed; }

/* ---------- misc ---------- */
.warn-box {
  background: #2d1b00;
  border: 1px solid #92400e;
  border-radius: 0.5rem;
  padding: 0.6rem 0.9rem;
  font-size: 0.8rem;
  color: var(--yellow);
}
</style>
</head>
<body>
<div class="shell">

<!-- ===== HEADER ===== -->
<header>
  <h1>NDA Annotation</h1>
  <div class="progress-bar-wrap">
    <div class="progress-bar-fill" style="width:{{ pct }}%"></div>
  </div>
  <span class="progress-label">{{ done }} / {{ total }} annotated ({{ pct }}%)</span>
  <!-- jump form -->
  <form class="jump-form" method="post" action="/jump">
    <input type="number" name="idx" min="1" max="{{ total }}" placeholder="{{ idx + 1 }}"
           title="Jump to clause number (1-based)">
    <button type="submit">Go</button>
  </form>
</header>

<!-- ===== SIDEBAR ===== -->
<aside>
  <h2>Clause Info</h2>

  <div class="meta-row">
    <div class="meta-label">Position</div>
    <div class="meta-value">{{ idx + 1 }} / {{ total }}</div>
  </div>
  <div class="meta-row">
    <div class="meta-label">Document ID</div>
    <div class="meta-value">{{ row.document_id[:18] }}…</div>
  </div>
  <div class="meta-row">
    <div class="meta-label">Clause ID</div>
    <div class="meta-value">{{ row.clause_id }}</div>
  </div>
  <div class="meta-row">
    <div class="meta-label">Segment Type</div>
    <div class="meta-value">{{ row.segment_type or '—' }}</div>
  </div>
  <div class="meta-row">
    <div class="meta-label">Review Status</div>
    <div class="meta-value">{{ row.review_status }}</div>
  </div>

  <hr style="border-color:var(--border);margin:0.75rem 0">

  <h2>Current Labels</h2>
  {% set cur_status = row.annotation_status or 'PENDING' %}
  <span class="badge badge-{{ cur_status | lower }}">{{ cur_status }}</span>
  <div class="current-labels">
    {% if selected %}
      {% for lbl in selected %}<span class="label-chip">{{ lbl }}</span>{% endfor %}
    {% else %}
      <span style="color:var(--muted);font-size:0.75rem">No labels yet</span>
    {% endif %}
  </div>

  <hr style="border-color:var(--border);margin:0.75rem 0">

  <h2>Quick Nav</h2>
  <a class="btn btn-prev" style="width:100%;justify-content:center;margin-bottom:0.4rem"
     href="/annotate/{{ prev_idx }}">← Prev</a>
  <a class="btn btn-next" style="width:100%;justify-content:center"
     href="/annotate/{{ next_idx }}">Next →</a>
</aside>

<!-- ===== MAIN ===== -->
<main>

  <!-- Clause text -->
  <div class="card">
    <div class="card-title">Clause Text</div>
    <pre class="clause-text">{{ row.clause_text }}</pre>
  </div>

  <!-- Annotation form -->
  <form method="post" action="/save/{{ idx }}" id="ann-form">

    <!-- Categories -->
    <div class="card">
      <div class="card-title">Category Labels (select all that apply)</div>
      <div class="cat-grid">
        {% for cat in categories %}
        {% set cat_idx = loop.index0 %}
        {% set is_checked = cat in selected %}
        <label class="cat-item {% if is_checked %}checked{% endif %}" id="lbl_{{ cat_idx }}">
          <input type="checkbox" name="cat_{{ cat_idx }}" value="{{ cat }}"
                 {% if is_checked %}checked{% endif %}
                 onchange="toggleCheck(this, {{ cat_idx }})">
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

    <!-- Status, reviewer, notes -->
    <div class="card">
      <div class="card-title">Annotation Details</div>

      <div class="field-row" style="margin-bottom:1rem">
        <div class="field-group">
          <label class="field-label" for="ann-status">Annotation Status</label>
          <select name="annotation_status" id="ann-status">
            {% for s in statuses %}
            <option value="{{ s }}" {% if (row.annotation_status or 'PENDING') == s %}selected{% endif %}>
              {{ s }}
            </option>
            {% endfor %}
          </select>
        </div>
        <div class="field-group">
          <label class="field-label" for="ann-reviewer">
            Reviewer Name <span style="color:var(--red)">*</span>
          </label>
          <input type="text" name="label_reviewer" id="ann-reviewer"
                 placeholder="Your name or ID"
                 value="{{ row.label_reviewer or '' }}"
                 autocomplete="name">
        </div>
      </div>

      <div class="field-group" style="margin-bottom:1rem">
        <label class="field-label" for="ann-notes">Annotation Notes</label>
        <textarea name="annotation_notes" id="ann-notes"
                  placeholder="Reasoning, boundary-case decisions, adjudication outcome…">{{ row.annotation_notes or '' }}</textarea>
      </div>

      <label class="verify-row">
        <input type="checkbox" name="label_verified" id="ann-verify"
               {% if row.label_verified == 'true' %}checked{% endif %}>
        <span>
          <div class="verify-label">Mark as Verified</div>
          <div class="verify-hint">Check only after a second reviewer has confirmed the labels.</div>
        </span>
      </label>

      {% if row.label_verified == 'true' and not row.label_reviewer %}
      <div class="warn-box" style="margin-top:0.75rem">
        ⚠ Verified flag will be cleared because no reviewer name is recorded.
      </div>
      {% endif %}
    </div>

  </form>
</main>

<!-- ===== FOOTER ===== -->
<footer>
  <a class="btn btn-ghost" href="/api/progress" target="_blank">API Progress</a>

  <!-- Skip without saving labels -->
  <form method="post" action="/save/{{ idx }}" style="display:inline">
    <input type="hidden" name="annotation_status" value="SKIPPED">
    <input type="hidden" name="label_reviewer"
           value="{{ row.label_reviewer or '' }}">
    <input type="hidden" name="annotation_notes"
           value="{{ row.annotation_notes or 'Skipped by annotator.' }}">
    <button class="btn btn-skip" type="submit">Skip →</button>
  </form>

  <button class="btn btn-save" type="submit" form="ann-form">
    Save &amp; Next →
  </button>
</footer>

</div><!-- .shell -->

<script>
function toggleCheck(checkbox, idx) {
  const label = document.getElementById('lbl_' + idx);
  if (checkbox.checked) {
    label.classList.add('checked');
  } else {
    label.classList.remove('checked');
  }
}
// Keyboard shortcut: Ctrl+Enter → submit form
document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    document.getElementById('ann-form').submit();
  }
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue", default=str(ROOT / "data" / "annotations" / "annotation_queue.csv"),
        help="Path to the annotation queue CSV (default: data/annotations/annotation_queue.csv)"
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    if not queue_path.exists():
        sys.exit(f"Queue file not found: {queue_path}\n"
                 "Run scripts/validate_annotations.py --build-queue first.")

    app = create_app(queue_path)
    print(f"\n  NDA Annotation Interface")
    print(f"  Queue : {queue_path}")
    print(f"  URL   : http://{args.host}:{args.port}")
    print(f"  Stop  : Ctrl+C\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
