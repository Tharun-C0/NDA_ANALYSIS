import sys
sys.path.append(r"f:\NDA")
import pandas as pd
from scripts.llm_auto_review import init_gemini, call_gemini

def main():
    print("Testing single segment...")
    df = pd.read_csv(r"f:\NDA\data\human_review\priority_review.csv")
    
    # Target segment
    target_clause_id = "0859334b76224ff82c1312ae7b2b5da1_clause_000"
    target_row = df[df["clause_id"] == target_clause_id]
    
    if target_row.empty:
        print(f"Clause {target_clause_id} not found.")
        return
        
    row = target_row.iloc[0]
    idx = target_row.index[0]
    doc_id = row["document_id"]
    
    c_type = row["segment_type"]
    length = row["length"]
    c_text = row["clause_text"]
    
    prev_text = "N/A"
    next_text = "N/A"
    
    for i in range(idx - 1, -1, -1):
        if df.iloc[i]["document_id"] == doc_id:
            prev_text = df.iloc[i]["clause_text"]
            break
            
    for i in range(idx + 1, len(df)):
        if df.iloc[i]["document_id"] == doc_id:
            next_text = df.iloc[i]["clause_text"]
            break
            
    print(f"Segment: {target_clause_id}")
    print(f"Model configured: gemini-3.1-pro-preview")
    
    try:
        client = init_gemini()
        print("API Client initialized.")
    except Exception as e:
        print(f"Failed to initialize API: {e}")
        return
        
    print("Calling Gemini API...")
    result, error = call_gemini(client, c_type, length, c_text, prev_text, next_text)
    
    if error:
        print(f"API Error: {error}")
    else:
        print(f"API Response: {result}")
        print("Success!")

# Offline regression tests use synthetic segments and a mocked client only.
import json
from unittest.mock import MagicMock

import httpx
import pytest
import scripts.llm_auto_review as review


@pytest.fixture
def review_run(tmp_path, monkeypatch):
    queue = pd.DataFrame([
        {"document_id": "fixture", "clause_id": f"fixture_clause_{i:03d}",
         "clause_text": f"segment {i}", "segment_type": "fixture", "length": 9}
        for i in range(4)
    ])
    queue.to_csv(tmp_path / "review_dataset.csv", index=False)
    queue.iloc[[0, 1, 3]].to_csv(tmp_path / "priority_review.csv", index=False)
    monkeypatch.setattr(review, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(review, "REPORTS_DIR", tmp_path)
    for name, filename in {
        "CSV_PATH": "priority_review.csv", "PSEUDO_LABELS_PATH": "llm_pseudo_labels.csv",
        "REVIEW_REQUIRED_PATH": "human_review_required.csv", "LOG_PATH": "llm_review_log.csv",
        "STATS_PATH": "llm_review_statistics.csv",
    }.items():
        monkeypatch.setattr(review, name, tmp_path / filename)
    review.append_records(review.PSEUDO_LABELS_PATH, [{
        **queue.iloc[0].to_dict(), "ai_suggested_action": "KEEP", "ai_confidence": 1.0,
        "ai_reason": "Synthetic fixture", "review_method": "llm_auto_pseudolabel",
        "review_timestamp": "fixture timestamp", "previous_text": "N/A", "next_text": "segment 1",
        "ai_confidence_level": "HIGH_CONFIDENCE", "needs_context": False,
    }])
    review.append_records(review.LOG_PATH, [{
        "timestamp": "fixture timestamp", "document_id": "fixture", "clause_id": "fixture_clause_000",
        "status": "SUCCESS", "suggested_action": "KEEP", "confidence": 1.0, "error": "",
    }])
    review.REVIEW_REQUIRED_PATH.write_text("\n", encoding="utf-8")
    client = MagicMock()
    monkeypatch.setattr(review, "init_gemini", lambda: client)
    monkeypatch.setattr(review.time, "sleep", MagicMock())
    monkeypatch.setattr(review.random, "random", lambda: 1.0)
    return client


def fixture_response(action="KEEP", confidence=0.95):
    return MagicMock(text=json.dumps({"suggested_action": action, "confidence": confidence,
                                    "reason": "Synthetic fixture", "needs_context": False}))


def test_resume_preserves_success_and_uses_adjacent_context(review_run):
    original = review.PSEUDO_LABELS_PATH.read_bytes()
    original_log = review.LOG_PATH.read_bytes()
    review_run.models.generate_content.side_effect = [MagicMock(), fixture_response(), fixture_response()]
    stats = review.process_segments()
    assert stats["total_segments"] == 3
    assert stats["already_completed_before_run"] == 1
    assert stats["successful"] == 3
    assert stats["newly_processed"] == 2
    assert review.PSEUDO_LABELS_PATH.read_bytes().startswith(original)
    assert review.LOG_PATH.read_bytes().startswith(original_log)
    assert review_run.models.generate_content.call_count == 3
    prompt = review_run.models.generate_content.call_args_list[1].kwargs["contents"]
    assert "CURRENT SEGMENT (Type: fixture, Length: 9):\nsegment 1" in prompt
    assert "PREVIOUS SEGMENT:\nsegment 0" in prompt
    assert "NEXT SEGMENT:\nsegment 2" in prompt
    assert all(row.get("review_timestamp") for row in review.read_records(review.PSEUDO_LABELS_PATH))
    review_run.models.generate_content.reset_mock()
    review.process_segments()
    review_run.models.generate_content.assert_not_called()


def test_daily_quota_preflight_stops_without_changing_labels(review_run):
    original = review.PSEUDO_LABELS_PATH.read_bytes()
    original_required = review.REVIEW_REQUIRED_PATH.read_bytes()
    review_run.models.generate_content.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED RequestsPerDay")
    stats = review.process_segments()
    assert review_run.models.generate_content.call_count == 1
    review.time.sleep.assert_not_called()
    assert stats["run_status"] == "PAUSED_DAILY_QUOTA"
    assert stats["newly_processed"] == 0
    assert stats["api_failures_this_run"] == 1
    assert review.PSEUDO_LABELS_PATH.read_bytes() == original
    assert review.REVIEW_REQUIRED_PATH.read_bytes() == original_required
    assert review.read_records(review.LOG_PATH)[-1]["status"] == "CONNECTIVITY_ERROR"


def test_daily_quota_mid_queue_checkpoints_and_stops(review_run):
    review_run.models.generate_content.side_effect = [
        MagicMock(), fixture_response(), RuntimeError("429 RESOURCE_EXHAUSTED per_day quota")
    ]
    stats = review.process_segments()
    assert review_run.models.generate_content.call_count == 3
    assert stats["run_status"] == "PAUSED_DAILY_QUOTA"
    assert stats["successful"] == 2
    assert stats["remaining"] == 1
    assert len(review.read_records(review.LOG_PATH)) == 3


@pytest.mark.parametrize("error,attempts,delays", [
    (RuntimeError("429 RESOURCE_EXHAUSTED RequestsPerMinute"), 3, [60, 60]),
    (RuntimeError("503 MODEL_CAPACITY_EXHAUSTED"), 3, [30, 30]),
    (httpx.ReadTimeout(""), 3, [10, 10]),
    (RuntimeError("429 RESOURCE_EXHAUSTED RequestsPerDay retryDelay 60s"), 1, []),
    (RuntimeError("429 RESOURCE_EXHAUSTED unknown quota"), 1, []),
])
def test_bounded_retries(review_run, error, attempts, delays):
    review_run.models.generate_content.side_effect = error
    result, failure = review.call_gemini(review_run, "fixture", 9, "segment 1", "segment 0", "segment 2")
    assert result is None and failure
    assert review_run.models.generate_content.call_count == attempts
    assert [call.args[0] for call in review.time.sleep.call_args_list] == delays


def test_exhausted_temporary_retries_continue_to_next_segment(review_run):
    error = RuntimeError("503 MODEL_CAPACITY_EXHAUSTED")
    review_run.models.generate_content.side_effect = [MagicMock(), error, error, error, fixture_response()]
    stats = review.process_segments()
    assert review_run.models.generate_content.call_count == 5
    assert stats["newly_processed"] == 2
    assert stats["successful"] == 2
    assert stats["api_failures_this_run"] == 1
    assert stats["human_review_required"] == 1
    assert review.read_records(review.LOG_PATH)[-1]["clause_id"] == "fixture_clause_003"


@pytest.mark.parametrize("response", ["not json", "[]", '{"suggested_action":"INVALID"}',
                                      '{"suggested_action":"KEEP","confidence":2}'])
def test_invalid_json_is_not_a_success(review_run, response):
    review_run.models.generate_content.return_value = MagicMock(text=response)
    result, error = review.call_gemini(review_run, "fixture", 9, "segment 1", "segment 0", "segment 2")
    assert result is None and error
    assert review_run.models.generate_content.call_count == 1


def test_api_key_is_redacted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-secret")
    assert "synthetic-secret" not in review.safe_error(RuntimeError("synthetic-secret"))


def test_dry_run_never_calls_api(review_run):
    stats = review.process_segments(dry_run=True)
    assert stats == {"total_segments": 3, "already_completed": 1, "remaining": 2}
    review_run.models.generate_content.assert_not_called()


if __name__ == "__main__":
    main()
