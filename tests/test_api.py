"""
tests/test_api.py
=================
pytest test suite for FinLens Document Intelligence API.

Tests cover:
  - Health check
  - /ingest endpoint
  - /analyze endpoint (standard)
  - /analyze/stream endpoint (SSE)
  - RAG evaluation scores presence
  - Edge cases (empty query, missing file)

Run with:
    pytest tests/test_api.py -v

Performance: session-scoped fixtures make only 1 ingest + 1 analyze + 1 stream
LLM call for the entire suite (~3-4 minutes on CPU Ollama).
"""

import json
import pytest
import httpx
from pathlib import Path
from fpdf import FPDF

BASE_URL = "http://localhost:8000"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION-SCOPED FIXTURES  (LLM called once per session)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    """Generate a minimal bank statement PDF for testing."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "test_bank_statement.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "HDFC Bank Statement - Test 2024", ln=True)
    pdf.cell(0, 10, "Account: ACC999999", ln=True)
    pdf.cell(0, 10, "Customer: Test User", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Date         Description              Amount    Balance", ln=True)
    pdf.cell(0, 10, "01-Jan-2024  Salary credit            +50000    50000", ln=True)
    pdf.cell(0, 10, "05-Jan-2024  ATM withdrawal           -20000    30000", ln=True)
    pdf.cell(0, 10, "10-Jan-2024  UPI to Amazon            -2000     28000", ln=True)
    pdf.cell(0, 10, "15-Jan-2024  UPI received Rahul       +10000    38000", ln=True)
    pdf.cell(0, 10, "20-Jan-2024  ATM withdrawal           -15000    23000", ln=True)
    pdf.cell(0, 10, "31-Jan-2024  Closing balance                    23000", ln=True)
    pdf.output(str(pdf_path))
    return pdf_path


@pytest.fixture(scope="session")
def ingest_response(sample_pdf) -> dict:
    """Ingest the sample PDF once and return the full response dict."""
    with open(sample_pdf, "rb") as f:
        response = httpx.post(
            f"{BASE_URL}/ingest",
            files={"file": (sample_pdf.name, f, "application/pdf")},
            timeout=60,
        )
    assert response.status_code == 200, f"Ingest setup failed: {response.text}"
    return response.json()


@pytest.fixture(scope="session")
def ingested_doc(ingest_response) -> str:
    """Return the ingested document name."""
    return ingest_response["document"]


@pytest.fixture(scope="session")
def analyze_response(ingested_doc) -> dict:
    """
    Call /analyze once and cache the response for the entire session.
    All TestAnalyze tests share this single LLM call.
    """
    response = httpx.post(
        f"{BASE_URL}/analyze",
        json={"query": "What are the large withdrawals?"},
        timeout=300,
    )
    assert response.status_code == 200, f"Analyze setup failed: {response.text}"
    return response.json()


@pytest.fixture(scope="session")
def stream_response_text(ingested_doc) -> str:
    """
    Call /analyze/stream once and cache the raw SSE text for the entire session.
    All TestAnalyzeStream tests share this single LLM call.
    """
    response = httpx.post(
        f"{BASE_URL}/analyze/stream",
        json={"query": "What are the large withdrawals?"},
        timeout=300,
    )
    assert response.status_code == 200, f"Stream setup failed: {response.text}"
    return response.text


@pytest.fixture(scope="session")
def stream_events(stream_response_text) -> list[dict]:
    """Parse SSE text into a list of event dicts."""
    events = []
    for line in stream_response_text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except Exception:
                pass
    return events


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_returns_200(self):
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert response.json()["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# INGEST
# ══════════════════════════════════════════════════════════════════════════════


class TestIngest:
    def test_ingest_returns_200(self, ingest_response):
        # ingest_response fixture already asserts 200; just verify it's accessible
        assert ingest_response is not None

    def test_ingest_returns_success_status(self, ingest_response):
        assert ingest_response["status"] == "success"

    def test_ingest_returns_document_name(self, ingest_response, sample_pdf):
        assert "document" in ingest_response
        assert ingest_response["document"] == sample_pdf.stem

    def test_ingest_returns_chunks_stored(self, ingest_response):
        assert "chunks_stored" in ingest_response
        assert ingest_response["chunks_stored"] >= 1

    def test_ingest_no_temp_filename_leak(self, ingest_response):
        """Verify the temp path is not returned as document name."""
        assert not ingest_response["document"].startswith("tmp")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE  (all tests share one cached LLM response)
# ══════════════════════════════════════════════════════════════════════════════


class TestAnalyze:
    def test_analyze_status_is_valid(self, analyze_response):
        assert analyze_response["status"] == "valid"

    def test_analyze_data_has_required_fields(self, analyze_response):
        data = analyze_response["data"]
        for field in [
            "answer",
            "transactions",
            "total_amount",
            "confidence",
            "source_document",
            "anomaly_flag",
        ]:
            assert field in data, f"Missing field: {field}"

    def test_analyze_answer_is_non_empty_string(self, analyze_response):
        answer = analyze_response["data"]["answer"]
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

    def test_analyze_transactions_is_list(self, analyze_response):
        assert isinstance(analyze_response["data"]["transactions"], list)

    def test_analyze_transaction_fields(self, analyze_response):
        for txn in analyze_response["data"]["transactions"]:
            assert "date" in txn
            assert "amount" in txn
            assert "type" in txn
            assert "description" in txn
            assert isinstance(txn["amount"], (int, float))

    def test_analyze_confidence_in_range(self, analyze_response):
        conf = analyze_response["data"]["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_analyze_total_amount_is_numeric(self, analyze_response):
        total = analyze_response["data"]["total_amount"]
        assert isinstance(total, (int, float))

    def test_analyze_anomaly_flag_is_bool(self, analyze_response):
        assert isinstance(analyze_response["data"]["anomaly_flag"], bool)

    def test_analyze_includes_evaluation_key(self, analyze_response):
        assert "evaluation" in analyze_response

    def test_analyze_evaluation_has_required_metrics(self, analyze_response):
        evaluation = analyze_response.get("evaluation")
        if evaluation is not None:
            for key in [
                "context_relevancy",
                "faithfulness",
                "answer_relevancy",
                "evaluator",
                "latency_ms",
            ]:
                assert key in evaluation, f"Missing evaluation key: {key}"

    def test_analyze_evaluation_scores_in_range(self, analyze_response):
        evaluation = analyze_response.get("evaluation")
        if evaluation is not None:
            for metric in ["context_relevancy", "faithfulness", "answer_relevancy"]:
                score = evaluation[metric]
                assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"

    def test_analyze_evaluation_evaluator_is_string(self, analyze_response):
        evaluation = analyze_response.get("evaluation")
        if evaluation is not None:
            assert isinstance(evaluation["evaluator"], str)
            assert len(evaluation["evaluator"]) > 0

    def test_analyze_evaluation_latency_is_positive(self, analyze_response):
        evaluation = analyze_response.get("evaluation")
        if evaluation is not None:
            assert isinstance(evaluation["latency_ms"], int)
            assert evaluation["latency_ms"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE STREAM  (all tests share one cached SSE response)
# ══════════════════════════════════════════════════════════════════════════════


class TestAnalyzeStream:
    def test_stream_has_events(self, stream_events):
        assert len(stream_events) > 0

    def test_stream_emits_all_expected_steps(self, stream_events):
        step_names = {e.get("step") for e in stream_events}
        for step in [
            "Query Received",
            "Retrieving Chunks",
            "Reasoning (LLM)",
            "Validation",
            "RAG Evaluation",
            "complete",
        ]:
            assert step in step_names, f"Missing step: {step}"

    def test_stream_steps_have_running_and_done(self, stream_events):
        for step in ["Retrieving Chunks", "Reasoning (LLM)", "Validation"]:
            statuses = {e["status"] for e in stream_events if e.get("step") == step}
            assert "running" in statuses, f"{step} missing 'running' event"
            assert "done" in statuses, f"{step} missing 'done' event"

    def test_stream_complete_event_exists(self, stream_events):
        complete = next((e for e in stream_events if e.get("step") == "complete"), None)
        assert complete is not None

    def test_stream_complete_status_is_done(self, stream_events):
        complete = next((e for e in stream_events if e.get("step") == "complete"), None)
        assert complete["status"] == "done"

    def test_stream_complete_has_data(self, stream_events):
        complete = next((e for e in stream_events if e.get("step") == "complete"), None)
        assert complete.get("data") is not None

    def test_stream_complete_data_has_answer(self, stream_events):
        complete = next((e for e in stream_events if e.get("step") == "complete"), None)
        data = complete["data"].get("data", {})
        assert "answer" in data
        assert isinstance(data["answer"], str)

    def test_stream_done_events_have_elapsed_ms(self, stream_events):
        done_events = [
            e
            for e in stream_events
            if e.get("status") == "done" and e.get("step") != "Query Received"
        ]
        assert len(done_events) > 0
        for evt in done_events:
            assert "elapsed_ms" in evt, f"Missing elapsed_ms in: {evt}"
            assert isinstance(evt["elapsed_ms"], int)
            assert evt["elapsed_ms"] >= 0

    def test_stream_complete_has_total_elapsed(self, stream_events):
        complete = next((e for e in stream_events if e.get("step") == "complete"), None)
        assert "elapsed_ms" in complete
        assert complete["elapsed_ms"] > 0

    def test_stream_complete_includes_evaluation(self, stream_events):
        complete = next((e for e in stream_events if e.get("step") == "complete"), None)
        assert "evaluation" in complete["data"]


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_analyze_missing_query_returns_422(self):
        response = httpx.post(f"{BASE_URL}/analyze", json={}, timeout=30)
        assert response.status_code == 422

    def test_ingest_missing_file_returns_422(self):
        response = httpx.post(f"{BASE_URL}/ingest", timeout=30)
        assert response.status_code == 422

    def test_analyze_with_null_source_document_returns_200(self, ingested_doc):
        """source_document is optional — null should not cause 422."""
        response = httpx.post(
            f"{BASE_URL}/analyze",
            json={"query": "What are the withdrawals?", "source_document": None},
            timeout=300,
        )
        assert response.status_code == 200

    def test_health_endpoint_always_available(self):
        """Health check must always respond regardless of agent state."""
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200
