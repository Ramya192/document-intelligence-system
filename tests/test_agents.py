# tests/test_agents.py
# Eval test suite for Project B — Document Intelligence System
# Written BEFORE the app is built — Mistake #6 practice.
# Run: pytest tests/test_agents.py -v

import pytest
from agents.document_loader_agent import DocumentLoaderAgent
from agents.retriever_agent import RetrieverAgent
from agents.reasoning_agent import ReasoningAgent, ReasoningOutput
from agents.validator_agent import ValidatorAgent, ValidationOutput

PDF_PATH = "data/sample_bank_statement.pdf"

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def loaded_doc():
    """Load the sample PDF once for all tests."""
    loader = DocumentLoaderAgent()
    result = loader.load_from_path(PDF_PATH)
    assert result["status"] == "success", f"Failed to load PDF: {result}"
    return result

@pytest.fixture(scope="module")
def retriever():
    return RetrieverAgent()

@pytest.fixture(scope="module")
def reasoner():
    return ReasoningAgent()

@pytest.fixture(scope="module")
def validator():
    return ValidatorAgent()

# ── DocumentLoaderAgent tests ─────────────────────────────────────────────────
def test_document_loads_successfully(loaded_doc):
    assert loaded_doc["status"] == "success"
    assert loaded_doc["chunks"] > 0
    assert loaded_doc["total_chars"] > 0
    assert "doc_id" in loaded_doc

def test_document_produces_multiple_chunks(loaded_doc):
    """Bank statement should produce at least 3 chunks."""
    assert loaded_doc["chunks"] >= 3

# ── RetrieverAgent tests ──────────────────────────────────────────────────────
def test_retriever_returns_chunks(loaded_doc, retriever):
    chunks = retriever.retrieve("closing balance", doc_id=loaded_doc["doc_id"])
    assert len(chunks) > 0

def test_retriever_chunk_has_required_fields(loaded_doc, retriever):
    chunks = retriever.retrieve("transactions", doc_id=loaded_doc["doc_id"])
    for chunk in chunks:
        assert "text" in chunk
        assert "source" in chunk
        assert "distance" in chunk

def test_retriever_formats_context(loaded_doc, retriever):
    chunks  = retriever.retrieve("opening balance", doc_id=loaded_doc["doc_id"])
    context = retriever.format_context(chunks)
    assert isinstance(context, str)
    assert len(context) > 0
    assert "[Chunk" in context

# ── ReasoningAgent tests ──────────────────────────────────────────────────────
def test_reasoning_returns_structured_output(loaded_doc, retriever, reasoner):
    chunks   = retriever.retrieve("closing balance", doc_id=loaded_doc["doc_id"])
    context  = retriever.format_context(chunks)
    output   = reasoner.reason("What is the closing balance?", context)
    assert isinstance(output, ReasoningOutput)

def test_reasoning_output_has_valid_confidence(loaded_doc, retriever, reasoner):
    chunks  = retriever.retrieve("closing balance", doc_id=loaded_doc["doc_id"])
    context = retriever.format_context(chunks)
    output  = reasoner.reason("What is the closing balance?", context)
    assert 0.0 <= output.confidence <= 1.0

def test_reasoning_answer_type_is_valid(loaded_doc, retriever, reasoner):
    chunks  = retriever.retrieve("account type", doc_id=loaded_doc["doc_id"])
    context = retriever.format_context(chunks)
    output  = reasoner.reason("What type of account is this?", context)
    assert output.answer_type in {"factual", "analytical", "not_found"}

def test_reasoning_not_found_for_irrelevant_question(loaded_doc, retriever, reasoner):
    """Question about something not in a bank statement should return not_found."""
    chunks  = retriever.retrieve("weather forecast tomorrow", doc_id=loaded_doc["doc_id"])
    context = retriever.format_context(chunks)
    output  = reasoner.reason("What is the weather forecast for tomorrow?", context)
    assert output.answer_type == "not_found" or output.confidence < 0.5

# ── ValidatorAgent tests ──────────────────────────────────────────────────────
def test_validator_returns_structured_output(loaded_doc, retriever, reasoner, validator):
    chunks     = retriever.retrieve("closing balance", doc_id=loaded_doc["doc_id"])
    context    = retriever.format_context(chunks)
    reasoning  = reasoner.reason("What is the closing balance?", context)
    validation = validator.validate("What is the closing balance?", context, reasoning)
    assert isinstance(validation, ValidationOutput)

def test_validator_verdict_is_valid(loaded_doc, retriever, reasoner, validator):
    chunks     = retriever.retrieve("closing balance", doc_id=loaded_doc["doc_id"])
    context    = retriever.format_context(chunks)
    reasoning  = reasoner.reason("What is the closing balance?", context)
    validation = validator.validate("What is the closing balance?", context, reasoning)
    assert validation.verdict in {"PASS", "WARN", "FAIL"}

def test_validator_has_final_answer(loaded_doc, retriever, reasoner, validator):
    chunks     = retriever.retrieve("account holder", doc_id=loaded_doc["doc_id"])
    context    = retriever.format_context(chunks)
    reasoning  = reasoner.reason("What is the account holder's occupation?", context)
    validation = validator.validate("What is the account holder's occupation?", context, reasoning)
    assert isinstance(validation.final_answer, str)
    assert len(validation.final_answer) > 0

# ── End-to-end pipeline test ──────────────────────────────────────────────────
def test_full_pipeline_closing_balance(loaded_doc, retriever, reasoner, validator):
    """
    End-to-end test: the pipeline should find the closing balance
    from the bank statement with PASS or WARN verdict.
    """
    question   = "What is the closing balance of the account?"
    chunks     = retriever.retrieve(question, doc_id=loaded_doc["doc_id"])
    context    = retriever.format_context(chunks)
    reasoning  = reasoner.reason(question, context)
    validation = validator.validate(question, context, reasoning)

    assert validation.verdict in {"PASS", "WARN"}
    assert reasoning.is_answerable is True
    assert reasoning.confidence > 0.5
