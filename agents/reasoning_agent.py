import json
from pydantic import BaseModel, Field
from typing import Optional
from settings import Settings


class Transaction(BaseModel):
    date: str = Field(description="Transaction date YYYY-MM-DD")
    amount: float = Field(description="Amount in INR")
    type: str = Field(description="ATM / UPI / NEFT etc.")
    description: Optional[str] = Field(default="")


class ReasoningOutput(BaseModel):
    answer: str = Field(description="Direct answer to user question")
    transactions: list[Transaction] = Field(default=[])
    total_amount: Optional[float] = Field(default=None)
    confidence: float = Field(description="0 to 1")
    source_document: str = Field(description="Source PDF filename")
    anomaly_flag: bool = Field(default=False)
    anomaly_reason: Optional[str] = Field(default=None)


class ReasoningAgent:
    def __init__(self):
        if Settings.LLM_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(
                model=Settings.OLLAMA_MODEL, base_url=Settings.OLLAMA_BASE_URL
            )
        else:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL)

    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        context = "\n\n".join(
            [
                f"[{i+1}] {chunk['text']}\nSource: {chunk['source']}"
                for i, chunk in enumerate(chunks)
            ]
        )
        return f"""You are a BFSI document analyst.
Use the context below to answer the question.
Flag anomalies if you detect suspicious patterns.

Context:
{context}

Question: {query}

Respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.
The "answer" field MUST be a plain text string — NOT a list or array.
Do not include closing balance or opening balance as transactions.
The "answer" must mention ALL transactions listed in the "transactions" array. Do not omit any.
Write the "answer" as a human-readable sentence. Use ₹ symbol and readable dates (e.g. '4th March' not '04-Mar-2024').
{{
    "answer": "answer": "A clear, readable sentence answering the question. Example: 'There were 3 large withdrawals in March: ₹18,000 on 2nd March (Rent), ₹60,000 on 4th March (ATM), and ₹50,000 on 20th March (ATM).'",
    "transactions": [
        {{"date": "YYYY-MM-DD", "amount": 0.0, "type": "ATM/UPI/NEFT", "description": "details"}}
    ],
    "total_amount": 0.0,
    "confidence": 0.9,
    "source_document": "filename",
    "anomaly_flag": false,
    "anomaly_reason": null
}}"""

    def _parse_ollama_response(self, raw: str) -> ReasoningOutput:
        raw = raw.strip()

        # Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        # Extract JSON object bounds
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        data = json.loads(raw)

        # Repair: if answer is a list, flatten to string
        if isinstance(data.get("answer"), list):
            items = data["answer"]
            # If list of transaction-like dicts, summarise them
            if items and isinstance(items[0], dict):
                data["answer"] = " | ".join(
                    f"{t.get('date', '')} {t.get('description', '')} ₹{t.get('amount', '')}"
                    for t in items
                )
            else:
                # List of strings
                data["answer"] = " ".join(str(i) for i in items)

        # Repair: ensure answer is always a non-empty string
        if not data.get("answer") or not isinstance(data["answer"], str):
            data["answer"] = "No answer could be extracted."

        # Repair: transactions must be a list
        if not data.get("transactions") or not isinstance(data["transactions"], list):
            data["transactions"] = []  # ← this line was missing entirely

        # Repair: fix None fields in transactions
        for t in data["transactions"]:
            if t.get("description") is None:
                t["description"] = ""
            if t.get("type") is None:
                t["type"] = "UNKNOWN"

        # Repair: recalculate total_amount from actual transactions
        data["total_amount"] = (
            sum(t["amount"] for t in data["transactions"])
            if data["transactions"]
            else 0.0
        )

        # Repair: confidence must be float 0–1

        # Repair: confidence must be float 0–1
        conf = data.get("confidence", 0.8)
        if isinstance(conf, str):
            try:
                conf = float(conf)
            except ValueError:
                conf = 0.8
        data["confidence"] = max(0.0, min(1.0, float(conf)))

        return ReasoningOutput(**data)

    def reason(self, query: str, chunks: list[dict]) -> ReasoningOutput:
        prompt = self._build_prompt(query, chunks)
        if Settings.LLM_PROVIDER == "ollama":
            response = self.llm.invoke(prompt)
            return self._parse_ollama_response(response.content)
        else:
            structured_llm = self.llm.with_structured_output(ReasoningOutput)
            return structured_llm.invoke(prompt)
