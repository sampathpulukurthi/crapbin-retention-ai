from fastapi import FastAPI, Request, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Any, Literal
import re
import os
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

app = FastAPI(title="Crapbin Retention AI Webhook", version="0.1.0")

class TranscriptInput(BaseModel):
    transcript: str
    call_id: str | None = None
    call_status: str = "answered"

class PickupLocation(BaseModel):
    locality: str | None = None
    city: str | None = None

class ExtractionResult(BaseModel):
    """Structured data extracted from a Bolna call transcript by the LLM."""
    customer_intent: Literal["interested", "not_interested", "callback_requested", "undecided", "unknown"]
    scrap_types: list[str] = Field(
        default_factory=list,
        description="Normalized lowercase scrap categories the customer mentioned, e.g. iron, e_waste, plastic, newspaper, cardboard, metal",
    )
    estimated_weight_kg: float | None = None
    estimated_boxes: str | None = Field(
        default=None,
        description="Box count as the customer phrased it, e.g. '4-5'; string, not a number",
    )
    pickup_location: PickupLocation | None = None
    preferred_pickup_date: str | None = Field(
        default=None, description="ISO date YYYY-MM-DD only if the customer stated one explicitly"
    )
    callback_requested: bool = False
    regional_manager_requested: bool = False
    next_action: Literal[
        "schedule_pickup", "coordinator_callback", "regional_manager_followup", "no_action", "unclear"
    ]
    whatsapp_draft: str | None = None

EXTRACTION_PROMPT = """You extract structured data from Bolna AI voice-agent call transcripts for Crapbin, a scrap pickup service.

Transcripts may be in English, Telugu, Hindi, or mixed language. Lines are usually prefixed with roles such as "assistant"/"user" or "agent"/"customer".

Rules:
- Base every field ONLY on what the CUSTOMER said. Treat the assistant's questions and statements as context, never as facts about the customer.
- customer_intent: the customer's overall stance - interested, not_interested, callback_requested, undecided, or unknown. Requests for a callback or manager go in the dedicated flags below, so an interested customer who also asks for a callback stays "interested". Use "callback_requested" only when the customer's response is essentially "call me later" without expressing interest or refusal.
- scrap_types: normalize to lowercase snake_case categories (e.g. iron, e_waste, plastic, newspaper, cardboard, metal).
- estimated_boxes keeps the customer's own phrasing (e.g. "4-5"), so it stays a string.
- preferred_pickup_date: ISO YYYY-MM-DD only if the customer stated a date explicitly; otherwise null.
- callback_requested: true if the customer asked to be called back, OR if the customer engaged with a pending executive/coordinator callback (e.g. asked when the executive will call or come). False if the customer never discussed a callback.
- regional_manager_requested: true only if the customer asked to speak with a manager or senior person.
- Do not assume a pickup is confirmed unless the customer clearly confirmed it.
- Never invent prices, dates, quantities, or addresses. Use null/false when not provided.
- whatsapp_draft: a short (1-2 sentence) polite thank-you message in English, only when the customer is interested or requested a callback; otherwise null. Do not include prices.
- next_action: schedule_pickup if the customer confirmed a pickup, coordinator_callback if a coordinator should call them back, regional_manager_followup if they asked for a manager, no_action if not interested, unclear otherwise."""

def extract_with_llm(transcript: str) -> ExtractionResult:
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": transcript},
        ],
        text_format=ExtractionResult,
    )
    return response.output_parsed

def normalize_transcript(payload: dict[str, Any]) -> str:
    """Accept plain transcript or common nested conversation formats.
    Adjust mappings after inspecting a real Bolna webhook sample."""
    for key in ("transcript", "conversation", "call_transcript"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(
                f"{item.get('role', item.get('speaker', 'unknown'))}: {item.get('content', item.get('text', ''))}"
                for item in value if isinstance(item, dict)
            )
    for container in ("data", "payload", "call"):
        nested = payload.get(container)
        if isinstance(nested, dict):
            found = normalize_transcript(nested)
            if found:
                return found
    return ""

def extract(transcript: str, call_status: str = "answered") -> dict[str, Any]:
    """Simple baseline heuristics for local testing; NOT an LLM or production classifier."""
    text = transcript.lower()
    if call_status.lower() in ("no_answer", "not_answered", "missed", "failed") or not text.strip():
        intent = "unknown"
        status = "no_answer" if call_status.lower() in ("no_answer", "not_answered", "missed") else call_status
    elif re.search(r"\b(not interested|don't call|do not call|stop calling|no scrap|not now)\b", text):
        intent, status = "not_interested", "answered"
    elif re.search(r"\b(call me back|callback|call later|busy now)\b", text):
        intent, status = "callback_requested", "answered"
    elif re.search(r"\b(yes|interested|pickup|pick up|schedule|collect|available)\b", text):
        intent, status = "interested", "answered"
    else:
        intent, status = "unknown", "answered"

    types = {
        "newspapers": ["newspaper", "papers"],
        "cardboard": ["cardboard", "carton", "boxes"],
        "metal": ["iron", "metal", "steel"],
        "plastic": ["plastic"],
        "e_waste": ["e-waste", "ewaste", "electronics", "computer"],
    }
    scrap_types = [name for name, terms in types.items() if any(t in text for t in terms)]
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", transcript)
    if intent == "interested":
        draft = "Thank you for speaking with Crapbin! Our coordinator will follow up about your scrap pickup."
    elif intent == "callback_requested":
        draft = "Thank you for your time. Crapbin can follow up when convenient."
    elif intent == "not_interested":
        draft = None
    else:
        draft = None
    return {
        "call_status": status,
        "customer_intent": intent,
        "scrap_types": scrap_types,
        "preferred_pickup_date": dates[0] if dates else None,
        "callback_requested": intent == "callback_requested",
        "whatsapp_draft": draft,
        "whatsapp_sent": False,
        "extraction_method": "keyword_baseline",
        "needs_review": True,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post(
    "/extract",
    openapi_extra={
        "requestBody": {
            "required": True,
            "description": 'Paste the full Bolna transcript as plain text, or send JSON: {"call_id": "...", "transcript": "..."}',
            "content": {
                "text/plain": {"schema": {"type": "string"}},
                "application/json": {"schema": TranscriptInput.model_json_schema()},
            },
        }
    },
)
async def extract_test(request: Request):
    raw = (await request.body()).decode("utf-8", errors="replace")
    body = TranscriptInput(transcript=raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "transcript" in parsed:
            body = TranscriptInput(**parsed)
    except (ValueError, TypeError):
        pass
    transcript = body.transcript.strip()
    no_answer = body.call_status.lower() in ("no_answer", "not_answered", "missed", "failed")
    if no_answer or not transcript:
        return {
            "call_id": body.call_id,
            "customer_intent": "unknown",
            "scrap_types": [],
            "estimated_weight_kg": None,
            "estimated_boxes": None,
            "pickup_location": None,
            "preferred_pickup_date": None,
            "callback_requested": False,
            "regional_manager_requested": False,
            "next_action": "no_action",
            "whatsapp_draft": None,
        }
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    try:
        result = extract_with_llm(transcript)
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")
    return {"call_id": body.call_id, **result.model_dump()}

@app.post("/webhooks/bolna")
async def bolna_webhook(request: Request):
    """POC: extract completed Bolna transcripts using the same OpenAI pipeline as /extract."""
    secret = os.getenv("LOCAL_WEBHOOK_TOKEN")
    if secret and request.headers.get("X-Local-Webhook-Token") != secret:
        raise HTTPException(status_code=401, detail="Invalid local webhook token")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Expected JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object")

    call_id = payload.get("id") or payload.get("call_id")
    status = str(payload.get("status") or payload.get("call_status") or "").lower()
    transcript = normalize_transcript(payload).strip()
    completed = {"completed", "complete", "ended", "finished", "call_completed"}
    if status not in completed:
        return {"call_id": call_id, "call_status": status, "processed": False, "reason": "call_not_completed"}
    if not transcript:
        return {"call_id": call_id, "call_status": status, "processed": False, "reason": "transcript_missing"}
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    try:
        analysis = extract_with_llm(transcript)
    except OpenAIError:
        raise HTTPException(status_code=502, detail="OpenAI extraction failed")
    return {"call_id": call_id, "call_status": status, "processed": True,
            "whatsapp_sent": False, **analysis.model_dump()}
