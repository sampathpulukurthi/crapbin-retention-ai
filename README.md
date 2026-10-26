# Crapbin Retention AI FastAPI POC

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and set OPENAI_API_KEY
uvicorn main:app --reload
```

Visit http://127.0.0.1:8000/docs

## Test the OpenAI-powered extraction

`POST /extract` accepts either the raw transcript as plain text, or JSON `{"call_id": "...", "transcript": "..."}`.

In Swagger: expand `POST /extract` -> Try it out -> paste the full transcript (e.g. the contents of `sample_transcript.txt`) into the body -> Execute.

From the terminal:

```bash
# raw transcript
curl -X POST http://127.0.0.1:8000/extract \
  -H 'Content-Type: text/plain' \
  --data-binary @sample_transcript.txt

# JSON with call_id
curl -X POST http://127.0.0.1:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"call_id": "test-001", "transcript": "assistant: Hi, I am Priya from Crapbin.\nuser: Yes, I have iron, e-waste and plastic scrap. Around 28 kg and 4 to 5 boxes. I am in Secunderabad, Hyderabad."}'
```

`/extract` sends the transcript to `gpt-4.1-mini` with a structured Pydantic output schema. It supports English, Telugu, Hindi and mixed-language transcripts, bases fields only on the customer's statements, and returns `null`/`false` for anything not provided. Requires `OPENAI_API_KEY` in `.env` (loaded via python-dotenv) or the environment; returns 503 if unset.

The `/webhooks/bolna` endpoint still uses the local keyword baseline.

Test from another terminal:
```bash
curl -X POST http://127.0.0.1:8000/webhooks/bolna -H 'Content-Type: application/json' --data-binary @sample_payload.json
```

This is a local prototype, not yet mapped to Bolna's actual webhook schema. Extraction uses basic English keyword heuristics, not AI; multilingual conversations and natural language dates require an LLM later. It does not send WhatsApp messages. Do not use with real customer data until authentication, privacy safeguards and deduplication are implemented.
