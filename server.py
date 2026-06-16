"""
Terra -- Semantic Embedding Server
Part of the Aetheris/Strata/Sentari stack.
"""
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import uvicorn, logging, json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Terra] %(message)s")
log = logging.getLogger("terra")

app = FastAPI(title="Terra", version="1.0.0")

log.info("Loading model: all-MiniLM-L6-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")
log.info("Terra ready")

@app.get("/health")
def health():
    return {"status": "ok", "service": "Terra", "model": "all-MiniLM-L6-v2"}

@app.post("/api/embeddings")
async def embed(request: Request):
    body = await request.body()
    log.info(f"Raw request: {body.decode()[:200]}")
    data = json.loads(body)
    # Handle all formats: {prompt}, {input}, {text}, or nested {texts:[]}
    text = (data.get("prompt") or data.get("input") or 
            data.get("text") or data.get("content") or "")
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    texts = data.get("texts", [])
    if texts and not text:
        text = " ".join(str(t) for t in texts)
    log.info(f"Embedding: {len(text)} chars")
    vector = model.encode(text).tolist()
    return {"embedding": vector}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11436)
