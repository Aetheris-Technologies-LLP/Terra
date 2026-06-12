"""
Terra — Semantic Embedding Server
Part of the Aetheris/Strata/Sentari stack.
Exposes /api/embeddings compatible with Weaviate text2vec-ollama format.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Terra] %(message)s")
log = logging.getLogger("terra")

app = FastAPI(title="Terra", description="Semantic Embedding Server", version="1.0.0")

log.info("Loading embedding model: all-MiniLM-L6-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")
log.info("Model loaded — Terra is ready")

class EmbedRequest(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    prompt: str

class EmbedResponse(BaseModel):
    embedding: list[float]

@app.get("/health")
def health():
    return {"status": "ok", "service": "Terra", "model": "all-MiniLM-L6-v2"}

@app.post("/api/embeddings")
def embed(req: EmbedRequest):
    log.info(f"Embedding request: {len(req.prompt)} chars")
    vector = model.encode(req.prompt).tolist()
    return {"embedding": vector}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11436)
