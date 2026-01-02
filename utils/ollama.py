import sys
from concurrent.futures import ThreadPoolExecutor

try:
    import ollama
except ImportError as e:
    print("Install dependencies: pip install ollama")
    sys.exit(1)

DEFAULT_EMBED_MODEL = "bge-m3"
EMBED_WORKERS = 6


def extract_embedding(resp):
    # возьмём список/список значений, либо ключ "embedding"
    if resp is None:
        return []
    if isinstance(resp, dict):
        if "embedding" in resp:
            return resp["embedding"]
        if "embeddings" in resp:
            return resp["embeddings"]
    # возможен объект с атрибутом embedding
    if hasattr(resp, "embedding"):
        return resp.embedding
    if hasattr(resp, "embeddings"):
        return resp.embeddings
    # если возвращён список напрямую
    if isinstance(resp, (list, tuple)):
        return resp
    return []


def embed(text):
    try:
        resp = ollama.embeddings(model=DEFAULT_EMBED_MODEL, prompt=text)
        return extract_embedding(resp)
    except Exception as e:
        print("Error embeddings:", e)
        return []


def embed_chunks(chunks, workers=EMBED_WORKERS):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(embed, chunks))
