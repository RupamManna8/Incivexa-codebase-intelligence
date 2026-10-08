import json
import urllib.request
import urllib.error
import numpy as np
from typing import List, Union
from .config import AICREDITS_BASE_URL, AICREDITS_API_KEY, AICREDITS_EMBEDDING_MODEL

class Embedder:
    def __init__(
        self,
        api_key: str = AICREDITS_API_KEY,
        base_url: str = AICREDITS_BASE_URL,
        model: str = AICREDITS_EMBEDDING_MODEL
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._cache = {}

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates a single normalized embedding vector for a string."""
        if text in self._cache:
            return self._cache[text]

        results = self.get_embeddings([text])
        if len(results) > 0:
            self._cache[text] = results[0]
            return results[0]
        return np.zeros((1024,), dtype=np.float32)

    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Generates normalized embeddings for a list of texts in batches."""
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        all_vectors = []
        url = f"{self.base_url}/embeddings"

        for i in range(0, len(texts), batch_size):
            batch = [t[:3000] for t in texts[i : i + batch_size]]  # Trim very large texts
            payload = {
                "model": self.model,
                "input": batch
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    embeddings_data = data.get("data", [])
                    # Sort by index to maintain ordering
                    embeddings_data.sort(key=lambda x: x.get("index", 0))
                    for item in embeddings_data:
                        vec = np.array(item["embedding"], dtype=np.float32)
                        # Normalize vector for fast cosine similarity via dot product
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        all_vectors.append(vec)
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="replace")
                print(f"[Embedder] API HTTP Error {e.code}: {err_msg}")
                # Fallback zero vectors for this batch
                for _ in batch:
                    all_vectors.append(np.zeros((1024,), dtype=np.float32))
            except Exception as e:
                print(f"[Embedder] Error getting embeddings: {e}")
                for _ in batch:
                    all_vectors.append(np.zeros((1024,), dtype=np.float32))

        return np.array(all_vectors, dtype=np.float32)
