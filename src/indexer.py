import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from .config import INDEX_DIR
from .git_manager import GitManager
from .parser import CodeParser
from .embedder import Embedder

class CodebaseIndexer:
    def __init__(self, git_mgr: GitManager, embedder: Optional[Embedder] = None):
        self.git_mgr = git_mgr
        self.embedder = embedder or Embedder()
        self.index_dir = INDEX_DIR

    def _get_index_file(self, commit_sha: str) -> Path:
        repo_id = hashlib.sha256(self.git_mgr.repo_url.encode()).hexdigest()[:12]
        return self.index_dir / f"index_{repo_id}_{commit_sha[:12]}.npz"

    def is_indexed(self, commit_sha: str) -> bool:
        return self._get_index_file(commit_sha).exists()

    def build_index(self, commit_sha: str) -> Dict[str, Any]:
        """Builds and caches code embeddings for a specific commit."""
        # 1. Ensure repo is cloned/fetched and checkout commit
        resolved_sha = self.git_mgr.checkout_commit(commit_sha)
        target_file = self._get_index_file(resolved_sha)
        if target_file.exists():
            return {"status": "ALREADY_INDEXED", "commit_sha": resolved_sha}

        # 2. Extract code chunks
        code_files = self.git_mgr.list_code_files()
        all_chunks: List[Dict[str, Any]] = []

        for rel_path in code_files:
            content = self.git_mgr.get_file_content(rel_path)
            if content:
                chunks = CodeParser.extract_chunks_for_indexing(rel_path, content)
                all_chunks.extend(chunks)

        if not all_chunks:
            return {"status": "EMPTY", "chunk_count": 0, "commit_sha": resolved_sha}

        # 3. Embed chunks in batches
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedder.get_embeddings(texts)

        # 4. Save to compressed npz
        np.savez_compressed(
            target_file,
            embeddings=embeddings,
            metadata=json.dumps(all_chunks).encode("utf-8")
        )

        return {
            "status": "INDEXED",
            "commit_sha": resolved_sha,
            "chunk_count": len(all_chunks),
            "file_count": len(code_files)
        }

    def search(self, commit_sha: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs cosine similarity search against indexed chunks."""
        index_file = self._get_index_file(commit_sha)
        if not index_file.exists():
            # Build index on demand
            self.build_index(commit_sha)
            if not index_file.exists():
                return []

        # Load index
        with np.load(index_file, allow_pickle=True) as data:
            embeddings = data["embeddings"]
            metadata = json.loads(data["metadata"].tobytes().decode("utf-8"))

        if len(embeddings) == 0:
            return []

        # Embed query and compute cosine similarity
        query_vec = self.embedder.get_embedding(query)
        scores = np.dot(embeddings, query_vec)

        # Get top K indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = dict(metadata[idx])
            item["similarity_score"] = float(scores[idx])
            results.append(item)

        return results
