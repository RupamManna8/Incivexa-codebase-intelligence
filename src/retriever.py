from typing import List, Dict, Any, Optional
from pathlib import Path
from .git_manager import GitManager
from .parser import CodeParser, CodeSymbol
from .indexer import CodebaseIndexer

class MultiTierRetriever:
    def __init__(self, git_mgr: GitManager, indexer: CodebaseIndexer):
        self.git_mgr = git_mgr
        self.indexer = indexer

    def _is_user_frame(self, file_path: str) -> bool:
        """Filters out library and runtime internal frames."""
        normalized = file_path.replace("\\", "/").lower()
        ignore_patterns = [
            "node_modules",
            "node:internal",
            "internal/modules",
            "/lib/python",
            "site-packages",
            "venv/",
            ".venv/"
        ]
        return not any(p in normalized for p in ignore_patterns)

    def retrieve(
        self,
        commit_sha: str,
        stack_frames: List[Dict[str, Any]],
        error_message: str
    ) -> Dict[str, Any]:
        """Runs the 4-level retrieval pipeline."""
        # 1. Checkout commit
        resolved_sha = self.git_mgr.checkout_commit(commit_sha)

        retrieval_result = {
            "commit_sha": resolved_sha,
            "crash_location": None,
            "code_context": None,
            "enclosing_symbol": None,
            "dependencies": [],
            "semantic_matches": [],
            "git_blame": None,
            "commit_diff": None,
            "test_files": []
        }

        # 2. Tier 1: Deterministic Stack Locator
        target_frame = None
        for frame in stack_frames:
            file_path = frame.get("file", "")
            if file_path and self._is_user_frame(file_path):
                target_frame = frame
                break

        if not target_frame and stack_frames:
            target_frame = stack_frames[0]

        if target_frame:
            raw_file = target_frame.get("file", "")
            line = target_frame.get("line", 1)
            content = self.git_mgr.get_file_content(raw_file)

            if content:
                lines = content.splitlines()
                # Line context: 15 lines before and after
                start_l = max(1, line - 15)
                end_l = min(len(lines), line + 15)
                context_lines = [
                    f"{idx+1:4d}: {lines[idx]}"
                    for idx in range(start_l - 1, end_l)
                ]
                snippet = "\n".join(context_lines)

                # Find AST symbols in target file
                symbols = CodeParser.parse_file(raw_file, content)
                enclosing = CodeParser.find_enclosing_symbol(symbols, line)

                retrieval_result["crash_location"] = {
                    "file": raw_file,
                    "line": line,
                    "function": enclosing.name if enclosing else target_frame.get("function", "anonymous")
                }
                retrieval_result["code_context"] = snippet
                retrieval_result["enclosing_symbol"] = enclosing.to_dict() if enclosing else None

                # Tier 4: Git Blame & Diff
                blame = self.git_mgr.get_file_blame(raw_file, line)
                retrieval_result["git_blame"] = blame.get("raw")
                retrieval_result["commit_diff"] = self.git_mgr.get_commit_diff(resolved_sha)[:4000]

                # Tier 2: Tracing calls made in enclosing function
                if enclosing and enclosing.calls:
                    for call_name in enclosing.calls[:5]:
                        # Look for definition of call_name in codebase via semantic search
                        query = f"function {call_name}"
                        matches = self.indexer.search(resolved_sha, query, top_k=2)
                        for m in matches:
                            if m["symbol_name"] != enclosing.name:
                                retrieval_result["dependencies"].append(m)

        # 3. Tier 3: Semantic Code Search using error message
        query_text = f"{error_message} {retrieval_result['crash_location']['function'] if retrieval_result['crash_location'] else ''}"
        semantic_results = self.indexer.search(resolved_sha, query_text, top_k=3)
        retrieval_result["semantic_matches"] = semantic_results

        # 4. Find matching test files in repo
        code_files = self.git_mgr.list_code_files()
        for f in code_files:
            lower_f = f.lower()
            if "test" in lower_f or "spec" in lower_f:
                retrieval_result["test_files"].append(f)

        return retrieval_result
