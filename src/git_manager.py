import os
import re
import subprocess
import hashlib
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from .config import CACHE_DIR

class GitManager:
    def __init__(self, repo_url: str, cache_dir: Optional[Path] = None, access_token: Optional[str] = None):
        raw_url = repo_url.strip()
        self.access_token = access_token or self._extract_token(raw_url) or os.getenv("GITHUB_TOKEN")
        self.repo_url = self._clean_url(raw_url)
        self.cache_dir = cache_dir or CACHE_DIR
        self.repo_path = self._resolve_repo_path()

    @staticmethod
    def _clean_url(url: str) -> str:
        """Strips embedded credentials from URL for deterministic hashing and secure storage."""
        return re.sub(r'https?://[^@]+@', 'https://', url.strip())

    @staticmethod
    def _extract_token(url: str) -> Optional[str]:
        """Extracts token if embedded in URL format like https://x-access-token:<token>@github.com."""
        match = re.search(r'https?://[^:]+:([^@]+)@github\.com', url)
        if match:
            return match.group(1)
        match = re.search(r'https?://([^@]+)@github\.com', url)
        if match:
            return match.group(1)
        return None

    def get_authenticated_url(self) -> str:
        """Builds an authenticated clone URL for private repositories."""
        if self.access_token and "github.com" in self.repo_url and "@" not in self.repo_url:
            return self.repo_url.replace("https://github.com", f"https://x-access-token:{self.access_token}@github.com")
        return self.repo_url

    def _get_git_env(self) -> Dict[str, str]:
        """Non-interactive git environment to prevent hanging on auth prompts."""
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_ASKPASS"] = ""
        return env

    def _resolve_repo_path(self) -> Path:
        """Resolves local repository path or clones remote into cache_dir."""
        # 1. Check if repo_url is an existing local directory or path
        candidate_path = Path(self.repo_url)
        if candidate_path.is_dir() and (candidate_path / ".git").exists():
            return candidate_path.resolve()

        # 2. Remote URL - create a deterministic folder name from clean URL
        url_hash = hashlib.sha256(self.repo_url.encode()).hexdigest()[:12]
        safe_name = "".join(c if c.isalnum() else "_" for c in self.repo_url.split("/")[-1].replace(".git", ""))
        repo_dir = self.cache_dir / f"{safe_name}_{url_hash}"
        return repo_dir

    def fetch_latest(self) -> None:
        """Fetches the latest commits from remote, with auth headers if private."""
        if not (self.repo_path / ".git").exists():
            return
        fetch_cmd = ["git", "fetch", "--all", "--tags"]
        if self.access_token and "github.com" in self.repo_url:
            auth_b64 = base64.b64encode(f"x-access-token:{self.access_token}".encode()).decode()
            fetch_cmd = ["git", "-c", f"http.https://github.com/.extraHeader=AUTHORIZATION: basic {auth_b64}", "fetch", "--all", "--tags"]
        subprocess.run(
            fetch_cmd,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
            check=False,
            env=self._get_git_env()
        )

    def ensure_repo(self) -> Path:
        """Ensures repository is cloned and up to date."""
        if not self.repo_path.exists():
            # Clone repo with authentication if private
            clone_url = self.get_authenticated_url()
            cmd = ["git", "clone", clone_url, str(self.repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=self._get_git_env())
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository {self.repo_url}: {result.stderr}")

            # Reset origin URL to clean URL so credentials are never saved to .git/config on disk
            if self.access_token:
                subprocess.run(
                    ["git", "remote", "set-url", "origin", self.repo_url],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    check=False
                )
        else:
            self.fetch_latest()
        return self.repo_path

    def checkout_commit(self, commit_sha: str) -> str:
        """Checks out a specific commit in detached HEAD mode. If unknown or HEAD, returns resolved SHA."""
        self.ensure_repo()
        if not commit_sha or commit_sha == "unknown" or commit_sha == "HEAD":
            # Get current HEAD sha
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout.strip()

        # Checkout target commit
        result = subprocess.run(
            ["git", "checkout", commit_sha],
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            # If commit not found, fetch latest and retry checkout
            self.fetch_latest()
            retry = subprocess.run(
                ["git", "checkout", commit_sha],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                check=False
            )
            if retry.returncode == 0:
                return commit_sha
            return self.get_head_commit()
        return commit_sha

    def get_head_commit(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout.strip()

    def get_commit_diff(self, commit_sha: str) -> str:
        """Returns unified git diff for a commit."""
        self.ensure_repo()
        cmd = ["git", "show", "--patch", "--stat", commit_sha]
        result = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True, check=False)
        return result.stdout if result.returncode == 0 else ""

    def get_file_blame(self, rel_path: str, line_number: int, context_lines: int = 3) -> Dict[str, Any]:
        """Runs git blame for a specific line."""
        self.ensure_repo()
        start = max(1, line_number - context_lines)
        end = line_number + context_lines
        cmd = ["git", "blame", f"-L{start},{end}", "--line-porcelain", rel_path]
        result = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return {"raw": "", "commit": None, "author": None}
        return {"raw": result.stdout}

    def get_file_content(self, rel_path: str) -> Optional[str]:
        """Reads file content from the checked-out repository."""
        file_path = self.repo_path / rel_path
        if not file_path.exists() or not file_path.is_file():
            # Try finding the file by suffix match if path was absolute from stack trace
            matched = list(self.repo_path.glob(f"**/{Path(rel_path).name}"))
            if matched:
                file_path = matched[0]
            else:
                return None
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def list_code_files(self, extensions: Optional[List[str]] = None) -> List[str]:
        """Lists tracked code files in repository."""
        self.ensure_repo()
        exts = extensions or [".js", ".mjs", ".ts", ".py", ".json", ".jsx", ".tsx"]
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            # Ignore .git, node_modules, venv, dist, build
            dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "venv", ".venv", "dist", "build", "__pycache__"]]
            for f in filenames:
                if any(f.endswith(ext) for ext in exts):
                    full_path = Path(root) / f
                    rel_path = full_path.relative_to(self.repo_path).as_posix()
                    files.append(rel_path)
        return files
