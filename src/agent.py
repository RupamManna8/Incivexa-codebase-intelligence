import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .config import (
    AICREDITS_BASE_URL,
    AICREDITS_API_KEY,
    AICREDITS_AGENT_MODEL
)

class TargetLocation(BaseModel):
    filePath: str = ""
    functionName: str = ""
    startLine: int = 0
    endLine: int = 0

class EvidenceItem(BaseModel):
    type: str = Field(..., pattern="^(FACT|HYPOTHESIS)$")
    description: str
    source: Optional[str] = None

class RootCauseAnalysis(BaseModel):
    summary: str
    mechanism: str
    contributingFactors: List[str] = []

class SuggestedFix(BaseModel):
    description: str
    diff: str
    affectedFiles: List[str] = []

class InvestigationReport(BaseModel):
    targetLocation: TargetLocation
    executionTrace: List[Dict[str, Any]] = []
    evidence: List[EvidenceItem]
    rootCauseAnalysis: RootCauseAnalysis
    confidence: str = Field(..., pattern="^(HIGH|MEDIUM|LOW|INSUFFICIENT_EVIDENCE)$")
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    suggestedFix: SuggestedFix
    relatedTests: List[str] = []


SYSTEM_PROMPT = """You are an elite autonomous Site Reliability Engineering (SRE) and Codebase Investigation Agent.
Your job is to investigate production incidents by analyzing stack traces, source code context, AST symbols, and git history.

CRITICAL RULES:
1. STRICT EVIDENCE SEPARATION:
   - FACT: Directly observed data (e.g. error message, stack frame line, variable access on line X, exact compiler exception).
   - HYPOTHESIS: Inferred reasoning (e.g. why variable was undefined, race condition likelihood, missing validation).
2. CONFIDENCE RATING:
   - "HIGH": Target line and mechanism are completely verified by the provided code context.
   - "MEDIUM": Likely root cause, but some dependency or upstream caller context is missing.
   - "LOW": Plausible guess, but insufficient context.
   - "INSUFFICIENT_EVIDENCE": When the error cannot be pinpointed in the provided code snippets.
3. SUGGESTED FIX:
   - Provide a valid, clean unified git diff (--- a/path +++ b/path) that fixes the root cause safely without introducing regressions.
4. OUTPUT FORMAT:
   - You MUST output ONLY valid JSON matching this schema:
{
  "targetLocation": {
    "filePath": "string",
    "functionName": "string",
    "startLine": number,
    "endLine": number
  },
  "executionTrace": [
    { "file": "string", "function": "string", "line": number, "snippet": "string" }
  ],
  "evidence": [
    { "type": "FACT", "description": "string", "source": "string" },
    { "type": "HYPOTHESIS", "description": "string", "source": "string" }
  ],
  "rootCauseAnalysis": {
    "summary": "string",
    "mechanism": "string",
    "contributingFactors": ["string"]
  },
  "confidence": "HIGH" | "MEDIUM" | "LOW" | "INSUFFICIENT_EVIDENCE",
  "confidenceScore": number (0.0 to 1.0),
  "suggestedFix": {
    "description": "string",
    "diff": "unified diff string",
    "affectedFiles": ["string"]
  },
  "relatedTests": ["string"]
}
Do not wrap your output in markdown unless standard json fences are used.
"""

class InvestigationAgent:
    def __init__(
        self,
        api_key: str = AICREDITS_API_KEY,
        base_url: str = AICREDITS_BASE_URL,
        model: str = AICREDITS_AGENT_MODEL
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def investigate(
        self,
        incident_data: Dict[str, Any],
        retrieval_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs the LLM investigation loop on retrieved evidence."""
        user_prompt = f"""Investigate this incident:
Incident Title: {incident_data.get('title', 'Unknown')}
Severity: {incident_data.get('severity', 'error')}
Commit SHA: {retrieval_data.get('commit_sha')}

Crash Location:
{json.dumps(retrieval_data.get('crash_location') or {}, indent=2)}

Source Code Around Crash Site:
```
{retrieval_data.get('code_context') or 'No source available'}
```

Enclosing AST Symbol:
{json.dumps(retrieval_data.get('enclosing_symbol') or {}, indent=2)}

Related Code Dependencies:
{json.dumps([d.get('text', '')[:300] for d in (retrieval_data.get('dependencies') or [])], indent=2)}

Semantic Code Matches:
{json.dumps([m.get('text', '')[:300] for m in (retrieval_data.get('semantic_matches') or [])], indent=2)}

Git Blame on Line:
{retrieval_data.get('git_blame') or 'None'}

Recent Commit Diff:
```
{(retrieval_data.get('commit_diff') or 'None')[:1500]}
```

Relevant Test Files Found:
{json.dumps(retrieval_data.get('test_files') or [])}

Generate the structured root cause analysis JSON report now.
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"].strip()
                return self._parse_json_response(content, retrieval_data)
        except Exception as e:
            print(f"[InvestigationAgent] LLM API Call Error: {e}")
            return self._build_fallback_report(retrieval_data, str(e))

    def _parse_json_response(self, content: str, retrieval_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts and validates JSON from model reply."""
        clean_content = content
        if "```json" in clean_content:
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_content:
            clean_content = clean_content.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(clean_content)
            # Validate via Pydantic model
            report = InvestigationReport(**parsed)
            return report.model_dump()
        except Exception as err:
            print(f"[InvestigationAgent] Schema validation failed ({err}), raw: {clean_content[:200]}")
            return self._build_fallback_report(retrieval_data, f"Schema error: {err}")

    def _build_fallback_report(self, retrieval_data: Dict[str, Any], err_msg: str) -> Dict[str, Any]:
        loc = retrieval_data.get("crash_location") or {}
        return {
            "targetLocation": {
                "filePath": loc.get("file", "unknown"),
                "functionName": loc.get("function", "anonymous"),
                "startLine": loc.get("line", 0),
                "endLine": loc.get("line", 0)
            },
            "executionTrace": [],
            "evidence": [
                {
                    "type": "FACT",
                    "description": f"Incident crashed at {loc.get('file')}:{loc.get('line')}",
                    "source": "Stack Trace"
                }
            ],
            "rootCauseAnalysis": {
                "summary": "Automatic parsing completed; detailed LLM inference fell back.",
                "mechanism": err_msg,
                "contributingFactors": []
            },
            "confidence": "LOW",
            "confidenceScore": 0.3,
            "suggestedFix": {
                "description": "Manual inspection required.",
                "diff": "",
                "affectedFiles": [loc.get("file", "")]
            },
            "relatedTests": retrieval_data.get("test_files", [])
        }
