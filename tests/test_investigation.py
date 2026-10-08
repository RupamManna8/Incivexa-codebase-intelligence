import os
import sys
from pathlib import Path
import pytest

# Add parent directory to sys.path
engine_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(engine_root))

from src.git_manager import GitManager
from src.parser import CodeParser
from src.embedder import Embedder
from src.indexer import CodebaseIndexer
from src.retriever import MultiTierRetriever
from src.agent import InvestigationAgent

DEMO_REPO_PATH = (engine_root.parent / "demo-repo").resolve()

def test_git_manager():
    git_mgr = GitManager(str(DEMO_REPO_PATH))
    head_commit = git_mgr.get_head_commit()
    assert len(head_commit) == 40

    files = git_mgr.list_code_files()
    assert any("order_service.js" in f for f in files)

    content = git_mgr.get_file_content("src/order_service.js")
    assert content is not None
    assert "processOrder" in content

def test_ast_parser():
    content = """
    export function calculateTotal(items) {
        return items.length;
    }
    """
    symbols = CodeParser.parse_javascript("calc.js", content)
    assert len(symbols) >= 1
    assert symbols[0].name == "calculateTotal"

def test_embedder_bge_m3():
    embedder = Embedder()
    vec = embedder.get_embedding("Order processing failure in billingAddress")
    assert vec.shape == (1024,)
    assert abs(float((vec ** 2).sum()) - 1.0) < 1e-3

def test_indexer_and_search():
    git_mgr = GitManager(str(DEMO_REPO_PATH))
    embedder = Embedder()
    indexer = CodebaseIndexer(git_mgr, embedder)
    head_commit = git_mgr.get_head_commit()

    index_res = indexer.build_index(head_commit)
    assert index_res["status"] in ["INDEXED", "ALREADY_INDEXED"]

    results = indexer.search(head_commit, "billingAddress zipCode", top_k=2)
    assert len(results) > 0
    assert any("order_service.js" in r["file_path"] for r in results)

def test_retriever_pipeline():
    git_mgr = GitManager(str(DEMO_REPO_PATH))
    embedder = Embedder()
    indexer = CodebaseIndexer(git_mgr, embedder)
    retriever = MultiTierRetriever(git_mgr, indexer)
    head_commit = git_mgr.get_head_commit()

    stack = [
        {
            "file": "src/order_service.js",
            "line": 23,
            "function": "processOrder"
        }
    ]
    error_msg = "TypeError: Cannot read properties of undefined (reading 'zipCode')"

    res = retriever.retrieve(head_commit, stack, error_msg)
    assert res["crash_location"] is not None
    assert res["crash_location"]["file"] == "src/order_service.js"
    assert res["crash_location"]["function"] == "processOrder"
    assert "billingAddress" in res["code_context"]

def test_investigation_agent_qwen():
    git_mgr = GitManager(str(DEMO_REPO_PATH))
    embedder = Embedder()
    indexer = CodebaseIndexer(git_mgr, embedder)
    retriever = MultiTierRetriever(git_mgr, indexer)
    agent = InvestigationAgent()
    head_commit = git_mgr.get_head_commit()

    stack = [
        {
            "file": "src/order_service.js",
            "line": 23,
            "function": "processOrder"
        }
    ]
    error_msg = "TypeError: Cannot read properties of undefined (reading 'zipCode')"

    retrieval_data = retriever.retrieve(head_commit, stack, error_msg)
    incident_data = {
        "title": "TypeError in processOrder",
        "severity": "critical",
        "incident_id": "test-incident-123"
    }

    report = agent.investigate(incident_data, retrieval_data)
    assert "targetLocation" in report
    assert "confidence" in report
    assert report["confidence"] in ["HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE"]
    assert "evidence" in report
    assert any(e["type"] == "FACT" for e in report["evidence"])
    assert "suggestedFix" in report
    assert "diff" in report["suggestedFix"]
    print("\n--- AGENT GENERATED REPORT ---")
    print("Summary:", report["rootCauseAnalysis"]["summary"])
    print("Confidence:", report["confidence"], f"({report['confidenceScore']})")
    print("Suggested Diff:\n", report["suggestedFix"]["diff"])
