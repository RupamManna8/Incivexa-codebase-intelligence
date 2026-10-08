import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId

from .config import (
    SERVICE_PORT,
    SERVICE_HOST,
    MONGO_URI,
    MONGO_DB_NAME
)
from fastapi.middleware.cors import CORSMiddleware
from .git_manager import GitManager
from .embedder import Embedder
from .indexer import CodebaseIndexer
from .retriever import MultiTierRetriever
from .agent import InvestigationAgent

app = FastAPI(title="Autonomous Investigation Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
incidents_col = db["incidents"]
investigations_col = db["investigations"]
repositories_col = db["repositories"]

# Shared instances
embedder = Embedder()
agent = InvestigationAgent()


class InvestigateRequest(BaseModel):
    incident_id: str
    project_id: Optional[str] = None
    repo_url: str
    commit_sha: Optional[str] = "HEAD"
    title: Optional[str] = "Incident"
    severity: Optional[str] = "error"
    stack: List[Dict[str, Any]] = []
    error_message: Optional[str] = ""


class IndexRequest(BaseModel):
    repo_url: str
    commit_sha: Optional[str] = "HEAD"


def run_investigation_job(investigation_id: str, req: InvestigateRequest):
    """Background task performing git retrieval, parsing, indexing, and LLM reasoning."""
    inv_obj_id = ObjectId(investigation_id)
    try:
        git_mgr = GitManager(req.repo_url)
        indexer = CodebaseIndexer(git_mgr, embedder)
        retriever = MultiTierRetriever(git_mgr, indexer)

        # 1. Multi-tier retrieval
        retrieval_data = retriever.retrieve(
            commit_sha=req.commit_sha or "HEAD",
            stack_frames=req.stack,
            error_message=req.error_message or req.title
        )

        # 2. Agent reasoning
        incident_data = {
            "title": req.title,
            "severity": req.severity,
            "incident_id": req.incident_id
        }
        report = agent.investigate(incident_data, retrieval_data)

        # 3. Update investigation record in MongoDB
        investigations_col.update_one(
            {"_id": inv_obj_id},
            {
                "$set": {
                    "status": "COMPLETED",
                    "commitSha": retrieval_data.get("commit_sha", req.commit_sha),
                    "targetLocation": report.get("targetLocation"),
                    "executionTrace": report.get("executionTrace"),
                    "evidence": report.get("evidence"),
                    "rootCauseAnalysis": report.get("rootCauseAnalysis"),
                    "confidence": report.get("confidence"),
                    "confidenceScore": report.get("confidenceScore"),
                    "suggestedFix": report.get("suggestedFix"),
                    "relatedTests": report.get("relatedTests"),
                    "completedAt": datetime.utcnow()
                }
            }
        )

        # 4. Update Incident record in MongoDB
        if ObjectId.is_valid(req.incident_id):
            incidents_col.update_one(
                {"_id": ObjectId(req.incident_id)},
                {
                    "$set": {
                        "status": "DIAGNOSED",
                        "investigationId": inv_obj_id,
                        "diagnostics.investigation": {
                            "investigation_id": str(inv_obj_id),
                            "confidence": report.get("confidence"),
                            "confidence_score": report.get("confidenceScore"),
                            "summary": report.get("rootCauseAnalysis", {}).get("summary"),
                            "target_file": report.get("targetLocation", {}).get("filePath"),
                            "target_function": report.get("targetLocation", {}).get("functionName"),
                            "timestamp": datetime.utcnow()
                        }
                    },
                    "$push": {
                        "timeline": {
                            "timestamp": datetime.utcnow(),
                            "action": "DIAGNOSIS_COMPLETED",
                            "actor": "INVESTIGATION_ENGINE",
                            "message": f"Root cause diagnosed ({report.get('confidence')} confidence). Root cause: {report.get('rootCauseAnalysis', {}).get('summary', 'Identified')}",
                            "metadata": {
                                "investigation_id": str(inv_obj_id),
                                "confidence": report.get("confidence"),
                                "target": f"{report.get('targetLocation', {}).get('filePath')}:{report.get('targetLocation', {}).get('startLine')}"
                            }
                        }
                    }
                }
            )

        print(f"[InvestigationEngine] Completed investigation {investigation_id} for incident {req.incident_id}")

    except Exception as e:
        print(f"[InvestigationEngine] Investigation error: {e}")
        investigations_col.update_one(
            {"_id": inv_obj_id},
            {
                "$set": {
                    "status": "FAILED",
                    "error": str(e),
                    "completedAt": datetime.utcnow()
                }
            }
        )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "investigation-engine",
        "models": {
            "embedding": embedder.model,
            "agent": agent.model
        }
    }


@app.post("/investigate")
def investigate(req: InvestigateRequest, background_tasks: BackgroundTasks):
    """Triggers an autonomous investigation for an incident."""
    inv_doc = {
        "incidentId": ObjectId(req.incident_id) if ObjectId.is_valid(req.incident_id) else req.incident_id,
        "projectId": ObjectId(req.project_id) if req.project_id and ObjectId.is_valid(req.project_id) else None,
        "repoUrl": req.repo_url,
        "commitSha": req.commit_sha or "HEAD",
        "status": "IN_PROGRESS",
        "createdAt": datetime.utcnow()
    }
    result = investigations_col.insert_one(inv_doc)
    inv_id = str(result.inserted_id)

    # Mark incident as INVESTIGATING
    if ObjectId.is_valid(req.incident_id):
        incidents_col.update_one(
            {"_id": ObjectId(req.incident_id)},
            {
                "$set": {
                    "status": "INVESTIGATING",
                    "investigationId": result.inserted_id
                },
                "$push": {
                    "timeline": {
                        "timestamp": datetime.utcnow(),
                        "action": "INVESTIGATION_STARTED",
                        "actor": "INVESTIGATION_ENGINE",
                        "message": f"Autonomous investigation started for repo {req.repo_url} at {req.commit_sha}",
                        "metadata": {"investigation_id": inv_id}
                    }
                }
            }
        )

    background_tasks.add_task(run_investigation_job, inv_id, req)
    return {
        "status": "ACCEPTED",
        "investigation_id": inv_id,
        "message": "Investigation job dispatched."
    }


@app.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str):
    """Fetches full investigation report and status."""
    if not ObjectId.is_valid(investigation_id):
        raise HTTPException(status_code=400, detail="Invalid investigation ID format")

    doc = investigations_col.find_one({"_id": ObjectId(investigation_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Investigation not found")

    doc["_id"] = str(doc["_id"])
    if "incidentId" in doc:
        doc["incidentId"] = str(doc["incidentId"])
    if "projectId" in doc and doc["projectId"]:
        doc["projectId"] = str(doc["projectId"])
    return doc


@app.post("/index")
def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    """Explicitly triggers codebase vector indexing for a commit."""
    def _do_index():
        git_mgr = GitManager(req.repo_url)
        indexer = CodebaseIndexer(git_mgr, embedder)
        indexer.build_index(req.commit_sha or "HEAD")

    background_tasks.add_task(_do_index)
    return {"status": "INDEXING_STARTED", "repo_url": req.repo_url}


if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
