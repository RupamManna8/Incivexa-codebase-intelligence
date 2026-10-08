import os
import sys
import time
from pathlib import Path
from bson import ObjectId
from pymongo import MongoClient

# Add parent directory to sys.path
engine_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(engine_root))

from src.config import MONGO_URI, MONGO_DB_NAME
from src.api import app, InvestigateRequest, run_investigation_job

DEMO_REPO_PATH = (engine_root.parent / "demo-repo").resolve()

def test_full_investigation_pipeline_e2e():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    incidents_col = db["incidents"]
    investigations_col = db["investigations"]

    # 1. Create a dummy incident in MongoDB representing a high-risk error
    dummy_project_id = ObjectId()
    incident_doc = {
        "projectId": dummy_project_id,
        "primaryHash": "test_hash_order_service_bug",
        "title": "TypeError: Cannot read properties of undefined (reading 'zipCode')",
        "service": "demo-payment-app",
        "environment": "production",
        "releaseSha": "HEAD",
        "repoUrl": str(DEMO_REPO_PATH),
        "status": "OPEN",
        "severity": "critical",
        "eventCount": 5,
        "timeline": [
            {
                "timestamp": time.time(),
                "action": "INCIDENT_CREATED",
                "actor": "CORRELATOR",
                "message": "Incident detected and fingerprinted."
            }
        ],
        "evidence": [
            {
                "trace_id": "test-trace-123",
                "message": "TypeError: Cannot read properties of undefined (reading 'zipCode')",
                "stack": [
                    {
                        "file": "src/order_service.js",
                        "line": 23,
                        "function": "processOrder"
                    }
                ]
            }
        ],
        "diagnostics": {
            "ml_diagnosis": {
                "risk_level": "CRITICAL",
                "risk_score": 0.98,
                "confidence": 0.95
            }
        }
    }
    inserted_incident = incidents_col.insert_one(incident_doc)
    incident_id = str(inserted_incident.inserted_id)

    print(f"\n[E2E] Created incident: {incident_id}")

    # 2. Prepare investigation payload
    req = InvestigateRequest(
        incident_id=incident_id,
        project_id=str(dummy_project_id),
        repo_url=str(DEMO_REPO_PATH),
        commit_sha="HEAD",
        title=incident_doc["title"],
        severity=incident_doc["severity"],
        stack=incident_doc["evidence"][0]["stack"],
        error_message=incident_doc["evidence"][0]["message"]
    )

    # 3. Create investigation document
    inv_doc = {
        "incidentId": ObjectId(incident_id),
        "projectId": dummy_project_id,
        "repoUrl": str(DEMO_REPO_PATH),
        "commitSha": "HEAD",
        "status": "IN_PROGRESS"
    }
    inv_res = investigations_col.insert_one(inv_doc)
    investigation_id = str(inv_res.inserted_id)

    # 4. Run investigation job synchronously for test validation
    run_investigation_job(investigation_id, req)

    # 5. Verify Investigation Document in MongoDB
    saved_inv = investigations_col.find_one({"_id": ObjectId(investigation_id)})
    assert saved_inv is not None
    assert saved_inv["status"] == "COMPLETED"
    assert saved_inv["targetLocation"]["functionName"] == "processOrder"
    assert "zipCode" in saved_inv["rootCauseAnalysis"]["summary"] or "billingAddress" in saved_inv["rootCauseAnalysis"]["summary"]
    assert saved_inv["confidence"] in ["HIGH", "MEDIUM"]
    assert len(saved_inv["suggestedFix"]["diff"]) > 0

    # 6. Verify Incident Document status updated in MongoDB
    updated_incident = incidents_col.find_one({"_id": ObjectId(incident_id)})
    assert updated_incident is not None
    assert updated_incident["status"] == "DIAGNOSED"
    assert updated_incident["investigationId"] == ObjectId(investigation_id)
    assert updated_incident["diagnostics"]["investigation"]["investigation_id"] == investigation_id

    # Verify timeline entry added
    timeline_actions = [t["action"] for t in updated_incident["timeline"]]
    assert "DIAGNOSIS_COMPLETED" in timeline_actions

    print("[E2E] Full autonomous investigation lifecycle verified successfully!")
    print(f"[E2E] Incident status: {updated_incident['status']}")
    print(f"[E2E] Investigation confidence: {saved_inv['confidence']} ({saved_inv['confidenceScore']})")
    print(f"[E2E] Diff length: {len(saved_inv['suggestedFix']['diff'])} chars")

    # Cleanup test records
    incidents_col.delete_one({"_id": ObjectId(incident_id)})
    investigations_col.delete_one({"_id": ObjectId(investigation_id)})
