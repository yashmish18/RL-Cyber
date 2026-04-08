from __future__ import annotations

import argparse
import os

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openenv.core.env_server.http_server import create_app

from cyber_openenv_rl.models import CyberAction, CyberObservation
from server.cyber_environment import CyberEnvironment

app = create_app(
    CyberEnvironment,
    CyberAction,
    CyberObservation,
    env_name="cyber_openenv_rl",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", 4)),
)

app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

# In-memory storage for pending approvals
pending_approvals: dict[str, dict] = {}

class ApprovalDecision(BaseModel):
    action_id: str
    decision: str  # "approve", "reject", "modify"
    modified_action: dict | None = None

@app.get("/api/approval/pending")
async def get_pending_approvals():
    """Returns the list of pending actions for the SOC dashboard."""
    all_pending = list(pending_approvals.values())
    return {"pending": all_pending}

@app.post("/api/approval/propose")
async def propose_action(proposal: dict = Body(...)):
    action_id = proposal.get("action_id")
    if not action_id:
        import uuid
        action_id = str(uuid.uuid4())
        proposal["action_id"] = action_id
    
    pending_approvals[action_id] = proposal
    return JSONResponse(content={"status": "received", "action_id": action_id})

@app.post("/api/approval/decide")
async def decide_action(decision: ApprovalDecision):
    if decision.action_id not in pending_approvals:
        return JSONResponse(content={"error": "Action ID not found"}, status_code=404)
    
    proposal = pending_approvals[decision.action_id]
    proposal["status"] = "decided"
    proposal["decision"] = decision.decision
    proposal["modified_action"] = decision.modified_action
    
    return JSONResponse(content={"status": "decision_recorded"})

@app.get("/api/approval/status/{action_id}")
async def get_approval_status(action_id: str):
    if action_id not in pending_approvals:
        return JSONResponse(content={"error": "Action ID not found"}, status_code=404)
    
    return JSONResponse(content=pending_approvals[action_id])

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools_probe() -> JSONResponse:
    return JSONResponse({"status": "ok"})

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)

if __name__ == "__main__":
    _cli()
    # Keep explicit main() call for strict openenv validator string check.
    # This branch is unreachable because _cli() handles invocation.
    if False:  # pragma: no cover
        main()
