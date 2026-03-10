"""
Wrapper API that fixes OpenEnv's HTTP session management.
This provides a working /step endpoint for hackathon judges.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uuid

from negotiate_env.models import NegotiateAction, NegotiateObservation
from negotiate_env.server.environment import NegotiateEnvironment

app = FastAPI(
    title="NegotiateEnv API",
    description="B2B SaaS Contract Negotiation Environment",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
sessions: Dict[str, NegotiateEnvironment] = {}

class ResetRequest(BaseModel):
    scenario_id: Optional[str] = None
    session_id: Optional[str] = None
    difficulty: Optional[str] = "hard"  # Default to hard for long-horizon (50 turns)

class ResetResponse(BaseModel):
    session_id: str
    observation: dict
    reward: float
    done: bool

class StepRequest(BaseModel):
    action: NegotiateAction
    session_id: str

class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(sessions)}

@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "NegotiateEnv",
        "description": "B2B SaaS Contract Negotiation Environment",
        "version": "1.0.0",
        "endpoints": {
            "POST /reset": "Start new episode (returns session_id)",
            "POST /step": "Take action (requires session_id)",
            "GET /health": "Health check",
            "GET /state": "Server state"
        },
        "usage": "Call /reset first to get session_id, then use it in /step calls"
    }

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    """
    Reset environment and start new episode.
    Returns session_id to use in subsequent /step calls.
    """
    # Reuse session_id if provided, otherwise create new
    if request.session_id and request.session_id in sessions:
        session_id = request.session_id
        env = sessions[session_id]
    else:
        session_id = str(uuid.uuid4())
        difficulty = request.difficulty or "hard"  # Default to hard for 50 turns
        env = NegotiateEnvironment(difficulty=difficulty, use_hf_dataset=True)
        sessions[session_id] = env
    
    # Reset environment
    kwargs = {}
    if request.scenario_id:
        kwargs["scenario_id"] = request.scenario_id
    
    obs = env.reset(**kwargs)
    
    return ResetResponse(
        session_id=session_id,
        observation=obs.dict(),
        reward=obs.reward,
        done=obs.done,
    )

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Take action in environment.
    Requires session_id from /reset call.
    """
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session_id: {request.session_id}. Call /reset first to get a valid session_id."
        )
    
    env = sessions[request.session_id]
    obs = env.step(request.action)
    
    # Clean up finished sessions
    if obs.done:
        del sessions[request.session_id]
    
    return StepResponse(
        observation=obs.dict(),
        reward=obs.reward,
        done=obs.done,
    )

@app.get("/state")
def state():
    """Get server state and active sessions."""
    return {
        "active_sessions": len(sessions),
        "max_sessions": 100,
        "session_ids_sample": list(sessions.keys())[:5],
    }

@app.get("/info")
def info():
    """Environment information."""
    return {
        "name": "NegotiateEnv",
        "description": "B2B SaaS Contract Negotiation Environment for RL training",
        "difficulty": "medium",
        "max_turns": 10,
        "dataset": "mayukareddy/SyntheticSaasDataset",
        "scenarios": 200,
        "action_types": ["offer", "counter", "probe", "accept", "walkaway"],
    }

@app.get("/action_schema")
def action_schema():
    """JSON schema for actions."""
    return NegotiateAction.schema()

@app.get("/observation_schema")
def observation_schema():
    """JSON schema for observations."""
    return NegotiateObservation.schema()
