"""
FastAPI application for the Multi-Agent Debate framework.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from loguru import logger

from app import solve, MADSystem

app = FastAPI(
    title="Multi-Agent Debate API",
    description="Evidence-triggered debate with conflict classification",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["POST"], allow_headers=["Content-Type"])

class QueryRequest(BaseModel):
    """Request model for /solve endpoint."""
    query: str = Field(..., description="The question to answer")
    demo_mode: bool = Field(default=False, description="Use demo mode without API keys")


class SourceResponse(BaseModel):
    """Source information."""
    url: str
    title: str
    snippet: str


class ConflictResponse(BaseModel):
    """Conflict information."""
    type: str
    confidence: float
    explanation: str


class JudgeResponse(BaseModel):
    """Judge result."""
    winner: str
    reason: str
    evidence_score: float
    confidence: float


class QueryResponse(BaseModel):
    """Response model for /solve endpoint."""
    query: str
    answer: str
    reasoning: str
    debate_triggered: bool
    conflict_type: Optional[str] = None
    conflict: Optional[ConflictResponse] = None
    evidence: list = Field(default_factory=list)
    sources: list = Field(default_factory=list)
    judge: Optional[JudgeResponse] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "Multi-Agent Debate API",
        "version": "1.0.0",
        "status": "running",
        "description": "Evidence-triggered debate with conflict classification",
    }


@app.post("/api/solve", response_model=QueryResponse)
@app.post("/solve", response_model=QueryResponse)
async def solve_query(request: QueryRequest):
    """
    Solve a query using the Multi-Agent Debate system.
    
    The system will:
    1. Route the query to appropriate agents
    2. Retrieve evidence from multiple sources
    3. Compare evidence for disagreements
    4. If evidence agrees → direct answer
    5. If evidence disagrees → classify conflict and run adaptive debate
    """
    try:
        logger.info(f"API request: {request.query}")
        
        trace = await solve(request.query, demo_mode=request.demo_mode)
        
        if trace.error:
            raise HTTPException(status_code=500, detail=trace.error)
        
        # Build response
        response = QueryResponse(
            query=trace.query,
            answer=trace.final_answer.answer if trace.final_answer else "No answer generated",
            reasoning=trace.final_answer.reasoning if trace.final_answer else "",
            debate_triggered=trace.debate_triggered,
            conflict_type=trace.conflict.type.value if trace.conflict else None,
            metrics=trace.metrics,
        )
        
        # Add conflict info
        if trace.conflict:
            response.conflict = ConflictResponse(
                type=trace.conflict.type.value,
                confidence=trace.conflict.confidence,
                explanation=trace.conflict.explanation,
            )
        
        # Add evidence
        if trace.agents:
            for ev in trace.agents:
                evidence_item = {
                    "agent_id": ev.agent_id,
                    "answer": ev.answer,
                    "claims": [{"text": c.text} for c in ev.claims[:5]],
                    "sources": [
                        {"url": s.url, "title": s.title} 
                        for s in ev.sources[:5]
                    ],
                }
                response.evidence.append(evidence_item)
        
        # Add sources
        if trace.final_answer and trace.final_answer.sources:
            for source in trace.final_answer.sources:
                response.sources.append({
                    "url": source.url,
                    "title": source.title,
                })
        
        # Add judge result
        if trace.judge_result:
            response.judge = JudgeResponse(
                winner=trace.judge_result.winner,
                reason=trace.judge_result.reason,
                evidence_score=trace.judge_result.evidence_score,
                confidence=trace.judge_result.confidence,
            )
        
        return response
        
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)