"""
Query router for determining which agents/tools to use.
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel
from loguru import logger

from agents.base_agent import BaseAgent, registry
from agents.search_agent import SearchAgentA, SearchAgentB, SearchAgentC
from config import settings


class ToolType(str, Enum):
    """Types of tools available."""
    WEB_SEARCH = "web_search"
    RAG = "rag"
    CALCULATOR = "calculator"
    DATABASE = "database"


class RoutingDecision(BaseModel):
    """Result of routing a query."""
    tools: List[ToolType] = [ToolType.WEB_SEARCH]
    agents: List[str] = []  # agent IDs
    reasoning: str = ""
    confidence: float = 1.0


class QueryRouter:
    """
    Routes queries to appropriate agents/tools.
    
    Currently supports web_search. Designed to be extensible for:
    - RAG (retrieval-augmented generation)
    - Calculator
    - Database queries
    - Code execution
    - etc.
    """
    
    def __init__(self):
        self._initialized = False
        self._mode: Optional[bool] = None
    
    def initialize(self, demo_mode: Optional[bool] = None) -> None:
        """Initialize default agents."""
        mode = (self._mode if demo_mode is None and self._initialized else (settings.demo_mode if demo_mode is None else demo_mode))
        if self._initialized and self._mode == mode:
            return
        if self._initialized and self._mode != mode:
            registry.clear()
        
        # Register default search agents
        registry.register(SearchAgentA(demo_mode=mode))
        registry.register(SearchAgentB(demo_mode=mode))
        
        # Register third agent if not in demo mode (to save resources)
        if not mode:
            registry.register(SearchAgentC(demo_mode=mode))
        
        self._initialized = True
        self._mode = mode
        logger.info(f"Router initialized with agents: {registry.list_ids()}")
    
    def route(self, query: str) -> RoutingDecision:
        """
        Determine which tools and agents to use for a query.
        
        Args:
            query: User query
            
        Returns:
            RoutingDecision with selected tools and agents
        """
        self.initialize()
        
        # Simple routing logic - can be enhanced with LLM-based routing
        query_lower = query.lower()
        
        # Default to web search for all queries
        tools = [ToolType.WEB_SEARCH]
        agents = registry.list_ids()
        reasoning = "Default routing to web search agents"
        confidence = 0.9
        
        # Could add more sophisticated routing here:
        # - Detect math queries -> calculator
        # - Detect code queries -> code execution
        # - Detect factual queries -> web search + RAG
        # - etc.
        
        return RoutingDecision(
            tools=tools,
            agents=agents,
            reasoning=reasoning,
            confidence=confidence,
        )
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute the routing decision.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with routing decision and agent results
        """
        decision = self.route(query)
        
        # Get selected agents
        agents = [registry.get(aid) for aid in decision.agents if registry.get(aid)]
        
        if not agents:
            return {
                "decision": decision.model_dump(),
                "evidence": [],
                "error": "No agents available",
            }
        
        # Retrieve evidence from all selected agents in parallel
        import asyncio
        tasks = [agent.retrieve(query) for agent in agents]
        evidence_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = []
        for i, result in enumerate(evidence_list):
            agent = agents[i]
            if isinstance(result, Exception):
                logger.error(f"Agent {agent.agent_id} failed: {result}")
                results.append({
                    "agent_id": agent.agent_id,
                    "error": str(result),
                })
            else:
                results.append({
                    "agent_id": agent.agent_id,
                    "evidence": result.model_dump(),
                })
        
        return {
            "decision": decision.model_dump(),
            "evidence": results,
        }


# Global router instance
router = QueryRouter()