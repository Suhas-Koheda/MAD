"""
Base agent class for the Multi-Agent Debate framework.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
import asyncio
from loguru import logger

from evidence.models import Evidence, Source, Claim


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    agent_id: str = Field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    name: str = "BaseAgent"
    description: str = "Base agent for evidence retrieval"
    max_results: int = 5
    temperature: float = 0.3
    timeout: float = 30.0


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.agent_id = self.config.agent_id
        self.logger = logger.bind(agent_id=self.agent_id)
    
    @abstractmethod
    async def retrieve(self, query: str) -> Evidence:
        """
        Retrieve evidence for a query.
        
        Args:
            query: The user query to retrieve evidence for
            
        Returns:
            Evidence object containing claims, sources, and answer
        """
        raise NotImplementedError("Subclasses must implement retrieve")
    
    @abstractmethod
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for relevant information.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        raise NotImplementedError("Subclasses must implement search")
    
    async def extract_claims(self, text: str, sources: List[Source]) -> List[Claim]:
        """
        Extract claims from retrieved text.
        This is a simple implementation - can be overridden by subclasses.
        """
        # Simple claim extraction - split by sentences and create claims
        import re
        sentences = re.split(r'[.!?]+', text)
        claims = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short fragments
                source = sources[i % len(sources)] if sources else None
                claim = Claim(
                    text=sentence,
                    source=source or Source(url="", title="", snippet=sentence),
                    confidence=0.8
                )
                claims.append(claim)
        return claims[:10]  # Limit number of claims
    
    async def generate_answer(self, query: str, claims: List[Claim], sources: List[Source]) -> str:
        """
        Generate an answer from claims and sources.
        This is a simple implementation - can be overridden by subclasses.
        """
        if not claims:
            return "No relevant evidence found."
        
        # Simple answer generation - combine top claims
        answer_parts = [f"Based on {len(sources)} sources:"]
        for claim in claims[:5]:
            answer_parts.append(f"- {claim.text}")
        return "\n".join(answer_parts)
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "description": self.config.description,
        }


class AgentRegistry:
    """Registry for managing agents."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self._agents[agent.agent_id] = agent
    def clear(self) -> None:
        """Clear registered agents when switching execution modes."""
        self._agents.clear()
    
    def get(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_all(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def list_ids(self) -> List[str]:
        """List all agent IDs."""
        return list(self._agents.keys())
    
    async def retrieve_all(self, query: str) -> List[Evidence]:
        """Retrieve evidence from all agents in parallel."""
        tasks = [agent.retrieve(query) for agent in self._agents.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        evidence_list = []
        for i, result in enumerate(results):
            agent_id = list(self._agents.keys())[i]
            if isinstance(result, Exception):
                self.logger.error(f"Agent {agent_id} failed: {result}")
                # Create empty evidence for failed agent
                evidence_list.append(Evidence(
                    agent_id=agent_id,
                    query=query,
                    answer=f"Error: {result}",
                    claims=[],
                    sources=[],
                ))
            else:
                evidence_list.append(result)
        
        return evidence_list


# Global registry instance
registry = AgentRegistry()