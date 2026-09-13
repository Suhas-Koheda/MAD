"""
Agents package for the Multi-Agent Debate framework.
"""
from .base_agent import BaseAgent, AgentConfig, AgentRegistry, registry
from .search_agent import SearchAgent, SearchAgentConfig, SearchAgentA, SearchAgentB, SearchAgentC
from .router import QueryRouter, ToolType, RoutingDecision, router

__all__ = [
    "BaseAgent",
    "AgentConfig", 
    "AgentRegistry",
    "registry",
    "SearchAgent",
    "SearchAgentConfig",
    "SearchAgentA",
    "SearchAgentB", 
    "SearchAgentC",
    "QueryRouter",
    "ToolType",
    "RoutingDecision",
    "router",
]