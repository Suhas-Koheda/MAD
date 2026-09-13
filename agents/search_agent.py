"""
import html
import re
Search agent for web-based evidence retrieval.
"""
import os
import json
import html
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
from loguru import logger

from agents.base_agent import BaseAgent, AgentConfig
from evidence.models import Evidence, Source, Claim
from config import settings


class SearchAgentConfig(AgentConfig):
    """Configuration for search agent."""
    search_engine: str = "google"  # google, duckduckgo, bing
    max_results: int = 5
    language: str = "en"
    safe_search: bool = True
    query_suffix: str = ""
    provider: str = "duckduckgo"  # duckduckgo, wikipedia, google


class SearchAgent(BaseAgent):
    """Agent that retrieves evidence using web search."""
    
    def __init__(self, config: Optional[SearchAgentConfig] = None, demo_mode: Optional[bool] = None):
        super().__init__(config or SearchAgentConfig())
        self.config: SearchAgentConfig = self.config
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        self.demo_mode = settings.demo_mode if demo_mode is None else demo_mode
        self._demo_data = self._load_demo_data()
    
    def _load_demo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load demo data for offline testing."""
        return {
            "default": [
                {
                    "title": "Demo Source 1",
                    "snippet": "This is demo evidence for testing purposes. The policy was introduced in 2019.",
                    "link": "https://example.com/demo1",
                },
                {
                    "title": "Demo Source 2", 
                    "snippet": "Another demo source. The policy was introduced in 2020.",
                    "link": "https://example.com/demo2",
                },
                {
                    "title": "Demo Source 3",
                    "snippet": "Third demo source with contextual information. The policy applies to enterprises only.",
                    "link": "https://example.com/demo3",
                },
            ]
        }
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        effective_query = query if self.demo_mode else f"{query} {self.config.query_suffix}".strip()
        if self.demo_mode:
            self.logger.info(f"DEMO MODE: Searching for {query!r}")
            q = query.lower()
            # The offline fixture has an agreement path and a conflict path so
            # the evidence gate can be demonstrated honestly.
            if any(k in q for k in ("policy", "introduced", "event happened", "when did")):
                year = "2019" if self.agent_id.endswith("a") else "2020"
                return [{"title": f"Demo {self.agent_id} source", "snippet": f"The policy was introduced in {year}.", "link": f"https://example.com/{self.agent_id}/{year}"}]
            elif "version" in q:
                label = "Version 1 supports feature X" if self.agent_id.endswith("a") else "Version 2 does not support feature X"
                return [{"title": f"Demo {self.agent_id} version source", "snippet": label, "link": f"https://example.com/{self.agent_id}/version"}]
            elif "coffee" in q or "healthy for everyone" in q:
                label = "Coffee is effective for adults" if self.agent_id.endswith("a") else "Coffee is not effective for children"
                return [{"title": f"Demo {self.agent_id} health source", "snippet": label, "link": f"https://example.com/{self.agent_id}/health"}]
            elif "blog" in q or "peer reviewed" in q:
                label = "A blog says the claim is secure" if self.agent_id.endswith("a") else "A peer reviewed study says the claim is insecure"
                return [{"title": f"Demo {self.agent_id} source-quality source", "snippet": label, "link": f"https://example.com/{self.agent_id}/source-quality"}]
            return [{
                "title": f"Demo {self.agent_id} source",
                "snippet": "Python is a programming language used for programming and web development.",
                "link": f"https://example.com/{self.agent_id}/python",
            }]
        if self.config.provider == "wikipedia":
            return await self._wikipedia_search(effective_query)
        if self.config.provider == "crossref":
            return await self._crossref_search(effective_query)
        if self.config.provider == "google" and settings.search_api_key and settings.search_engine_id:
            return await self._google_search(effective_query)
        if self.config.provider == "duckduckgo":
            return await self._duckduckgo_search(effective_query)
        # Fallback to simple search (could be extended)
        self.logger.warning("No search API configured, using demo data")
        return self._demo_data.get("default", [])
    
    async def _duckduckgo_search(self, query: str) -> List[Dict[str, Any]]:
        """Use DuckDuckGo HTML search without an API key."""
        response = await self.client.get("https://html.duckduckgo.com/html/", params={"q": query}, headers={"User-Agent": "evidence-mad/1.0"})
        response.raise_for_status()
        matches = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.I | re.S)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', response.text, re.I | re.S)
        return [{"title": re.sub(r"<[^>]+>", "", html.unescape(title)).strip(), "snippet": re.sub(r"<[^>]+>", "", html.unescape(snippets[i] if i < len(snippets) else title)).strip(), "link": html.unescape(link)} for i, (link, title) in enumerate(matches[:self.config.max_results])]
    async def _wikipedia_search(self, query: str) -> List[Dict[str, Any]]:
        """Search Wikipedia using its free public API; no API key required."""
        try:
            response = await self.client.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": query, "format": "json", "utf8": 1, "srlimit": self.config.max_results}, headers={"User-Agent": "EvidenceMAD/1.0 research contact@example.com"})
            response.raise_for_status()
            items = response.json().get("query", {}).get("search", [])
            return [{"title": item.get("title", ""), "snippet": re.sub(r"<[^>]+>", "", html.unescape(item.get("snippet", ""))).strip(), "link": "https://en.wikipedia.org/?curid=" + str(item.get("pageid", ""))} for item in items]
        except Exception as exc:
            self.logger.warning(f"Wikipedia search unavailable: {exc}; using DuckDuckGo fallback")
            return await self._duckduckgo_search(f"{query} site:wikipedia.org")

    async def _crossref_search(self, query: str) -> List[Dict[str, Any]]:
        """Search Crossref's free scholarly metadata API."""
        try:
            response = await self.client.get(
                "https://api.crossref.org/works",
                params={
                    "query.bibliographic": query,
                    "rows": self.config.max_results,
                    "select": "title,URL,container-title,published",
                },
                headers={"User-Agent": "EvidenceMAD/1.0 (mailto:research@example.com)"},
            )
            response.raise_for_status()
            items = response.json().get("message", {}).get("items", [])
            results = []
            for item in items:
                title = (item.get("title") or ["Untitled"])[0]
                venue = (item.get("container-title") or [""])[0]
                published = item.get("published", {}).get("date-parts", [[""]])[0]
                year = published[0] if published else ""
                results.append({
                    "title": title,
                    "snippet": f"Scholarly work indexed by Crossref{f' · {venue}' if venue else ''}{f' · {year}' if year else ''}",
                    "link": item.get("URL", ""),
                })
            return results
        except Exception as exc:
            self.logger.warning(f"Crossref search unavailable: {exc}; using DuckDuckGo fallback")
            return await self._duckduckgo_search(f"{query} academic research")


    async def _google_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform Google Custom Search."""
        params = {
            "key": settings.search_api_key,
            "cx": settings.search_engine_id,
            "q": query,
            "num": min(self.config.max_results, 10),
            "safe": "active" if self.config.safe_search else "off",
            "lr": f"lang_{self.config.language}",
        }
        
        try:
            response = await self.client.get(settings.search_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                })
            return results[:self.config.max_results]
        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return self._demo_data.get("default", [])
    
    async def retrieve(self, query: str) -> Evidence:
        """
        Retrieve evidence for a query using web search.
        
        Args:
            query: User query
            
        Returns:
            Evidence object
        """
        self.logger.info(f"Retrieving evidence for: {query}")
        
        # Search for relevant information
        search_results = await self.search(query)
        
        # Convert to sources
        sources = []
        retrieved_texts = []
        for result in search_results:
            source = Source(
                url=result.get("link", ""),
                title=result.get("title", ""),
                snippet=result.get("snippet", ""),
                metadata={"search_query": query},
            )
            sources.append(source)
            retrieved_texts.append(f"{source.title}: {source.snippet}")
        
        # Combine retrieved text
        combined_text = "\n\n".join(retrieved_texts)
        
        # Extract claims
        claims = await self.extract_claims(combined_text, sources)
        
        # Generate answer
        answer = await self.generate_answer(query, claims, sources)
        
        return Evidence(
            agent_id=self.agent_id,
            query=query,
            answer=answer,
            claims=claims,
            sources=sources,
            retrieved_text=combined_text,
            metadata={
                "search_results_count": len(search_results),
                "search_engine": self.config.search_engine,
                "provider": self.config.provider,
                "query_suffix": self.config.query_suffix,
            },
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class SearchAgentA(SearchAgent):
    """First search agent with default configuration."""
    
    def __init__(self, demo_mode: Optional[bool] = None):
        config = SearchAgentConfig(
            agent_id="search_agent_a",
            name="SearchAgentA",
            description="Primary web search agent",
            max_results=5,
            query_suffix="general web sources",
            provider="duckduckgo",
        )
        super().__init__(config, demo_mode=demo_mode)


class SearchAgentB(SearchAgent):
    """Second search agent - can use different search parameters."""
    
    def __init__(self, demo_mode: Optional[bool] = None):
        config = SearchAgentConfig(
            agent_id="search_agent_b",
            name="SearchAgentB",
            description="Secondary web search agent with different parameters",
            max_results=5,
            query_suffix="official government and primary sources",
            provider="wikipedia",
        )
        super().__init__(config, demo_mode=demo_mode)


class SearchAgentC(SearchAgent):
    """Third search agent for additional coverage."""
    
    def __init__(self, demo_mode: Optional[bool] = None):
        config = SearchAgentConfig(
            agent_id="search_agent_c",
            name="SearchAgentC",
            description="Tertiary web search agent",
            max_results=5,
            query_suffix="academic historical timeline sources",
            provider="crossref",
        )
        super().__init__(config, demo_mode=demo_mode)
