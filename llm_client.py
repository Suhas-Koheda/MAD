"""
LLM client wrapper for interacting with OpenAI-compatible APIs.
"""
import json
from typing import Optional, Dict, Any
from loguru import logger
import httpx

from config import settings


class LLMClient:
    """
    Client for OpenAI-compatible LLM APIs.
    Supports chat completions with structured output.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.api_key = api_key or settings.llm_api_key
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.client = httpx.Client(timeout=60.0)
        self.call_count = 0
    
    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat completion request.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        if not self.api_key:
            logger.warning("No LLM API key configured")
            return "No LLM available - using demo mode"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            data = response.json()
            self.call_count += 1
            
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def chat_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a chat completion request expecting JSON output.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response
        """
        response = self.chat(prompt, system_prompt)
        
        # Try to extract JSON from response
        try:
            # Look for JSON object in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If JSON parsing fails, return the raw response
        return {"raw_response": response}
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()


class DemoLLMClient(LLMClient):
    """Demo LLM client that doesn't require API keys."""
    
    def __init__(self):
        super().__init__()
        self.api_key = "demo"
    
    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Demo chat that returns a generic response."""
        self.call_count += 1
        
        # Simple demo responses based on prompt content
        if "judge" in prompt.lower() or "evaluate" in prompt.lower():
            return json.dumps({
                "winner": "candidate_1",
                "reason": "Evidence from candidate 1 has stronger support",
                "evidence_score": 0.85,
                "source_score": 0.82,
                "reasoning_score": 0.80,
                "confidence": 0.83,
            })
        
        if "classify" in prompt.lower() or "conflict" in prompt.lower():
            return json.dumps({
                "type": "TEMPORAL",
                "confidence": 0.85,
                "explanation": "Claims refer to different time periods",
            })
        
        return f"Demo response for prompt starting with: {prompt[:50]}..."


def get_llm_client(demo_mode: bool = False) -> LLMClient:
    """Get or create LLM client instance."""
    if demo_mode:
        return DemoLLMClient()
    return LLMClient()