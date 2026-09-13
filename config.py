import os
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    
    # Search API Configuration
    search_api_key: str = Field(default="", alias="SEARCH_API_KEY")
    search_engine_id: str = Field(default="", alias="SEARCH_ENGINE_ID")
    search_base_url: str = Field(default="https://www.googleapis.com/customsearch/v1", alias="SEARCH_BASE_URL")
    
    # NLI Model Configuration
    nli_model: str = Field(default="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", alias="NLI_MODEL")
    nli_device: str = Field(default="cpu", alias="NLI_DEVICE")
    nli_mode: str = Field(default="transformers", alias="NLI_MODE")
    
    # Debate Configuration
    disagreement_threshold: float = Field(default=0.70, alias="DISAGREEMENT_THRESHOLD")
    max_debate_rounds: int = Field(default=3, alias="MAX_DEBATE_ROUNDS")
    max_search_results: int = Field(default=5, alias="MAX_SEARCH_RESULTS")
    
    # Demo Mode
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()