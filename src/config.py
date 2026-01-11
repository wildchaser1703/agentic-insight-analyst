"""
Configuration Module - Centralized configuration management.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048")))
    
    def validate(self) -> bool:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env or as an environment variable.")
        return True


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    max_iterations: int = field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10")))
    min_improvement_threshold: float = field(default_factory=lambda: float(os.getenv("MIN_IMPROVEMENT_THRESHOLD", "0.1")))
    verbose_logging: bool = field(default_factory=lambda: os.getenv("VERBOSE_LOGGING", "true").lower() == "true")
    tool_timeout_seconds: int = 30
    max_tool_retries: int = 2
    max_consecutive_no_progress: int = 2
    min_confidence_to_stop: float = 0.8


@dataclass
class APIConfig:
    """Configuration for the FastAPI server."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    cors_allow_all: bool = field(default_factory=lambda: os.getenv("CORS_ALLOW_ALL", "true").lower() == "true")


@dataclass
class Config:
    """Master configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls()


config = Config.from_env()


def get_config() -> Config:
    return config
