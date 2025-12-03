import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application configuration settings."""
    
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-2A7b2ww9Vbw-644Gi4B6QA")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://litellm.ai.paas.htec.rs")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "l2-gpt-4o-mini")
    )
    confidence_threshold: float = 0.85
    embed_model: str = "all-MiniLM-L6-v2"
    data_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "health_dataset.json"
    )
    use_mock_llm: bool = field(
        default_factory=lambda: os.getenv("USE_MOCK_LLM", "false").lower() == "true"
    )


def get_settings() -> Settings:
    """Factory function to create settings instance."""
    return Settings()
