import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application configuration settings."""
    
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-...")
    )
    llm_model: str = "gpt-4"
    confidence_threshold: float = 0.85
    embed_model: str = "all-MiniLM-L6-v2"
    data_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "health_dataset.json"
    )
    use_mock_llm: bool = field(
        default_factory=lambda: os.getenv("USE_MOCK_LLM", "true").lower() == "true"
    )


def get_settings() -> Settings:
    """Factory function to create settings instance."""
    return Settings()
