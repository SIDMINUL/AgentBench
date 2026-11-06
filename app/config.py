from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    default_provider: str = "groq"
    default_model: str = "openai/gpt-oss-120b"
    default_judge_model: str = "openai/gpt-oss-120b"
    ollama_base_url: str = "http://localhost:11434"
    request_timeout_seconds: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
