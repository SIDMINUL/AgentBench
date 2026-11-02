from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2:3b"
    request_timeout_seconds: int = 60
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
