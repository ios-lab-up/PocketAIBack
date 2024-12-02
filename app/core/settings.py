from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_BASE_URL: str
    TIMEOUT: int
    openai_api_key: str


    class Config:
        env_file = ".env"

settings = Settings()
