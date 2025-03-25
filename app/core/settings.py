from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_BASE_URL: str
    API_KEY: str
    STUDENT_BASE_URL: str
    LOCAL_URL: str
    LLM_MODEL: str

    class Config:
        env_file = ".env"

settings = Settings()
