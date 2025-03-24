from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_BASE_URL: str
    API_KEY: str
    STUDENT_BASE_URL: str

    class Config:
        env_file = ".env"

settings = Settings()
