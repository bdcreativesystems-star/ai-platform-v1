from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    PROJECT_NAME: str = "AI Platform V1 (Tabular Classification)"
    DB_URL: str = "sqlite:///./backend.db"
    UPLOAD_DIR: Path = Path("data/uploads")
    ARTIFACT_DIR: Path = Path("artifacts")

settings = Settings()