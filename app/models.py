from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class Dataset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    filename: str
    path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrainingRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(index=True)
    target_col: str
    status: str = Field(default="queued")  # queued|running|succeeded|failed
    best_model: Optional[str] = None
    metrics_json: Optional[str] = None
    artifact_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

class Deployment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    is_active: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PredictionLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    deployment_id: int = Field(index=True)
    latency_ms: int
    input_json: str
    output_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)