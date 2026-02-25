from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

class DatasetOut(BaseModel):
    id: int
    name: str
    filename: str

class ProfileOut(BaseModel):
    columns: List[str]
    dtypes: Dict[str, str]
    missing: Dict[str, int]
    sample_rows: List[Dict[str, Any]]
    suggested_target_cols: List[str]

class TrainRequest(BaseModel):
    dataset_id: int
    target_col: str
    positive_label: Optional[str] = None  # optional for string targets

class TrainOut(BaseModel):
    run_id: int
    status: str

class RunOut(BaseModel):
    id: int
    dataset_id: int
    target_col: str
    status: str
    best_model: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None
    error: Optional[str] = None

class DeployOut(BaseModel):
    deployment_id: int
    run_id: int
    is_active: bool

class PredictRequest(BaseModel):
    row: Dict[str, Any] = Field(..., description="One row of feature inputs, as a dict")

class PredictOut(BaseModel):
    deployment_id: int
    label: int
    probability: float