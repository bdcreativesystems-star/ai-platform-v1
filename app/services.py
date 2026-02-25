from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

from pathlib import Path
from sqlmodel import Session, select

from .models import Dataset, TrainingRun, Deployment, PredictionLog
from .storage import run_artifact_dir, wipe_run_artifacts
from .ml import train_best_model, load_pipeline

_PIPELINE_CACHE: dict[int, Any] = {}  # deployment_id -> pipeline

def create_dataset(session: Session, name: str, filename: str, path: str) -> Dataset:
    ds = Dataset(name=name, filename=filename, path=path)
    session.add(ds)
    session.commit()
    session.refresh(ds)
    return ds

def create_run(session: Session, dataset_id: int, target_col: str) -> TrainingRun:
    run = TrainingRun(dataset_id=dataset_id, target_col=target_col, status="queued")
    session.add(run)
    session.commit()
    session.refresh(run)
    return run

def run_training_job(
    session: Session,
    run_id: int,
    positive_label: Optional[str] = None,
) -> None:
    run = session.get(TrainingRun, run_id)
    if not run:
        return

    ds = session.get(Dataset, run.dataset_id)
    if not ds:
        run.status = "failed"
        run.error = "Dataset not found"
        run.finished_at = datetime.utcnow()
        session.add(run)
        session.commit()
        return

    run.status = "running"
    session.add(run)
    session.commit()

    try:
        wipe_run_artifacts(run_id)
        art_dir = run_artifact_dir(run_id)
        schema = train_best_model(ds.path, run.target_col, art_dir, positive_label=positive_label)

        run.status = "succeeded"
        run.best_model = schema["best_model"]
        run.metrics_json = json.dumps(schema["best_metrics"])
        run.artifact_path = str(art_dir)
        run.finished_at = datetime.utcnow()
        session.add(run)
        session.commit()

    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        run.finished_at = datetime.utcnow()
        session.add(run)
        session.commit()

def deploy_run(session: Session, run_id: int) -> Deployment:
    run = session.get(TrainingRun, run_id)
    if not run or run.status != "succeeded" or not run.artifact_path:
        raise ValueError("Run not deployable (must be succeeded with artifacts).")

    # deactivate any existing active deployment
    active = session.exec(select(Deployment).where(Deployment.is_active == True)).all()
    for d in active:
        d.is_active = False
        session.add(d)

    dep = Deployment(run_id=run_id, is_active=True)
    session.add(dep)
    session.commit()
    session.refresh(dep)

    _PIPELINE_CACHE.pop(dep.id, None)
    return dep

def get_active_deployment(session: Session) -> Deployment:
    dep = session.exec(select(Deployment).where(Deployment.is_active == True)).first()
    if not dep:
        raise ValueError("No active deployment. Deploy a run first.")
    return dep

def predict_one(session: Session, row: Dict[str, Any]) -> Dict[str, Any]:
    dep = get_active_deployment(session)
    run = session.get(TrainingRun, dep.run_id)
    if not run or not run.artifact_path:
        raise ValueError("Active deployment is missing run artifacts.")

    start = time.perf_counter()

    if dep.id in _PIPELINE_CACHE:
        pipe = _PIPELINE_CACHE[dep.id]
    else:
        pipe = load_pipeline(Path(run.artifact_path))
        _PIPELINE_CACHE[dep.id] = pipe

    import pandas as pd
    X = pd.DataFrame([row])
    proba = float(pipe.predict_proba(X)[:, 1][0])
    label = int(proba >= 0.5)

    latency_ms = int((time.perf_counter() - start) * 1000)

    out = {"deployment_id": dep.id, "label": label, "probability": proba}

    log = PredictionLog(
        deployment_id=dep.id,
        latency_ms=latency_ms,
        input_json=json.dumps(row),
        output_json=json.dumps(out),
    )
    session.add(log)
    session.commit()

    return out