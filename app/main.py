from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks, HTTPException
from sqlmodel import Session, select
import json

from .config import settings
from .db import init_db, get_session
from .storage import save_upload, ensure_dirs
from .models import Dataset, TrainingRun, Deployment, PredictionLog
from .schemas import DatasetOut, ProfileOut, TrainRequest, TrainOut, RunOut, DeployOut, PredictRequest, PredictOut
from .ml import profile_csv
from .services import create_dataset, create_run, run_training_job, deploy_run, predict_one

app = FastAPI(title=settings.PROJECT_NAME)

@app.on_event("startup")
def _startup():
    ensure_dirs()
    init_db()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/datasets/upload", response_model=DatasetOut)
async def upload_dataset(
    name: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    data = await file.read()
    path = save_upload(data, file.filename)
    ds = create_dataset(session, name=name, filename=file.filename, path=str(path))
    return DatasetOut(id=ds.id, name=ds.name, filename=ds.filename)

@app.get("/datasets/{dataset_id}/profile", response_model=ProfileOut)
def dataset_profile(dataset_id: int, session: Session = Depends(get_session)):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return profile_csv(ds.path)

@app.post("/train", response_model=TrainOut)
def train(
    req: TrainRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
):
    ds = session.get(Dataset, req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    run = create_run(session, dataset_id=req.dataset_id, target_col=req.target_col)

    # training runs async (simple background task for V1)
    background.add_task(_train_in_background, run.id, req.positive_label)

    return TrainOut(run_id=run.id, status="queued")

def _train_in_background(run_id: int, positive_label: str | None):
    # BackgroundTasks runs outside request context; create a fresh session
    from sqlmodel import Session as SQLSession
    from .db import engine
    with SQLSession(engine) as s:
        run_training_job(s, run_id, positive_label=positive_label)

@app.get("/runs", response_model=list[RunOut])
def list_runs(session: Session = Depends(get_session)):
    runs = session.exec(select(TrainingRun).order_by(TrainingRun.id.desc())).all()
    out = []
    for r in runs:
        metrics = json.loads(r.metrics_json) if r.metrics_json else None
        out.append(RunOut(
            id=r.id,
            dataset_id=r.dataset_id,
            target_col=r.target_col,
            status=r.status,
            best_model=r.best_model,
            metrics=metrics,
            artifact_path=r.artifact_path,
            error=r.error,
        ))
    return out

@app.post("/deploy/{run_id}", response_model=DeployOut)
def deploy(run_id: int, session: Session = Depends(get_session)):
    try:
        dep = deploy_run(session, run_id)
        return DeployOut(deployment_id=dep.id, run_id=dep.run_id, is_active=dep.is_active)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/deployments")
def deployments(session: Session = Depends(get_session)):
    deps = session.exec(select(Deployment).order_by(Deployment.id.desc())).all()
    return deps

@app.post("/predict", response_model=PredictOut)
def predict(req: PredictRequest, session: Session = Depends(get_session)):
    try:
        out = predict_one(session, req.row)
        return out
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/monitoring/recent")
def monitoring_recent(limit: int = 25, session: Session = Depends(get_session)):
    logs = session.exec(select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit)).all()
    return [
        {
            "id": l.id,
            "deployment_id": l.deployment_id,
            "latency_ms": l.latency_ms,
            "input": json.loads(l.input_json),
            "output": json.loads(l.output_json),
            "created_at": l.created_at,
        }
        for l in logs
    ]