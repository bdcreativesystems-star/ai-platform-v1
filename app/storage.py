import shutil
from pathlib import Path
from .config import settings

def ensure_dirs() -> None:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def save_upload(file_bytes: bytes, filename: str) -> Path:
    ensure_dirs()
    safe_name = filename.replace("..", "").replace("/", "_").replace("\\", "_")
    path = settings.UPLOAD_DIR / safe_name
    path.write_bytes(file_bytes)
    return path

def run_artifact_dir(run_id: int) -> Path:
    ensure_dirs()
    d = settings.ARTIFACT_DIR / f"run_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def wipe_run_artifacts(run_id: int) -> None:
    d = settings.ARTIFACT_DIR / f"run_{run_id}"
    if d.exists():
        shutil.rmtree(d)