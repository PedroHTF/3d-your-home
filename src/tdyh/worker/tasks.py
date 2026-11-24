"""Workers Celery. Filas: cpu (ingestão/frames) e gpu (reconstrução)."""

from __future__ import annotations

import os
from pathlib import Path

from celery import Celery

from .. import pipeline

BROKER = os.getenv("TDYH_REDIS_URL", "redis://localhost:6379/0")
app = Celery("tdyh", broker=BROKER, backend=BROKER)
app.conf.task_routes = {"tdyh.worker.tasks.run_pipeline": {"queue": "gpu"}}
app.conf.task_track_started = True


@app.task(bind=True)
def run_pipeline(self, src: str, project_dir: str, recipe: str = "auto") -> dict:
    def progress(stage: str, pct: float, info: dict) -> None:
        self.update_state(state="PROGRESS", meta={"stage": stage, "progress": pct, **info})

    return pipeline.run(Path(src), Path(project_dir), recipe, progress)
