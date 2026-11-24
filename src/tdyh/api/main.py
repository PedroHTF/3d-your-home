"""API FastAPI v1 (M0–M1). Armazena projetos em disco; PostgreSQL entra em M3."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import pipeline
from ..scenegraph import Constraint, SceneGraph

DATA = Path(os.getenv("TDYH_DATA", "./data")).resolve()
USE_CELERY = os.getenv("TDYH_USE_CELERY", "0") == "1"

app = FastAPI(title="3D Your Home API", version="0.1.0")
_jobs: dict[str, dict] = {}  # modo local sem Celery


class ProjectIn(BaseModel):
    name: str


class JobIn(BaseModel):
    recipe: Literal["auto", "classic", "neural", "hybrid"] = "auto"


def _pdir(pid: str) -> Path:
    d = DATA / "projects" / pid
    if not d.is_dir():
        raise HTTPException(404, "projeto não encontrado")
    return d


@app.get("/health")
def health() -> dict:
    return {"ok": True, "colmap": shutil.which("colmap") is not None,
            "openmvs": shutil.which("DensifyPointCloud") is not None,
            "gpu": shutil.which("nvidia-smi") is not None}


@app.get("/markers.pdf")
def markers_pdf(n: int = 6, start: int = 0, side_cm: float = 15.0):
    """PDF A4 com marcadores ArUco (um por página) para escala métrica."""
    from .. import markers
    if not (1 <= n and 0 <= start and start + n <= 50):
        raise HTTPException(422, "IDs devem ficar entre 0 e 49")
    out = DATA / "markers" / f"aruco_{start}_{n}_{side_cm:g}cm.pdf"
    if not out.exists():
        markers.make_pdf(out, range(start, start + n), side_cm * 10)
    return FileResponse(out, media_type="application/pdf", filename="marcadores.pdf")


@app.post("/projects", status_code=201)
def create_project(body: ProjectIn) -> dict:
    pid = uuid.uuid4().hex[:12]
    d = DATA / "projects" / pid
    (d / "uploads").mkdir(parents=True)
    meta = {"id": pid, "name": body.name}
    (d / "project.json").write_text(json.dumps(meta))
    return meta


@app.get("/projects/{pid}")
def get_project(pid: str) -> dict:
    return json.loads((_pdir(pid) / "project.json").read_text())


@app.post("/projects/{pid}/media", status_code=201)
async def upload_media(pid: str, files: list[UploadFile]) -> dict:
    up = _pdir(pid) / "uploads"
    saved = []
    for f in files:
        name = Path(f.filename or "file").name
        with open(up / name, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)
    return {"saved": saved}


@app.post("/projects/{pid}/jobs", status_code=202)
def start_job(pid: str, body: JobIn) -> dict:
    d = _pdir(pid)
    src, work = d / "uploads", d / "work"
    if USE_CELERY:
        from ..worker.tasks import run_pipeline
        r = run_pipeline.delay(str(src), str(work), body.recipe)
        return {"job_id": r.id, "backend": "celery"}
    jid = uuid.uuid4().hex[:12]
    job = _jobs[jid] = {"id": jid, "project": pid, "status": "queued", "stage": None, "progress": 0.0}

    def prog(stage: str, pct: float, _info: dict) -> None:
        job.update(stage=stage, progress=pct, status="running")

    def worker() -> None:
        try:
            job["result"] = pipeline.run(src, work, body.recipe, prog)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job.update(status="failed", error=str(e))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": jid, "backend": "thread"}


@app.get("/jobs/{jid}")
def get_job(jid: str) -> dict:
    if jid in _jobs:
        return _jobs[jid]
    if USE_CELERY:
        from ..worker.tasks import app as capp
        r = capp.AsyncResult(jid)
        return {"id": jid, "status": r.state.lower(), "meta": r.info if isinstance(r.info, dict) else None}
    raise HTTPException(404, "job não encontrado")


@app.get("/projects/{pid}/report")
def get_report(pid: str) -> dict:
    p = _pdir(pid) / "work" / "report.json"
    if not p.exists():
        raise HTTPException(404, "relatório ainda não gerado")
    return json.loads(p.read_text())


@app.get("/projects/{pid}/scene")
def get_scene(pid: str) -> SceneGraph:
    p = _pdir(pid) / "work" / "scene.json"
    return SceneGraph.model_validate_json(p.read_text()) if p.exists() else SceneGraph()


class ScaleIn(BaseModel):
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]
    real_m: float


@app.post("/projects/{pid}/plan")
def build_plan(pid: str, ref: ScaleIn | None = None) -> dict:
    """(Re)gera a planta a partir da nuvem densa; opcionalmente com medida de referência."""
    from .. import scale as sc
    rep = get_report(pid)
    dense = rep.get("stages", {}).get("mvs", {}).get("dense")
    if not dense:
        raise HTTPException(409, "nuvem densa ainda não existe")
    s, src = 1.0, "unknown"
    if ref:
        est = sc.from_reference(ref.p1, ref.p2, ref.real_m)
        s, src = est.factor, est.source
    return pipeline.make_plan(Path(dense), _pdir(pid) / "work", s, src)


# ------------------------------------------------------------- edição a posteriori

def _work(pid: str) -> Path:
    w = _pdir(pid) / "work"
    if not (w / "scene_measured.json").exists():
        raise HTTPException(409, "planta ainda não gerada")
    return w


def _load_measured(pid: str) -> SceneGraph:
    return SceneGraph.model_validate_json((_work(pid) / "scene_measured.json").read_text())


def _save_and_publish(pid: str, sg: SceneGraph) -> dict:
    w = _work(pid)
    (w / "scene_measured.json").write_text(sg.model_dump_json(indent=2))
    return pipeline.publish(w)


@app.get("/projects/{pid}/scene/elements")
def list_elements(pid: str) -> list[dict]:
    """Lista elementos da cena FINAL com suas medidas atuais (para escolher o que editar)."""
    import numpy as np
    sg = get_scene(pid)
    out = []
    for e in sg.entities:
        d = {"id": e.id, "type": e.type, "confidence": e.confidence, "source": e.source}
        if e.type == "wall":
            d |= {"length": round(float(np.linalg.norm(np.subtract(e.end, e.start))), 3),
                  "start": e.start, "end": e.end, "thickness": e.thickness, "height": e.height}
        elif e.type == "opening":
            d |= {"kind": e.kind.value, "host_wall": e.host_wall, "offset": e.offset,
                  "width": e.width, "height": e.height, "sill": e.sill}
        out.append(d)
    return out


@app.get("/projects/{pid}/scene/constraints")
def list_constraints(pid: str) -> list[Constraint]:
    return _load_measured(pid).constraints


@app.post("/projects/{pid}/scene/constraints", status_code=201)
def add_constraint(pid: str, c: Constraint) -> dict:
    """Ex.: {"kind":"length","refs":["<parede>"],"value":4.20,"mode":"scale"} corrige a escala de tudo;
    {"kind":"perpendicular","refs":["<a>","<b>"]} endireita um canto."""
    sg = _load_measured(pid)
    for r in c.refs:
        if sg.get(r) is None:
            raise HTTPException(422, f"elemento {r} não existe")
    sg.constraints.append(c)
    return {"constraint": c, **_save_and_publish(pid, sg)}


@app.patch("/projects/{pid}/scene/constraints/{cid}")
def edit_constraint(pid: str, cid: str, patch: dict) -> dict:
    sg = _load_measured(pid)
    c = next((x for x in sg.constraints if x.id == cid), None)
    if c is None:
        raise HTTPException(404, "restrição não encontrada")
    for k in ("value", "enabled", "mode", "note"):
        if k in patch:
            setattr(c, k, patch[k])
    return _save_and_publish(pid, sg)


@app.delete("/projects/{pid}/scene/constraints/{cid}")
def delete_constraint(pid: str, cid: str) -> dict:
    sg = _load_measured(pid)
    sg.constraints = [x for x in sg.constraints if x.id != cid]
    return _save_and_publish(pid, sg)


@app.patch("/projects/{pid}/scene/entities/{eid}")
def edit_entity(pid: str, eid: str, params: dict) -> dict:
    """Ajuste manual de UM elemento (ex.: espessura de uma parede, tipo/largura de uma abertura).
    Valores null removem o ajuste e voltam ao medido."""
    sg = _load_measured(pid)
    if sg.get(eid) is None:
        raise HTTPException(404, "elemento não encontrado")
    cur = sg.overrides.get(eid, {})
    for k, v in params.items():
        if v is None:
            cur.pop(k, None)
        else:
            cur[k] = v
    if cur:
        sg.overrides[eid] = cur
    else:
        sg.overrides.pop(eid, None)
    return _save_and_publish(pid, sg)


@app.get("/projects/{pid}/plan/{fmt}")
def get_plan(pid: str, fmt: Literal["dxf", "svg"]):
    p = _pdir(pid) / "work" / "plan" / f"planta.{fmt}"
    if not p.exists():
        raise HTTPException(404, "planta ainda não gerada")
    return FileResponse(p)


@app.get("/projects/{pid}/artifacts/{kind}")
def get_artifact(pid: str, kind: Literal["dense", "mesh", "textured"]):
    rep = get_report(pid)
    path = rep.get("stages", {}).get("mvs", {}).get(kind)
    if not path or not Path(path).exists():
        raise HTTPException(404, "artefato indisponível")
    return FileResponse(path)
