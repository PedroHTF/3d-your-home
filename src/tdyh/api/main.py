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


@app.get("/charuco.pdf")
def charuco_pdf():
    """PDF A4 do tabuleiro ChArUco (calibração de lentes)."""
    from .. import charuco
    out = DATA / "markers" / "charuco_7x10_25mm.pdf"
    if not out.exists():
        charuco.make_pdf(out)
    return FileResponse(out, media_type="application/pdf", filename="tabuleiro_charuco.pdf")


@app.get("/lenses")
def list_lenses() -> list[dict]:
    from ..lens import ProfileStore
    return [p.model_dump() for p in ProfileStore(DATA / "lenses.json").load()]


@app.delete("/lenses")
def delete_lens(key: str) -> dict:
    from ..lens import ProfileStore
    if not ProfileStore(DATA / "lenses.json").delete(key):
        raise HTTPException(404, "perfil não encontrado")
    return {"deleted": key}


@app.post("/lenses/calibrate", status_code=201)
async def calibrate_lens(files: list[UploadFile], square_mm: float | None = None, fisheye: bool = False) -> dict:
    """Fotos (ou um vídeo) do tabuleiro ChArUco tiradas com UMA lente -> perfil salvo."""
    import tempfile

    from ..cli import _calibrate
    tmp = Path(tempfile.mkdtemp())
    for f in files:
        (tmp / Path(f.filename or "x.jpg").name).write_bytes(await f.read())
    vids = [p for p in tmp.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".m4v", ".mkv")]
    try:
        return _calibrate(vids[0] if vids else tmp, square_mm, fisheye, DATA / "lenses.json")
    except (ValueError, SystemExit) as e:
        raise HTTPException(422, str(e))


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
    from ..constraints import LENGTH_PARAMS, mark_drawn
    if LENGTH_PARAMS & params.keys() or f"override:{eid}" in sg.drawn_at_scale:
        mark_drawn(sg, eid, override=True)  # valores em metros na escala ATUAL
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


class NewElement(BaseModel):
    element: dict          # {"type":"wall","start":[x,y],"end":[x,y],...} | {"type":"opening",...} | {"type":"box",...}
    snap: bool = True      # encaixa extremidades de paredes em cantos/paredes existentes (10 cm)


@app.post("/projects/{pid}/scene/entities", status_code=201)
def create_entity(pid: str, body: NewElement) -> dict:
    """Cria parede, abertura (porta/janela) ou objeto. Coordenadas em metros, no sistema da planta."""
    from pydantic import TypeAdapter, ValidationError

    from ..constraints import snap_point
    from ..scenegraph import AnyEntity, Opening, Wall
    from typing import Annotated
    from pydantic import Field as F
    try:
        e = TypeAdapter(Annotated[AnyEntity, F(discriminator="type")]).validate_python(
            {**body.element, "source": "user"})
    except ValidationError as err:
        raise HTTPException(422, err.errors(include_url=False))
    final = get_scene(pid)
    sg = _load_measured(pid)
    info = {}
    if isinstance(e, Wall):
        if body.snap:
            e.start, info["start"] = snap_point(final, e.start)
            e.end, info["end"] = snap_point(final, e.end)
        import numpy as np
        if np.linalg.norm(np.subtract(e.end, e.start)) < 0.05:
            raise HTTPException(422, "parede com menos de 5 cm")
    if isinstance(e, Opening) and not isinstance(final.get(e.host_wall), Wall) and \
            not any(a.id == e.host_wall for a in sg.added):
        raise HTTPException(422, f"parede {e.host_wall} não existe")
    from ..constraints import mark_drawn
    sg.added.append(e)
    mark_drawn(sg, e.id)
    return {"created": e.id, "snapped": info, **_save_and_publish(pid, sg)}


@app.delete("/projects/{pid}/scene/entities/{eid}")
def delete_entity(pid: str, eid: str) -> dict:
    """Remove um elemento. Aberturas da parede removida saem junto. Reversível com /restore
    (elementos detectados); elementos criados pelo usuário são apagados de vez."""
    sg = _load_measured(pid)
    if any(a.id == eid for a in sg.added):
        sg.added = [a for a in sg.added if a.id != eid and getattr(a, "host_wall", None) != eid]
    elif any(e.id == eid for e in sg.entities):
        if eid not in sg.deleted:
            sg.deleted.append(eid)
    else:
        raise HTTPException(404, "elemento não encontrado")
    return _save_and_publish(pid, sg)


@app.post("/projects/{pid}/scene/entities/{eid}/restore")
def restore_entity(pid: str, eid: str) -> dict:
    sg = _load_measured(pid)
    if eid not in sg.deleted:
        raise HTTPException(404, "elemento não está removido")
    sg.deleted.remove(eid)
    return _save_and_publish(pid, sg)


class SplitIn(BaseModel):
    at: float  # metros a partir do início da parede


@app.post("/projects/{pid}/scene/entities/{eid}/split")
def split_entity(pid: str, eid: str, body: SplitIn) -> dict:
    """Divide uma parede em duas (ex.: para ligar uma parede interna nesse ponto)."""
    from ..constraints import split_wall
    sg, final = _load_measured(pid), get_scene(pid)
    if final.get(eid) is None or final.get(eid).type != "wall":
        raise HTTPException(404, "parede não encontrada")
    try:
        ids = split_wall(sg, final, eid, body.at)
    except ValueError as err:
        raise HTTPException(422, str(err))
    return {"created": list(ids), **_save_and_publish(pid, sg)}


# ------------------------------------------------------------- cômodos (opcional)

class RoomsToggle(BaseModel):
    enabled: bool


class LabelIn(BaseModel):
    point: tuple[float, float]
    name: str
    use: str | None = None


@app.put("/projects/{pid}/scene/rooms")
def set_rooms(pid: str, body: RoomsToggle) -> dict:
    """Liga/desliga a divisão em cômodos (desligada: a planta tem uma área total só)."""
    sg = _load_measured(pid)
    sg.rooms_enabled = body.enabled
    return _save_and_publish(pid, sg)


@app.get("/projects/{pid}/scene/rooms")
def list_rooms(pid: str) -> list[dict]:
    sg = get_scene(pid)
    return [{"id": e.id, "name": e.name, "area": e.area, "net_area": e.net_area,
             "perimeter": e.perimeter, "label_id": e.label_id, "polygon": e.polygon}
            for e in sg.entities if e.type == "space"]


@app.post("/projects/{pid}/scene/rooms/labels", status_code=201)
def add_label(pid: str, body: LabelIn) -> dict:
    """Dá nome ao cômodo que contém `point`. O nome continua valendo após editar paredes."""
    from ..scenegraph import RoomLabel
    sg = _load_measured(pid)
    lb = RoomLabel(point=body.point, name=body.name, use=body.use)
    sg.room_labels.append(lb)
    return {"label": lb, **_save_and_publish(pid, sg)}


@app.patch("/projects/{pid}/scene/rooms/labels/{lid}")
def edit_label(pid: str, lid: str, patch: dict) -> dict:
    sg = _load_measured(pid)
    lb = next((x for x in sg.room_labels if x.id == lid), None)
    if lb is None:
        raise HTTPException(404, "nome não encontrado")
    for k in ("name", "use", "point"):
        if k in patch:
            setattr(lb, k, tuple(patch[k]) if k == "point" else patch[k])
    return _save_and_publish(pid, sg)


@app.delete("/projects/{pid}/scene/rooms/labels/{lid}")
def delete_label(pid: str, lid: str) -> dict:
    sg = _load_measured(pid)
    sg.room_labels = [x for x in sg.room_labels if x.id != lid]
    return _save_and_publish(pid, sg)


@app.get("/projects/{pid}/plan/{fmt}")
def get_plan(pid: str, fmt: Literal["dxf", "svg"]):
    p = _pdir(pid) / "work" / "plan" / f"planta.{fmt}"
    if not p.exists():
        raise HTTPException(404, "planta ainda não gerada")
    return FileResponse(p)


def _ifc_response(sg: SceneGraph, out: Path, **names):
    from ..exporters.ifc import to_ifc
    try:
        path, warn = to_ifc(sg, out, **names)
    except RuntimeError as e:
        raise HTTPException(501, str(e))
    return FileResponse(path, media_type="application/x-step", filename=out.name,
                        headers={"X-TDYH-Warnings": json.dumps(warn, ensure_ascii=True)})


@app.get("/projects/{pid}/ifc")
def get_ifc(pid: str):
    """IFC4 da cena final do levantamento (gerado na hora, sempre a partir da cena atual)."""
    sg = get_scene(pid)
    if not sg.entities:
        raise HTTPException(409, "levantamento ainda não tem planta")
    name = json.loads((_pdir(pid) / "project.json").read_text()).get("name", "3D Your Home") \
        if (_pdir(pid) / "project.json").exists() else "3D Your Home"
    return _ifc_response(sg, _pdir(pid) / "work" / "plan" / "modelo.ifc", project_name=name, building_name=name)


@app.get("/projects/{pid}/artifacts/{kind}")
def get_artifact(pid: str, kind: Literal["dense", "mesh", "textured"]):
    rep = get_report(pid)
    path = rep.get("stages", {}).get("mvs", {}).get(kind)
    if not path or not Path(path).exists():
        raise HTTPException(404, "artefato indisponível")
    return FileResponse(path)


# ============================================================= edificação (vários levantamentos)

def _bfile(bid: str) -> Path:
    f = DATA / "buildings" / f"{bid}.json"
    if not f.exists():
        raise HTTPException(404, "edificação não encontrada")
    return f


def _bload(bid: str):
    from ..building import Building
    return Building.model_validate_json(_bfile(bid).read_text())


def _bsave(b) -> None:
    (DATA / "buildings").mkdir(parents=True, exist_ok=True)
    (DATA / "buildings" / f"{b.id}.json").write_text(b.model_dump_json(indent=2))


def _bscenes(b) -> dict:
    out = {}
    for c in b.captures:
        out[c.id] = get_scene(c.project_id)
        if not out[c.id].entities:
            raise HTTPException(409, f"levantamento '{c.name}' ainda não tem planta")
    return out


class BuildingIn(BaseModel):
    name: str


class CaptureIn(BaseModel):
    project_id: str
    name: str | None = None


@app.post("/buildings", status_code=201)
def create_building(body: BuildingIn) -> dict:
    from ..building import Building
    b = Building(name=body.name)
    _bsave(b)
    return b.model_dump()


@app.get("/buildings/{bid}")
def get_building(bid: str) -> dict:
    return _bload(bid).model_dump()


@app.post("/buildings/{bid}/captures", status_code=201)
def add_capture(bid: str, body: CaptureIn) -> dict:
    """Adiciona um levantamento (projeto já com planta) à edificação."""
    from ..building import Capture
    b = _bload(bid)
    meta = get_project(body.project_id)
    c = Capture(name=body.name or meta["name"], project_id=body.project_id)
    b.captures.append(c)
    _bsave(b)
    return c.model_dump()


@app.delete("/buildings/{bid}/captures/{cid}")
def remove_capture(bid: str, cid: str) -> dict:
    b = _bload(bid)
    b.captures = [c for c in b.captures if c.id != cid]
    b.links = [lk for lk in b.links if cid not in (lk.a.capture, lk.b.capture)]
    _bsave(b)
    return b.model_dump()


@app.put("/buildings/{bid}/captures/{cid}/pose")
def set_pose(bid: str, cid: str, pose: dict) -> dict:
    """Posição manual (x, y em m; theta_deg) e/ou trava (locked)."""
    b = _bload(bid)
    c = next((c for c in b.captures if c.id == cid), None)
    if c is None:
        raise HTTPException(404, "levantamento não encontrado")
    for k in ("x", "y", "theta_deg", "locked"):
        if k in pose:
            setattr(c.pose, k, pose[k])
    _bsave(b)
    return c.model_dump()


@app.get("/buildings/{bid}/suggestions")
def building_suggestions(bid: str, limit: int = 10) -> list[dict]:
    """Pares de portas que provavelmente são a mesma, com pontuação e motivos."""
    from dataclasses import asdict

    from ..building import suggest_door_matches
    b = _bload(bid)
    return [asdict(s) for s in suggest_door_matches(b, _bscenes(b))[:limit]]


@app.post("/buildings/{bid}/links", status_code=201)
def add_link(bid: str, body: dict) -> dict:
    """{"kind":"door_match","a":{"capture":..,"element":..},"b":{...}} | wall_contact | collinear |
    point_match (com "point":[x,y]); "thickness" opcional."""
    from pydantic import ValidationError

    from ..building import Link
    b = _bload(bid)
    try:
        lk = Link.model_validate(body)
    except ValidationError as err:
        raise HTTPException(422, err.errors(include_url=False))
    ids = {c.id for c in b.captures}
    if lk.a.capture not in ids or lk.b.capture not in ids:
        raise HTTPException(422, "levantamento inexistente na ligação")
    b.links.append(lk)
    _bsave(b)
    return lk.model_dump()


@app.delete("/buildings/{bid}/links/{lid}")
def delete_link(bid: str, lid: str) -> dict:
    b = _bload(bid)
    b.ignored_auto += [lk.auto_key for lk in b.links if lk.id == lid and lk.auto_key]
    b.links = [lk for lk in b.links if lk.id != lid]
    _bsave(b)
    return b.model_dump()


@app.post("/buildings/{bid}/solve")
def solve_building(bid: str) -> dict:
    """Ajuste global das posições + planta unificada (DXF/SVG)."""
    from dataclasses import asdict

    from ..building import apply_poses, auto_marker_links, merge, solve_poses
    from ..exporters import plan2d
    b = _bload(bid)
    if not b.captures:
        raise HTTPException(409, "edificação sem levantamentos")
    scenes = _bscenes(b)
    marker_info = auto_marker_links(b, scenes)
    res = solve_poses(b, scenes)
    apply_poses(b, res)
    _bsave(b)
    merged, info = merge(b, scenes, res)
    out = DATA / "buildings" / b.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.json").write_text(merged.model_dump_json(indent=2))
    plan2d.to_dxf(merged, out / "planta.dxf")
    plan2d.to_svg(merged, out / "planta.svg")
    scales = {c.name: get_scene(c.project_id).scale_source for c in b.captures}
    warn = res.warnings + [f"Levantamento '{n}' com escala {s}: corrija a escala antes de juntar."
                           for n, s in scales.items() if s in ("prior", "unknown")]
    return {"poses": {c.id: c.pose.model_dump() for c in b.captures},
            "links": [asdict(s) for s in res.statuses], "ok": all(s.ok for s in res.statuses),
            "shared_walls": info, "markers": marker_info, "warnings": warn,
            "rooms": [{"name": e.name, "area": e.area, "net_area": e.net_area}
                      for e in merged.entities if e.type == "space"]}


@app.get("/buildings/{bid}/ifc")
def building_ifc(bid: str):
    """IFC4 da planta unificada (paredes compartilhadas uma vez só, portas duplicadas uma vez só)."""
    b = _bload(bid)
    f = DATA / "buildings" / bid / "scene.json"
    if not f.exists():
        raise HTTPException(404, "planta unificada ainda não gerada (POST /solve)")
    sg = SceneGraph.model_validate_json(f.read_text())
    return _ifc_response(sg, DATA / "buildings" / bid / "modelo.ifc", project_name=b.name, building_name=b.name)


@app.get("/buildings/{bid}/plan/{fmt}")
def building_plan(bid: str, fmt: Literal["dxf", "svg"]):
    f = DATA / "buildings" / bid / f"planta.{fmt}"
    if not f.exists():
        raise HTTPException(404, "planta unificada ainda não gerada (POST /solve)")
    return FileResponse(f)
