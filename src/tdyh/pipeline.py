"""Orquestração M0–M1: ingestão -> frames -> planejamento -> SfM -> MVS."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import frames, ingest, planner
from . import layout as layout_mod
from .exporters import plan2d
from .pointcloud import read_ply_xyz
from .recon import colmap, openmvs

Progress = Callable[[str, float, dict], None]


def _noop(stage: str, pct: float, info: dict) -> None:
    pass


def run(src: Path, project_dir: Path, recipe: str = "auto", progress: Progress = _noop,
        scale: float = 1.0, scale_source: str = "unknown") -> dict:
    report: dict = {"stages": {}}
    progress("ingest", 0.0, {})
    media = ingest.ingest_folder(src, project_dir)
    report["stages"]["ingest"] = {"images": sum(m.kind == "image" for m in media),
                                  "videos": sum(m.kind == "video" for m in media)}

    progress("frames", 0.15, {})
    images = project_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    for m in media:
        if m.kind == "video":
            n_frames += len(frames.process_video(Path(m.path), images))
    report["stages"]["frames"] = {"selected_from_video": n_frames}

    progress("plan", 0.3, {})
    q = planner.assess(images)
    plan = planner.choose(q, has_gpu=shutil.which("nvidia-smi") is not None)
    if recipe != "auto":
        plan.recipe = recipe
    report["quality"], report["plan"] = asdict(q), asdict(plan)

    if not colmap.available():
        report["stopped"] = "COLMAP não instalado — etapas 3+ ignoradas."
        _save(project_dir, report)
        return report

    progress("sfm", 0.4, {})
    sparse = colmap.run_sparse(images, project_dir / "colmap")
    report["stages"]["sfm"] = {**asdict(sparse), "model_dir": str(sparse.model_dir),
                               "registration_ratio": sparse.registration_ratio}
    if sparse.registration_ratio < 0.6:
        report["warnings"] = [f"Só {sparse.registration_ratio:.0%} das imagens registradas: "
                              "grave novamente com mais sobreposição (~70%) e movimento lento."]

    if "mvs" in plan.steps and openmvs.available():
        progress("mvs", 0.6, {})
        und = colmap.undistort(images, sparse.model_dir, project_dir / "colmap" / "dense")
        out = openmvs.run_dense(und, project_dir / "mvs")
        report["stages"]["mvs"] = {k: str(v) for k, v in out.items()}
        progress("layout", 0.85, {})
        report["stages"]["layout"] = make_plan(out["dense"], project_dir, scale, scale_source)
    progress("done", 1.0, {})
    _save(project_dir, report)
    return report


def _save(project_dir: Path, report: dict) -> None:
    (project_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))


def make_plan(dense_ply: Path, project_dir: Path, scale: float = 1.0, scale_source: str = "unknown",
              up_hint=None) -> dict:
    """Nuvem densa -> grafo de cena (paredes, aberturas, lajes) -> scene.json + DXF + SVG."""
    pts = read_ply_xyz(Path(dense_ply))
    res = layout_mod.build_layout(pts, scale=scale, up_hint=up_hint)
    res.scene.scale_source = scale_source
    if scale_source == "unknown":
        res.notes.append("Sem referência de escala: medidas relativas. Informe uma medida ou use marcador ArUco.")
    (project_dir / "scene.json").write_text(res.scene.model_dump_json(indent=2))
    plan_dir = project_dir / "plan"
    dxf = plan2d.to_dxf(res.scene, plan_dir / "planta.dxf")
    svg = plan2d.to_svg(res.scene, plan_dir / "planta.svg")
    return {"walls": sum(e.type == "wall" for e in res.scene.entities),
            "openings": sum(e.type == "opening" for e in res.scene.entities),
            "height_m": round(res.ceiling_z - res.floor_z, 3),
            "dxf": str(dxf), "svg": str(svg), "notes": res.notes}
