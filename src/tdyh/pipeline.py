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
        txt = colmap.export_txt(sparse.model_dir, project_dir / "colmap" / "sparse_txt")
        report["stages"]["layout"] = make_plan(out["dense"], project_dir, scale, scale_source, colmap_txt=txt)
    progress("done", 1.0, {})
    _save(project_dir, report)
    return report


def _save(project_dir: Path, report: dict) -> None:
    (project_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))


def make_plan(dense_ply: Path, project_dir: Path, scale: float = 1.0, scale_source: str = "unknown",
              up_hint=None, colmap_txt: Path | None = None) -> dict:
    """Nuvem densa -> grafo de cena medido -> (restrições) -> scene.json + DXF + SVG.

    Sem escala conhecida, estima uma escala APROXIMADA (pé-direito e portas típicos) para que os
    algoritmos funcionem; o usuário corrige depois com uma medida (mode="scale").
    colmap_txt fornece as câmeras (necessárias para portas/janelas) e a vertical."""
    from . import scale as sc
    pts = read_ply_xyz(Path(dense_ply))
    cams = None
    if colmap_txt:
        poses = sc.read_colmap_text(Path(colmap_txt))
        cams = sc.camera_centers(poses)
        up_hint = up_hint if up_hint is not None else sc.camera_up_hint(poses)
    unc = None
    if scale_source == "unknown":
        scale = sc.coarse_from_extent(pts, up_hint)
        res = layout_mod.build_layout(pts, scale=scale, up_hint=up_hint, cameras=cams)
        doors = [e.height for e in res.scene.entities
                 if e.type == "opening" and e.kind.value == "door" and e.confidence >= 0.8]
        est = sc.from_priors(res.ceiling_z - res.floor_z, doors)
        scale *= est.factor
        scale_source, unc = "prior", est.uncertainty
    res = layout_mod.build_layout(pts, scale=scale, up_hint=up_hint, cameras=cams)
    res.scene.scale_source, res.scene.scale_factor, res.scene.scale_uncertainty = scale_source, scale, unc
    if scale_source == "prior":
        res.notes.append(f"Escala ESTIMADA por tamanhos típicos (±{unc:.0%}). Informe uma medida real "
                         "(ex.: comprimento de uma parede) para corrigir.")
    (project_dir / "scene_measured.json").write_text(res.scene.model_dump_json(indent=2))
    out = publish(project_dir)
    out["notes"] = res.notes + out.get("warnings", [])
    out["height_m"] = round(res.ceiling_z - res.floor_z, 3)
    return out


def publish(project_dir: Path) -> dict:
    """Aplica restrições/ajustes sobre a cena medida e regenera scene.json, DXF e SVG."""
    from dataclasses import asdict as _asdict

    from .constraints import rebuild
    from .scenegraph import SceneGraph
    measured = SceneGraph.model_validate_json((project_dir / "scene_measured.json").read_text())
    final, rep = rebuild(measured)
    (project_dir / "scene.json").write_text(final.model_dump_json(indent=2))
    plan_dir = project_dir / "plan"
    dxf = plan2d.to_dxf(final, plan_dir / "planta.dxf")
    svg = plan2d.to_svg(final, plan_dir / "planta.svg")
    return {"walls": sum(e.type == "wall" for e in final.entities),
            "openings": [{"id": e.id, "kind": e.kind.value, "width": round(e.width, 3),
                          "height": round(e.height, 3), "sill": round(e.sill, 3),
                          "confidence": e.confidence} for e in final.entities if e.type == "opening"],
            "scale": {"source": final.scale_source, "factor": final.scale_factor,
                      "uncertainty": final.scale_uncertainty},
            "constraints": [_asdict(s) for s in rep.statuses], "ok": rep.ok,
            "warnings": rep.warnings, "dxf": str(dxf), "svg": str(svg)}
