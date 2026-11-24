"""Reconstrução densa + malha com OpenMVS (AGPL, executado como processo externo)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

TOOLS = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"]


def available() -> bool:
    return all(shutil.which(t) for t in TOOLS[:3])


def _run(cmd: list[str], cwd: Path) -> None:
    log = cwd / "openmvs.log"
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        f.write(res.stdout + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Falhou: {cmd[0]} (ver {log})")


def run_dense(undistorted: Path, work: Path, refine: bool = True, texture: bool = True) -> dict[str, Path]:
    """undistorted = saída de colmap.undistort. Retorna caminhos de nuvem densa e malhas."""
    if not available():
        raise RuntimeError("OpenMVS não encontrado no PATH")
    work.mkdir(parents=True, exist_ok=True)
    scene = work / "scene.mvs"
    _run(["InterfaceCOLMAP", "-i", str(undistorted), "-o", str(scene),
          "--image-folder", str(undistorted / "images")], work)
    _run(["DensifyPointCloud", str(scene)], work)
    dense = work / "scene_dense.mvs"
    _run(["ReconstructMesh", str(dense)], work)
    out = {"dense": work / "scene_dense.ply", "mesh": work / "scene_dense_mesh.ply"}
    mesh_mvs = work / "scene_dense_mesh.mvs"
    if refine and shutil.which("RefineMesh"):
        _run(["RefineMesh", str(mesh_mvs)], work)
        mesh_mvs = work / "scene_dense_mesh_refine.mvs"
        out["mesh"] = work / "scene_dense_mesh_refine.ply"
    if texture and shutil.which("TextureMesh"):
        _run(["TextureMesh", str(mesh_mvs), "--export-type", "obj"], work)
        out["textured"] = mesh_mvs.with_name(mesh_mvs.stem + "_texture.obj")
    return out
