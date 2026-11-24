"""Reconstrução esparsa com COLMAP (CLI). Usa GLOMAP para o mapper se instalado."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SparseResult:
    model_dir: Path
    registered_images: int
    total_images: int
    points3d: int
    mean_reproj_error: float | None

    @property
    def registration_ratio(self) -> float:
        return self.registered_images / self.total_images if self.total_images else 0.0


def _run(cmd: list[str], log: Path) -> str:
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        res = subprocess.run(cmd, capture_output=True, text=True)
        f.write(res.stdout + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Falhou: {cmd[0]} {cmd[1]} (ver {log})")
    return res.stdout + res.stderr


def available() -> bool:
    return shutil.which("colmap") is not None


def run_sparse(
    images: Path,
    work: Path,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    gpu: bool = True,
    matcher: str = "sequential",  # "sequential" p/ vídeo, "exhaustive" p/ fotos soltas
) -> SparseResult:
    if not available():
        raise RuntimeError("colmap não encontrado no PATH")
    work.mkdir(parents=True, exist_ok=True)
    db, sparse, log = work / "database.db", work / "sparse", work / "colmap.log"
    sparse.mkdir(exist_ok=True)
    g = "1" if gpu else "0"
    _run(["colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(images),
          "--ImageReader.camera_model", camera_model,
          "--ImageReader.single_camera", "1" if single_camera else "0",
          "--SiftExtraction.use_gpu", g], log)
    _run(["colmap", f"{matcher}_matcher", "--database_path", str(db),
          "--SiftMatching.use_gpu", g], log)
    if shutil.which("glomap"):
        _run(["glomap", "mapper", "--database_path", str(db), "--image_path", str(images),
              "--output_path", str(sparse)], log)
    else:
        _run(["colmap", "mapper", "--database_path", str(db), "--image_path", str(images),
              "--output_path", str(sparse)], log)
    model = _largest_model(sparse)
    stats = _run(["colmap", "model_analyzer", "--path", str(model)], log)
    total = len([p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    return parse_analyzer(stats, model, total)


def _largest_model(sparse: Path) -> Path:
    models = [d for d in sparse.iterdir() if d.is_dir()]
    if not models:
        raise RuntimeError("COLMAP não gerou nenhum modelo (pouca sobreposição ou textura)")
    return max(models, key=lambda d: (d / "images.bin").stat().st_size if (d / "images.bin").exists() else 0)


def parse_analyzer(text: str, model: Path, total: int) -> SparseResult:
    def num(pattern: str, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return SparseResult(
        model_dir=model,
        registered_images=num(r"Registered images:\s*(\d+)", int) or 0,
        total_images=total,
        points3d=num(r"Points:\s*(\d+)", int) or 0,
        mean_reproj_error=num(r"Mean reprojection error:\s*([\d.]+)"),
    )


def undistort(images: Path, model: Path, out: Path) -> Path:
    """Gera imagens sem distorção + modelo no formato que o OpenMVS consome."""
    _run(["colmap", "image_undistorter", "--image_path", str(images), "--input_path", str(model),
          "--output_path", str(out), "--output_type", "COLMAP"], out.parent / "colmap.log")
    return out


def export_txt(model: Path, out: Path) -> Path:
    """Converte o modelo binário para TXT (usado para poses/câmeras)."""
    out.mkdir(parents=True, exist_ok=True)
    _run(["colmap", "model_converter", "--input_path", str(model), "--output_path", str(out),
          "--output_type", "TXT"], out.parent / "colmap.log")
    return out
