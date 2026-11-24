"""Etapa 2 — Extração de frames: FFmpeg + seleção por nitidez + deduplicação."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameScore:
    path: Path
    sharpness: float
    phash: int
    selected: bool = False


def sharpness(gray: np.ndarray) -> float:
    """Variância do Laplaciano — maior = mais nítido."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def phash(gray: np.ndarray) -> int:
    """Perceptual hash de 64 bits (DCT 32x32 -> 8x8)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    bits = (dct > np.median(dct)).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_raw(video: Path, out_dir: Path, fps: float = 4.0, max_side: int = 2048) -> list[Path]:
    """Extrai frames com FFmpeg (respeita a rotação do celular automaticamente)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg não encontrado no PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-vf", f"fps={fps},{scale}", "-q:v", "2", str(out_dir / "f_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("f_*.jpg"))


def select_frames(
    frames: list[Path],
    window: int = 4,
    min_sharpness_ratio: float = 0.35,
    dedup_hamming: int = 6,
) -> list[FrameScore]:
    """Mantém o frame mais nítido de cada janela, descarta borrados e quase-duplicados.

    - window: nº de frames brutos por janela (com fps=4, window=4 ≈ 1 frame/s).
    - min_sharpness_ratio: descarta frames abaixo de ratio * mediana de nitidez.
    - dedup_hamming: distância máxima de pHash para considerar duplicado.
    """
    scores: list[FrameScore] = []
    for p in frames:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        g = cv2.resize(gray, (640, int(640 * gray.shape[0] / gray.shape[1])))
        scores.append(FrameScore(p, sharpness(g), phash(g)))
    if not scores:
        return []
    median = float(np.median([s.sharpness for s in scores]))
    threshold = median * min_sharpness_ratio
    kept: list[FrameScore] = []
    for i in range(0, len(scores), window):
        best = max(scores[i : i + window], key=lambda s: s.sharpness)
        if best.sharpness < threshold:
            continue
        if kept and hamming(best.phash, kept[-1].phash) <= dedup_hamming:
            if best.sharpness > kept[-1].sharpness:
                kept[-1] = best
            continue
        kept.append(best)
    for s in kept:
        s.selected = True
    return scores


def process_video(video: Path, out_dir: Path, fps: float = 4.0, **kw) -> list[Path]:
    """Extrai, seleciona e copia os frames escolhidos para out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = extract_raw(video, Path(tmp), fps=fps)
        scored = select_frames(raw, **kw)
        chosen = []
        for s in scored:
            if s.selected:
                dst = out_dir / f"{video.stem}_{s.path.stem}.jpg"
                shutil.copy2(s.path, dst)
                chosen.append(dst)
    return chosen
