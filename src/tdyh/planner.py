"""Planejador de receita (regras determinísticas; o agente LLM entra depois sobre estas métricas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .frames import sharpness


@dataclass
class InputQuality:
    n_images: int
    median_sharpness: float
    texture_score: float  # densidade de cantos ORB por megapixel (proxy de textura)
    notes: list[str] = field(default_factory=list)


@dataclass
class Plan:
    recipe: str  # "classic" | "neural" | "hybrid"
    steps: list[str]
    reasons: list[str]


def assess(images: Path, sample: int = 30) -> InputQuality:
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    step = max(1, len(paths) // sample)
    orb = cv2.ORB_create(nfeatures=5000)
    sharp, tex = [], []
    for p in paths[::step]:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        sharp.append(sharpness(g))
        kps = orb.detect(g, None)
        tex.append(len(kps) / (g.shape[0] * g.shape[1] / 1e6))
    q = InputQuality(len(paths), float(np.median(sharp) if sharp else 0), float(np.median(tex) if tex else 0))
    if q.n_images < 20:
        q.notes.append("Poucas imagens (<20): cobertura provavelmente insuficiente.")
    if q.texture_score < 300:
        q.notes.append("Pouca textura (paredes lisas/brancas): MVS clássico tende a falhar.")
    return q


def choose(q: InputQuality, has_gpu: bool = True) -> Plan:
    reasons = list(q.notes)
    if not has_gpu:
        return Plan("classic", ["sfm", "mvs"], reasons + ["Sem GPU CUDA: apenas fotogrametria clássica."])
    if q.texture_score >= 800 and q.n_images >= 40:
        return Plan("classic", ["sfm", "mvs"], reasons + ["Textura e cobertura boas: MVS dá a melhor métrica."])
    if q.texture_score < 300:
        return Plan("neural", ["sfm", "gsplat"], reasons + ["Superfícies lisas: splatting com priors de profundidade."])
    return Plan("hybrid", ["sfm", "mvs", "gsplat"], reasons + ["Padrão: MVS para estrutura, splatting para objetos."])
