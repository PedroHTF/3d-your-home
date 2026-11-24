"""Escala métrica: marcador ArUco triangulado, medida de referência do usuário ou ARCore.

Uma reconstrução SfM só tem escala relativa. O fator s converte unidades do modelo em metros.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ScaleEstimate:
    factor: float          # multiplicar coordenadas do modelo por isso -> metros
    source: str            # "marker" | "user_measure" | "arcore"
    uncertainty: float     # desvio relativo estimado (0.01 = 1%)
    samples: int = 1


def from_reference(p1: np.ndarray, p2: np.ndarray, real_m: float) -> ScaleEstimate:
    """Usuário marca dois pontos no modelo e informa a distância real (ex.: largura de porta)."""
    model = float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))
    if model <= 0:
        raise ValueError("pontos coincidentes")
    # incerteza típica de clicar pontos: ~1 cm em cada extremidade
    return ScaleEstimate(real_m / model, "user_measure", min(0.5, 0.014 / real_m))


# ---------- ArUco ----------

ARUCO_DICT = cv2.aruco.DICT_4X4_50


def detect_markers(image: np.ndarray, dictionary: int = ARUCO_DICT) -> dict[int, np.ndarray]:
    """Retorna {id: cantos 4x2 em pixels}."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary),
                                  cv2.aruco.DetectorParameters())
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.flatten(), corners)}


def triangulate(projections: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray:
    """DLT linear multi-vista. projections: P 3x4; pixels: (u, v)."""
    rows = []
    for P, (u, v) in zip(projections, pixels):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    X = np.linalg.svd(np.asarray(rows))[2][-1]
    return X[:3] / X[3]


def from_markers(observations: dict[int, list[tuple[np.ndarray, np.ndarray]]],
                 marker_size_m: float) -> ScaleEstimate:
    """observations[id] = [(P 3x4, cantos 4x2), ...] em >= 2 vistas.

    Triangula os 4 cantos e compara o comprimento médio das arestas com o tamanho real.
    """
    ratios = []
    for obs in observations.values():
        if len(obs) < 2:
            continue
        Ps = [o[0] for o in obs]
        X = np.array([triangulate(Ps, [o[1][k] for o in obs]) for k in range(4)])
        edges = [np.linalg.norm(X[k] - X[(k + 1) % 4]) for k in range(4)]
        ratios.extend(marker_size_m / e for e in edges if e > 0)
    if not ratios:
        raise RuntimeError("nenhum marcador visto em 2+ imagens")
    r = np.array(ratios)
    med = float(np.median(r))
    r = r[np.abs(r - med) < 0.05 * med]  # descarta outliers
    return ScaleEstimate(float(r.mean()), "marker", float(r.std() / r.mean()) if len(r) > 1 else 0.02,
                         samples=len(r))


# ---------- leitura de poses do COLMAP (formato texto) ----------

def _qvec2rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_colmap_text(model_dir: Path) -> dict[str, dict]:
    """Lê cameras.txt + images.txt -> {nome_imagem: {"P": 3x4, "R": 3x3, "t": 3}}.

    Para modelos binários: `colmap model_converter --output_type TXT`.
    """
    cams = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cid, model, w, h, *params = line.split()
        p = list(map(float, params))
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if model.startswith("SIMPLE") else p[:4]
        cams[int(cid)] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    out = {}
    lines = [ln for ln in (model_dir / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    for line in lines[::2]:
        parts = line.split()
        q, t = np.array(parts[1:5], float), np.array(parts[5:8], float)
        R = _qvec2rot(q)
        K = cams[int(parts[8])]
        out[parts[9]] = {"P": K @ np.hstack([R, t[:, None]]), "R": R, "t": t}
    return out


def camera_centers(poses: dict[str, dict]) -> np.ndarray:
    return np.array([-(p["R"].T @ p["t"]) for p in poses.values()])


def camera_up_hint(poses: dict[str, dict]) -> np.ndarray:
    """Direção 'para cima' do mundo = média de -Y das câmeras (celular segurado em pé ou deitado
    aponta Y para baixo na imagem em qualquer caso de foto 'normal')."""
    ups = np.array([-(p["R"].T @ np.array([0.0, 1.0, 0.0])) for p in poses.values()])
    u = ups.mean(axis=0)
    return u / np.linalg.norm(u)


def scale_from_images(images_dir: Path, model_dir: Path, marker_size_m: float) -> ScaleEstimate:
    poses = read_colmap_text(model_dir)
    obs: dict[int, list] = {}
    for name, pose in poses.items():
        img = cv2.imread(str(images_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for mid, corners in detect_markers(img).items():
            obs.setdefault(mid, []).append((pose["P"], corners))
    return from_markers(obs, marker_size_m)
