"""Detecção de portas e janelas numa parede, distinguindo VAZIO de NÃO OBSERVADO.

Para cada parede monta-se uma grade 2D (posição ao longo da parede t x altura z, células de 5 cm)
com quatro estados:

  OCC     pontos sobre a face da parede (|d| < face_tol)
  RECESS  pontos um pouco atrás da face (folha de porta fechada, 2–9 cm)
  FREE    algum raio câmera -> ponto observado ATRÁS da parede cruza essa célula sem obstáculo
          (ou seja, a câmera viu através dela)
  UNKNOWN nada disso (não filmado, ou escondido atrás de um móvel)

Abertura = componente de células FREE, estendida para cima/baixo através de células não ocupadas.
Regiões UNKNOWN nunca viram abertura; são relatadas como "não observadas".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .scenegraph import Opening, OpeningKind, Wall

OCC, RECESS, FREE, UNKNOWN = 1, 2, 3, 0


@dataclass
class WallGrid:
    state: np.ndarray       # (nt, nz) uint8
    cell: float
    length: float
    height: float


class Occluders:
    """Grade de voxels ocupados da cena inteira, para testar se um raio é bloqueado."""

    def __init__(self, pts: np.ndarray, voxel: float = 0.08):
        self.voxel = voxel
        keys = np.unique(np.floor(pts / voxel).astype(np.int64), axis=0)
        self.codes = np.sort(self._encode(keys))

    @staticmethod
    def _encode(k: np.ndarray) -> np.ndarray:
        k = k + (1 << 20)
        return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]

    def blocked(self, a: np.ndarray, b: np.ndarray, skip_end: float = 0.15, step: float = 0.04) -> np.ndarray:
        """Para cada par (a_i, b_i), True se algum voxel ocupado está no segmento (exceto o final)."""
        v = b - a
        length = np.linalg.norm(v, axis=1)
        n = max(2, int(np.ceil(length.max() / step)))
        s = np.linspace(0.05, 1.0, n)[None, :, None]
        samples = a[:, None, :] + s * v[:, None, :]
        valid = (s[0, :, 0][None, :] * length[:, None]) < (length[:, None] - skip_end)
        codes = self._encode(np.floor(samples.reshape(-1, 3) / self.voxel).astype(np.int64))
        idx = np.clip(np.searchsorted(self.codes, codes), 0, len(self.codes) - 1)
        hit = (self.codes[idx] == codes).reshape(samples.shape[:2]) & valid
        return hit.any(axis=1)


def wall_frame(w: Wall, center_xy: np.ndarray):
    a, b = np.array(w.start), np.array(w.end)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    if ((a + b) / 2 - center_xy) @ n < 0:
        n = -n  # n aponta para FORA do cômodo
    return a, d, n, L


def build_grid(w: Wall, pts: np.ndarray, height: float, center_xy: np.ndarray,
               cameras: np.ndarray | None, vis_pairs: np.ndarray | None,
               occluders: Occluders | None, cell: float = 0.05, face_tol: float = 0.02,
               recess: tuple[float, float] = (0.025, 0.09), beyond: float = 0.12,
               max_rays: int = 60000, rng: np.random.Generator | None = None) -> WallGrid:
    rng = rng or np.random.default_rng(0)
    a, d, n, L = wall_frame(w, center_xy)
    nt, nz = max(1, int(round(L / cell))), max(1, int(round(height / cell)))
    rel = pts[:, :2] - a
    t, dist, z = rel @ d, rel @ n, pts[:, 2]
    inside = (t >= 0) & (t < L) & (z >= 0) & (z < height)

    def cells(tt, zz):
        return np.clip((tt / cell).astype(int), 0, nt - 1), np.clip((zz / cell).astype(int), 0, nz - 1)

    state = np.full((nt, nz), UNKNOWN, np.uint8)
    # FREE primeiro; RECESS e OCC sobrescrevem (pontos na parede são evidência mais forte)
    if cameras is not None and len(cameras):
        far = np.where(dist > beyond)[0]
        if vis_pairs is not None and len(vis_pairs):
            pairs = vis_pairs[np.isin(vis_pairs[:, 0], far)]
        else:  # sem visibilidade: todas as câmeras do lado de dentro (aproximação)
            cin = np.where(((cameras[:, :2] - a) @ n) < 0)[0]
            pairs = np.array(np.meshgrid(far, cin)).reshape(2, -1).T if len(far) and len(cin) else np.empty((0, 2), int)
        if len(pairs) > max_rays:
            pairs = pairs[rng.choice(len(pairs), max_rays, replace=False)]
        if len(pairs):
            P, C = pts[pairs[:, 0]], cameras[pairs[:, 1]]
            dc = (C[:, :2] - a) @ n
            dp = (P[:, :2] - a) @ n
            ok = dc < -0.05
            s = np.where(ok, dc / (dc - dp), 0.0)
            X = C + s[:, None] * (P - C)
            tx, zx = (X[:, :2] - a) @ d, X[:, 2]
            ok &= (tx >= 0) & (tx < L) & (zx >= 0) & (zx < height)
            if occluders is not None and ok.any():
                idx = np.where(ok)[0]
                ok[idx[occluders.blocked(C[idx], X[idx])]] = False
            i, k = cells(tx[ok], zx[ok])
            state[i, k] = FREE
    m = inside & (dist > recess[0]) & (dist < recess[1])
    i, k = cells(t[m], z[m])
    state[i, k] = RECESS
    m = inside & (np.abs(dist) < face_tol)
    i, k = cells(t[m], z[m])
    state[i, k] = OCC
    return WallGrid(state, cell, L, height)


def _components(mask: np.ndarray, close: int = 3):
    m = mask.astype(np.uint8)
    if close > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # stats com eixos do cv2: x = coluna (z), y = linha (t)
    for c in range(1, n):
        z0, t0, wz, wt, area = stats[c]
        yield (t0, t0 + wt - 1, z0, z0 + wz - 1, area, lab == c)


def openings_from_grid(w: Wall, g: WallGrid, min_width: float = 0.45, min_height: float = 0.4,
                       door_floor: float = 0.15, min_evidence_m2: float = 0.1) -> tuple[list[Opening], list[str]]:
    st, c = g.state, g.cell
    nt, nz = st.shape
    out: list[Opening] = []
    occupied = (st == OCC) | (st == RECESS)
    # furo isolado numa parede sólida (célula sem ponto por acaso) não é vão:
    # célula FREE só vale se a vizinhança 5x5 for majoritariamente não-parede
    occ_frac = cv2.blur(occupied.astype(np.float32), (5, 5), borderType=cv2.BORDER_REPLICATE)
    free = (st == FREE) & (occ_frac < 0.3)
    min_cells = int(min_evidence_m2 / (c * c))

    # 1) aberturas vistas através (FREE)
    for i0, i1, k0, k1, area, comp in _components(free):
        raw = int((free & comp).sum())
        if raw < min_cells or raw < 0.3 * (i1 - i0 + 1) * (k1 - k0 + 1):
            continue
        # recupera as bordas cortadas pelo filtro: cresce na horizontal até achar parede (máx. 15 cm)
        rows = slice(k0, k1 + 1)
        for _ in range(3):
            if i0 > 0 and occupied[i0 - 1, rows].mean() < 0.3:
                i0 -= 1
            if i1 < nt - 1 and occupied[i1 + 1, rows].mean() < 0.3:
                i1 += 1
        cols = slice(i0, i1 + 1)
        # estende verticalmente enquanto a linha seguinte não for parede
        while k0 > 0 and occupied[cols, k0 - 1].mean() < 0.3:
            k0 -= 1
        while k1 < nz - 1 and occupied[cols, k1 + 1].mean() < 0.3:
            k1 += 1
        width, top, bottom = (i1 - i0 + 1) * c, (k1 + 1) * c, k0 * c
        if width < min_width or top - bottom < min_height:
            continue
        if occupied[i0:i1 + 1, k0:k1 + 1].mean() > 0.25:
            continue
        is_door = bottom <= door_floor
        out.append(Opening(kind=OpeningKind.door if is_door else OpeningKind.window, host_wall=w.id,
                           offset=round(i0 * c, 3), width=round(width, 3),
                           height=round(top - (0.0 if is_door else bottom), 3),
                           sill=0.0 if is_door else round(bottom, 3), confidence=0.8))

    # 2) folha de porta fechada (RECESSO retangular encostado no piso) — heurística fraca
    for i0, i1, k0, k1, area, comp in _components(st == RECESS):
        width, top, bottom = (i1 - i0 + 1) * c, (k1 + 1) * c, k0 * c
        fill = area / ((i1 - i0 + 1) * (k1 - k0 + 1))
        if 0.55 <= width <= 1.3 and 1.8 <= top <= 2.5 and bottom <= door_floor and fill > 0.6:
            out.append(Opening(kind=OpeningKind.door, host_wall=w.id, offset=round(i0 * c, 3),
                               width=round(width, 3), height=round(top, 3), confidence=0.45,
                               source="inferred"))

    notes = []
    unk = (st == UNKNOWN).mean()
    if unk > 0.15:
        notes.append(f"Parede {w.id}: {unk:.0%} da área não observada (não filmada ou escondida por móvel).")
    return out, notes
