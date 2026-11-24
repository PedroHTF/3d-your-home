"""Etapa 9 — Layout: piso/teto, paredes como segmentos 2D e grafo de cena.

Entrada: nuvem densa (N x 3) já em metros (ou com fator de escala).
Saída: SceneGraph com Wall + Slab (piso e teto), e transform para o sistema da planta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pointcloud import fit_plane_ransac, rotation_to_z, voxel_downsample
from . import openings as op
from .scenegraph import SceneGraph, Slab, Wall


@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray
    support: int

    @property
    def direction(self) -> np.ndarray:
        d = self.p1 - self.p0
        return d / np.linalg.norm(d)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


@dataclass
class LayoutResult:
    scene: SceneGraph
    floor_z: float                  # sempre 0 no sistema da planta
    floor_offset: float             # p_planta = R @ (s * p) - [0, 0, floor_offset]
    ceiling_z: float
    rotation: np.ndarray
    segments: list[Segment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def align_to_gravity(pts: np.ndarray, up_hint: np.ndarray | None, thresh: float = 0.02):
    """Encontra o piso (plano dominante perpendicular a up) e roda a nuvem para Z para cima."""
    sample = voxel_downsample(pts, 0.03)
    if up_hint is not None:
        up_hint = up_hint / np.linalg.norm(up_hint)
    n, d, _ = fit_plane_ransac(sample, thresh, iters=800, normal_hint=up_hint)
    if up_hint is not None and n @ up_hint < 0:
        n, d = -n, -d
    R = rotation_to_z(n)
    rot = pts @ R.T
    if up_hint is None:  # sem dica: piso = plano com mais pontos acima dele
        z_plane = -d
        if (rot[:, 2] > z_plane + 0.5).sum() < (rot[:, 2] < z_plane - 0.5).sum():
            R = np.diag([1.0, -1.0, -1.0]) @ R
            rot = pts @ R.T
    return R, rot


def floor_ceiling(z: np.ndarray, bin_m: float = 0.02) -> tuple[float, float]:
    hist, edges = np.histogram(z, bins=max(10, int((z.max() - z.min()) / bin_m)))
    centers = (edges[:-1] + edges[1:]) / 2
    lo = centers < np.percentile(z, 30)
    hi = centers > np.percentile(z, 70)
    floor = float(centers[lo][np.argmax(hist[lo])])
    ceil = float(centers[hi][np.argmax(hist[hi])])
    return floor, ceil


def _fit_line(pts2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts2.mean(axis=0)
    d = np.linalg.svd(pts2 - c, full_matrices=False)[2][0]
    return c, d


def extract_segments(xy: np.ndarray, thresh: float = 0.025, min_pts: int = 150,
                     min_len: float = 0.4, gap: float = 0.35, max_lines: int = 40,
                     rng: np.random.Generator | None = None) -> list[Segment]:
    """RANSAC sequencial de retas 2D + divisão por lacunas (portas, vãos)."""
    rng = rng or np.random.default_rng(0)
    remaining = xy.copy()
    segs: list[Segment] = []
    for _ in range(max_lines):
        if len(remaining) < min_pts:
            break
        best = None
        for _ in range(300):
            a, b = remaining[rng.choice(len(remaining), 2, replace=False)]
            d = b - a
            nrm = np.linalg.norm(d)
            if nrm < 0.2:
                continue
            nvec = np.array([-d[1], d[0]]) / nrm
            mask = np.abs((remaining - a) @ nvec) < thresh
            if best is None or mask.sum() > best.sum():
                best = mask
        if best is None or best.sum() < min_pts:
            break
        c, d = _fit_line(remaining[best])
        nvec = np.array([-d[1], d[0]])
        mask = np.abs((remaining - c) @ nvec) < thresh
        inl = remaining[mask]
        t = np.sort((inl - c) @ d)
        breaks = np.where(np.diff(t) > gap)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(t) - 1]
        for s, e in zip(starts, ends):
            if e - s + 1 >= min_pts // 3 and t[e] - t[s] >= min_len:
                segs.append(Segment(c + t[s] * d, c + t[e] * d, int(e - s + 1)))
        remaining = remaining[~mask]
    return segs


def snap_manhattan(segs: list[Segment], tol_deg: float = 6.0) -> list[Segment]:
    """Endireita segmentos quase alinhados à direção dominante (ou perpendicular). Não força os demais."""
    if not segs:
        return segs
    ang = np.array([np.arctan2(s.direction[1], s.direction[0]) % (np.pi / 2) for s in segs])
    w = np.array([s.length for s in segs])
    # média circular com período 90°
    dom = (np.angle(np.sum(w * np.exp(1j * 4 * ang))) / 4) % (np.pi / 2)
    out = []
    for s in segs:
        a = np.arctan2(s.direction[1], s.direction[0])
        k = np.round((a - dom) / (np.pi / 2))
        target = dom + k * np.pi / 2
        if abs(a - target) < np.radians(tol_deg):
            mid = (s.p0 + s.p1) / 2
            dvec = np.array([np.cos(target), np.sin(target)])
            half = s.length / 2
            s = Segment(mid - half * dvec, mid + half * dvec, s.support)
        out.append(s)
    return out


def merge_collinear(segs: list[Segment], ang_tol_deg: float = 4.0, dist_tol: float = 0.05,
                    gap: float = 0.35) -> list[Segment]:
    segs = sorted(segs, key=lambda s: -s.length)
    merged: list[Segment] = []
    for s in segs:
        for i, m in enumerate(merged):
            if abs(abs(s.direction @ m.direction) - 1) > 1 - np.cos(np.radians(ang_tol_deg)):
                continue
            nvec = np.array([-m.direction[1], m.direction[0]])
            if max(abs((s.p0 - m.p0) @ nvec), abs((s.p1 - m.p0) @ nvec)) > dist_tol:
                continue
            ts = sorted([(m.p0 - m.p0) @ m.direction, (m.p1 - m.p0) @ m.direction])
            us = sorted([(s.p0 - m.p0) @ m.direction, (s.p1 - m.p0) @ m.direction])
            if us[0] > ts[1] + gap or us[1] < ts[0] - gap:
                continue
            lo, hi = min(ts[0], us[0]), max(ts[1], us[1])
            merged[i] = Segment(m.p0 + lo * m.direction, m.p0 + hi * m.direction, m.support + s.support)
            break
        else:
            merged.append(s)
    return merged


def join_corners(segs: list[Segment], reach: float = 0.4, min_angle_deg: float = 30) -> list[Segment]:
    """Estende/apara extremidades até a interseção com a parede vizinha (fecha cantos)."""
    segs = [Segment(s.p0.copy(), s.p1.copy(), s.support) for s in segs]
    for i, a in enumerate(segs):
        for j, b in enumerate(segs):
            if j <= i or abs(a.direction @ b.direction) > np.cos(np.radians(min_angle_deg)):
                continue
            A = np.array([a.direction, -b.direction]).T
            try:
                ta, tb = np.linalg.solve(A, b.p0 - a.p0)
            except np.linalg.LinAlgError:
                continue
            x = a.p0 + ta * a.direction
            for seg in (a, b):
                d0, d1 = np.linalg.norm(seg.p0 - x), np.linalg.norm(seg.p1 - x)
                if min(d0, d1) <= reach:
                    if d0 < d1:
                        seg.p0 = x.copy()
                    else:
                        seg.p1 = x.copy()
    return segs


def build_layout(pts: np.ndarray, scale: float = 1.0, up_hint: np.ndarray | None = None,
                 wall_thickness: float = 0.15, cameras: np.ndarray | None = None,
                 vis_pairs: np.ndarray | None = None) -> LayoutResult:
    """cameras: centros das câmeras (M x 3, mesmo sistema de pts antes da escala).
    vis_pairs: pares (índice do ponto, índice da câmera) de quem viu cada ponto (OpenMVS/COLMAP).
    Sem câmeras não há como separar vão de região não filmada: aberturas não são detectadas."""
    pts = np.asarray(pts, float) * scale
    R, rot = align_to_gravity(pts, up_hint)
    floor_z, ceil_z = floor_ceiling(rot[:, 2])
    floor_off = floor_z
    rot[:, 2] -= floor_z
    cams = None
    if cameras is not None:
        cams = (np.asarray(cameras, float) * scale) @ R.T
        cams[:, 2] -= floor_z
    ceil_z, floor_z = ceil_z - floor_z, 0.0
    height = ceil_z - floor_z
    notes = []
    if not 2.0 <= height <= 5.0:
        notes.append(f"Pé-direito estimado {height:.2f} m fora do usual — verificar escala.")
    # faixa de parede: evita móveis (baixo) e sancas/luminárias (alto)
    band = (rot[:, 2] > floor_z + 0.25 * height + 0.3) & (rot[:, 2] < ceil_z - 0.15)
    xy = voxel_downsample(rot[band][:, :2], 0.01)
    segs = extract_segments(xy)
    segs = join_corners(merge_collinear(snap_manhattan(segs)))
    segs = [sg_ for sg_ in segs if reaches_ceiling(sg_, rot, ceil_z)]
    inner = cams[:, :2].mean(axis=0) if cams is not None else np.median(
        rot[np.abs(rot[:, 2]) < 0.03][:, :2], axis=0)
    segs = join_corners(enclosing_only(segs, inner))
    sg = SceneGraph(scale_source="unknown")
    for s in segs:
        sg.entities.append(Wall(start=tuple(map(float, s.p0)), end=tuple(map(float, s.p1)),
                                thickness=wall_thickness, height=round(height, 3),
                                confidence=min(1.0, s.support / 1500)))
    walls = [e for e in sg.entities if isinstance(e, Wall)]
    center = np.mean([np.r_[w.start] for w in walls] + [np.r_[w.end] for w in walls], axis=0) if walls else np.zeros(2)
    if cams is None:
        notes.append("Sem poses de câmera: portas e janelas não detectadas (não dá para separar vão de área não filmada).")
    else:
        occl = op.Occluders(voxel_downsample(rot, 0.04))
        for w in walls:
            g = op.build_grid(w, rot, height, center, cams, vis_pairs, occl)
            found, wn = op.openings_from_grid(w, g)
            for o in found:
                if o.source != "inferred":
                    o.depth = op.reveal_depth(w, o, rot, center)
            sg.entities.extend(found)
            notes.extend(wn)
    poly = room_polygon(segs, inner)
    if poly is None:  # paredes não fecham: usa pontos do piso perto do interior
        flat = rot[(np.abs(rot[:, 2]) < 0.03) & (np.linalg.norm(rot[:, :2] - inner, axis=1) < 6)][:, :2]
        poly = _hull(flat) if len(flat) >= 3 else None
        notes.append("Paredes não formam contorno fechado: piso estimado pelos pontos (pode incluir áreas vizinhas).")
    if poly:
        sg.entities.append(Slab(polygon=poly, elevation=0.0, role="floor"))
        sg.entities.append(Slab(polygon=poly, elevation=round(height, 3), role="ceiling"))
    notes.append("Espessura de parede é padrão (inferida); eixo das paredes = face interna medida.")
    return LayoutResult(sg, floor_z, floor_off, ceil_z, R, segs, notes)


def enclosing_only(segs: list[Segment], inner: np.ndarray) -> list[Segment]:
    """Remove superfícies vistas ATRÁS de outra parede (corredor pela porta, prédio pela janela):
    descarta o segmento se a linha do interior até ele cruzar outro segmento antes."""
    def crosses(p, q, s):
        d1, d2 = q - p, s.p1 - s.p0
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            return False
        w = s.p0 - p
        u = (w[0] * d2[1] - w[1] * d2[0]) / den
        v = (w[0] * d1[1] - w[1] * d1[0]) / den
        return 0 < u < 0.97 and -0.02 <= v <= 1.02

    keep = []
    for s in segs:
        probes = [s.p0 + f * (s.p1 - s.p0) for f in (0.25, 0.5, 0.75)]
        hidden = sum(any(crosses(inner, q, o) for o in segs if o is not s) for q in probes)
        if hidden < 2:
            keep.append(s)
    return keep


def reaches_ceiling(seg: Segment, rot: np.ndarray, ceil_z: float, near: float = 0.04,
                    top_band: float = 0.35, bin_m: float = 0.1, min_cov: float = 0.5) -> bool:
    """Parede de verdade vai até o teto; frente de armário/estante não."""
    d = seg.direction
    nvec = np.array([-d[1], d[0]])
    rel = rot[:, :2] - seg.p0
    m = (np.abs(rel @ nvec) < near) & (rot[:, 2] > ceil_z - top_band) & (rot[:, 2] < ceil_z - 0.03)
    t = rel[m] @ d
    nb = max(1, int(seg.length / bin_m))
    hist, _ = np.histogram(t, bins=nb, range=(0, seg.length))
    return (hist > 0).mean() >= min_cov


def room_polygon(segs: list[Segment], inner: np.ndarray, snap: float = 0.03):
    """Contorno do cômodo a partir dos cantos das paredes (ordenados pelo ângulo em torno do interior).
    Vale para cômodos 'estrelados' a partir do interior (retangulares, em L, a maioria dos reais)."""
    if len(segs) < 3:
        return None
    pts = []
    for s in segs:
        for q in (s.p0, s.p1):
            if not any(np.linalg.norm(q - r) < snap for r in pts):
                pts.append(q)
    # cada canto deve ser compartilhado por 2 paredes; senão o contorno não fecha
    shared = [q for q in pts if sum(min(np.linalg.norm(q - s.p0), np.linalg.norm(q - s.p1)) < snap
                                    for s in segs) >= 2]
    if len(shared) != len(pts) or len(pts) < 3:
        return None
    ang = [np.arctan2(*(q - inner)[::-1]) for q in pts]
    return [tuple(map(float, pts[i])) for i in np.argsort(ang)]


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _hull(p: np.ndarray) -> list[tuple[float, float]]:
    """Fecho convexo (monotone chain) — contorno inicial do piso."""
    p = np.unique(np.round(p, 3), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h

    lower, upper = half(p), half(p[::-1])
    return [tuple(map(float, q)) for q in lower[:-1] + upper[:-1]]
