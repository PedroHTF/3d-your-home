"""Edição a posteriori: calibração de escala por medidas, ajustes por elemento e restrições
geométricas (paralelismo, perpendicularidade, ângulo, orientação, distância, comprimento, trava).

Fluxo sempre reprodutível (nada se acumula; apagar uma restrição restaura o estado anterior):

    cena medida  ->  escala (medidas mode="scale")  ->  ajustes por elemento (overrides)
                 ->  restrições geométricas (mínimos quadrados)  ->  cena final

As paredes ligadas por um canto compartilham o mesmo ponto, então mexer numa parede arrasta
as vizinhas. A geometria medida entra como âncora fraca: o resolvedor muda o mínimo possível.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .scenegraph import Constraint, Opening, SceneGraph, Slab, Wall

MEASURE_KINDS = {"length", "distance", "ceiling_height", "opening_width", "opening_height",
                 "opening_sill", "opening_offset"}
GEOM_KINDS = {"length", "distance", "parallel", "perpendicular", "angle", "orientation", "fixed"}
OPENING_KINDS = {"opening_width", "opening_height", "opening_sill", "opening_offset"}
WALL_EDITABLE = {"thickness", "height", "start", "end"}
OPENING_EDITABLE = {"kind", "offset", "width", "height", "sill"}


@dataclass
class ConstraintStatus:
    id: str
    kind: str
    satisfied: bool
    error: float          # m (medidas) ou graus (ângulos)
    message: str = ""


@dataclass
class SolveReport:
    scale_factor: float = 1.0
    scale_uncertainty: float | None = None
    statuses: list[ConstraintStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.satisfied for s in self.statuses)


# ---------------------------------------------------------------- medição

def measure(sg: SceneGraph, c: Constraint) -> float | None:
    """Valor atual, na cena, da grandeza que a restrição fixa."""
    if c.kind == "length":
        w = sg.get(c.refs[0])
        return float(np.linalg.norm(np.subtract(w.end, w.start)))
    if c.kind == "distance":
        a, b = sg.get(c.refs[0]), sg.get(c.refs[1])
        pa, da = np.array(a.start), _dir(a)
        n = np.array([-da[1], da[0]])
        mb = (np.array(b.start) + np.array(b.end)) / 2
        return float(abs((mb - pa) @ n))
    if c.kind == "ceiling_height":
        hs = [w.height for w in sg.entities if isinstance(w, Wall)]
        return float(np.median(hs)) if hs else None
    if c.kind in OPENING_KINDS:
        o = sg.get(c.refs[0])
        return float(getattr(o, c.kind.removeprefix("opening_")))
    if c.kind in ("parallel", "perpendicular", "angle"):
        return _line_angle(sg.get(c.refs[0]), sg.get(c.refs[1]))
    if c.kind == "orientation":
        d = _dir(sg.get(c.refs[0]))
        return float(np.degrees(np.arctan2(d[1], d[0])) % 180)
    return None


def _dir(w: Wall) -> np.ndarray:
    d = np.subtract(w.end, w.start)
    return d / np.linalg.norm(d)


def _line_angle(a: Wall, b: Wall) -> float:
    """Ângulo entre as linhas, em [0, 180)."""
    da, db = _dir(a), _dir(b)
    return float(np.degrees(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db)) % 180)


def _target(c: Constraint) -> float | None:
    return {"parallel": 0.0, "perpendicular": 90.0}.get(c.kind, c.value)


def _wrap180(x: float) -> float:
    """Diferença angular entre linhas (período 180°) em (-90, 90]."""
    return (x + 90.0) % 180.0 - 90.0


# ---------------------------------------------------------------- escala

def calibrate(sg: SceneGraph) -> tuple[float, float | None, list[ConstraintStatus]]:
    """Fator global a partir das medidas mode='scale' (mediana ponderada, descarta incoerentes)."""
    cs = [c for c in sg.constraints if c.enabled and c.mode == "scale" and c.kind in MEASURE_KINDS]
    ratios, used = [], []
    for c in cs:
        m = measure(sg, c)
        if m and c.value and m > 0:
            ratios.append(c.value / m)
            used.append(c)
    if not ratios:
        return 1.0, None, []
    r = np.array(ratios)
    med = float(np.median(r))
    good = np.abs(r / med - 1) < 0.03
    s = float(r[good].mean())
    unc = float(r[good].std() / s) if good.sum() > 1 else None
    st = []
    for c, ri, g in zip(used, r, good):
        err = abs(ri / s - 1) * (c.value or 0)
        st.append(ConstraintStatus(c.id, c.kind, bool(g), round(err, 4),
                                   "" if g else f"Medida incoerente com as demais ({(ri / s - 1):+.1%}); ignorada na escala."))
    return s, unc, st


def apply_scale(sg: SceneGraph, s: float) -> None:
    for e in sg.entities:
        if isinstance(e, Wall):
            e.start = tuple(np.multiply(e.start, s))
            e.end = tuple(np.multiply(e.end, s))
            e.height *= s
        elif isinstance(e, Opening):
            e.offset *= s; e.width *= s; e.height *= s; e.sill *= s
        elif isinstance(e, Slab):
            e.polygon = [tuple(np.multiply(p, s)) for p in e.polygon]
            e.elevation *= s
    sg.scale_factor *= s


# ---------------------------------------------------------------- ajustes por elemento

def apply_overrides(sg: SceneGraph) -> list[str]:
    warn = []
    for eid, params in sg.overrides.items():
        e = sg.get(eid)
        if e is None:
            warn.append(f"Ajuste para elemento inexistente {eid} ignorado.")
            continue
        allowed = WALL_EDITABLE if isinstance(e, Wall) else OPENING_EDITABLE if isinstance(e, Opening) else set()
        for k, v in params.items():
            if k not in allowed:
                warn.append(f"{eid}: parâmetro '{k}' não editável.")
                continue
            setattr(e, k, tuple(v) if k in ("start", "end") else v)
    return warn


# ---------------------------------------------------------------- resolvedor geométrico

def _topology(walls: list[Wall], tol: float = 0.03):
    """Une extremidades coincidentes num mesmo ponto (canto)."""
    pts: list[np.ndarray] = []
    idx = {}
    for w in walls:
        for end in ("start", "end"):
            q = np.array(getattr(w, end), float)
            for i, r in enumerate(pts):
                if np.linalg.norm(q - r) < tol:
                    idx[(w.id, end)] = i
                    break
            else:
                idx[(w.id, end)] = len(pts)
                pts.append(q)
    return np.array(pts), idx


def solve_geometry(sg: SceneGraph, w_constraint: float = 1e3, w_anchor: float = 1.0,
                   tol_m: float = 0.001, tol_deg: float = 0.1) -> list[ConstraintStatus]:
    walls = [e for e in sg.entities if isinstance(e, Wall)]
    cs = [c for c in sg.constraints if c.enabled and c.kind in GEOM_KINDS
          and not (c.mode == "scale" and c.kind in MEASURE_KINDS)]
    bad = [c for c in cs if any(not isinstance(sg.get(r), Wall) for r in c.refs)]
    statuses = [ConstraintStatus(c.id, c.kind, False, float("nan"), "Referência inexistente ou não é parede.")
                for c in bad]
    cs = [c for c in cs if c not in bad]
    if not cs or not walls:
        return statuses
    P0, idx = _topology(walls)
    wid = {w.id: w for w in walls}
    fixed_pts = {idx[(r, e)] for c in cs if c.kind == "fixed" for r in c.refs for e in ("start", "end")}

    def seg(P, w_id):
        return P[idx[(w_id, "start")]], P[idx[(w_id, "end")]]

    def residuals(x):
        P = x.reshape(-1, 2)
        res = []
        for c in cs:
            if c.kind == "fixed":
                continue
            if c.kind == "length":
                a, b = seg(P, c.refs[0])
                res.append(w_constraint * (np.linalg.norm(b - a) - c.value))
            elif c.kind == "distance":
                a0, a1 = seg(P, c.refs[0])
                b0, b1 = seg(P, c.refs[1])
                da = (a1 - a0) / np.linalg.norm(a1 - a0)
                n = np.array([-da[1], da[0]])
                res.append(w_constraint * (abs(((b0 + b1) / 2 - a0) @ n) - c.value))
            else:  # ângulos: resíduo em radianos x comprimento médio (vira metros na ponta)
                a0, a1 = seg(P, c.refs[0])
                da = a1 - a0
                if c.kind == "orientation":
                    cur = np.degrees(np.arctan2(da[1], da[0]))
                    L = np.linalg.norm(da)
                else:
                    b0, b1 = seg(P, c.refs[1])
                    db = b1 - b0
                    cur = np.degrees(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db))
                    L = (np.linalg.norm(da) + np.linalg.norm(db)) / 2
                res.append(w_constraint * np.radians(_wrap180(cur - _target(c))) * L)
        anchor_w = np.where(np.isin(np.arange(len(P0)), list(fixed_pts)), w_constraint, w_anchor)
        res.extend((anchor_w[:, None] * (P - P0)).ravel())
        return np.array(res)

    sol = least_squares(residuals, P0.ravel(), method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    P = sol.x.reshape(-1, 2)
    # escreve de volta: paredes, aberturas acompanham a parede (offset preservado), lajes seguem os cantos
    old = {w.id: (np.array(w.start), np.array(w.end)) for w in walls}
    for w in walls:
        w.start, w.end = tuple(map(float, P[idx[(w.id, "start")]])), tuple(map(float, P[idx[(w.id, "end")]]))
    for s in sg.entities:
        if isinstance(s, Slab):
            s.polygon = [tuple(map(float, P[np.argmin(np.linalg.norm(P0 - np.array(q), axis=1))]))
                         if np.min(np.linalg.norm(P0 - np.array(q), axis=1)) < 0.05 else q for q in s.polygon]
    for c in cs:
        if c.kind == "fixed":
            moved = max(np.linalg.norm(np.array(getattr(wid[c.refs[0]], e)) - old[c.refs[0]][i])
                        for i, e in enumerate(("start", "end")))
            statuses.append(ConstraintStatus(c.id, c.kind, moved < tol_m, round(moved, 4),
                                             "" if moved < tol_m else "Parede travada precisou se mover: conflito."))
            continue
        cur = measure(sg, c)
        if c.kind in ("length", "distance"):
            err = abs(cur - c.value)
            ok = err < tol_m
        else:
            target = _target(c)
            if c.kind == "orientation":
                err = abs(_wrap180(cur - target))
            else:
                err = abs(_wrap180(cur - target))
            ok = err < tol_deg
        statuses.append(ConstraintStatus(c.id, c.kind, bool(ok), round(float(err), 4),
                                         "" if ok else "Não satisfeita: conflita com outras restrições."))
    return statuses


def apply_opening_constraints(sg: SceneGraph) -> list[ConstraintStatus]:
    st = []
    for c in sg.constraints:
        if not c.enabled or c.kind not in OPENING_KINDS or c.mode == "scale":
            continue
        o = sg.get(c.refs[0]) if c.refs else None
        if not isinstance(o, Opening):
            st.append(ConstraintStatus(c.id, c.kind, False, float("nan"), "Referência não é abertura."))
            continue
        setattr(o, c.kind.removeprefix("opening_"), float(c.value))
        w = sg.get(o.host_wall)
        L = float(np.linalg.norm(np.subtract(w.end, w.start))) if w else 0
        fits = o.offset >= 0 and o.offset + o.width <= L + 1e-6 and (not w or o.sill + o.height <= w.height + 1e-6)
        st.append(ConstraintStatus(c.id, c.kind, fits, 0.0, "" if fits else "Abertura não cabe na parede."))
    return st


def ceiling_constraint(sg: SceneGraph) -> list[ConstraintStatus]:
    st = []
    for c in sg.constraints:
        if c.enabled and c.kind == "ceiling_height" and c.mode == "local":
            for e in sg.entities:
                if isinstance(e, Wall):
                    e.height = float(c.value)
                elif isinstance(e, Slab) and e.role == "ceiling":
                    e.elevation = float(c.value)
            st.append(ConstraintStatus(c.id, c.kind, True, 0.0))
    return st


# ---------------------------------------------------------------- tudo junto

def rebuild(measured: SceneGraph) -> tuple[SceneGraph, SolveReport]:
    """Recalcula a cena final a partir da medida + restrições + ajustes (que vivem em `measured`)."""
    sg = measured.model_copy(deep=True)
    rep = SolveReport()
    s, unc, st = calibrate(sg)
    if st:
        apply_scale(sg, s)
        sg.scale_source = "user_measure"
        sg.scale_uncertainty = unc
        rep.scale_factor, rep.scale_uncertainty = s, unc
        if abs(s - 1) > 0.10:
            rep.warnings.append(f"Escala corrigida em {s - 1:+.0%}: recomendável refazer a detecção "
                                "(POST /plan) para que portas e janelas usem limites na escala certa.")
    rep.statuses += st
    rep.warnings += apply_overrides(sg)
    rep.statuses += ceiling_constraint(sg)
    rep.statuses += solve_geometry(sg)
    rep.statuses += apply_opening_constraints(sg)
    return sg, rep
