"""Junção de cômodos de levantamentos diferentes numa planta única.

Cada levantamento (captura) tem a própria planta, já em metros e com o piso alinhado. Juntar é
achar, para cada captura, uma pose 2D (x, y, θ) num sistema comum: p_mundo = R(θ)·p + (x, y).

As poses vêm de LIGAÇÕES entre capturas, resolvidas juntas por mínimos quadrados (ajuste global:
em ciclos o erro se distribui em vez de acumular):

  door_match    mesma porta vista dos dois lados (fixa a pose relativa e MEDE a espessura da parede)
  wall_contact  parede A encostada na parede B (faces opostas, distância = espessura)
  collinear     parede A e parede B na mesma linha (ex.: fachada/corredor contínuos): fixa ângulo e a
                posição perpendicular. NÃO fixa a espessura entre cômodos — quem fixa é a
                profundidade do batente (Opening.depth) ou o valor informado.
  point_match   mesmo ponto físico nas duas capturas (canto, marcador ArUco compartilhado...)
  pose          pose definida à mão (pode ser travada)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import least_squares

from .scenegraph import Opening, SceneGraph, Separator, Slab, Space, Wall, _id

T_PRIOR, T_MIN, T_MAX = 0.15, 0.05, 0.40   # espessura de parede (m)


class Pose2D(BaseModel):
    x: float = 0.0
    y: float = 0.0
    theta_deg: float = 0.0
    locked: bool = False


class Capture(BaseModel):
    id: str = Field(default_factory=_id)
    name: str
    project_id: str | None = None      # projeto (levantamento) de origem
    pose: Pose2D = Pose2D()


class Ref(BaseModel):
    capture: str
    element: str | None = None         # id da abertura/parede na captura
    point: tuple[float, float] | None = None  # ponto no sistema da captura (point_match)


class Link(BaseModel):
    id: str = Field(default_factory=_id)
    kind: Literal["door_match", "wall_contact", "collinear", "point_match"]
    a: Ref
    b: Ref
    thickness: float | None = None     # conhecida? senão é estimada (door_match / wall_contact)
    enabled: bool = True
    note: str | None = None
    auto_key: str | None = None        # criada automaticamente (ex.: "aruco:7:A:B")


class Building(BaseModel):
    id: str = Field(default_factory=_id)
    name: str
    captures: list[Capture] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    measured_thickness: dict[str, float] = Field(default_factory=dict)  # link id -> espessura estimada
    ignored_auto: list[str] = Field(default_factory=list)  # ligações automáticas que o usuário apagou

    def cap(self, cid: str) -> Capture:
        return next(c for c in self.captures if c.id == cid)


# ------------------------------------------------------------------ geometria

def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def to_world(p, pose: tuple[float, float, float]) -> np.ndarray:
    x, y, th = pose
    return rot(th) @ np.asarray(p, float) + np.array([x, y])


def room_center(sg: SceneGraph) -> np.ndarray:
    fl = [e for e in sg.entities if isinstance(e, Slab) and e.role == "floor"]
    if fl:
        return np.mean(fl[0].polygon, axis=0)
    ws = [e for e in sg.entities if isinstance(e, Wall)]
    return np.mean([w.start for w in ws] + [w.end for w in ws], axis=0)


@dataclass
class DoorGeom:
    center: np.ndarray       # centro do vão na face da parede (sistema da captura)
    direction: np.ndarray    # ao longo da parede
    normal: np.ndarray       # para FORA do cômodo
    width: float
    height: float


def door_geom(sg: SceneGraph, oid: str) -> DoorGeom:
    o = sg.get(oid)
    w = sg.get(o.host_wall)
    a, b = np.array(w.start), np.array(w.end)
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    if ((a + b) / 2 - room_center(sg)) @ n < 0:
        n = -n
    return DoorGeom(a + d * (o.offset + o.width / 2), d, n, o.width, o.height)


def wall_geom(sg: SceneGraph, wid: str):
    w = sg.get(wid)
    a, b = np.array(w.start), np.array(w.end)
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    if ((a + b) / 2 - room_center(sg)) @ n < 0:
        n = -n
    return a, b, d, n


# ------------------------------------------------------------------ ajuste global

@dataclass
class LinkStatus:
    id: str
    kind: str
    residual_m: float
    residual_deg: float
    thickness: float | None = None
    ok: bool = True
    message: str = ""


@dataclass
class SolveResult:
    poses: dict[str, tuple[float, float, float]]
    statuses: list[LinkStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def solve_poses(b: Building, scenes: dict[str, SceneGraph], tol_m: float = 0.02,
                tol_deg: float = 0.5) -> SolveResult:
    caps = b.captures
    links = [lk for lk in b.links if lk.enabled]
    warn = []
    # captura de referência: travada, ou a primeira
    ref = next((c for c in caps if c.pose.locked), caps[0])
    free = [c for c in caps if c is not ref and not c.pose.locked]
    fidx = {c.id: i for i, c in enumerate(free)}
    th_links = [lk for lk in links if lk.kind in ("door_match", "wall_contact") and lk.thickness is None]
    tidx = {lk.id: i for i, lk in enumerate(th_links)}
    nP = 3 * len(free)

    def pose_of(x, cid):
        if cid in fidx:
            i = fidx[cid]
            return x[3 * i], x[3 * i + 1], x[3 * i + 2]
        c = b.cap(cid)
        return c.pose.x, c.pose.y, np.radians(c.pose.theta_deg)

    def thick(x, lk):
        return lk.thickness if lk.thickness is not None else x[nP + tidx[lk.id]]

    # profundidade do batente medida nas capturas: vira a espessura da ligação
    for lk in links:
        if lk.kind == "door_match" and lk.thickness is None:
            ds = [d for d in (scenes[lk.a.capture].get(lk.a.element).depth,
                              scenes[lk.b.capture].get(lk.b.element).depth) if d]
            if ds:
                lk.thickness = float(np.mean(ds))
                lk.note = f"espessura pelo batente ({', '.join(f'{d * 100:.1f}' for d in ds)} cm)"
                b.measured_thickness[lk.id] = round(lk.thickness, 4)
    th_links = [lk for lk in links if lk.kind in ("door_match", "wall_contact") and lk.thickness is None]
    tidx = {lk.id: i for i, lk in enumerate(th_links)}
    nP = 3 * len(free)

    pre = {}
    for lk in links:
        sa, sb = scenes[lk.a.capture], scenes[lk.b.capture]
        if lk.kind == "door_match":
            pre[lk.id] = (door_geom(sa, lk.a.element), door_geom(sb, lk.b.element))
            ga, gb = pre[lk.id]
            if abs(ga.width - gb.width) > 0.06 or abs(ga.height - gb.height) > 0.08:
                warn.append(f"Ligação {lk.id}: portas com medidas diferentes "
                            f"({ga.width:.2f}x{ga.height:.2f} vs {gb.width:.2f}x{gb.height:.2f}) — conferir.")
        elif lk.kind in ("wall_contact", "collinear"):
            pre[lk.id] = (wall_geom(sa, lk.a.element), wall_geom(sb, lk.b.element))

    W_LINK, W_ANCHOR, W_T = 100.0, 1e-3, 1.0

    def residuals(x):
        r = []
        for lk in links:
            pa, pb = pose_of(x, lk.a.capture), pose_of(x, lk.b.capture)
            Ra, Rb = rot(pa[2]), rot(pb[2])
            if lk.kind == "door_match":
                ga, gb = pre[lk.id]
                t = thick(x, lk)
                ca, cb = to_world(ga.center, pa), to_world(gb.center, pb)
                na, nb = Ra @ ga.normal, Rb @ gb.normal
                r.extend(W_LINK * (ca + na * t - cb))          # vão de B = vão de A do outro lado
                # normais opostas; ângulo pesado pelo comprimento da parede (a direção vem da parede toda)
                La = np.linalg.norm(np.subtract(*[getattr(scenes[lk.a.capture].get(
                    scenes[lk.a.capture].get(lk.a.element).host_wall), e) for e in ("end", "start")]))
                r.append(W_LINK * La * (na[0] * nb[1] - na[1] * nb[0]))
                r.append(W_LINK * (na @ nb + 1.0))
            elif lk.kind == "wall_contact":
                (a0, a1, da, na_), (b0, b1, db, nb_) = pre[lk.id]
                t = thick(x, lk)
                A0, na = to_world(a0, pa), Ra @ na_
                B0, B1, nb = to_world(b0, pb), to_world(b1, pb), Rb @ nb_
                for q in (B0, B1):                              # face de B a t da face de A
                    r.append(W_LINK * ((q - A0) @ na - t))
                r.append(W_LINK * (na[0] * nb[1] - na[1] * nb[0]))
                r.append(W_LINK * (na @ nb + 1.0))
            elif lk.kind == "collinear":
                (a0, a1, da, na_), (b0, b1, db, nb_) = pre[lk.id]
                A0, na = to_world(a0, pa), Ra @ na_
                nb = Rb @ nb_
                for q in (b0, b1):                              # B sobre a linha de A, mesma face
                    r.append(W_LINK * ((to_world(q, pb) - A0) @ na))
                r.append(W_LINK * (na @ nb - 1.0))
            else:  # point_match
                r.extend(W_LINK * (to_world(lk.a.point, pa) - to_world(lk.b.point, pb)))
        for lk in th_links:  # espessura: prior fraco
            r.append(W_T * 0.1 * (thick(x, lk) - T_PRIOR))
        for c in free:  # âncora fraquíssima: evita graus de liberdade soltos
            p = pose_of(x, c.id)
            r.extend(W_ANCHOR * np.array([p[0] - c.pose.x, p[1] - c.pose.y,
                                          np.sin(p[2] - np.radians(c.pose.theta_deg))]))
        return np.array(r) if r else np.zeros(1)

    init = _chain_init(b, scenes, ref, links)
    x0 = []
    for c in free:
        x0 += list(init.get(c.id, (c.pose.x, c.pose.y, np.radians(c.pose.theta_deg))))
    x0 += [T_PRIOR] * len(th_links)
    if x0:
        lb = [-np.inf] * nP + [T_MIN] * len(th_links)
        ub = [np.inf] * nP + [T_MAX] * len(th_links)
        x0 = np.clip(np.array(x0, float), np.array(lb) + 1e-9, np.array(ub) - 1e-9)
        sol = least_squares(residuals, x0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
        x = sol.x
        # espessura determinada pela geometria (e não só pelo prior)? sensibilidade das ligações a t
        J = np.asarray(sol.jac)
        n_link_rows = len(residuals(x)) - len(th_links) - 3 * len(free)
        Jl = J[:n_link_rows]

        def rank(M):
            sv = np.linalg.svd(M, compute_uv=False) if M.size else np.zeros(0)
            return int((sv > 1e-6 * (sv[0] if len(sv) else 1)).sum())
        r_full = rank(Jl)
        determined = {lk.id: rank(np.delete(Jl, nP + tidx[lk.id], axis=1)) < r_full for lk in th_links}
    else:
        x = np.zeros(0)
        determined = {}
    poses = {c.id: pose_of(x, c.id) for c in caps}

    # ligações por captura: alguma solta?
    linked = {ref.id} | {c.id for c in caps if c.pose.locked}
    changed = True
    while changed:
        changed = False
        for lk in links:
            if (lk.a.capture in linked) != (lk.b.capture in linked):
                linked |= {lk.a.capture, lk.b.capture}
                changed = True
    for c in caps:
        if c.id not in linked:
            warn.append(f"Captura '{c.name}' sem ligação com as demais: posicionada só pela pose manual.")

    st = []
    for lk in links:
        pa, pb = poses[lk.a.capture], poses[lk.b.capture]
        t = thick(x, lk) if lk.kind in ("door_match", "wall_contact") else None
        if lk.kind == "door_match":
            ga, gb = pre[lk.id]
            na, nb = rot(pa[2]) @ ga.normal, rot(pb[2]) @ gb.normal
            em = float(np.linalg.norm(to_world(ga.center, pa) + na * t - to_world(gb.center, pb)))
            ed = float(np.degrees(np.arccos(np.clip(-na @ nb, -1, 1))))
        elif lk.kind in ("wall_contact", "collinear"):
            (a0, a1, da, na_), (b0, b1, db, nb_) = pre[lk.id]
            na = rot(pa[2]) @ na_
            off = t if lk.kind == "wall_contact" else 0.0
            sign = -1 if lk.kind == "wall_contact" else 1
            em = float(max(abs((to_world(q, pb) - to_world(a0, pa)) @ na - off) for q in (b0, b1)))
            ed = float(np.degrees(np.arccos(np.clip(sign * (na @ (rot(pb[2]) @ nb_)), -1, 1))))
        else:
            em, ed = float(np.linalg.norm(to_world(lk.a.point, pa) - to_world(lk.b.point, pb))), 0.0
        ok = em < tol_m and ed < tol_deg
        if t is not None and lk.thickness is None and (t <= T_MIN + 1e-4 or t >= T_MAX - 1e-4):
            ok = False
        msg = "" if ok else "Ligação não fecha: confira se é a porta/parede certa ou se há conflito."
        if ok and t is not None and lk.thickness is None and not determined.get(lk.id, False):
            msg = (f"Espessura não determinada pela geometria: usado padrão {T_PRIOR * 100:.0f} cm. "
                   "Ligue uma parede alinhada (fachada) ou informe a espessura.")
        st.append(LinkStatus(lk.id, lk.kind, round(em, 4), round(ed, 3),
                             round(float(t), 4) if t is not None else None, ok, msg))
        if t is not None and lk.thickness is None and determined.get(lk.id, False):
            b.measured_thickness[lk.id] = round(float(t), 4)
    return SolveResult(poses, st, warn)


def _door_pose(pa, ga: "DoorGeom", gb: "DoorGeom", t: float):
    """Pose de B dado A: normal de B oposta à de A e vãos alinhados com espessura t."""
    Ra = rot(pa[2])
    na_w, ca_w = Ra @ ga.normal, to_world(ga.center, pa)
    th = np.arctan2(-na_w[1], -na_w[0]) - np.arctan2(gb.normal[1], gb.normal[0])
    tr = ca_w + na_w * t - rot(th) @ gb.center
    return (float(tr[0]), float(tr[1]), float(th))


def _procrustes(P, Q):
    """Rotação+translação 2D que leva Q (local de B) em P (mundo)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    pc, qc = P.mean(0), Q.mean(0)
    H = (Q - qc).T @ (P - pc)
    th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    tr = pc - rot(th) @ qc
    return (float(tr[0]), float(tr[1]), float(th))


def _chain_init(b: Building, scenes, ref: Capture, links) -> dict:
    """Chute inicial: percorre as ligações a partir da referência (evita mínimos locais)."""
    known = {c.id: (c.pose.x, c.pose.y, np.radians(c.pose.theta_deg)) for c in b.captures
             if c is ref or c.pose.locked}
    changed = True
    while changed:
        changed = False
        for lk in links:
            for src, dst, rs, rd in ((lk.a, lk.b, "a", "b"), (lk.b, lk.a, "b", "a")):
                if src.capture not in known or dst.capture in known:
                    continue
                if lk.kind == "door_match":
                    gs = door_geom(scenes[src.capture], src.element)
                    gd = door_geom(scenes[dst.capture], dst.element)
                    known[dst.capture] = _door_pose(known[src.capture], gs, gd, lk.thickness or T_PRIOR)
                    changed = True
                elif lk.kind == "point_match":
                    pts = [(to_world(getattr(l2, rs).point, known[src.capture]), getattr(l2, rd).point)
                           for l2 in links if l2.kind == "point_match"
                           and getattr(l2, rs).capture == src.capture and getattr(l2, rd).capture == dst.capture]
                    if len(pts) >= 2:
                        known[dst.capture] = _procrustes([p for p, _ in pts], [q for _, q in pts])
                        changed = True
    return known


def apply_poses(b: Building, res: SolveResult) -> None:
    for c in b.captures:
        x, y, th = res.poses[c.id]
        c.pose.x, c.pose.y, c.pose.theta_deg = float(x), float(y), float(np.degrees(th) % 360)


# ------------------------------------------------------------------ sugestões automáticas

@dataclass
class Suggestion:
    a_capture: str
    a_door: str
    b_capture: str
    b_door: str
    score: float
    thickness: float
    pose_b: tuple[float, float, float]   # pose de B relativa a A (A na origem)
    reasons: list[str]
    ambiguous: bool = False


def _poly_overlap(pa, pb, n: int = 40) -> float:
    """Fração da área de B que cai dentro de A (amostragem em grade)."""
    from .rooms import _inside
    pb = np.array(pb)
    lo, hi = pb.min(axis=0), pb.max(axis=0)
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)), -1).reshape(-1, 2)
    inb = [q for q in g if _inside(q, pb)]
    if not inb:
        return 0.0
    return sum(_inside(q, pa) for q in inb) / len(inb)


def _floor(sg: SceneGraph):
    fl = [e for e in sg.entities if isinstance(e, Slab) and e.role == "floor"]
    return fl[0].polygon if fl else None


def suggest_door_matches(b: Building, scenes: dict[str, SceneGraph], min_conf: float = 0.6,
                         max_dw: float = 0.06, max_dh: float = 0.08) -> list[Suggestion]:
    out = []
    caps = b.captures
    already = {(lk.a.capture, lk.a.element, lk.b.capture, lk.b.element) for lk in b.links}
    for i, ca in enumerate(caps):
        for cb in caps[i + 1:]:
            sa, sb = scenes[ca.id], scenes[cb.id]
            da = [e for e in sa.entities if isinstance(e, Opening) and e.kind.value == "door" and e.confidence >= min_conf]
            db = [e for e in sb.entities if isinstance(e, Opening) and e.kind.value == "door" and e.confidence >= min_conf]
            for oa in da:
                for ob in db:
                    if (ca.id, oa.id, cb.id, ob.id) in already:
                        continue
                    if abs(oa.width - ob.width) > max_dw or abs(oa.height - ob.height) > max_dh:
                        continue
                    ga, gb = door_geom(sa, oa.id), door_geom(sb, ob.id)
                    depths = [d for d in (oa.depth, ob.depth) if d]
                    if len(depths) == 2 and abs(depths[0] - depths[1]) > 0.04:
                        continue  # batentes de profundidades diferentes: não é a mesma porta
                    # pose de B com A na origem: normal de B oposta à de A, vãos alinhados a T_PRIOR
                    th = np.arctan2(-ga.normal[1], -ga.normal[0]) - np.arctan2(gb.normal[1], gb.normal[0])
                    t_best, s_best, reasons_best, pose_best = None, -np.inf, [], None
                    ts = [float(np.mean(depths))] if depths else np.arange(0.08, 0.36, 0.005)
                    for t in ts:
                        tr = ga.center + ga.normal * t - rot(th) @ gb.center
                        pose = (tr[0], tr[1], th)
                        sc, why = _score(sa, sb, pose)
                        sc -= 1e-3 * abs(t - T_PRIOR)  # empate (espessura indeterminada) -> padrão
                        if sc > s_best:
                            t_best, s_best, reasons_best, pose_best = t, sc, why, pose
                    sc = s_best - 2.0 * abs(oa.width - ob.width) / max_dw - abs(oa.height - ob.height) / max_dh
                    out.append(Suggestion(ca.id, oa.id, cb.id, ob.id, round(float(sc), 3), round(float(t_best), 3),
                                          tuple(map(float, pose_best)), reasons_best))
    out.sort(key=lambda s: -s.score)
    # ambiguidade: outra hipótese para a MESMA porta com pontuação parecida
    for s_ in out:
        rivals = [o for o in out if o is not s_ and ((o.a_capture, o.a_door) == (s_.a_capture, s_.a_door)
                  or (o.b_capture, o.b_door) == (s_.b_capture, s_.b_door))]
        s_.ambiguous = any(abs(o.score - s_.score) < 0.5 for o in rivals)
        if s_.ambiguous:
            s_.reasons.append("ambígua: outra porta encaixa igualmente bem — confirme visualmente")
    return out


def _score(sa: SceneGraph, sb: SceneGraph, pose) -> tuple[float, list[str]]:
    """Plausibilidade de B posicionado em `pose` ao lado de A.
    + paredes de B paralelas e coladas (5–40 cm) às de A, com faces opostas
    - cômodos sobrepostos"""
    why = []
    fa, fb = _floor(sa), _floor(sb)
    overlap = 0.0
    if fa and fb:
        overlap = _poly_overlap(fa, [to_world(p, pose) for p in fb])
    s = -10.0 * overlap
    if overlap > 0.05:
        why.append(f"cômodos se sobrepõem {overlap:.0%}")
    shared = 0.0
    wa = [wall_geom(sa, w.id) for w in sa.entities if isinstance(w, Wall)]
    for w in sb.entities:
        if not isinstance(w, Wall):
            continue
        b0, b1, db, nb = wall_geom(sb, w.id)
        B0, B1, nbw = to_world(b0, pose), to_world(b1, pose), rot(pose[2]) @ nb
        for a0, a1, da, na in wa:
            if na @ nbw > -0.995:
                continue
            gap = [(q - a0) @ na for q in (B0, B1)]
            if not all(0.04 < g < 0.42 for g in gap) or abs(gap[0] - gap[1]) > 0.03:
                continue
            ta = sorted([0.0, (a1 - a0) @ da])
            tb = sorted([(B0 - a0) @ da, (B1 - a0) @ da])
            ov = min(ta[1], tb[1]) - max(ta[0], tb[0])
            if ov > 0.2:
                shared += ov
    s += shared
    if shared:
        why.append(f"{shared:.2f} m de parede compartilhada coerente")
    # paredes alinhadas (fachada/corredor contínuos): mesma face, mesma linha, pontas próximas
    aligned = 0
    for w in sb.entities:
        if not isinstance(w, Wall):
            continue
        b0, b1, db, nb = wall_geom(sb, w.id)
        B0, B1, nbw = to_world(b0, pose), to_world(b1, pose), rot(pose[2]) @ nb
        for a0, a1, da, na in wa:
            if na @ nbw < 0.995:
                continue
            off = max(abs((q - a0) @ na) for q in (B0, B1))
            near = min(np.linalg.norm(p - q) for p in (a0, a1) for q in (B0, B1))
            if off < 0.03 and near < 0.5:
                aligned += 1
                s += 0.5 * (1 - off / 0.03)
    if aligned:
        why.append(f"{aligned} parede(s) alinhada(s) entre os cômodos")
    return s, why


# ------------------------------------------------------------------ marcadores compartilhados

def auto_marker_links(b: Building, scenes: dict[str, SceneGraph]) -> list[str]:
    """Marcador ArUco (mesmo ID) visto em dois levantamentos = mesmo ponto físico: cria ligações
    point_match automáticas (encadeadas na ordem das capturas). Não recria as que o usuário apagou
    e não mexe nas que ele desligou."""
    info = []
    seen: dict[str, list[str]] = {}
    for c in b.captures:
        for mid in scenes[c.id].markers:
            seen.setdefault(mid, []).append(c.id)
    have = {lk.auto_key for lk in b.links if lk.auto_key}
    for mid, caps in sorted(seen.items(), key=lambda kv: int(kv[0])):
        if len(caps) > 2:
            info.append(f"Marcador {mid} aparece em {len(caps)} levantamentos: use IDs diferentes por "
                        "porta; ligado em cadeia mesmo assim.")
        for ca, cb in zip(caps, caps[1:]):
            key = f"aruco:{mid}:{ca}:{cb}"
            if key in have or key in b.ignored_auto:
                continue
            pa, pb = scenes[ca].markers[mid], scenes[cb].markers[mid]
            b.links.append(Link(kind="point_match", a=Ref(capture=ca, point=(pa[0], pa[1])),
                                b=Ref(capture=cb, point=(pb[0], pb[1])), auto_key=key,
                                note=f"marcador ArUco {mid} (automático)"))
            info.append(f"Ligação automática pelo marcador {mid}: {b.cap(ca).name} ↔ {b.cap(cb).name}.")
            dz = abs(pa[2] - pb[2])
            if dz > 0.10:
                info.append(f"Marcador {mid}: altura difere {dz * 100:.0f} cm entre os levantamentos "
                            "(escala ou piso incoerente?).")
    return info


# ------------------------------------------------------------------ planta unificada

def merge(b: Building, scenes: dict[str, SceneGraph], res: SolveResult | None = None) -> tuple[SceneGraph, list[str]]:
    poses = res.poses if res else {c.id: (c.pose.x, c.pose.y, np.radians(c.pose.theta_deg)) for c in b.captures}
    out = SceneGraph(scale_source="user_measure")
    warn = []
    for c in b.captures:
        sg, pose = scenes[c.id], poses[c.id]
        R = rot(pose[2])
        pid = lambda e: f"{c.id}:{e}"
        spaces = [e for e in sg.entities if isinstance(e, Space)]
        for e in sg.entities:
            e = e.model_copy(deep=True)
            e.id = pid(e.id)
            if isinstance(e, (Wall, Separator)):
                e.start, e.end = tuple(to_world(e.start, pose)), tuple(to_world(e.end, pose))
            elif isinstance(e, Opening):
                e.host_wall = pid(e.host_wall)
            elif isinstance(e, (Slab, Space)):
                e.polygon = [tuple(to_world(p, pose)) for p in e.polygon]
            out.entities.append(e)
        if not spaces:  # sem divisão em cômodos: o levantamento inteiro vira um cômodo com o nome da captura
            fl = _floor(sg)
            if fl:
                from .rooms import _signed_area
                poly = [tuple(to_world(p, pose)) for p in fl]
                a = abs(_signed_area(poly))
                per = float(sum(np.linalg.norm(np.subtract(poly[i], poly[(i + 1) % len(poly)])) for i in range(len(poly))))
                out.entities.append(Space(id=f"{c.id}:space", name=c.name, polygon=poly, area=round(a, 3),
                                          net_area=round(a, 3), perimeter=round(per, 3)))
    warn += _merge_shared_walls(out)
    _dedupe_doors(b, out)
    return out, warn


def _merge_shared_walls(sg: SceneGraph) -> list[str]:
    """Paredes de capturas diferentes, paralelas, com faces opostas a 4–42 cm: mesma parede física.
    Cada uma passa a ter a espessura medida (as duas desenham a mesma faixa)."""
    from .exporters.plan2d import _center
    walls = [e for e in sg.entities if isinstance(e, Wall)]
    centers = {}
    for w in walls:  # centro do cômodo de origem (para a normal externa)
        cap = w.id.split(":")[0]
        fl = [e for e in sg.entities if isinstance(e, Slab) and e.role == "floor" and e.id.startswith(cap + ":")]
        centers[w.id] = np.mean(fl[0].polygon, axis=0) if fl else _center(sg)
    info = []
    for i, wa in enumerate(walls):
        for wb in walls[i + 1:]:
            if wa.id.split(":")[0] == wb.id.split(":")[0]:
                continue
            a0, a1 = np.array(wa.start), np.array(wa.end)
            b0, b1 = np.array(wb.start), np.array(wb.end)
            da = (a1 - a0) / np.linalg.norm(a1 - a0)
            na = np.array([-da[1], da[0]])
            if ((a0 + a1) / 2 - centers[wa.id]) @ na < 0:
                na = -na
            db = (b1 - b0) / np.linalg.norm(b1 - b0)
            if abs(da @ db) < 0.999:
                continue
            gaps = [(q - a0) @ na for q in (b0, b1)]
            if not all(0.04 < g < 0.42 for g in gaps):
                continue
            ov = min(max(0, (a1 - a0) @ da), max((b0 - a0) @ da, (b1 - a0) @ da)) - \
                max(0, min((b0 - a0) @ da, (b1 - a0) @ da))
            if ov < 0.2:
                continue
            t = float(np.mean(gaps))
            wa.thickness = wb.thickness = round(t, 3)
            wb.shared_with = wa.id
            info.append(f"Parede compartilhada {wa.id} / {wb.id}: espessura medida {t * 100:.1f} cm.")
    return info


def _dedupe_doors(b: Building, sg: SceneGraph) -> None:
    """A mesma porta vista dos dois lados fica uma vez só nas contagens (a do lado B vira 'inferred')."""
    for lk in b.links:
        if lk.enabled and lk.kind == "door_match":
            o = sg.get(f"{lk.b.capture}:{lk.b.element}")
            if o is not None:
                o.duplicate_of = f"{lk.a.capture}:{lk.a.element}"
