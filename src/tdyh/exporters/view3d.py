"""Cena 3D para o visualizador web, gerada do grafo de cena final.

Formato leve (JSON) com primitivas que o navegador transforma em malha:
- "box":   caixa orientada — canto `o` (x, y, z), eixos horizontais `u` e `v` (vetores 2D com o
           comprimento do lado) e altura `h`. As paredes já vêm recortadas nos vãos (várias caixas).
- "prism": polígono 2D extrudado de `z0` a `z1` (lajes, cômodos).
Cada elemento leva tipo, nome, origem da medida (medida/inferida/usuário), confiança e parâmetros,
para o painel de propriedades. Coordenadas da planta, em metros, Z para cima.
"""

from __future__ import annotations

import numpy as np

from ..scenegraph import Opening, SceneGraph, Separator, Slab, Space, Wall
from .ifc import SOURCE_PT, _floors, _prefix, _rehost, _wall_center
from .plan2d import _l_corner, _outward, axis_flags


def _r(x, n=4):
    return [round(float(v), n) for v in x]


def _box(origin2d, z0, u2d, v2d, h):
    return {"t": "box", "o": _r([origin2d[0], origin2d[1], z0]), "u": _r(u2d), "v": _r(v2d), "h": round(float(h), 4)}


def _solid_spans(H: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Intervalos de altura sólidos em [0, H] tirando os vãos."""
    spans, z = [], 0.0
    for a, b in sorted(holes):
        if a > z + 1e-6:
            spans.append((z, a))
        z = max(z, b)
    if z < H - 1e-6:
        spans.append((z, H))
    return spans


def wall_boxes(L: float, e0: float, e1: float, H: float, openings: list[tuple[float, float, float, float]]):
    """Parede como caixas (no sistema local: x ao longo, z para cima), recortada nos vãos.
    openings: (x0, x1, z0, z1). Devolve [(x0, x1, z0, z1)] sólidos."""
    xs = sorted({-e0, L + e1, *[max(-e0, min(L + e1, v)) for o in openings for v in o[:2]]})
    out = []
    for a, b in zip(xs, xs[1:]):
        if b - a < 1e-6:
            continue
        m = (a + b) / 2
        holes = [(o[2], o[3]) for o in openings if o[0] < m < o[1]]
        for z0, z1 in _solid_spans(H, holes):
            out.append((a, b, z0, z1))
    # junta pedaços vizinhos com a mesma faixa de altura (menos caixas)
    merged = []
    for p in out:
        if merged and abs(merged[-1][1] - p[0]) < 1e-9 and merged[-1][2:] == p[2:]:
            merged[-1] = (merged[-1][0], p[1], *p[2:])
        else:
            merged.append(p)
    return merged


def _meta(e, sg: SceneGraph) -> dict:
    return {"id": e.id, "source": SOURCE_PT.get(e.source, e.source), "confidence": round(float(e.confidence), 3),
            "adjusted": e.id in sg.overrides}


def build_view(sg: SceneGraph, title: str = "") -> dict:
    floors = _floors(sg)
    walls = [e for e in sg.entities if isinstance(e, Wall)]
    by_id = {w.id: w for w in walls}
    kept = [w for w in walls if not (w.shared_with and w.shared_with in by_id)]
    ax = axis_flags(sg)
    ops: dict[str, list[Opening]] = {}
    for o in (e for e in sg.entities if isinstance(e, Opening)):
        if o.duplicate_of or o.host_wall not in by_id:
            continue
        h = by_id[o.host_wall]
        if h not in kept and h.shared_with in by_id:
            o = _rehost(o, h, by_id[h.shared_with])
        ops.setdefault(o.host_wall, []).append(o)

    elems, lo_pt, hi_pt = [], np.full(3, np.inf), np.full(3, -np.inf)
    for w in kept:
        a, b = np.array(w.start, float), np.array(w.end, float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        d = (b - a) / L
        nu = _outward(w, _wall_center(sg, w, floors))
        lo, hi = (-w.thickness / 2, w.thickness / 2) if ax[w.id] else (0.0, w.thickness)
        e0, e1 = _l_corner(w, "start", walls), _l_corner(w, "end", walls)
        mine = ops.get(w.id, [])
        holes = [(o.offset, o.offset + o.width, o.sill, min(o.sill + o.height, w.height)) for o in mine]
        prims = [_box(a + d * x0 + nu * lo, z0, d * (x1 - x0), nu * (hi - lo), z1 - z0)
                 for x0, x1, z0, z1 in wall_boxes(L, e0, e1, w.height, holes)]
        shared = [o.id for o in walls if o.shared_with == w.id]
        elems.append({**_meta(w, sg), "kind": "wall", "name": f"Parede {w.id}", "prims": prims,
                      "params": {"comprimento": round(L, 3), "espessura": w.thickness, "altura": w.height,
                                 "espessura_lado": "centralizada" if ax[w.id] else "para fora",
                                 **({"compartilhada_com": ", ".join(shared)} if shared else {})}})
        for o in mine:
            door = o.kind.value == "door"
            pw = 0.04 if door else 0.02
            mid = (lo + hi) / 2
            top = min(o.sill + o.height, w.height)
            prim = _box(a + d * o.offset + nu * (mid - pw / 2), o.sill, d * o.width, nu * pw, top - o.sill)
            dup = [x.id for x in sg.entities if isinstance(x, Opening) and x.duplicate_of == o.id]
            elems.append({**_meta(o, sg), "kind": "door" if door else "window",
                          "name": f"{'Porta' if door else 'Janela'} {o.id}", "prims": [prim],
                          "params": {"largura": o.width, "altura": o.height, "peitoril": o.sill,
                                     "parede": o.host_wall,
                                     "espessura_do_vao": o.depth if o.depth is not None else "não determinada",
                                     **({"mesma_porta_que": ", ".join(dup)} if dup else {})}})
        for p in (a, b):
            lo_pt = np.minimum(lo_pt, [*p, 0])
            hi_pt = np.maximum(hi_pt, [*p, w.height])

    for s in (e for e in sg.entities if isinstance(e, Slab)):
        if len(s.polygon) < 3:
            continue
        fl = s.role == "floor"
        z0 = s.elevation - s.thickness if fl else s.elevation
        elems.append({**_meta(s, sg), "kind": "floor" if fl else "ceiling",
                      "name": f"{'Piso' if fl else 'Teto'} {s.id}",
                      "prims": [{"t": "prism", "poly": [_r(p) for p in s.polygon], "z0": round(z0, 4),
                                 "z1": round(z0 + s.thickness, 4)}],
                      "params": {"cota": s.elevation, "espessura": s.thickness}})
    ceil = {_prefix(s.id): s.elevation for s in sg.entities if isinstance(s, Slab) and s.role == "ceiling"}
    hdef = max((w.height for w in walls), default=2.7)
    for sp in (e for e in sg.entities if isinstance(e, Space)):
        h = ceil.get(_prefix(sp.id), hdef)
        elems.append({**_meta(sp, sg), "kind": "space", "name": sp.name,
                      "prims": [{"t": "prism", "poly": [_r(p) for p in sp.polygon], "z0": 0.0, "z1": round(h, 4)}],
                      "params": {"uso": sp.use or "—", "area": sp.area, "area_util": sp.net_area,
                                 "perimetro": sp.perimeter}})
    for s in (e for e in sg.entities if isinstance(e, Separator)):
        elems.append({**_meta(s, sg), "kind": "separator", "name": f"Divisória {s.id}",
                      "prims": [{"t": "line", "a": _r([*s.start, 0.01]), "b": _r([*s.end, 0.01])}], "params": {}})
    if not np.isfinite(lo_pt).all():
        lo_pt, hi_pt = np.zeros(3), np.array([1.0, 1.0, 2.7])
    return {"title": title, "units": "m", "bounds": [_r(lo_pt, 3), _r(hi_pt, 3)],
            "scale": {"source": sg.scale_source, "uncertainty": sg.scale_uncertainty},
            "markers": {k: _r(v) for k, v in sg.markers.items()},
            "elements": elems}
