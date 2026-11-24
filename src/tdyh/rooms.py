"""Divisão em cômodos: faces fechadas do grafo plano formado por paredes e linhas divisórias.

1. Todas as linhas (paredes + divisórias) são quebradas nos cruzamentos e junções em T.
2. Pontas soltas são podadas (parede que não fecha nada não cria cômodo).
3. Percorrendo as semi-arestas sempre "virando à esquerda" obtêm-se as faces; as de área positiva
   são os cômodos (a face externa tem área negativa).
4. Os nomes vêm de rótulos presos a um ponto: o cômodo que contém o ponto recebe o nome.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .scenegraph import SceneGraph, Separator, Space, Wall

TOL = 0.03


def _split_points(segs: list[tuple[np.ndarray, np.ndarray]]) -> list[list[float]]:
    """Para cada segmento, parâmetros t em [0,1] onde ele deve ser quebrado."""
    cuts = [[0.0, 1.0] for _ in segs]
    for i, (a, b) in enumerate(segs):
        d = b - a
        L2 = d @ d
        for j, (c, e) in enumerate(segs):
            if i == j:
                continue
            for q in (c, e):  # junção em T: ponta de j sobre o meio de i
                t = (q - a) @ d / L2
                if 0 < t < 1 and np.linalg.norm(a + t * d - q) < TOL:
                    cuts[i].append(float(t))
            f = e - c  # cruzamento próprio
            den = d[0] * f[1] - d[1] * f[0]
            if abs(den) > 1e-12:
                w = c - a
                t = (w[0] * f[1] - w[1] * f[0]) / den
                u = (w[0] * d[1] - w[1] * d[0]) / den
                if 1e-6 < t < 1 - 1e-6 and 1e-6 < u < 1 - 1e-6:
                    cuts[i].append(float(t))
    return [sorted(set(np.round(c, 9))) for c in cuts]


def planar_faces(segs: list[tuple[np.ndarray, np.ndarray]]) -> list[list[np.ndarray]]:
    nodes: list[np.ndarray] = []

    def node(p):
        for k, q in enumerate(nodes):
            if np.linalg.norm(p - q) < TOL:
                return k
        nodes.append(p)
        return len(nodes) - 1

    adj = defaultdict(set)
    for (a, b), ts in zip(segs, _split_points(segs)):
        ids = [node(a + t * (b - a)) for t in ts]
        for u, v in zip(ids, ids[1:]):
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
    # poda pontas soltas
    changed = True
    while changed:
        changed = False
        for v in [v for v, nb in adj.items() if len(nb) == 1]:
            (u,) = adj[v]
            adj[u].discard(v)
            del adj[v]
            changed = True
    P = np.array(nodes) if nodes else np.zeros((0, 2))
    order = {v: sorted(nb, key=lambda u: np.arctan2(*(P[u] - P[v])[::-1])) for v, nb in adj.items()}
    seen, faces = set(), []
    for u in adj:
        for v in adj[u]:
            if (u, v) in seen:
                continue
            face, a, b = [], u, v
            while (a, b) not in seen:
                seen.add((a, b))
                face.append(b)
                nb = order[b]
                k = nb.index(a)
                a, b = b, nb[(k - 1) % len(nb)]  # vizinho imediatamente horário = vira à esquerda
            poly = [P[i] for i in face]
            if len(poly) >= 3 and _signed_area(poly) > 1e-4:
                faces.append(poly)
    return faces


def _signed_area(poly) -> float:
    p = np.array(poly)
    x, y = p[:, 0], p[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _inside(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def interior_point(poly) -> np.ndarray:
    """Ponto dentro do polígono para o rótulo (centroide se estiver dentro; senão varredura)."""
    p = np.array(poly)
    A = _signed_area(poly)
    x, y = p[:, 0], p[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    c = np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * A)
    if _inside(c, poly):
        return c
    best, bd = c, -1
    lo, hi = p.min(axis=0), p.max(axis=0)
    for gx in np.linspace(lo[0], hi[0], 25)[1:-1]:
        for gy in np.linspace(lo[1], hi[1], 25)[1:-1]:
            q = np.array([gx, gy])
            if _inside(q, poly):
                d = min(_dist_seg(q, p[i], p[(i + 1) % len(p)]) for i in range(len(p)))
                if d > bd:
                    best, bd = q, d
    return best


def _dist_seg(q, a, b) -> float:
    d = b - a
    t = np.clip((q - a) @ d / (d @ d), 0, 1)
    return float(np.linalg.norm(q - (a + t * d)))


def build_spaces(sg: SceneGraph) -> list[str]:
    lines = [e for e in sg.entities if isinstance(e, (Wall, Separator))]
    segs = [(np.array(e.start, float), np.array(e.end, float)) for e in lines]
    faces = planar_faces(segs)
    warn = []
    if not faces:
        return ["Divisão em cômodos ligada, mas as paredes não formam nenhum ambiente fechado."]
    # paredes internas: aparecem em 2 faces -> desconta metade da espessura de cada lado
    walls = [e for e in sg.entities if isinstance(e, Wall)]
    used_labels = set()
    spaces = []
    for k, poly in enumerate(sorted(faces, key=lambda f: -_signed_area(f))):
        area = _signed_area(poly)
        per = sum(np.linalg.norm(poly[i] - poly[(i + 1) % len(poly)]) for i in range(len(poly)))
        deduct = 0.0
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            mid = (a + b) / 2
            for w in walls:
                if _dist_seg(mid, np.array(w.start), np.array(w.end)) < TOL and _is_internal(w, faces):
                    deduct += np.linalg.norm(b - a) * w.thickness / 2
                    break
        labels = [lb for lb in sg.room_labels if _inside(lb.point, poly)]
        if len(labels) > 1:
            warn.append(f"Cômodo com {len(labels)} nomes ({', '.join(lb.name for lb in labels)}): usando '{labels[0].name}'.")
        lb = labels[0] if labels else None
        if lb:
            used_labels.add(lb.id)
        spaces.append(Space(name=lb.name if lb else f"Cômodo {k + 1}",
                            polygon=[tuple(map(float, q)) for q in poly],
                            area=round(area, 3), net_area=round(area - deduct, 3), perimeter=round(per, 3),
                            label_id=lb.id if lb else None, use=lb.use if lb else None))
    for lb in sg.room_labels:
        if lb.id not in used_labels:
            warn.append(f"Nome '{lb.name}' está fora de qualquer cômodo (o ambiente pode ter sido alterado).")
    sg.entities = [e for e in sg.entities if not isinstance(e, Space)] + spaces
    return warn


def _is_internal(w: Wall, faces) -> bool:
    mid = (np.array(w.start) + np.array(w.end)) / 2
    d = np.subtract(w.end, w.start)
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * 0.05
    return _inside(mid + n, _f(faces, mid + n)) and _inside(mid - n, _f(faces, mid - n))


def _f(faces, q):
    for f in faces:
        if _inside(q, f):
            return f
    return [(1e9, 1e9), (1e9 + 1, 1e9), (1e9, 1e9 + 1)]
