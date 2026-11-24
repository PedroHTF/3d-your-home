"""Cena sintética difícil: porta junto ao canto, janela com peitoril, porta fechada,
guarda-roupa encostado e um trecho de parede não filmado."""

import numpy as np
import pytest

from tdyh import layout
from tdyh.scenegraph import Opening, Wall

W, D, H = 4.20, 3.10, 2.70
DOOR = dict(wall="y0", a=0.10, b=0.90, z0=0.0, z1=2.10)           # junto ao canto x=0
WINDOW = dict(wall="xW", a=1.00, b=2.20, z0=1.00, z1=2.10)        # peitoril 1,0 m
CLOSED = dict(wall="yD", a=2.50, b=3.30, z0=0.0, z1=2.10)         # folha 4 cm atrás da face
UNSEEN = dict(wall="yD", a=0.30, b=1.00, z0=1.00, z1=2.00)        # não filmado
WARDROBE = dict(x0=0.0, x1=0.60, y0=0.80, y1=2.00, z1=2.00)
CAMS = np.array([[x, y, 1.4] for x in (1.0, 2.1, 3.2) for y in (0.8, 1.55, 2.3)])


def _in(r, u, z):
    return (u > r["a"]) & (u < r["b"]) & (z > r["z0"]) & (z < r["z1"])


def _wall(rng, dens, length, holes, to3d):
    p = rng.random((int(length * H * dens), 2)) * [length, H]
    keep = np.ones(len(p), bool)
    for r in holes:
        keep &= ~_in(r, p[:, 0], p[:, 1])
    p = p[keep]
    return to3d(p[:, 0], p[:, 1])


def _exit_crossing(C, P):
    """Onde o raio C->P sai do cômodo: (nome da parede, coordenada ao longo, z)."""
    best = (2.0, None)
    for name, axis, val in (("x0", 0, 0.0), ("xW", 0, W), ("y0", 1, 0.0), ("yD", 1, D)):
        den = P[axis] - C[axis]
        if abs(den) < 1e-9:
            continue
        s = (val - C[axis]) / den
        if 0 < s < best[0]:
            X = C + s * (P - C)
            best = (s, (name, X[1 - axis], X[2]))
    return best[1]


def make_scene(seed=0, dens=900, noise=0.003, neighbor_behind_wardrobe=False):
    rng = np.random.default_rng(seed)
    parts = []
    u = rng.random((int(W * D * dens), 2)) * [W, D]
    parts += [np.c_[u, np.zeros(len(u))], np.c_[u, np.full(len(u), H)]]
    parts.append(_wall(rng, dens, W, [DOOR], lambda a, z: np.c_[a, np.zeros_like(a), z]))
    parts.append(_wall(rng, dens, W, [CLOSED, UNSEEN], lambda a, z: np.c_[a, np.full_like(a, D), z]))
    parts.append(_wall(rng, dens, D, [WINDOW], lambda a, z: np.c_[np.full_like(a, W), a, z]))
    hidden = dict(a=WARDROBE["y0"], b=WARDROBE["y1"], z0=0, z1=WARDROBE["z1"])
    parts.append(_wall(rng, dens, D, [hidden], lambda a, z: np.c_[np.zeros_like(a), a, z]))
    # folha da porta fechada, 4 cm para fora
    q = rng.random((int(0.8 * 2.1 * dens), 2)) * [0.8, 2.1] + [CLOSED["a"], 0]
    parts.append(np.c_[q[:, 0], np.full(len(q), D + 0.04), q[:, 1]])
    # guarda-roupa: frente, laterais, topo
    wr = WARDROBE
    f = rng.random((int(1.2 * 2.0 * dens), 2)) * [1.2, 2.0] + [wr["y0"], 0]
    parts.append(np.c_[np.full(len(f), wr["x1"]), f])
    for y in (wr["y0"], wr["y1"]):
        sd = rng.random((int(0.6 * 2.0 * dens), 2)) * [0.6, 2.0]
        parts.append(np.c_[sd[:, 0], np.full(len(sd), y), sd[:, 1]])
    tp = rng.random((int(0.6 * 1.2 * dens), 2)) * [0.6, 1.2] + [0, wr["y0"]]
    parts.append(np.c_[tp, np.full(len(tp), wr["z1"])])
    room = np.vstack(parts)

    # lado de fora: corredor (atrás da porta) e fachada vizinha (atrás da janela)
    out = []
    c = rng.random((20000, 2)) * [6, 3.5] + [-1, 0]
    out.append(np.c_[c[:, 0], np.full(len(c), -1.5), c[:, 1]])
    cf = rng.random((8000, 2)) * [6, 1.5] + [-1, -1.5]
    out.append(np.c_[cf, np.zeros(len(cf))])
    fa = rng.random((20000, 2)) * [6, 5] + [-1.5, 0]
    out.append(np.c_[np.full(len(fa), W + 3), fa])
    outside = np.vstack(out)
    openings = {"y0": [DOOR], "xW": [WINDOW], "yD": [], "x0": []}
    pairs, keep = [], []
    for i, P in enumerate(outside):
        seen = False
        for j, C in enumerate(CAMS):
            hit = _exit_crossing(C, P)
            if hit and any(_in(r, np.array(hit[1]), np.array(hit[2])) for r in openings[hit[0]]):
                pairs.append((len(room) + len(keep), j))
                seen = True
        if seen:
            keep.append(i)
    outside = outside[keep]
    extra = []
    if neighbor_behind_wardrobe:  # pontos que só um raio "errado" atravessaria
        nb = rng.random((6000, 2)) * [3.1, 2.7]
        extra.append(np.c_[np.full(len(nb), -2.0), nb])
    pts = np.vstack([room, outside, *extra])
    return pts + rng.normal(0, noise, pts.shape), np.array(pairs)


def _openings(res):
    out = []
    for o in res.scene.entities:
        if isinstance(o, Opening):
            w = res.scene.get(o.host_wall)
            a, b = np.array(w.start), np.array(w.end)
            d = (b - a) / np.linalg.norm(b - a)
            p0, p1 = a + d * o.offset, a + d * (o.offset + o.width)
            out.append((o, p0, p1))
    return out


def _find(ops, x, y):
    """Abertura cujo centro está perto de (x, y) no sistema original."""
    for o, p0, p1 in ops:
        if np.linalg.norm((p0 + p1) / 2 - [x, y]) < 0.15:
            return o
    return None


@pytest.fixture(scope="module")
def scene_exact():
    pts, pairs = make_scene()
    # sistema já alinhado (sem rotação) para comparar posições diretamente
    return layout.build_layout(pts, up_hint=np.array([0, 0, 1.0]), cameras=CAMS, vis_pairs=pairs)


def test_walls_ignore_wardrobe(scene_exact):
    L = sorted(np.linalg.norm(np.subtract(w.end, w.start)) for w in scene_exact.scene.entities
               if isinstance(w, Wall))
    assert len(L) == 4, L
    assert np.allclose(L, [D, D, W, W], atol=0.03)


def test_door_next_to_corner(scene_exact):
    o = _find(_openings(scene_exact), 0.50, 0.0)
    assert o is not None and o.kind.value == "door"
    assert abs(o.width - 0.80) < 0.08 and abs(o.height - 2.10) < 0.08


def test_window_with_sill(scene_exact):
    o = _find(_openings(scene_exact), W, 1.60)
    assert o is not None and o.kind.value == "window", _openings(scene_exact)
    assert abs(o.width - 1.20) < 0.08 and abs(o.sill - 1.00) < 0.08 and abs(o.height - 1.10) < 0.1


def test_closed_door_low_confidence(scene_exact):
    o = _find(_openings(scene_exact), 2.90, D)
    assert o is not None and o.kind.value == "door" and o.confidence < 0.5 and o.source == "inferred"


def test_no_false_openings(scene_exact):
    ops = _openings(scene_exact)
    assert _find(ops, 0.0, 1.40) is None       # atrás do guarda-roupa
    assert _find(ops, 0.65, D) is None         # trecho não filmado
    assert len(ops) == 3, [(o.kind, o.offset, o.width) for o, *_ in ops]
    assert any("não observada" in n for n in scene_exact.notes)


def test_fallback_without_visibility_blocks_occluded_rays():
    pts, _ = make_scene(seed=1, neighbor_behind_wardrobe=True)
    res = layout.build_layout(pts, up_hint=np.array([0, 0, 1.0]), cameras=CAMS)
    ops = _openings(res)
    assert _find(ops, 0.0, 1.40) is None       # raios bloqueados pelo guarda-roupa
    assert _find(ops, 0.50, 0.0) is not None and _find(ops, W, 1.60) is not None


def test_no_cameras_no_openings():
    pts, _ = make_scene(seed=2)
    res = layout.build_layout(pts, up_hint=np.array([0, 0, 1.0]))
    assert not _openings(res) and any("Sem poses" in n for n in res.notes)


def test_floor_area_excludes_outside(scene_exact):
    from tdyh.exporters import plan2d
    floor = next(e for e in scene_exact.scene.entities if getattr(e, "role", "") == "floor")
    assert abs(plan2d._area(floor.polygon) - W * D) < 0.1  # corredor visto pela porta não entra
