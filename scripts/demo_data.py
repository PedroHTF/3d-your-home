"""Dados de demonstração para o visualizador (sem captura real).

Uso: TDYH_DATA=pasta python scripts/demo_data.py
Cria: um levantamento "Apartamento (demo)" — sala e quarto com divisória desenhada pelo usuário,
porta, janela inferida e nuvem de pontos sintética — e uma edificação "Casa (demo)" juntando dois
levantamentos por uma porta."""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from tdyh import building as bd  # noqa: E402
from tdyh import pipeline  # noqa: E402
from tdyh.scenegraph import Opening, RoomLabel, SceneGraph, Slab, Wall  # noqa: E402

DATA = Path(os.environ.get("TDYH_DATA", "data"))


def project(pid: str, name: str, sg: SceneGraph, points: np.ndarray | None = None) -> None:
    d = DATA / "projects" / pid
    (d / "work").mkdir(parents=True, exist_ok=True)
    (d / "project.json").write_text(json.dumps({"id": pid, "name": name}))
    (d / "work" / "scene_measured.json").write_text(sg.model_dump_json(indent=2))
    if points is not None:
        pipeline.save_view_points(points, np.eye(3), 1.0, 0.0, d / "work")
    pipeline.publish(d / "work")


def room_cloud(W, D, H, rng, n=60000, holes=()):
    """Pontos nas paredes, piso e teto de uma caixa W x D x H (com ruído), sem os vãos."""
    k = rng.integers(0, 6, n)
    u, v = rng.random(n), rng.random(n)
    pts = np.select(
        [k[:, None] == 0, k[:, None] == 1, k[:, None] == 2, k[:, None] == 3, k[:, None] == 4],
        [np.c_[u * W, 0 * u, v * H], np.c_[u * W, 0 * u + D, v * H], np.c_[0 * u, u * D, v * H],
         np.c_[0 * u + W, u * D, v * H], np.c_[u * W, v * D, 0 * u]],
        np.c_[u * W, v * D, 0 * u + H])
    keep = np.ones(n, bool)
    for (x0, x1, y0, y1, z0, z1) in holes:
        keep &= ~((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
                  & (pts[:, 2] >= z0) & (pts[:, 2] <= z1))
    return pts[keep] + rng.normal(0, 0.006, (keep.sum(), 3))


def main():
    rng = np.random.default_rng(1)
    c = [(0, 0), (7, 0), (7, 4), (0, 4)]
    ents = [Wall(id=f"w{i}", start=c[i], end=c[(i + 1) % 4], height=2.6) for i in range(4)]
    ents += [Opening(id="d1", kind="door", host_wall="w0", offset=1.0, width=0.9, height=2.1, depth=0.15),
             Opening(id="j1", kind="window", host_wall="w2", offset=1.0, width=1.5, height=1.2, sill=1.0,
                     source="inferred", confidence=0.4),
             Opening(id="j2", kind="window", host_wall="w1", offset=1.2, width=1.2, height=1.1, sill=1.0),
             Slab(id="piso", polygon=c), Slab(id="teto", polygon=c, elevation=2.6, role="ceiling")]
    sg = SceneGraph(entities=ents, scale_source="marker", scale_uncertainty=0.005, rooms_enabled=True,
                    markers={"3": (0.6, 0.0, 0.01), "5": (6.4, 4.0, 0.01)},
                    room_labels=[RoomLabel(point=(2, 2), name="Sala", use="estar"),
                                 RoomLabel(point=(6, 2), name="Quarto", use="dormitório")])
    sg.added += [Wall(id="div", start=(5.0, 0.0), end=(5.0, 4.0), source="user", thickness=0.10, height=2.6),
                 Opening(id="d2", kind="door", host_wall="div", offset=1.5, width=0.8, height=2.1, source="user")]
    pts = room_cloud(7, 4, 2.6, rng, holes=[(1.0, 1.9, -0.1, 0.1, 0, 2.1), (5.0, 6.5, 3.9, 4.1, 1.0, 2.2),
                                            (6.9, 7.1, 1.2, 2.4, 1.0, 2.1)])
    project("demo-apto", "Apartamento (demo)", sg, pts)

    from test_building import two_rooms
    b, sc = two_rooms()
    for k, s in sc.items():
        project(f"demo-{k.lower()}", f"Cômodo {k} (demo)", s)
    b.id, b.name = "demo-casa", "Casa (demo)"
    b.captures = [bd.Capture(id=c.id, name=c.name, project_id=f"demo-{c.id.lower()}") for c in b.captures]
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dAB"),
                       b=bd.Ref(capture="B", element="dBA"), thickness=0.15)]
    merged, _ = bd.merge(b, sc, bd.solve_poses(b, sc))
    (DATA / "buildings" / b.id).mkdir(parents=True, exist_ok=True)
    (DATA / "buildings" / f"{b.id}.json").write_text(b.model_dump_json(indent=2))
    (DATA / "buildings" / b.id / "scene.json").write_text(merged.model_dump_json(indent=2))
    print("ok:", DATA)


if __name__ == "__main__":
    main()
