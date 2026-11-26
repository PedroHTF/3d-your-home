"""Visualizador: cena 3D gerada do grafo de cena (paredes recortadas nos vãos, volumes conferidos),
endpoints da API, nuvem de pontos reescalada após recalibração e build servido em /viewer/."""

import json

import numpy as np
import pytest

from tdyh.constraints import rebuild
from tdyh.exporters.view3d import build_view, wall_boxes
from tdyh.scenegraph import Constraint, Opening, SceneGraph, Slab, Wall


def box_vol(p):
    u, v = np.array(p["u"]), np.array(p["v"])
    return abs(u[0] * v[1] - u[1] * v[0]) * p["h"]


def sample():
    c = [(0, 0), (5, 0), (5, 3), (0, 3)]
    ents = [Wall(id=f"w{i}", start=c[i], end=c[(i + 1) % 4], height=2.6) for i in range(4)]
    ents += [Opening(id="p", kind="door", host_wall="w0", offset=1.0, width=0.9, height=2.1),
             Opening(id="j", kind="window", host_wall="w0", offset=3.0, width=1.2, height=1.0, sill=1.1,
                     source="inferred", confidence=0.4),
             Slab(id="piso", polygon=c), Slab(id="teto", polygon=c, elevation=2.6, role="ceiling")]
    return SceneGraph(entities=ents, scale_source="marker", rooms_enabled=True)


def test_wall_boxes_partition():
    b = wall_boxes(5.0, 0.15, 0.15, 2.6, [(1.0, 1.9, 0.0, 2.1), (3.0, 4.2, 1.1, 2.1)])
    vol = sum((x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in b)
    assert vol == pytest.approx(5.3 * 2.6 - 0.9 * 2.1 - 1.2 * 1.0)
    # nenhum pedaço dentro de um vão
    for x0, x1, z0, z1 in b:
        m, zm = (x0 + x1) / 2, (z0 + z1) / 2
        assert not (1.0 < m < 1.9 and zm < 2.1) and not (3.0 < m < 4.2 and 1.1 < zm < 2.1)
    assert wall_boxes(3.0, 0, 0, 2.5, []) == [(0.0, 3.0, 0.0, 2.5)]


def test_view_elements_and_volumes():
    final, _ = rebuild(sample())
    v = build_view(final, "Teste")
    by = {e["id"]: e for e in v["elements"]}
    w0 = by["w0"]
    assert w0["kind"] == "wall" and w0["source"] == "medida"
    vol = sum(box_vol(p) for p in w0["prims"])
    assert vol == pytest.approx((5.3 * 2.6 - 0.9 * 2.1 - 1.2 * 1.0) * 0.15, rel=1e-3)
    # espessura para FORA (y <= 0)
    for p in w0["prims"]:
        ys = [p["o"][1], p["o"][1] + p["v"][1]]
        assert max(ys) <= 1e-9 and min(ys) == pytest.approx(-0.15)
    j = by["j"]
    assert j["kind"] == "window" and j["source"] == "inferida" and j["confidence"] == 0.4
    assert j["prims"][0]["o"][2] == pytest.approx(1.1) and j["prims"][0]["h"] == pytest.approx(1.0)
    assert {e["kind"] for e in v["elements"]} >= {"wall", "door", "window", "floor", "ceiling", "space"}
    sp = [e for e in v["elements"] if e["kind"] == "space"][0]
    assert sp["params"]["area"] == pytest.approx(15.0)
    assert v["bounds"][1][2] == pytest.approx(2.6)


def test_building_view_shared_and_duplicates():
    from test_building import two_rooms

    from tdyh import building as bd
    b, sc = two_rooms()
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dAB"),
                       b=bd.Ref(capture="B", element="dBA"), thickness=0.15)]
    merged, _ = bd.merge(b, sc, bd.solve_poses(b, sc))
    v = build_view(merged)
    ids = [e["id"] for e in v["elements"]]
    assert "B:w3" not in ids and "B:dBA" not in ids and "A:dAB" in ids
    assert [e for e in v["elements"] if e["id"] == "A:w1"][0]["params"]["compartilhada_com"] == "B:w3"


def test_api_view_points_rescale_and_static(tmp_path, monkeypatch):
    monkeypatch.setenv("TDYH_DATA", str(tmp_path / "d"))
    import importlib

    from fastapi.testclient import TestClient

    import tdyh.api.main as m
    from tdyh import pipeline
    importlib.reload(m)
    c = TestClient(m.app)
    pid = c.post("/projects", json={"name": "Sala"}).json()["id"]
    work = tmp_path / "d" / "projects" / pid / "work"
    work.mkdir(parents=True)
    assert c.get(f"/projects/{pid}/view.json").status_code == 409
    sg = sample()
    sg.scale_factor = 2.0  # escala usada na medição
    (work / "scene_measured.json").write_text(sg.model_dump_json())
    pts = np.array([[1.0, 1.0, 1.0], [2.0, 0.5, 2.0]])
    pipeline.save_view_points(pts / 2.0, np.eye(3), 2.0, 0.0, work)
    pipeline.publish(work)
    lst = c.get("/projects").json()
    assert lst == [{"id": pid, "name": "Sala", "has_plan": True}]
    v = c.get(f"/projects/{pid}/view.json").json()
    assert v["title"] == "Sala" and v["points"] == f"/projects/{pid}/points.bin"
    got = np.frombuffer(c.get(f"/projects/{pid}/points.bin").content, "<f4").reshape(-1, 3)
    assert got == pytest.approx(pts, abs=1e-6)
    # usuário mede a parede w0: 5,50 m (10% maior) -> a nuvem acompanha
    sgm = type(sg).model_validate_json((work / "scene_measured.json").read_text())
    sgm.constraints.append(Constraint(kind="length", refs=["w0"], value=5.5, mode="scale"))
    (work / "scene_measured.json").write_text(sgm.model_dump_json())
    pipeline.publish(work)
    got = np.frombuffer(c.get(f"/projects/{pid}/points.bin").content, "<f4").reshape(-1, 3)
    assert got == pytest.approx(pts * 1.1, abs=1e-5)
    assert c.get("/buildings").json() == []
    if m.VIEWER_DIR.exists():
        r = c.get("/viewer/")
        assert r.status_code == 200 and "3D Your Home" in r.text


def test_demo_data(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    monkeypatch.setenv("TDYH_DATA", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["demo"])
    runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "demo_data.py"), run_name="__main__")
    assert (tmp_path / "projects" / "demo-apto" / "work" / "view_points.bin").exists()
    assert json.loads((tmp_path / "buildings" / "demo-casa.json").read_text())["name"] == "Casa (demo)"
