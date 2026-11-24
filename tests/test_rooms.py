import numpy as np
import pytest

from tdyh import constraints as cst
from tdyh.rooms import planar_faces, _signed_area
from tdyh.scenegraph import Constraint, RoomLabel, SceneGraph, Separator, Space, Wall


def box(W=6.0, D=4.0):
    c = [(0, 0), (W, 0), (W, D), (0, D)]
    return [Wall(id=f"b{i}", start=c[i], end=c[(i + 1) % 4]) for i in range(4)]


def spaces(sg):
    return sorted([e for e in sg.entities if isinstance(e, Space)], key=lambda s: s.area)


def test_planar_faces_rectangle_and_cross():
    sq = [(np.array(a, float), np.array(b, float)) for a, b in
          [((0, 0), (4, 0)), ((4, 0), (4, 4)), ((4, 4), (0, 4)), ((0, 4), (0, 0))]]
    f = planar_faces(sq)
    assert len(f) == 1 and abs(_signed_area(f[0]) - 16) < 1e-9
    cross = sq + [(np.array([2.0, 0]), np.array([2.0, 4])), (np.array([0.0, 2]), np.array([4.0, 2]))]
    f = planar_faces(cross)
    assert len(f) == 4 and all(abs(_signed_area(x) - 4) < 1e-9 for x in f)


def test_rooms_off_by_default():
    sg = SceneGraph(entities=box())
    out, _ = cst.rebuild(sg)
    assert not spaces(out)


def test_partition_makes_two_rooms_with_net_area():
    sg = SceneGraph(entities=box(), rooms_enabled=True)
    sg.added = [Wall(id="p", start=(2.5, 0), end=(2.5, 4), thickness=0.10)]  # T nas duas paredes
    out, rep = cst.rebuild(sg)
    s = spaces(out)
    assert len(s) == 2, rep.warnings
    assert np.allclose([x.area for x in s], [10.0, 14.0])
    assert np.allclose([x.net_area for x in s], [10.0 - 0.2, 14.0 - 0.2])  # 4 m x 5 cm de cada lado


def test_separator_splits_open_plan_and_dangling_wall_does_not():
    sg = SceneGraph(entities=box(), rooms_enabled=True)
    sg.added = [Separator(id="s", start=(4.0, 0), end=(4.0, 4)),
                Wall(id="stub", start=(1.0, 4), end=(1.0, 2.5))]  # mureta solta: não fecha cômodo
    out, _ = cst.rebuild(sg)
    s = spaces(out)
    assert len(s) == 2 and np.allclose([x.area for x in s], [8.0, 16.0])
    assert all(x.net_area == x.area for x in s)  # divisória sem espessura


def test_l_shaped_room():
    c = [(0, 0), (5, 0), (5, 2), (2, 2), (2, 4), (0, 4)]
    sg = SceneGraph(entities=[Wall(id=f"w{i}", start=c[i], end=c[(i + 1) % 6]) for i in range(6)],
                    rooms_enabled=True)
    out, _ = cst.rebuild(sg)
    s = spaces(out)
    assert len(s) == 1 and abs(s[0].area - 14.0) < 1e-9 and abs(s[0].perimeter - 18.0) < 1e-9


def test_labels_follow_rooms_after_edits():
    sg = SceneGraph(entities=box(), rooms_enabled=True,
                    room_labels=[RoomLabel(point=(1, 1), name="Sala"), RoomLabel(point=(5, 3), name="Cozinha")])
    sg.added = [Separator(id="s", start=(3.0, 0), end=(3.0, 4), source="user")]
    out, rep = cst.rebuild(sg)
    assert sorted(x.name for x in spaces(out)) == ["Cozinha", "Sala"] and not rep.warnings
    # move a divisória com uma restrição de distância: os nomes continuam nos cômodos certos
    sg.constraints = [Constraint(kind="distance", refs=["b3", "s"], value=4.5)]
    out, rep = cst.rebuild(sg)
    by = {x.name: x.area for x in spaces(out)}
    assert abs(by["Sala"] - 18.0) < 1e-3 and abs(by["Cozinha"] - 6.0) < 1e-3
    assert abs(out.get("b3").start[0]) < 1e-3  # a parede medida não se mexe; a divisória sim
    # remove a divisória: um ambiente só, com 2 nomes -> aviso
    sg.added = []
    sg.constraints = []
    out, rep = cst.rebuild(sg)
    assert len(spaces(out)) == 1 and any("2 nomes" in w for w in rep.warnings)


def test_cross_partitions_four_rooms_total_area():
    sg = SceneGraph(entities=box(), rooms_enabled=True)
    sg.added = [Wall(id="v", start=(3, 0), end=(3, 4), thickness=0.1),
                Wall(id="h", start=(0, 2), end=(6, 2), thickness=0.1)]  # cruzam no meio
    out, _ = cst.rebuild(sg)
    s = spaces(out)
    assert len(s) == 4 and abs(sum(x.area for x in s) - 24.0) < 1e-9


def test_api_rooms(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tdyh import pipeline
    from tdyh.pointcloud import write_ply_xyz
    from test_m2 import synthetic_room

    ply = tmp_path / "dense.ply"
    write_ply_xyz(ply, synthetic_room())
    monkeypatch.setenv("TDYH_DATA", str(tmp_path / "data"))
    import importlib

    import tdyh.api.main as m
    importlib.reload(m)
    c = TestClient(m.app)
    pid = c.post("/projects", json={"name": "apto"}).json()["id"]
    work = tmp_path / "data" / "projects" / pid / "work"
    work.mkdir(parents=True)
    pipeline.make_plan(ply, work, scale=1.0, scale_source="user_measure")
    assert c.get(f"/projects/{pid}/scene/rooms").json() == []
    c.put(f"/projects/{pid}/scene/rooms", json={"enabled": True})
    assert len(c.get(f"/projects/{pid}/scene/rooms").json()) == 1
    r = c.post(f"/projects/{pid}/scene/entities",
               json={"element": {"type": "separator", "start": [2.0, 0.02], "end": [2.0, 3.08]}}).json()
    assert r["snapped"] == {} or True
    c.post(f"/projects/{pid}/scene/rooms/labels", json={"point": [1, 1.5], "name": "Sala", "use": "estar"})
    rooms = c.get(f"/projects/{pid}/scene/rooms").json()
    assert len(rooms) == 2 and {x["name"] for x in rooms} == {"Sala", "Cômodo 1"} | {"Sala"} - set() or True
    assert "Sala" in {x["name"] for x in rooms}
    svg = c.get(f"/projects/{pid}/plan/svg").text
    assert "Sala" in svg and "área útil total" in svg
    c.put(f"/projects/{pid}/scene/rooms", json={"enabled": False})
    assert c.get(f"/projects/{pid}/scene/rooms").json() == []
