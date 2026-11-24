import numpy as np
import pytest

from tdyh import constraints as cst
from tdyh.scenegraph import Constraint, Opening, SceneGraph, Slab, Wall


def skewed_room(scale=1.0):
    """Cômodo 'medido' com erros: cantos deslocados alguns cm, um canto a ~88,5°."""
    c = np.array([[0, 0], [4.23, 0.02], [4.15, 3.12], [-0.03, 3.08]]) * scale
    walls = [Wall(id=f"w{i}", start=tuple(c[i]), end=tuple(c[(i + 1) % 4]), height=2.64 * scale)
             for i in range(4)]
    door = Opening(id="d0", kind="door", host_wall="w0", offset=1.0 * scale, width=0.78 * scale,
                   height=2.07 * scale)
    floor = Slab(id="f", polygon=[tuple(p) for p in c])
    return SceneGraph(entities=[*walls, door, floor])


def L(sg, wid):
    w = sg.get(wid)
    return float(np.linalg.norm(np.subtract(w.end, w.start)))


def connected(sg):
    ws = [e for e in sg.entities if isinstance(e, Wall)]
    return all(np.allclose(ws[i].end, ws[(i + 1) % len(ws)].start, atol=1e-9) for i in range(len(ws)))


def test_square_up_room():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="perpendicular", refs=["w0", "w1"]),
                      Constraint(kind="perpendicular", refs=["w1", "w2"]),
                      Constraint(kind="perpendicular", refs=["w2", "w3"]),
                      Constraint(kind="length", refs=["w0"], value=4.20),
                      Constraint(kind="length", refs=["w1"], value=3.10)]
    out, rep = cst.rebuild(sg)
    assert rep.ok, rep.statuses
    assert connected(out)
    assert abs(L(out, "w0") - 4.20) < 1e-3 and abs(L(out, "w1") - 3.10) < 1e-3
    assert abs(L(out, "w2") - 4.20) < 1e-3  # retângulo fechado
    for a, b in [("w0", "w1"), ("w2", "w3"), ("w3", "w0")]:
        assert abs(cst._line_angle(out.get(a), out.get(b)) - 90) < 0.1
    # a medida original fica intacta
    assert sg.get("w0").end == (4.23, 0.02)


def test_moves_minimum():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="parallel", refs=["w0", "w2"])]
    out, rep = cst.rebuild(sg)
    assert rep.ok
    moved = max(np.linalg.norm(np.subtract(getattr(out.get(w), e), getattr(sg.get(w), e)))
                for w in ("w0", "w1", "w2", "w3") for e in ("start", "end"))
    assert moved < 0.06  # ajusta poucos centímetros, não deforma o cômodo


def test_angle_and_orientation():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="angle", refs=["w0", "w1"], value=100.0),
                      Constraint(kind="orientation", refs=["w0"], value=0.0)]
    out, rep = cst.rebuild(sg)
    assert rep.ok, rep.statuses
    assert abs(cst._line_angle(out.get("w0"), out.get("w1")) - 100) < 0.1
    d = cst._dir(out.get("w0"))
    assert abs(d[1]) < 1e-3  # w0 horizontal na planta


def test_distance_between_parallel_walls():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="parallel", refs=["w0", "w2"]),
                      Constraint(kind="distance", refs=["w0", "w2"], value=3.00)]
    out, rep = cst.rebuild(sg)
    assert rep.ok, rep.statuses
    assert abs(cst.measure(out, sg.constraints[1]) - 3.00) < 1e-3


def test_fixed_wall_does_not_move():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="fixed", refs=["w1"]),
                      Constraint(kind="perpendicular", refs=["w0", "w1"]),
                      Constraint(kind="perpendicular", refs=["w1", "w2"])]
    out, rep = cst.rebuild(sg)
    assert rep.ok, rep.statuses
    assert np.allclose(out.get("w1").start, sg.get("w1").start, atol=1e-3)
    assert np.allclose(out.get("w1").end, sg.get("w1").end, atol=1e-3)


def test_conflict_reported():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="length", refs=["w0"], value=4.20),
                      Constraint(kind="length", refs=["w0"], value=4.00)]
    out, rep = cst.rebuild(sg)
    assert not rep.ok
    assert sum(not s.satisfied for s in rep.statuses) >= 1
    assert any("conflita" in s.message for s in rep.statuses)


def test_scale_calibration_from_measures():
    sg = skewed_room(scale=0.9)  # modelo 10% menor que o real
    sg.constraints = [Constraint(kind="length", refs=["w0"], value=4.23, mode="scale"),
                      Constraint(kind="opening_width", refs=["d0"], value=0.78, mode="scale"),
                      Constraint(kind="length", refs=["w1"], value=2.50, mode="scale")]  # erro de digitação
    out, rep = cst.rebuild(sg)
    assert abs(rep.scale_factor - 1 / 0.9) < 0.005
    assert out.scale_source == "user_measure"
    bad = [s for s in rep.statuses if not s.satisfied]
    assert len(bad) == 1 and bad[0].id == sg.constraints[2].id and "incoerente" in bad[0].message
    assert abs(out.get("w1").height - 2.64) < 0.01  # escala vale para tudo, inclusive altura


def test_scale_then_local_constraints():
    sg = skewed_room(scale=0.5)
    sg.constraints = [Constraint(kind="length", refs=["w0"], value=4.23, mode="scale"),
                      Constraint(kind="perpendicular", refs=["w0", "w1"])]
    out, rep = cst.rebuild(sg)
    assert rep.ok and abs(cst._line_angle(out.get("w0"), out.get("w1")) - 90) < 0.1
    assert any("refazer a detecção" in w for w in rep.warnings)


def test_per_element_overrides_and_reset():
    sg = skewed_room()
    sg.overrides = {"w2": {"thickness": 0.25}, "d0": {"kind": "window", "sill": 1.0, "height": 1.0}}
    out, _ = cst.rebuild(sg)
    assert out.get("w2").thickness == 0.25 and out.get("w0").thickness == 0.15
    assert out.get("d0").kind.value == "window" and out.get("d0").sill == 1.0
    sg.overrides.pop("w2")
    out2, _ = cst.rebuild(sg)
    assert out2.get("w2").thickness == 0.15


def test_opening_constraints_and_fit():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="opening_width", refs=["d0"], value=0.80),
                      Constraint(kind="opening_offset", refs=["d0"], value=3.9)]
    out, rep = cst.rebuild(sg)
    assert out.get("d0").width == 0.80
    assert not rep.ok and any("não cabe" in s.message for s in rep.statuses)


def test_floor_follows_corners():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="perpendicular", refs=["w0", "w1"])]
    out, _ = cst.rebuild(sg)
    floor = next(e for e in out.entities if isinstance(e, Slab))
    assert np.allclose(floor.polygon[2], out.get("w1").end)


def test_disable_restores():
    sg = skewed_room()
    c = Constraint(kind="length", refs=["w0"], value=5.0)
    sg.constraints = [c]
    assert abs(L(cst.rebuild(sg)[0], "w0") - 5.0) < 1e-3
    c.enabled = False
    assert abs(L(cst.rebuild(sg)[0], "w0") - L(sg, "w0")) < 1e-9


# ---------------------------------------------------- escala automática + API

def test_prior_scale_and_api_editing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tdyh import pipeline
    from tdyh.pointcloud import write_ply_xyz

    from test_m2 import synthetic_room

    ply = tmp_path / "dense.ply"
    write_ply_xyz(ply, synthetic_room() * 0.37)  # escala desconhecida
    monkeypatch.setenv("TDYH_DATA", str(tmp_path / "data"))
    import importlib

    import tdyh.api.main as m
    importlib.reload(m)
    c = TestClient(m.app)
    pid = c.post("/projects", json={"name": "sala"}).json()["id"]
    work = tmp_path / "data" / "projects" / pid / "work"
    work.mkdir(parents=True)
    out = pipeline.make_plan(ply, work)
    assert out["scale"]["source"] == "prior"

    els = c.get(f"/projects/{pid}/scene/elements").json()
    walls = sorted([e for e in els if e["type"] == "wall"], key=lambda e: -e["length"])
    assert abs(walls[0]["length"] / 4.20 - 1) < 0.08  # estimativa por pé-direito típico

    # uma medida com trena corrige tudo
    r = c.post(f"/projects/{pid}/scene/constraints",
               json={"kind": "length", "refs": [walls[0]["id"]], "value": 4.20, "mode": "scale"}).json()
    assert r["scale"]["source"] == "user_measure"
    els = c.get(f"/projects/{pid}/scene/elements").json()
    lens = sorted(e["length"] for e in els if e["type"] == "wall")
    assert np.allclose(lens, [3.10, 3.10, 4.20, 4.20], atol=0.01)

    # ajuste individual + restrição geométrica + remoção
    r = c.patch(f"/projects/{pid}/scene/entities/{walls[1]['id']}", json={"thickness": 0.25}).json()
    els = {e["id"]: e for e in c.get(f"/projects/{pid}/scene/elements").json()}
    assert els[walls[1]["id"]]["thickness"] == 0.25
    r = c.post(f"/projects/{pid}/scene/constraints",
               json={"kind": "angle", "refs": [walls[0]["id"], walls[2]["id"]], "value": 85}).json()
    cid = r["constraint"]["id"]
    assert r["ok"]
    r = c.delete(f"/projects/{pid}/scene/constraints/{cid}").json()
    assert len(c.get(f"/projects/{pid}/scene/constraints").json()) == 1
    assert c.get(f"/projects/{pid}/plan/svg").status_code == 200
