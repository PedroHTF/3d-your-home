import numpy as np
import pytest

from tdyh import constraints as cst
from tdyh.scenegraph import Constraint, Opening, Slab, Wall

from test_constraints import L, skewed_room


def walls(sg):
    return [e for e in sg.entities if isinstance(e, Wall)]


def area(sg):
    from tdyh.exporters.plan2d import _area
    return _area(next(e for e in sg.entities if isinstance(e, Slab) and e.role == "floor").polygon)


def test_delete_wall_cascades_and_restores():
    sg = skewed_room()
    sg.deleted = ["w0"]  # w0 hospeda a porta d0
    out, rep = cst.rebuild(sg)
    assert len(walls(out)) == 3 and out.get("d0") is None
    assert any("contorno fechado" in w for w in rep.warnings)
    sg.deleted = []
    out, _ = cst.rebuild(sg)
    assert len(walls(out)) == 4 and out.get("d0") is not None


def test_replace_wall_and_floor_follows():
    """Apaga uma parede e cria outra no lugar, recuada: o piso é refeito pelos cantos."""
    sg = skewed_room()
    base = cst.rebuild(sg)[0]
    sg.deleted = ["w1"]
    # nova parede fechando o cômodo 1 m antes (cantos ligam em w0 e w2 por junção em T)
    a = np.array(base.get("w0").start) + (np.array(base.get("w0").end) - base.get("w0").start) * (3.2 / L(base, "w0"))
    b = np.array(base.get("w2").end) + (np.array(base.get("w2").start) - base.get("w2").end) * (3.2 / L(base, "w2"))
    sg.added = [Wall(id="n1", start=tuple(a), end=tuple(b), source="user")]
    out, rep = cst.rebuild(sg)
    assert out.get("n1") is not None and out.get("w1") is None
    # sem dividir w0/w2 o contorno não fecha nos cantos -> mantém piso antigo e avisa
    assert any("contorno" in w for w in rep.warnings)


def test_partition_wall_t_junction_stays_attached():
    sg = skewed_room()
    base = cst.rebuild(sg)[0]
    w0, w2 = base.get("w0"), base.get("w2")
    p = np.array(w0.start) + (np.array(w0.end) - w0.start) * 0.6
    q = np.array(w2.end) + (np.array(w2.start) - w2.end) * 0.6
    sg.added = [Wall(id="p", start=tuple(p), end=tuple(q), thickness=0.10, source="user")]
    # endireitar o cômodo move w0/w2; a divisória deve continuar encostada e ficar perpendicular
    sg.constraints = [Constraint(kind="perpendicular", refs=["w0", "w1"]),
                      Constraint(kind="perpendicular", refs=["w1", "w2"]),
                      Constraint(kind="perpendicular", refs=["w2", "w3"]),
                      Constraint(kind="perpendicular", refs=["p", "w0"])]
    out, rep = cst.rebuild(sg)
    assert rep.ok, rep.statuses
    for host, end in (("w0", "start"), ("w2", "end")):
        h = out.get(host)
        a, b = np.array(h.start), np.array(h.end)
        x = np.array(getattr(out.get("p"), end)) - a
        d = (b - a) / np.linalg.norm(b - a)
        assert abs(d[0] * x[1] - d[1] * x[0]) < 1e-3  # continua sobre a parede
    assert abs(cst._line_angle(out.get("p"), out.get("w0")) - 90) < 0.1


def test_user_opening_on_user_wall_and_fit_check():
    sg = skewed_room()
    sg.added = [Wall(id="n", start=(1.0, 1.0), end=(3.0, 1.0), source="user"),
                Opening(id="o1", kind="door", host_wall="n", offset=0.5, width=0.8, height=2.1, source="user"),
                Opening(id="o2", kind="window", host_wall="n", offset=1.6, width=0.8, height=1.0, sill=1.0)]
    out, rep = cst.rebuild(sg)
    assert out.get("o1") is not None
    assert any("o2" in w and "não cabe" in w for w in rep.warnings)


def test_split_wall_keeps_openings():
    sg = skewed_room()
    final, _ = cst.rebuild(sg)
    a, b = cst.split_wall(sg, final, "w0", 2.5)
    out, rep = cst.rebuild(sg)
    assert out.get("w0") is None and abs(L(out, a) - 2.5) < 1e-9
    assert abs(L(out, a) + L(out, b) - L(final, "w0")) < 1e-9
    door = [e for e in out.entities if isinstance(e, Opening)][0]
    assert door.host_wall == a and door.offset == 1.0  # porta estava a 1,0 m: fica na 1ª metade
    assert len(walls(out)) == 5 and abs(area(out) - area(final)) < 1e-6
    with pytest.raises(ValueError):
        cst.split_wall(sg, final, "w0", 1.3)  # cortaria a porta


def test_snap():
    sg = cst.rebuild(skewed_room())[0]
    p, how = cst.snap_point(sg, (4.18, 0.07))
    assert how == "corner" and np.allclose(p, sg.get("w0").end)
    p, how = cst.snap_point(sg, (2.0, 0.06))
    assert how == "wall" and abs(p[1] - 0.0095) < 0.01
    p, how = cst.snap_point(sg, (2.0, 1.5))
    assert how is None


def test_api_create_delete_split(tmp_path, monkeypatch):
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
    pid = c.post("/projects", json={"name": "sala"}).json()["id"]
    work = tmp_path / "data" / "projects" / pid / "work"
    work.mkdir(parents=True)
    pipeline.make_plan(ply, work, scale=1.0, scale_source="user_measure")

    def els():
        return c.get(f"/projects/{pid}/scene/elements").json()

    ws = sorted([e for e in els() if e["type"] == "wall"], key=lambda e: (round(e["start"][1], 1), e["length"]))
    longw = next(e for e in ws if e["length"] > 4)
    # divide a parede longa em 2,0 m e cria divisória até a parede oposta (encaixe automático)
    r = c.post(f"/projects/{pid}/scene/entities/{longw['id']}/split", json={"at": 2.0}).json()
    s1 = next(e for e in els() if e["id"] == r["created"][0])
    p0 = s1["end"]
    far = [p0[0], 3.06 if p0[1] < 1 else 0.04]  # 4 cm fora da parede oposta: deve encaixar
    r = c.post(f"/projects/{pid}/scene/entities",
               json={"element": {"type": "wall", "start": p0, "end": far, "thickness": 0.1}}).json()
    assert r["snapped"] == {"start": "corner", "end": "wall"}
    wid = r["created"]
    r = c.post(f"/projects/{pid}/scene/entities",
               json={"element": {"type": "opening", "kind": "door", "host_wall": wid,
                                 "offset": 1.0, "width": 0.7, "height": 2.1}}).json()
    door = r["created"]
    assert sum(e["type"] == "wall" for e in els()) == 6
    # apagar a divisória leva a porta junto
    c.delete(f"/projects/{pid}/scene/entities/{wid}")
    ids = {e["id"] for e in els()}
    assert wid not in ids and door not in ids
    # apagar e restaurar uma parede detectada
    other = next(e for e in els() if e["type"] == "wall" and e["length"] < 3.2)
    c.delete(f"/projects/{pid}/scene/entities/{other['id']}")
    assert other["id"] not in {e["id"] for e in els()}
    c.post(f"/projects/{pid}/scene/entities/{other['id']}/restore")
    assert other["id"] in {e["id"] for e in els()}
    bad = c.post(f"/projects/{pid}/scene/entities", json={"element": {"type": "opening", "kind": "door",
                 "host_wall": "naoexiste", "offset": 0, "width": 0.8, "height": 2.1}})
    assert bad.status_code == 422
    assert c.get(f"/projects/{pid}/plan/dxf").status_code == 200


def test_split_transfers_constraints_and_keeps_collinear():
    sg = skewed_room()
    sg.constraints = [Constraint(kind="perpendicular", refs=["w0", "w1"]),
                      Constraint(kind="perpendicular", refs=["w3", "w0"])]
    final, _ = cst.rebuild(sg)
    a, b = cst.split_wall(sg, final, "w0", 2.0)
    out, rep = cst.rebuild(sg)
    assert rep.ok, [(s.kind, s.message) for s in rep.statuses if not s.satisfied]
    assert abs(cst._line_angle(out.get(a), out.get(b))) < 0.1 or abs(cst._line_angle(out.get(a), out.get(b)) - 180) < 0.1
    assert abs(cst._line_angle(out.get(b), out.get("w1")) - 90) < 0.1
    assert all(not c.enabled for c in sg.constraints if "w0" in c.refs)


def test_partition_not_drawn_through_walls():
    from tdyh.exporters.plan2d import _l_corner
    sg = skewed_room()
    final, _ = cst.rebuild(sg)
    a, _b = cst.split_wall(sg, final, "w0", 2.0)
    final, _ = cst.rebuild(sg)
    q, _h = cst.snap_point(final, (final.get(a).end[0], 3.1))
    sg.added.append(Wall(id="part", start=final.get(a).end, end=q, source="user"))
    out, _ = cst.rebuild(sg)
    ws = walls(out)
    assert _l_corner(out.get("part"), "start", ws) == 0 and _l_corner(out.get("part"), "end", ws) == 0
    assert _l_corner(out.get("w1"), "start", ws) > 0  # canto em L continua fechado
