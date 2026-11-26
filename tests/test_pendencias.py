"""Pendências: reescala de elementos do usuário, parede desenhada centralizada, restrições de
posição de aberturas e ligação automática por marcadores ArUco compartilhados."""

import numpy as np
import pytest

from tdyh import building as bd
from tdyh import scale
from tdyh.constraints import mark_drawn, rebuild, split_wall
from tdyh.exporters import plan2d
from tdyh.scenegraph import Constraint, Opening, SceneGraph, Slab, Wall

from test_building import TRUE, pose_err, rect, room  # noqa: F401


def box(W=4.0, D=3.0, rot=0.0, shift=(0.0, 0.0), k=1.0):
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    T = lambda p: tuple(R @ (np.array(p, float) * k) + shift)  # noqa: E731
    c = [(0, 0), (W, 0), (W, D), (0, D)]
    ents = [Wall(id=f"w{i}", start=T(c[i]), end=T(c[(i + 1) % 4]), height=2.7 * k) for i in range(4)]
    ents += [Slab(id="piso", polygon=[T(p) for p in c]),
             Slab(id="teto", polygon=[T(p) for p in c], elevation=2.7 * k, role="ceiling")]
    return SceneGraph(entities=ents, scale_source="prior"), T


# ------------------------------------------------------------ reescala do que o usuário desenhou

def test_user_elements_follow_later_scale_change():
    sg, _ = box()
    div = Wall(id="div", start=(2.0, 0.0), end=(2.0, 3.0), source="user", thickness=0.10)
    door = Opening(id="pd", kind="door", host_wall="div", offset=1.0, width=0.8, height=2.0, source="user")
    sg.added += [div, door]
    mark_drawn(sg, "div"), mark_drawn(sg, "pd")
    sg.overrides["w2"] = {"height": 2.5}
    mark_drawn(sg, "w2", override=True)
    # depois o usuário mede a parede de baixo: 4,40 m (o modelo estava 10% pequeno)
    sg.constraints.append(Constraint(kind="length", refs=["w0"], value=4.4, mode="scale"))
    final, rep = rebuild(sg)
    d, p = final.get("div"), final.get("pd")
    assert d.start == pytest.approx((2.2, 0.0)) and d.end == pytest.approx((2.2, 3.3))
    assert d.thickness == pytest.approx(0.10)  # espessura não segue a escala (igual às medidas)
    assert (p.offset, p.width, p.height) == pytest.approx((1.1, 0.88, 2.2))
    assert final.get("w2").height == pytest.approx(2.75)
    assert any("reescalados" in w for w in rep.warnings)
    # a medida original continua intacta
    assert sg.added[0].start == (2.0, 0.0) and sg.overrides["w2"]["height"] == 2.5


def test_drawing_after_calibration_is_not_rescaled_twice():
    sg, _ = box()
    sg.constraints.append(Constraint(kind="length", refs=["w0"], value=4.4, mode="scale"))
    sg.added.append(Wall(id="div", start=(2.2, 0.0), end=(2.2, 3.3), source="user"))
    mark_drawn(sg, "div")  # desenhada já na escala 1,1
    final, _ = rebuild(sg)
    assert final.get("div").end == pytest.approx((2.2, 3.3))
    # e se a medida for corrigida para 4,80 m, acompanha 4,8/4,4
    sg.constraints[0].value = 4.8
    final, _ = rebuild(sg)
    assert final.get("div").start == pytest.approx((2.4, 0.0)) and final.get("div").end == pytest.approx((2.4, 3.6))


def test_editing_override_after_scale_change_keeps_units_consistent():
    sg, _ = box()
    sg.overrides["w1"] = {"height": 2.5}
    mark_drawn(sg, "w1", override=True)
    sg.constraints.append(Constraint(kind="length", refs=["w0"], value=4.4, mode="scale"))
    mark_drawn(sg, "w1", override=True)  # novo ajuste: converte o guardado para a escala atual
    assert sg.overrides["w1"]["height"] == pytest.approx(2.75)
    final, _ = rebuild(sg)
    assert final.get("w1").height == pytest.approx(2.75)


def test_legacy_without_record_is_untouched():
    sg, _ = box()
    sg.added.append(Wall(id="div", start=(2.0, 0.0), end=(2.0, 3.0), source="user"))
    sg.constraints.append(Constraint(kind="length", refs=["w0"], value=4.4, mode="scale"))
    final, _ = rebuild(sg)
    assert final.get("div").end == pytest.approx((2.0, 3.0))


# ------------------------------------------------------------ espessura centralizada

def test_user_internal_wall_is_centered_and_contour_stays_face():
    sg, _ = box()
    sg.added.append(Wall(id="div", start=(2.0, 0.0), end=(2.0, 3.0), source="user", thickness=0.10))
    sg.rooms_enabled = True
    final, _ = rebuild(sg)
    ax = plan2d.axis_flags(final)
    assert ax["div"] and not any(ax[f"w{i}"] for i in range(4))
    poly = plan2d._wall_polygon(final.get("div"), plan2d._center(final), axis=True)
    assert poly[:, 0].min() == pytest.approx(1.95) and poly[:, 0].max() == pytest.approx(2.05)
    # área útil de cada lado desconta metade da espessura — agora bate com o desenho
    sp = sorted(e.net_area for e in final.entities if e.type == "space")
    assert sp == pytest.approx([2 * 3 - 3 * 0.05] * 2)
    # parede de contorno desenhada pelo usuário continua com a face na linha
    sg2, _ = box()
    sg2.deleted.append("w3")
    sg2.added.append(Wall(id="nova", start=(0.0, 3.0), end=(0.0, 0.0), source="user"))
    f2, _ = rebuild(sg2)
    assert not plan2d.is_axis(f2.get("nova"), f2)


def test_explicit_alignment_and_split_keeps_side():
    sg, _ = box()
    sg.overrides = {}
    final, _ = rebuild(sg)
    w = final.get("w0")
    w.alignment = "center"
    assert plan2d.is_axis(w, final)
    sg, _ = box()
    final, _ = rebuild(sg)
    a, b = split_wall(sg, final, "w0", 1.5)
    f2, _ = rebuild(sg)
    assert not plan2d.is_axis(f2.get(a), f2) and not plan2d.is_axis(f2.get(b), f2)


def test_centered_wall_in_svg_and_ifc(tmp_path):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    import ifcopenshell.geom
    import ifcopenshell.util.element as uel

    from tdyh.exporters.ifc import to_ifc
    sg, _ = box()
    sg.added += [Wall(id="div", start=(2.0, 0.0), end=(2.0, 3.0), source="user", thickness=0.10),
                 Opening(id="pd", kind="door", host_wall="div", offset=1.0, width=0.8, height=2.1, source="user")]
    final, _ = rebuild(sg)
    plan2d.to_svg(final, tmp_path / "p.svg")
    f = ifcopenshell.open(str(to_ifc(final, tmp_path / "c.ifc")[0]))
    st = ifcopenshell.geom.settings()
    st.set("use-world-coords", True)
    w = next(e for e in f.by_type("IfcWall") if uel.get_pset(e, "TDYH_Origem")["IdCena"] == "div")
    v = np.array(ifcopenshell.geom.create_shape(st, w).geometry.verts).reshape(-1, 3)
    assert v[:, 0].min() == pytest.approx(1.95) and v[:, 0].max() == pytest.approx(2.05)
    assert uel.get_pset(w, "TDYH_Parametros")["LadoEspessura"] == "centralizada"
    d = np.array(ifcopenshell.geom.create_shape(st, f.by_type("IfcDoor")[0]).geometry.verts).reshape(-1, 3)
    assert d[:, 0].mean() == pytest.approx(2.0, abs=1e-6)


# ------------------------------------------------------------ posição de aberturas

@pytest.mark.parametrize("seed", range(4))
def test_center_and_align_openings(seed):
    rng = np.random.default_rng(seed)
    rot, shift, k = rng.uniform(-np.pi, np.pi), tuple(rng.uniform(-20, 20, 2)), rng.uniform(0.8, 1.2)
    sg, _ = box(5.0, 3.0, rot, shift, k)
    sg.entities += [Opening(id="porta", kind="door", host_wall="w0", offset=0.7 * k, width=0.9 * k, height=2.1 * k),
                    Opening(id="jan", kind="window", host_wall="w2", offset=0.3 * k, width=1.2 * k, height=1.0 * k,
                            sill=1.0 * k)]
    sg.constraints += [Constraint(kind="opening_center", refs=["porta"]),
                       Constraint(kind="opening_align", refs=["jan", "porta"])]
    final, rep = rebuild(sg)
    assert rep.ok
    p, j = final.get("porta"), final.get("jan")
    assert p.offset == pytest.approx((5.0 * k - 0.9 * k) / 2)
    # w2 vai de (5,3) a (0,3): centro da janela no mesmo x local que o centro da porta (2,5)
    assert j.offset + j.width / 2 == pytest.approx(5.0 * k - 2.5 * k)
    # e a largura medida da janela pode ser corrigida antes: o alinhamento usa a nova largura
    sg.constraints.append(Constraint(kind="opening_width", refs=["jan"], value=1.6 * k))
    final, rep = rebuild(sg)
    j = final.get("jan")
    assert j.width == pytest.approx(1.6 * k) and j.offset + j.width / 2 == pytest.approx(2.5 * k)


def test_opening_placement_reports_problems():
    sg, _ = box()
    sg.entities += [Opening(id="a", kind="door", host_wall="w0", offset=0.2, width=0.8, height=2.1),
                    Opening(id="b", kind="window", host_wall="w0", offset=2.5, width=1.0, height=1.0, sill=1.0),
                    Opening(id="c", kind="window", host_wall="w1", offset=0.5, width=1.0, height=1.0, sill=1.0),
                    Opening(id="big", kind="window", host_wall="w1", offset=0.1, width=0.5, height=1.0, sill=1.0)]
    sg.constraints += [Constraint(id="same", kind="opening_align", refs=["b", "a"]),
                       Constraint(id="perp", kind="opening_align", refs=["c", "a"]),
                       Constraint(id="wall", kind="opening_center", refs=["w0"])]
    _, rep = rebuild(sg)
    st = {s.id: s for s in rep.statuses}
    assert not st["same"].satisfied and "mesma parede" in st["same"].message
    assert not st["perp"].satisfied and "não paralelas" in st["perp"].message
    assert not st["wall"].satisfied
    # alinhar a uma abertura perto da ponta de uma parede mais curta: não cabe e avisa
    sg2, _ = box(6.0, 3.0)
    sg2.deleted.append("w2")
    sg2.added.append(Wall(id="curta", start=(2.0, 3.0), end=(0.0, 3.0), source="user"))
    sg2.entities += [Opening(id="x", kind="door", host_wall="w0", offset=5.0, width=0.8, height=2.1),
                     Opening(id="y", kind="window", host_wall="curta", offset=0.5, width=1.0, height=1.0, sill=1.0)]
    sg2.constraints.append(Constraint(id="fora", kind="opening_align", refs=["y", "x"]))
    _, rep2 = rebuild(sg2)
    assert not {s.id: s for s in rep2.statuses}["fora"].satisfied


# ------------------------------------------------------------ marcadores ArUco compartilhados

def test_marker_centers_triangulation():
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1.0]])
    c0 = np.array([0.3, -0.2, 2.5])
    corners = c0 + np.array([[0, 0, 0], [0.15, 0, 0], [0.15, 0.15, 0], [0, 0.15, 0]])
    obs = []
    for tx in (-0.4, 0.1, 0.5):
        P = K @ np.hstack([np.eye(3), [[tx], [0.05], [0]]])
        pix = np.array([(P @ np.r_[X, 1])[:2] / (P @ np.r_[X, 1])[2] for X in corners])
        obs.append((P, pix))
    out = scale.marker_centers({4: obs, 9: obs[:1]})
    assert set(out) == {4} and out[4] == pytest.approx(c0 + [0.075, 0.075, 0], abs=1e-9)


def test_make_plan_stores_marker_in_plan_coordinates(tmp_path):
    from test_m2 import random_rotation, synthetic_room

    from tdyh import pipeline
    from tdyh.pointcloud import write_ply_xyz
    R = random_rotation()
    s_model = 0.37
    write_ply_xyz(tmp_path / "d.ply", (synthetic_room() @ R.T) * s_model)
    marker_world = np.array([1.4, 0.0, 1.0])  # na parede y=0, a 1,4 m da parede x=0, a 1 m do piso
    (tmp_path / "p").mkdir()
    pipeline.make_plan(tmp_path / "d.ply", tmp_path / "p", 1 / s_model, "marker",
                       up_hint=R @ [0, 0, 1.0], marker_points={11: (R @ marker_world) * s_model})
    sg = SceneGraph.model_validate_json((tmp_path / "p" / "scene_measured.json").read_text())
    m = np.array(sg.markers["11"])
    assert m[2] == pytest.approx(1.0, abs=0.03)
    dist = sorted(abs((m[:2] - np.array(w.start)) @ np.array([-(w.end[1] - w.start[1]), w.end[0] - w.start[0]])
                      / np.linalg.norm(np.subtract(w.end, w.start))) for w in sg.entities if w.type == "wall")
    assert dist == pytest.approx([0.0, 1.4, 2.8, 3.1], abs=0.03)


def _with_markers(sc, world_pts: dict[str, tuple], caps=("A", "B")):
    for c in caps:
        x, y, th = TRUE[c]
        Rinv = bd.rot(-th)
        for mid, (px, py, pz) in world_pts.items():
            q = Rinv @ (np.array([px, py]) - [x, y])
            sc[c].markers[mid] = (float(q[0]), float(q[1]), pz)


def two_rooms_no_doors():
    A = room(rect(0, 0, 4, 3), [], TRUE["A"], "A")
    B = room(rect(4.15, -0.5, 7.15, 3.5), [], TRUE["B"], "B")
    return bd.Building(name="apto", captures=[bd.Capture(id="A", name="Sala"), bd.Capture(id="B", name="Quarto")]), \
        {"A": A, "B": B}


def test_two_shared_markers_solve_poses_without_any_manual_link():
    b, sc = two_rooms_no_doors()
    _with_markers(sc, {"3": (4.07, 1.2, 1.0), "5": (4.07, 2.4, 0.3)})  # no vão da porta, dois pontos
    info = bd.auto_marker_links(b, sc)
    assert len(b.links) == 2 and all(lk.kind == "point_match" and lk.auto_key for lk in b.links)
    assert sum("automática" in m for m in info) == 2
    res = bd.solve_poses(b, sc)
    e_pos, e_ang = pose_err(res.poses["B"], TRUE["B"])
    assert e_pos < 1e-6 and e_ang < 1e-6
    # rodar de novo não duplica; apagar fica lembrado
    bd.auto_marker_links(b, sc)
    assert len(b.links) == 2
    b.ignored_auto.append(b.links[0].auto_key)
    b.links = b.links[1:]
    bd.auto_marker_links(b, sc)
    assert len(b.links) == 1


def test_marker_height_mismatch_and_triple_use_warn():
    b, sc = two_rooms_no_doors()
    _with_markers(sc, {"3": (4.07, 1.2, 1.0)})
    sc["B"].markers["3"] = (*sc["B"].markers["3"][:2], 1.3)
    sc["C"] = room(rect(-3, 0, 0, 3), [], TRUE["C"], "C")
    b.captures.append(bd.Capture(id="C", name="Cozinha"))
    _with_markers(sc, {"3": (4.07, 1.2, 1.0)}, caps=("C",))
    info = bd.auto_marker_links(b, sc)
    assert any("altura difere 30 cm" in m for m in info)
    assert any("aparece em 3" in m for m in info) and len(b.links) == 2


def test_marker_scale_follows_calibration():
    sg, _ = box()
    sg.markers["7"] = (1.0, 0.0, 1.5)
    sg.constraints.append(Constraint(kind="length", refs=["w0"], value=4.4, mode="scale"))
    final, _ = rebuild(sg)
    assert final.markers["7"] == pytest.approx((1.1, 0.0, 1.65))


def test_api_marker_links_and_license(tmp_path, monkeypatch):
    from pathlib import Path

    from fastapi.testclient import TestClient
    monkeypatch.setenv("TDYH_DATA", str(tmp_path / "data"))
    import importlib

    import tdyh.api.main as m
    importlib.reload(m)
    c = TestClient(m.app)
    _, sc = two_rooms_no_doors()
    _with_markers(sc, {"3": (4.07, 1.2, 1.0), "5": (4.07, 2.4, 0.3)})
    bid = c.post("/buildings", json={"name": "Apto"}).json()["id"]
    from tdyh import pipeline
    for k in "AB":
        pid = c.post("/projects", json={"name": k}).json()["id"]
        work = tmp_path / "data" / "projects" / pid / "work"
        work.mkdir(parents=True)
        sc[k].scale_source = "marker"
        (work / "scene_measured.json").write_text(sc[k].model_dump_json())
        pipeline.publish(work)
        c.post(f"/buildings/{bid}/captures", json={"project_id": pid})
    r = c.post(f"/buildings/{bid}/solve").json()
    assert r["ok"] and len(r["markers"]) == 2
    links = c.get(f"/buildings/{bid}").json()["links"]
    c.delete(f"/buildings/{bid}/links/{links[0]['id']}")
    c.post(f"/buildings/{bid}/solve")
    assert len(c.get(f"/buildings/{bid}").json()["links"]) == 1
    lic = (Path(__file__).parents[1] / "LICENSE").read_text()
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in lic and "Version 3" in lic
