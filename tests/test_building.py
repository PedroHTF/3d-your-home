"""Junção de levantamentos: cômodos sintéticos com posição real conhecida, cada um entregue
num sistema de coordenadas próprio (rotação/translação aleatórias)."""

import numpy as np
import pytest

from tdyh import building as bd
from tdyh.scenegraph import Opening, SceneGraph, Slab, Space, Wall


def room(corners_world, doors, pose_true, name):
    """corners em ordem anti-horária (mundo); doors = [(id, índice da parede, offset, largura)].
    Devolve a cena no sistema LOCAL da captura (mundo -> local pela inversa de pose_true)."""
    x, y, th = pose_true
    Rinv = bd.rot(-th)
    loc = [tuple(Rinv @ (np.array(c, float) - [x, y])) for c in corners_world]
    n = len(loc)
    ents = [Wall(id=f"w{i}", start=loc[i], end=loc[(i + 1) % n]) for i in range(n)]
    ents += [Opening(id=d, kind="door", host_wall=f"w{wi}", offset=off, width=w, height=2.1)
             for d, wi, off, w in doors]
    ents.append(Slab(polygon=loc))
    return SceneGraph(entities=ents)


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


TRUE = {"A": (0.0, 0.0, 0.0), "B": (10.0, -3.0, np.radians(37)), "C": (-5.0, 8.0, np.radians(-120)),
        "D": (2.0, 20.0, np.radians(200))}


def two_rooms():
    # A: 4 x 3; porta para B na parede direita (w1: (4,0)->(4,3)), y 1.0–1.8; porta de entrada em w0
    A = room(rect(0, 0, 4, 3), [("dAB", 1, 1.0, 0.8), ("dEnt", 0, 0.5, 0.9)], TRUE["A"], "A")
    # B: 3 x 4 à direita, parede de 15 cm; porta em w3 ((4.15,3.5)->(4.15,-0.5)) y 1.0–1.8 => offset 1.7
    # e uma porta "isca" de mesma largura na parede oposta
    B = room(rect(4.15, -0.5, 7.15, 3.5), [("dBA", 3, 1.7, 0.8), ("dIsca", 1, 1.0, 0.8)], TRUE["B"], "B")
    b = bd.Building(name="apto", captures=[bd.Capture(id="A", name="Sala"), bd.Capture(id="B", name="Quarto")])
    return b, {"A": A, "B": B}


def pose_err(p, q):
    return np.hypot(p[0] - q[0], p[1] - q[1]), abs(np.degrees((p[2] - q[2] + np.pi) % (2 * np.pi) - np.pi))


def test_ambiguous_suggestion_is_flagged():
    A = room(rect(0, 0, 4, 3), [("dAB", 1, 1.0, 0.8), ("dEnt", 0, 0.5, 0.8)], TRUE["A"], "A")
    B = room(rect(4.15, -0.5, 7.15, 3.5), [("dBA", 3, 1.7, 0.8)], TRUE["B"], "B")
    b = bd.Building(name="x", captures=[bd.Capture(id="A", name="A"), bd.Capture(id="B", name="B")])
    top = bd.suggest_door_matches(b, {"A": A, "B": B})[:2]
    assert all(s.ambiguous for s in top)  # entrada de A (mesma largura) encaixa igual: não dá para saber


def test_door_suggestion_ranks_true_pair_first():
    b, sc = two_rooms()
    sug = bd.suggest_door_matches(b, sc)
    best = sug[0]
    assert (best.a_door, best.b_door) == ("dAB", "dBA"), [(s.a_door, s.b_door, s.score) for s in sug]
    assert abs(best.thickness - 0.15) < 0.006  # sem fachada alinhada: fica no padrão
    # B girado 180° usando a porta-isca também encaixa: empate real, que precisa ser sinalizado
    assert best.ambiguous
    em, ed = pose_err(best.pose_b, TRUE["B"])
    assert em < 0.02 and ed < 0.1


def test_solve_from_door_link_measures_thickness():
    b, sc = two_rooms()
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dAB"),
                       b=bd.Ref(capture="B", element="dBA"))]
    # chute inicial longe: o ajuste não depende de pose manual
    b.cap("B").pose = bd.Pose2D(x=0, y=0, theta_deg=0)
    res = bd.solve_poses(b, sc)
    em, ed = pose_err(res.poses["B"], TRUE["B"])
    assert em < 1e-3 and ed < 0.01, res.poses
    assert all(s.ok for s in res.statuses)
    # a porta sozinha (sem batente medido) não determina a espessura: avisa que usou o padrão
    assert "não determinada" in res.statuses[0].message and not b.measured_thickness


def test_door_reveal_depth_sets_thickness_and_collinear_aligns():
    """A e B com fachada alinhada; parede interna de 12 cm medida pelo batente da porta."""
    A = room(rect(0, 0, 4, 3), [("dAB", 1, 1.0, 0.8)], TRUE["A"], "A")
    B = room(rect(4.12, 0, 7.12, 3), [("dBA", 3, 1.2, 0.8)], TRUE["B"], "B")
    A.get("dAB").depth = 0.121
    B.get("dBA").depth = 0.119
    b = bd.Building(name="x", captures=[bd.Capture(id="A", name="A"), bd.Capture(id="B", name="B")])
    sc = {"A": A, "B": B}
    s = bd.suggest_door_matches(b, sc)[0]
    assert abs(s.thickness - 0.12) < 1e-3 and any("alinhada" in r for r in s.reasons)
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dAB"), b=bd.Ref(capture="B", element="dBA")),
               bd.Link(kind="collinear", a=bd.Ref(capture="A", element="w0"), b=bd.Ref(capture="B", element="w0"))]
    res = bd.solve_poses(b, sc)
    assert all(st.ok for st in res.statuses), res.statuses
    assert abs(b.measured_thickness[b.links[0].id] - 0.12) < 1e-3
    em, ed = pose_err(res.poses["B"], TRUE["B"])
    assert em < 1e-3 and ed < 0.01


def test_reveal_depth_from_points():
    from tdyh.openings import reveal_depth
    from tdyh.scenegraph import Opening, Wall
    rng = np.random.default_rng(0)
    w = Wall(start=(0, 0), end=(4, 0))
    o = Opening(kind="door", host_wall=w.id, offset=1.0, width=0.8, height=2.1)
    jambs = []
    for x in (1.0, 1.8):  # faces laterais do batente, de y=0 a y=-0.14 (lado de fora)
        q = rng.random((600, 2)) * [0.14, 2.1]
        jambs.append(np.c_[np.full(600, x), -q[:, 0], q[:, 1]])
    wall = np.c_[rng.random(3000) * 4, np.zeros(3000), rng.random(3000) * 2.7]
    pts = np.vstack([wall, *jambs])
    assert abs(reveal_depth(w, o, pts, np.array([2.0, 1.5])) - 0.14) < 0.01


def test_point_match_manual_alignment():
    b, sc = two_rooms()
    # dois cantos físicos clicados nas duas capturas
    A, B = sc["A"], sc["B"]
    to_loc = lambda pw, pose: bd.rot(-pose[2]) @ (np.array(pw) - pose[:2])
    pairs = [((4.15, 3.0), (4.15, 3.0)), ((4.15, 0.0), (4.15, 0.0))]
    b.links = [bd.Link(kind="point_match", a=bd.Ref(capture="A", point=tuple(to_loc(pa, np.array(TRUE["A"])))),
                       b=bd.Ref(capture="B", point=tuple(to_loc(pb, np.array(TRUE["B"])))))
               for pa, pb in pairs]
    res = bd.solve_poses(b, sc)
    em, ed = pose_err(res.poses["B"], TRUE["B"])
    assert em < 1e-3 and ed < 0.01


def test_merge_shared_wall_dedupe_and_spaces():
    b, sc = two_rooms()
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dAB"),
                       b=bd.Ref(capture="B", element="dBA"), thickness=0.15)]
    res = bd.solve_poses(b, sc)
    merged, info = bd.merge(b, sc, res)
    wa, wb = merged.get("A:w1"), merged.get("B:w3")
    assert wa.thickness == pytest.approx(0.15, abs=1e-3) and wb.thickness == pytest.approx(0.15, abs=1e-3)
    assert any("15.0 cm" in m for m in info)
    assert merged.get("B:dBA").duplicate_of == "A:dAB"
    sp = sorted((e.name, round(e.area, 2)) for e in merged.entities if isinstance(e, Space))
    assert sp == [("Quarto", 12.0), ("Sala", 12.0)]
    # paredes externas não compartilhadas mantêm o padrão
    assert merged.get("A:w3").thickness == 0.15 and merged.get("B:w1").thickness == 0.15


def test_unlinked_capture_warns():
    b, sc = two_rooms()
    res = bd.solve_poses(b, sc)
    assert any("sem ligação" in w for w in res.warnings)


def test_loop_of_four_rooms_distributes_error():
    """2 x 2 cômodos ligados em ciclo A-B-D-C-A; portas com erro de ~1 cm em cada captura."""
    rng = np.random.default_rng(0)
    g = 0.15
    rects = {"A": rect(0, 0, 4, 3), "B": rect(4 + g, 0, 8 + g, 3),
             "C": rect(0, 3 + g, 4, 6 + g), "D": rect(4 + g, 3 + g, 8 + g, 6 + g)}
    # paredes (anti-horário): w0 baixo, w1 direita, w2 cima, w3 esquerda. Portas de 0,8 m no meio das paredes comuns
    spec = {"A": [("AB", 1, 1.1), ("AC", 2, 1.6)], "B": [("BA", 3, 1.1), ("BD", 2, 1.6)],
            "C": [("CA", 0, 1.6), ("CD", 1, 1.1)], "D": [("DB", 0, 1.6), ("DC", 3, 1.1)]}
    sc = {}
    for k, r in rects.items():
        doors = [(d, wi, off + rng.normal(0, 0.01), 0.8) for d, wi, off in spec[k]]
        sc[k] = room(r, doors, TRUE[k], k)
    b = bd.Building(name="apto", captures=[bd.Capture(id=k, name=k) for k in "ABCD"])
    pairs = [("A", "AB", "B", "BA"), ("B", "BD", "D", "DB"), ("D", "DC", "C", "CD"), ("C", "CA", "A", "AC")]
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture=a, element=da), b=bd.Ref(capture=c, element=dc),
                       thickness=0.15) for a, da, c, dc in pairs]
    res = bd.solve_poses(b, sc)
    for k in "BCD":
        em, ed = pose_err(res.poses[k], TRUE[k])
        assert em < 0.03 and ed < 0.5, (k, em, ed)
    assert max(s.residual_m for s in res.statuses) < 0.02  # erro repartido, nenhuma ligação estoura


def test_wrong_link_is_flagged():
    b, sc = two_rooms()
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dAB"),
                       b=bd.Ref(capture="B", element="dBA")),
               bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dEnt"),  # errada
                       b=bd.Ref(capture="B", element="dIsca"))]
    res = bd.solve_poses(b, sc)
    assert not all(s.ok for s in res.statuses)


def test_api_building_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TDYH_DATA", str(tmp_path / "data"))
    import importlib

    import tdyh.api.main as m
    importlib.reload(m)
    c = TestClient(m.app)
    _, sc = two_rooms()
    pids = {}
    for k, name in (("A", "Sala"), ("B", "Quarto")):
        pid = c.post("/projects", json={"name": name}).json()["id"]
        work = tmp_path / "data" / "projects" / pid / "work"
        work.mkdir(parents=True)
        sc[k].scale_source = "user_measure"
        (work / "scene_measured.json").write_text(sc[k].model_dump_json())
        from tdyh import pipeline
        pipeline.publish(work)
        pids[k] = pid
    bid = c.post("/buildings", json={"name": "Apto"}).json()["id"]
    caps = {k: c.post(f"/buildings/{bid}/captures", json={"project_id": pids[k]}).json()["id"] for k in "AB"}
    sug = c.get(f"/buildings/{bid}/suggestions").json()
    top = sug[0]
    assert (top["a_door"], top["b_door"]) == ("dAB", "dBA")
    c.post(f"/buildings/{bid}/links", json={"kind": "door_match",
           "a": {"capture": top["a_capture"], "element": top["a_door"]},
           "b": {"capture": top["b_capture"], "element": top["b_door"]}})
    r = c.post(f"/buildings/{bid}/solve").json()
    assert r["ok"] and sorted(x["name"] for x in r["rooms"]) == ["Quarto", "Sala"]
    p = r["poses"][caps["B"]]
    assert abs(p["x"] - 10.0) < 1e-3 and abs(p["theta_deg"] - 37) < 0.01
    assert c.get(f"/buildings/{bid}/plan/svg").status_code == 200
