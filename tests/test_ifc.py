"""M3 — exportação IFC4: reabre o arquivo e confere entidades, dimensões (pela geometria tesselada,
com os vazios aplicados), Psets e robustez a rotação/escala/sementes."""

import json

import numpy as np
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.geom  # noqa: E402
import ifcopenshell.util.element as uel  # noqa: E402

from tdyh import building as bd  # noqa: E402
from tdyh.constraints import rebuild  # noqa: E402
from tdyh.exporters.ifc import to_ifc  # noqa: E402
from tdyh.scenegraph import Opening, RoomLabel, SceneGraph, Slab, Space, Wall  # noqa: E402

from test_building import two_rooms  # noqa: E402

ST = ifcopenshell.geom.settings()
ST.set("use-world-coords", True)


def verts(e):
    return np.array(ifcopenshell.geom.create_shape(ST, e).geometry.verts).reshape(-1, 3)


def volume(e):
    g = ifcopenshell.geom.create_shape(ST, e).geometry
    v = np.array(g.verts).reshape(-1, 3)
    t = np.array(g.faces).reshape(-1, 3)
    return abs(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6)


def by_scene_id(f, cls):
    return {uel.get_pset(e, "TDYH_Origem")["IdCena"]: e for e in f.by_type(cls)}


def apartment(rot=0.0, shift=(0.0, 0.0), k=1.0):
    """Sala 5x4 com divisória até 3 m (quarto 2x4 à direita), porta, janela e rótulos com uso."""
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    T = lambda p: tuple(R @ (np.array(p, float) * k) + shift)  # noqa: E731
    c = [(0, 0), (7, 0), (7, 4), (0, 4)]
    ents = [Wall(id=f"w{i}", start=T(c[i]), end=T(c[(i + 1) % 4]), height=2.6 * k) for i in range(4)]
    ents.append(Wall(id="wi", start=T((5, 0)), end=T((5, 4)), thickness=0.10, height=2.6 * k))
    ents += [Opening(id="d1", kind="door", host_wall="w0", offset=1.0 * k, width=0.9 * k, height=2.1 * k),
             Opening(id="j1", kind="window", host_wall="w2", offset=1.0 * k, width=1.5 * k, height=1.2 * k,
                     sill=1.0 * k, source="inferred", confidence=0.4),
             Opening(id="d2", kind="door", host_wall="wi", offset=1.5 * k, width=0.8 * k, height=2.1 * k,
                     depth=0.10)]
    ents += [Slab(id="piso", polygon=[T(p) for p in c]),
             Slab(id="teto", polygon=[T(p) for p in c], elevation=2.6 * k, role="ceiling")]
    sg = SceneGraph(entities=ents, scale_source="marker", scale_uncertainty=0.01, rooms_enabled=True,
                    room_labels=[RoomLabel(point=T((2, 2)), name="Sala", use="estar"),
                                 RoomLabel(point=T((6, 2)), name="Quarto", use="dormitório")])
    final, _ = rebuild(sg)
    return final


def test_single_capture_entities_dimensions_and_psets(tmp_path):
    sg = apartment()
    path, warn = to_ifc(sg, tmp_path / "a.ifc", project_name="Apto")
    f = ifcopenshell.open(str(path))
    assert f.schema == "IFC4"
    for cls in ("IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey"):
        assert len(f.by_type(cls)) == 1
    u = [x for x in f.by_type("IfcSIUnit") if x.UnitType == "LENGTHUNIT"][0]
    assert u.Name == "METRE" and u.Prefix is None
    walls = by_scene_id(f, "IfcWall")
    assert len(walls) == 5 and len(f.by_type("IfcDoor")) == 2 and len(f.by_type("IfcWindow")) == 1
    assert len(f.by_type("IfcOpeningElement")) == 3 and len(f.by_type("IfcSlab")) == 2
    # parede w0 (0,0)->(7,0), espessura 15 cm para FORA (y<0), estendida nos cantos em L
    v = verts(walls["w0"])
    assert v[:, 1].min() == pytest.approx(-0.15, abs=1e-6) and v[:, 1].max() == pytest.approx(0, abs=1e-6)
    assert v[:, 0].min() == pytest.approx(-0.15, abs=1e-6) and v[:, 0].max() == pytest.approx(7.15, abs=1e-6)
    assert v[:, 2].max() == pytest.approx(2.6, abs=1e-6)
    # volume já com o vão da porta: (7.3*0.15*2.6) - (0.9*0.15*2.1)
    assert volume(walls["w0"]) == pytest.approx(7.3 * 0.15 * 2.6 - 0.9 * 0.15 * 2.1, rel=1e-4)
    q = uel.get_psets(walls["w0"])
    assert q["Qto_WallBaseQuantities"]["Length"] == pytest.approx(7.0)
    assert q["Qto_WallBaseQuantities"]["Width"] == pytest.approx(0.15)
    assert q["Pset_WallCommon"]["IsExternal"] is True
    assert uel.get_psets(walls["wi"])["Pset_WallCommon"]["IsExternal"] is False
    # janela: peitoril 1.0, altura 1.2, origem inferida com confiança baixa
    j = f.by_type("IfcWindow")[0]
    vj = verts(j)
    assert vj[:, 2].min() == pytest.approx(1.0) and vj[:, 2].max() == pytest.approx(2.2)
    assert j.OverallWidth == pytest.approx(1.5) and j.OverallHeight == pytest.approx(1.2)
    o = uel.get_pset(j, "TDYH_Origem")
    assert o["Fonte"] == "inferida" and o["Confianca"] == pytest.approx(0.4) and o["FonteEscala"] == "marker"
    assert uel.get_pset(j, "TDYH_Parametros")["Peitoril"] == pytest.approx(1.0)
    # porta preenche o vão, que corta a parede hospedeira
    for d in f.by_type("IfcDoor"):
        op = d.FillsVoids[0].RelatingOpeningElement
        assert op.VoidsElements[0].RelatingBuildingElement.is_a("IfcWall")
    # lajes: piso abaixo de 0, teto acima de 2.6
    sl = {s.ObjectType: verts(s) for s in f.by_type("IfcSlab")}
    assert sl["piso"][:, 2].max() == pytest.approx(0) and sl["teto"][:, 2].min() == pytest.approx(2.6)
    # cômodos com nome, uso e áreas
    sp = {s.Name: s for s in f.by_type("IfcSpace")}
    assert set(sp) == {"Sala", "Quarto"}
    assert sp["Quarto"].ObjectType == "dormitório"
    qs = uel.get_pset(sp["Sala"], "Qto_SpaceBaseQuantities")
    assert qs["GrossFloorArea"] == pytest.approx(20.0) and qs["NetFloorArea"] == pytest.approx(20 - 4 * 0.05)
    assert sp["Sala"].Decomposes[0].RelatingObject.is_a("IfcBuildingStorey")
    assert not [w for w in warn if "inferida" in w]  # tem laje de teto medida


def test_rooms_disabled_makes_one_space_and_infers_ceiling(tmp_path):
    sg = apartment()
    sg.entities = [e for e in sg.entities if not isinstance(e, Space) and not (isinstance(e, Slab) and e.role == "ceiling")]
    path, warn = to_ifc(sg, tmp_path / "b.ifc", project_name="Sala")
    f = ifcopenshell.open(str(path))
    assert [s.Name for s in f.by_type("IfcSpace")] == ["Sala"]
    teto = [s for s in f.by_type("IfcSlab") if s.ObjectType == "teto"][0]
    assert uel.get_pset(teto, "TDYH_Origem")["Fonte"] == "inferida"
    assert any("teto inferida" in w for w in warn)


def test_building_shared_wall_once_and_duplicate_door_once(tmp_path):
    b, sc = two_rooms()
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dAB"),
                       b=bd.Ref(capture="B", element="dBA"), thickness=0.15)]
    merged, _ = bd.merge(b, sc, bd.solve_poses(b, sc))
    path, _ = to_ifc(merged, tmp_path / "e.ifc")
    f = ifcopenshell.open(str(path))
    walls = by_scene_id(f, "IfcWall")
    assert len(walls) == 7 and "B:w3" not in walls and "A:w1" in walls
    assert uel.get_pset(walls["A:w1"], "TDYH_Parametros")["CompartilhadaCom"] == "B:w3"
    assert uel.get_pset(walls["A:w1"], "Pset_WallCommon")["IsExternal"] is False
    doors = by_scene_id(f, "IfcDoor")
    assert set(doors) == {"A:dAB", "A:dEnt", "B:dIsca"}
    assert uel.get_pset(doors["A:dAB"], "TDYH_Parametros")["MesmaPortaQue"] == "B:dBA"
    # a parede compartilhada ocupa a faixa x 4.00–4.15 (face da sala até a face do quarto)
    v = verts(walls["A:w1"])
    assert v[:, 0].min() == pytest.approx(4.0, abs=1e-3) and v[:, 0].max() == pytest.approx(4.15, abs=1e-3)
    assert sorted(s.Name for s in f.by_type("IfcSpace")) == ["Quarto", "Sala"]


def test_opening_on_discarded_shared_wall_is_rehosted(tmp_path):
    b, sc = two_rooms()
    sc["B"].entities.append(Opening(id="jB", kind="window", host_wall="w3", offset=2.6, width=0.6,
                                    height=1.0, sill=1.1))  # janela só vista do lado do quarto
    b.links = [bd.Link(kind="door_match", a=bd.Ref(capture="A", element="dAB"),
                       b=bd.Ref(capture="B", element="dBA"), thickness=0.15)]
    merged, _ = bd.merge(b, sc, bd.solve_poses(b, sc))
    f = ifcopenshell.open(str(to_ifc(merged, tmp_path / "r.ifc")[0]))
    j = f.by_type("IfcWindow")[0]
    assert uel.get_pset(j, "TDYH_Parametros")["ParedeHospedeira"] == "A:w1"
    v = verts(j)
    # B:w3 vai de y=3.5 a -0.5 no mundo; offset 2.6..3.2 => y 0.9..0.3
    assert v[:, 1].min() == pytest.approx(0.3, abs=1e-3) and v[:, 1].max() == pytest.approx(0.9, abs=1e-3)


@pytest.mark.parametrize("seed", range(5))
def test_robust_rotation_translation_scale(tmp_path, seed):
    rng = np.random.default_rng(seed)
    rot, shift, k = rng.uniform(-np.pi, np.pi), tuple(rng.uniform(-50, 50, 2)), rng.uniform(0.8, 1.3)
    sg = apartment(rot, shift, k)
    f = ifcopenshell.open(str(to_ifc(sg, tmp_path / f"s{seed}.ifc")[0]))
    walls = by_scene_id(f, "IfcWall")
    for wid, w in walls.items():
        sw = sg.get(wid)
        L = np.linalg.norm(np.subtract(sw.end, sw.start))
        assert uel.get_pset(w, "Qto_WallBaseQuantities")["Length"] == pytest.approx(L, rel=1e-9)
        # a geometria fica do lado de fora da face medida (nunca invade o cômodo)
        v = verts(w)[:, :2]
        a, b = np.array(sw.start), np.array(sw.end)
        d = (b - a) / L
        n = np.array([-d[1], d[0]])
        side = (v - a) @ n
        assert (side.min() > -1e-6) or (side.max() < 1e-6)
        assert np.ptp(side) == pytest.approx(sw.thickness, abs=1e-6)
    sp = {s.Name: uel.get_pset(s, "Qto_SpaceBaseQuantities") for s in f.by_type("IfcSpace")}
    assert sp["Quarto"]["GrossFloorArea"] == pytest.approx(8 * k * k, abs=1e-3)  # área arredondada a 3 casas
    j = f.by_type("IfcWindow")[0]
    assert verts(j)[:, 2].min() == pytest.approx(1.0 * k, abs=1e-6)


def test_prior_scale_warns(tmp_path):
    sg = apartment()
    sg.scale_source = "prior"
    _, warn = to_ifc(sg, tmp_path / "p.ifc")
    assert any("aproximadas" in w for w in warn)


def test_cli_and_api(tmp_path, monkeypatch):
    from tdyh import cli
    sg = apartment()
    (tmp_path / "scene.json").write_text(sg.model_dump_json())
    cli.main(["ifc", str(tmp_path / "scene.json"), str(tmp_path / "c.ifc"), "--name", "Casa"])
    f = ifcopenshell.open(str(tmp_path / "c.ifc"))
    assert f.by_type("IfcProject")[0].Name == "Casa" and len(f.by_type("IfcWall")) == 5

    from fastapi.testclient import TestClient
    monkeypatch.setenv("TDYH_DATA", str(tmp_path / "data"))
    import importlib

    import tdyh.api.main as m
    importlib.reload(m)
    c = TestClient(m.app)
    pid = c.post("/projects", json={"name": "Apto 12"}).json()["id"]
    work = tmp_path / "data" / "projects" / pid / "work"
    work.mkdir(parents=True)
    assert c.get(f"/projects/{pid}/ifc").status_code == 409
    raw = apartment()
    raw.entities = [e for e in raw.entities if not isinstance(e, Space)]
    (work / "scene_measured.json").write_text(raw.model_dump_json())
    from tdyh import pipeline
    pipeline.publish(work)
    r = c.get(f"/projects/{pid}/ifc")
    assert r.status_code == 200 and r.content.startswith(b"ISO-10303-21")
    assert json.loads(r.headers["x-tdyh-warnings"]) == []
    (tmp_path / "api.ifc").write_bytes(r.content)
    f = ifcopenshell.open(str(tmp_path / "api.ifc"))
    assert f.by_type("IfcProject")[0].Name == "Apto 12"
    assert sorted(s.Name for s in f.by_type("IfcSpace")) == ["Quarto", "Sala"]


def test_valid_schema(tmp_path):
    import ifcopenshell.validate
    for sg in (apartment(), apartment(0.7, (3, -2), 1.1)):
        f = ifcopenshell.open(str(to_ifc(sg, tmp_path / "v.ifc")[0]))
        log = ifcopenshell.validate.json_logger()
        ifcopenshell.validate.validate(f, log, express_rules=True)
        assert log.statements == []
