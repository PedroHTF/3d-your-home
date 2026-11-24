"""Exportação IFC4 (BIM) a partir da cena final — de um levantamento ou da planta unificada.

Geometria (metros, Z para cima, piso em 0):
- IfcWall: perfil = mesma faixa da planta (face interna na linha medida, espessura para fora do
  cômodo, pontas estendidas nos cantos em L), extrudado pela altura. Paredes compartilhadas
  (`shared_with`) viram uma só; aberturas hospedadas na parede descartada são transferidas.
- IfcOpeningElement (vazio que atravessa a parede) + IfcDoor/IfcWindow (painel no meio do vão).
  Portas duplicadas (`duplicate_of`) entram uma vez só.
- IfcSlab de piso (para baixo a partir de 0) e de teto (para cima a partir da altura). Se a cena não
  tem laje de teto, ela é criada a partir do piso e marcada como inferida.
- IfcSpace por cômodo (nome, uso, área, área útil, perímetro), extrudado até o teto.
- Psets: parâmetros (TDYH_Parametros) e origem da medida (TDYH_Origem: medida/inferida/usuário,
  confiança, escala), além dos Psets/Qtos padrão do IFC4.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..scenegraph import Opening, SceneGraph, Slab, Space, Wall
from .plan2d import _center, _l_corner, _outward

SOURCE_PT = {"auto": "medida", "inferred": "inferida", "user": "usuário"}


def _require():
    try:
        import ifcopenshell  # noqa: F401
        import ifcopenshell.api  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Exportação IFC requer ifcopenshell: pip install 'tdyh[ifc]'") from e


def _prefix(eid: str) -> str | None:
    return eid.split(":")[0] if ":" in eid else None


def _floors(sg: SceneGraph) -> dict[str | None, Slab]:
    out = {}
    for s in sg.entities:
        if isinstance(s, Slab) and s.role == "floor":
            out.setdefault(_prefix(s.id), s)
    return out


def _wall_center(sg: SceneGraph, w: Wall, floors) -> np.ndarray:
    """Centro do cômodo de origem (na planta unificada, o piso da mesma captura)."""
    fl = floors.get(_prefix(w.id))
    return np.mean(fl.polygon, axis=0) if fl is not None else _center(sg)


def _project_on(w: Wall, p) -> float:
    a, b = np.array(w.start), np.array(w.end)
    d = (b - a) / np.linalg.norm(b - a)
    return float((np.asarray(p) - a) @ d)


def _rehost(o: Opening, src: Wall, dst: Wall) -> Opening:
    a, b = np.array(src.start), np.array(src.end)
    d = (b - a) / np.linalg.norm(b - a)
    t0, t1 = _project_on(dst, a + d * o.offset), _project_on(dst, a + d * (o.offset + o.width))
    o = o.model_copy()
    o.host_wall, o.offset = dst.id, round(min(t0, t1), 4)
    return o


def _matrix(origin2d, d2d, z: float = 0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, 0] = [d2d[0], d2d[1], 0]
    m[:3, 1] = [-d2d[1], d2d[0], 0]
    m[:3, 3] = [origin2d[0], origin2d[1], z]
    return m


def _is_internal(w: Wall, outward: np.ndarray, spaces: list[Space]) -> bool | None:
    """Externa se do outro lado da espessura não há cômodo. Sem cômodos: indeterminado (None)."""
    if not spaces:
        return None
    from ..rooms import _inside
    mid = (np.array(w.start) + np.array(w.end)) / 2
    q = mid + outward * (w.thickness + 0.05)
    return any(_inside(q, s.polygon) for s in spaces)


class _Writer:
    def __init__(self, project_name: str, building_name: str, storey_name: str):
        import importlib

        import ifcopenshell
        import ifcopenshell.api as api
        for mod in ("project", "root", "unit", "context", "aggregate", "geometry", "spatial", "pset", "feature"):
            importlib.import_module(f"ifcopenshell.api.{mod}")
        self.api = api
        f = self.f = api.project.create_file(version="IFC4")
        self.project = api.root.create_entity(f, ifc_class="IfcProject", name=project_name)
        api.unit.assign_unit(f, length={"is_metric": True, "raw": "METERS"},
                             area={"is_metric": True, "raw": "METERS"},
                             volume={"is_metric": True, "raw": "METERS"})
        model = api.context.add_context(f, context_type="Model")
        self.body = api.context.add_context(f, context_type="Model", context_identifier="Body",
                                            target_view="MODEL_VIEW", parent=model)
        self.site = api.root.create_entity(f, ifc_class="IfcSite", name="Terreno")
        self.building = api.root.create_entity(f, ifc_class="IfcBuilding", name=building_name)
        self.storey = api.root.create_entity(f, ifc_class="IfcBuildingStorey", name=storey_name)
        api.aggregate.assign_object(f, products=[self.site], relating_object=self.project)
        api.aggregate.assign_object(f, products=[self.building], relating_object=self.site)
        api.aggregate.assign_object(f, products=[self.storey], relating_object=self.building)
        for p in (self.site, self.building, self.storey):
            api.geometry.edit_object_placement(f, product=p, matrix=np.eye(4))
        self.storey.Elevation = 0.0
        self.ifcopenshell = ifcopenshell

    # --- geometria
    def extrusion(self, poly2d, depth: float, z0: float = 0.0):
        f = self.f
        pts = [tuple(map(float, p)) for p in poly2d]
        if _signed(pts) < 0:
            pts = pts[::-1]
        curve = f.createIfcPolyline([f.createIfcCartesianPoint(p) for p in pts + [pts[0]]])
        prof = f.createIfcArbitraryClosedProfileDef("AREA", None, curve)
        pos = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, float(z0))), None, None)
        solid = f.createIfcExtrudedAreaSolid(prof, pos, f.createIfcDirection((0.0, 0.0, 1.0)), float(depth))
        return f.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])

    def product(self, ifc_class: str, name: str, matrix: np.ndarray, rep, **attrs):
        api, f = self.api, self.f
        e = api.root.create_entity(f, ifc_class=ifc_class, name=name, **({"predefined_type": attrs.pop("predefined_type")}
                                                                        if "predefined_type" in attrs else {}))
        for k, v in attrs.items():
            setattr(e, k, v)
        api.geometry.edit_object_placement(f, product=e, matrix=matrix)
        api.geometry.assign_representation(f, product=e, representation=rep)
        return e

    def contain(self, elems):
        if elems:
            self.api.spatial.assign_container(self.f, products=elems, relating_structure=self.storey)

    def pset(self, e, name: str, props: dict):
        props = {k: v for k, v in props.items() if v is not None}
        if not props:
            return
        if name.startswith("Qto_"):
            q = self.api.pset.add_qto(self.f, product=e, name=name)
            self.api.pset.edit_qto(self.f, qto=q, properties=props)
        else:
            p = self.api.pset.add_pset(self.f, product=e, name=name)
            self.api.pset.edit_pset(self.f, pset=p, properties=props)

    def origin(self, e, ent, sg: SceneGraph, extra: dict | None = None):
        self.pset(e, "TDYH_Origem", {
            "Fonte": SOURCE_PT.get(ent.source, ent.source), "Confianca": float(ent.confidence),
            "IdCena": ent.id, "AjustadoPeloUsuario": ent.id in sg.overrides,
            "FonteEscala": sg.scale_source,
            "IncertezaEscala": float(sg.scale_uncertainty) if sg.scale_uncertainty is not None else None,
            **(extra or {})})


def _signed(pts) -> float:
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])) / 2


def to_ifc(sg: SceneGraph, path: Path, project_name: str = "3D Your Home",
           building_name: str = "Edificação", storey_name: str = "Térreo") -> tuple[Path, list[str]]:
    """Gera o IFC4 da cena final. Devolve (caminho, avisos)."""
    _require()
    w_ = _Writer(project_name, building_name, storey_name)
    api, f = w_.api, w_.f
    warn: list[str] = []
    floors = _floors(sg)
    all_walls = [e for e in sg.entities if isinstance(e, Wall)]
    by_id = {w.id: w for w in all_walls}
    kept = [w for w in all_walls if not (w.shared_with and w.shared_with in by_id)]
    spaces = [e for e in sg.entities if isinstance(e, Space)]
    if not spaces:  # sem divisão em cômodos: o levantamento inteiro é um ambiente (como na planta unificada)
        for pref, fl in floors.items():
            a = abs(_signed([tuple(p) for p in fl.polygon]))
            per = sum(math.dist(fl.polygon[i], fl.polygon[(i + 1) % len(fl.polygon)]) for i in range(len(fl.polygon)))
            spaces.append(Space(id=f"{fl.id}:ambiente", name=project_name if pref is None else pref,
                                polygon=fl.polygon, area=round(a, 3), net_area=round(a, 3), perimeter=round(per, 3)))
    if sg.scale_source in ("prior", "unknown"):
        warn.append(f"Escala '{sg.scale_source}': as dimensões do IFC são aproximadas.")

    # ---- paredes
    ifc_walls, wall_frame = {}, {}
    for w in kept:
        a, b = np.array(w.start, float), np.array(w.end, float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        d = (b - a) / L
        out = _outward(w, _wall_center(sg, w, floors))
        s = 1.0 if out @ np.array([-d[1], d[0]]) > 0 else -1.0
        e0, e1 = _l_corner(w, "start", all_walls), _l_corner(w, "end", all_walls)
        prof = [(-e0, 0.0), (L + e1, 0.0), (L + e1, s * w.thickness), (-e0, s * w.thickness)]
        M = _matrix(a, d)
        name = f"Parede {w.id}"
        ifc = w_.product("IfcWall", name, M, w_.extrusion(prof, w.height), predefined_type="STANDARD")
        internal = _is_internal(w, out, spaces) or bool(w.shared_with) or any(o.shared_with == w.id for o in all_walls)
        if not spaces and not internal:
            internal = None
        shared = [o.id for o in all_walls if o.shared_with == w.id]
        w_.pset(ifc, "Pset_WallCommon", {"IsExternal": (not internal) if internal is not None else None})
        w_.pset(ifc, "Qto_WallBaseQuantities", {"Length": L, "Height": float(w.height), "Width": float(w.thickness),
                                                "GrossSideArea": L * w.height})
        w_.pset(ifc, "TDYH_Parametros", {"Inicio": f"{a[0]:.4f}, {a[1]:.4f}", "Fim": f"{b[0]:.4f}, {b[1]:.4f}",
                                         "Comprimento": L, "Espessura": float(w.thickness), "Altura": float(w.height),
                                         "LadoEspessura": "esquerda" if s > 0 else "direita",
                                         "CompartilhadaCom": ", ".join(shared) or None})
        w_.origin(ifc, w, sg)
        ifc_walls[w.id], wall_frame[w.id] = ifc, (M, s, L)
    w_.contain(list(ifc_walls.values()))

    # ---- aberturas
    elems = []
    for o in [e for e in sg.entities if isinstance(e, Opening)]:
        if o.duplicate_of:
            continue
        host = by_id.get(o.host_wall)
        if host is None:
            warn.append(f"Abertura {o.id}: parede {o.host_wall} não existe; ignorada.")
            continue
        if host.id not in ifc_walls and host.shared_with in ifc_walls:
            o = _rehost(o, host, by_id[host.shared_with])
            host = by_id[o.host_wall]
        if host.id not in ifc_walls:
            continue
        M, s, L = wall_frame[host.id]
        x0, x1 = max(0.0, o.offset), min(L, o.offset + o.width)
        top = min(o.sill + o.height, host.height)
        if x1 - x0 < 1e-3 or top - o.sill < 1e-3:
            warn.append(f"Abertura {o.id} fora da parede {host.id}; ignorada.")
            continue
        if x0 != o.offset or x1 != o.offset + o.width or top != o.sill + o.height:
            warn.append(f"Abertura {o.id} excede a parede {host.id}; recortada no IFC.")
        t = host.thickness
        y0, y1 = sorted((-0.02 * s, s * (t + 0.02)))
        void = w_.product("IfcOpeningElement", f"Vão {o.id}", M,
                          w_.extrusion([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], top - o.sill, o.sill),
                          predefined_type="OPENING")
        api.feature.add_feature(f, feature=void, element=ifc_walls[host.id])
        is_door = o.kind.value == "door"
        pw = 0.04 if is_door else 0.02
        yc = s * t / 2
        panel = [(x0, yc - pw / 2), (x1, yc - pw / 2), (x1, yc + pw / 2), (x0, yc + pw / 2)]
        cls = "IfcDoor" if is_door else "IfcWindow"
        fill = w_.product(cls, f"{'Porta' if is_door else 'Janela'} {o.id}", M,
                          w_.extrusion(panel, top - o.sill, o.sill),
                          predefined_type="DOOR" if is_door else "WINDOW",
                          OverallWidth=float(x1 - x0), OverallHeight=float(top - o.sill))
        api.feature.add_filling(f, opening=void, element=fill)
        w_.pset(fill, "Qto_DoorBaseQuantities" if is_door else "Qto_WindowBaseQuantities",
                {"Width": float(x1 - x0), "Height": float(top - o.sill), "Area": float((x1 - x0) * (top - o.sill))})
        dup = [x.id for x in sg.entities if isinstance(x, Opening) and x.duplicate_of == o.id]
        w_.pset(fill, "TDYH_Parametros", {"Largura": float(o.width), "Altura": float(o.height), "Peitoril": float(o.sill),
                                          "Deslocamento": float(o.offset), "ParedeHospedeira": host.id,
                                          "ProfundidadeVao": float(o.depth) if o.depth is not None else None,
                                          "MesmaPortaQue": ", ".join(dup) or None})
        w_.origin(fill, o, sg, {"EspessuraDeterminada": o.depth is not None or None})
        elems.append(fill)
    w_.contain(elems)

    # ---- lajes
    slabs = [e for e in sg.entities if isinstance(e, Slab)]
    ceil_pref = {_prefix(s.id) for s in slabs if s.role == "ceiling"}
    hmax = {}
    for w in all_walls:
        hmax[_prefix(w.id)] = max(hmax.get(_prefix(w.id), 0.0), w.height)
    for pref, fl in floors.items():
        if pref not in ceil_pref:
            slabs.append(Slab(id=f"{fl.id}:teto", polygon=fl.polygon, elevation=hmax.get(pref, 2.7),
                              role="ceiling", source="inferred", confidence=0.5))
            warn.append(f"Laje de teto inferida a partir do piso {fl.id} (altura {hmax.get(pref, 2.7):.2f} m).")
    selems = []
    for s in slabs:
        if len(s.polygon) < 3:
            continue
        is_floor = s.role == "floor"
        z0 = s.elevation - s.thickness if is_floor else s.elevation
        e = w_.product("IfcSlab", f"{'Piso' if is_floor else 'Laje de teto'} {s.id}", np.eye(4),
                       w_.extrusion(s.polygon, s.thickness, z0), predefined_type="FLOOR",
                       ObjectType="piso" if is_floor else "teto")
        a = abs(_signed([tuple(p) for p in s.polygon]))
        w_.pset(e, "Qto_SlabBaseQuantities", {"Depth": float(s.thickness), "GrossArea": a})
        w_.pset(e, "TDYH_Parametros", {"Funcao": "piso" if is_floor else "teto", "Cota": float(s.elevation),
                                       "Espessura": float(s.thickness)})
        w_.origin(e, s, sg)
        selems.append(e)
    w_.contain(selems)

    # ---- cômodos
    ifc_spaces = []
    ceil_z = {_prefix(s.id): s.elevation for s in slabs if s.role == "ceiling"}
    default_h = max(hmax.values()) if hmax else 2.7
    for sp in spaces:
        h = ceil_z.get(_prefix(sp.id), default_h)
        e = w_.product("IfcSpace", sp.name, np.eye(4), w_.extrusion(sp.polygon, h, 0.0),
                       predefined_type="INTERNAL", LongName=sp.name, ObjectType=sp.use)
        w_.pset(e, "Qto_SpaceBaseQuantities", {"GrossFloorArea": float(sp.area), "NetFloorArea": float(sp.net_area),
                                               "GrossPerimeter": float(sp.perimeter), "Height": float(h)})
        w_.pset(e, "TDYH_Parametros", {"Nome": sp.name, "Uso": sp.use, "Area": float(sp.area),
                                       "AreaUtil": float(sp.net_area), "Perimetro": float(sp.perimeter)})
        w_.origin(e, sp, sg)
        ifc_spaces.append(e)
    if ifc_spaces:
        api.aggregate.assign_object(f, products=ifc_spaces, relating_object=w_.storey)
    else:
        warn.append("Sem cômodos (divisão em cômodos desligada): nenhum IfcSpace gerado.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(path))
    return path, warn
