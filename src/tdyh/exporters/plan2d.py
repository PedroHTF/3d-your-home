"""Planta baixa 2D: DXF (ezdxf) e SVG, em metros (DXF) / escala 1:50 em mm (SVG)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..scenegraph import Opening, SceneGraph, Slab, Wall


def _center(sg: SceneGraph) -> np.ndarray:
    floor = [s for s in sg.entities if isinstance(s, Slab) and s.role == "floor"]
    if floor:
        return np.mean(floor[0].polygon, axis=0)
    ws = [np.array(p) for w in sg.entities if isinstance(w, Wall) for p in (w.start, w.end)]
    return np.mean(ws, axis=0) if ws else np.zeros(2)


def _outward(w: Wall, center: np.ndarray) -> np.ndarray:
    a, b = np.array(w.start), np.array(w.end)
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    return n if (((a + b) / 2 - center) @ n) > 0 else -n


def _wall_polygon(w: Wall, center: np.ndarray | None = None) -> np.ndarray:
    """Face interna = linha medida; espessura para fora do cômodo; pontas estendidas para fechar cantos."""
    a, b = np.array(w.start), np.array(w.end)
    d = (b - a) / np.linalg.norm(b - a)
    n = (_outward(w, center) if center is not None else np.array([-d[1], d[0]])) * w.thickness
    a, b = a - d * w.thickness, b + d * w.thickness
    return np.array([a, b, b + n, a + n])


def _opening_geom(o: Opening, w: Wall, center: np.ndarray):
    a, b = np.array(w.start), np.array(w.end)
    d = (b - a) / np.linalg.norm(b - a)
    n = _outward(w, center) * w.thickness
    p0, p1 = a + d * o.offset, a + d * (o.offset + o.width)
    return p0, p1, n, d


def _bounds(sg: SceneGraph) -> tuple[np.ndarray, np.ndarray]:
    c = _center(sg)
    pts = [p for w in sg.entities if isinstance(w, Wall) for p in _wall_polygon(w, c)]
    pts += [p for s in sg.entities if isinstance(s, Slab) for p in s.polygon]
    arr = np.array(pts) if pts else np.zeros((1, 2))
    return arr.min(axis=0), arr.max(axis=0)


def to_dxf(sg: SceneGraph, path: Path) -> Path:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment

    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.M
    for name, color in (("PAREDES", 7), ("PISO", 8), ("ABERTURAS", 3), ("COTAS", 1)):
        doc.layers.add(name, color=color)
    st = doc.dimstyles.new("COTA_M")
    st.dxf.dimlfac, st.dxf.dimdec, st.dxf.dimtxt = 1.0, 2, 0.12   # metros, 2 casas
    st.dxf.dimasz, st.dxf.dimexo, st.dxf.dimexe, st.dxf.dimgap = 0.08, 0.05, 0.05, 0.03
    st.dxf.dimtsz, st.dxf.dimtad, st.dxf.dimzin = 0.06, 1, 0                         # tique, texto acima, mantém zeros (4,20)
    msp = doc.modelspace()
    walls = {e.id: e for e in sg.entities if isinstance(e, Wall)}
    center = _center(sg)
    for w in walls.values():
        poly = _wall_polygon(w, center)
        msp.add_lwpolyline([tuple(p) for p in poly], close=True, dxfattribs={"layer": "PAREDES"})
        hatch = msp.add_hatch(color=7, dxfattribs={"layer": "PAREDES"})
        hatch.paths.add_polyline_path([tuple(p) for p in poly], is_closed=True)
        a, b = np.array(w.start), np.array(w.end)
        if (b[0] - a[0]) < -1e-6 or (abs(b[0] - a[0]) <= 1e-6 and b[1] < a[1]):
            a, b = b, a  # texto sempre legível (esquerda->direita / baixo->cima)
        d = (b - a) / np.linalg.norm(b - a)
        left = np.array([-d[1], d[0]])
        side = 1 if (-_outward(w, center) @ left) > 0 else -1  # cota do lado de dentro
        msp.add_aligned_dim(p1=tuple(a), p2=tuple(b), distance=0.35 * side,
                            dimstyle="COTA_M", dxfattribs={"layer": "COTAS"}).render()
    for s in sg.entities:
        if isinstance(s, Slab) and s.role == "floor":
            msp.add_lwpolyline(s.polygon, close=True, dxfattribs={"layer": "PISO", "linetype": "DASHED"})
            c = np.mean(s.polygon, axis=0)
            area = _area(s.polygon)
            msp.add_text(f"{area:.2f} m²", height=0.15, dxfattribs={"layer": "COTAS"}) \
               .set_placement(tuple(c), align=TextEntityAlignment.MIDDLE_CENTER)
    for o in sg.entities:
        if isinstance(o, Opening) and o.host_wall in walls:
            p0, p1, n, d = _opening_geom(o, walls[o.host_wall], center)
            msp.add_lwpolyline([tuple(p0), tuple(p1), tuple(p1 + n), tuple(p0 + n)], close=True,
                               dxfattribs={"layer": "ABERTURAS"})
            if o.kind.value == "door":
                inward = -n / np.linalg.norm(n)
                if o.source == "inferred":
                    msp.add_text("PORTA?", height=0.1, dxfattribs={"layer": "ABERTURAS"}).set_placement(
                        tuple((p0 + p1) / 2 + inward * 0.2), align=TextEntityAlignment.MIDDLE_CENTER)
                a0 = np.degrees(np.arctan2(d[1], d[0]))
                a1 = np.degrees(np.arctan2(inward[1], inward[0]))
                cross = d[0] * inward[1] - d[1] * inward[0]
                start, end = (a0, a1) if cross > 0 else (a1, a0)  # arcos DXF são anti-horários
                msp.add_arc(tuple(p0), o.width, start, end, dxfattribs={"layer": "ABERTURAS"})
                msp.add_line(tuple(p0), tuple(p0 + inward * o.width), dxfattribs={"layer": "ABERTURAS"})
            else:
                mid = n / 2
                msp.add_line(tuple(p0 + mid), tuple(p1 + mid), dxfattribs={"layer": "ABERTURAS"})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path


def _area(poly) -> float:
    p = np.array(poly)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def to_svg(sg: SceneGraph, path: Path, scale: int = 50, margin_mm: float = 20) -> Path:
    """SVG em mm na escala 1:scale (padrão 1:50), Y invertido para leitura em planta."""
    lo, hi = _bounds(sg)
    k = 1000 / scale  # mm de papel por metro
    W, H = (hi - lo) * k + 2 * margin_mm

    def tx(p):
        return f"{(p[0] - lo[0]) * k + margin_mm:.2f},{(hi[1] - p[1]) * k + margin_mm:.2f}"

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.1f}mm" height="{H:.1f}mm" '
           f'viewBox="0 0 {W:.1f} {H:.1f}" font-family="sans-serif">',
           f'<rect width="100%" height="100%" fill="white"/>']
    for s in sg.entities:
        if isinstance(s, Slab) and s.role == "floor":
            pts = " ".join(tx(p) for p in s.polygon)
            out.append(f'<polygon points="{pts}" fill="#f4f1ea" stroke="#bbb" stroke-dasharray="2,1" stroke-width="0.2"/>')
            c = np.mean(s.polygon, axis=0)
            x, y = tx(c).split(",")
            out.append(f'<text x="{x}" y="{y}" font-size="3" text-anchor="middle">{_area(s.polygon):.2f} m²</text>')
    center = _center(sg)
    walls = {e.id: e for e in sg.entities if isinstance(e, Wall)}
    for w in walls.values():
        pts = " ".join(tx(p) for p in _wall_polygon(w, center))
        out.append(f'<polygon points="{pts}" fill="#222" stroke="#000" stroke-width="0.25"/>')
    for o in sg.entities:
        if isinstance(o, Opening) and o.host_wall in walls:
            p0, p1, n, d = _opening_geom(o, walls[o.host_wall], center)
            e = n / np.linalg.norm(n) * 0.01
            gap = " ".join(tx(p) for p in (p0 - e, p1 - e, p1 + n + e, p0 + n + e))
            out.append(f'<polygon points="{gap}" fill="white"/>')
            if o.kind.value == "door":
                inward = -n / np.linalg.norm(n)
                tip, arc_end = p0 + inward * o.width, p1
                r = o.width * k
                cross = d[0] * inward[1] - d[1] * inward[0]
                sweep = 1 if cross > 0 else 0  # Y do SVG é invertido: CCW no mundo = horário na tela
                out.append(f'<path d="M {tx(p0)} L {tx(tip)} A {r:.2f} {r:.2f} 0 0 {sweep} {tx(arc_end)}" '
                           f'fill="none" stroke="#2a7" stroke-width="0.3"'
                           + (' stroke-dasharray="1.5,1"' if o.source == "inferred" else "") + '/>')
            else:
                m0, m1 = p0 + n / 2, p1 + n / 2
                out.append(f'<polyline points="{tx(m0)} {tx(m1)}" stroke="#27a" stroke-width="0.4"/>')
    for w in walls.values():
        if True:
            a, b = np.array(w.start), np.array(w.end)
            m = (a + b) / 2
            d = (b - a) / np.linalg.norm(b - a)
            lab = m - _outward(w, center) * 0.25
            x, y = tx(lab).split(",")
            ang = -np.degrees(np.arctan2(d[1], d[0]))
            if ang > 90 or ang < -90:
                ang += 180
            out.append(f'<text x="{x}" y="{y}" font-size="2.5" text-anchor="middle" '
                       f'transform="rotate({ang:.1f} {x} {y})">{np.linalg.norm(b - a):.2f}</text>')
    out.append(f'<text x="{margin_mm}" y="{H - 5:.1f}" font-size="3">Escala 1:{scale} — cotas em metros</text>')
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    return path
