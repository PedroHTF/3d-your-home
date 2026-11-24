"""Pré-visualização de um IFC: tessela com ifcopenshell.geom (aplica os vazios) e desenha em
perspectiva com matplotlib. Uso: python scripts/render_ifc.py arquivo.ifc saida.png"""

import sys

import ifcopenshell
import ifcopenshell.geom
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COL = {"IfcWall": "#d9d4c7", "IfcDoor": "#8b5a2b", "IfcWindow": "#6fb7e8", "IfcSlab": "#9e9e9e",
       "IfcSpace": "#7fc97f"}


def render(path, out, elev=35, azim=-60, skip=("IfcSpace",), ceiling=False):
    f = ifcopenshell.open(path)
    st = ifcopenshell.geom.settings()
    st.set("use-world-coords", True)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    allv = []
    for e in f.by_type("IfcProduct"):
        if not e.Representation or e.is_a("IfcOpeningElement") or e.is_a() in skip:
            continue
        if e.is_a("IfcSlab") and e.ObjectType == "teto" and not ceiling:
            continue
        sh = ifcopenshell.geom.create_shape(st, e)
        v = np.array(sh.geometry.verts).reshape(-1, 3)
        t = np.array(sh.geometry.faces).reshape(-1, 3)
        ax.add_collection3d(Poly3DCollection(v[t], facecolor=COL.get(e.is_a(), "#ccc"), edgecolor="k",
                                             linewidths=0.1, alpha=0.95))
        allv.append(v)
    V = np.vstack(allv)
    lo, hi = V.min(0), V.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(0, 2 * r)
    ax.view_init(elev, azim)
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(out, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
