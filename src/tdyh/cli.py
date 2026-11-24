"""CLI: tdyh run | plan | markers | ifc"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pipeline


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tdyh")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="executa o pipeline local")
    r.add_argument("src", type=Path)
    r.add_argument("project", type=Path)
    r.add_argument("--recipe", default="auto", choices=["auto", "classic", "neural", "hybrid"])
    pl = sub.add_parser("plan", help="gera planta (DXF/SVG) a partir de uma nuvem densa PLY")
    pl.add_argument("ply", type=Path)
    pl.add_argument("project", type=Path)
    g = pl.add_mutually_exclusive_group()
    g.add_argument("--scale", type=float, help="fator modelo->metros já conhecido")
    g.add_argument("--marker", type=float, metavar="LADO_M",
                   help="lado do marcador ArUco 4x4 impresso (m); requer --images e --colmap-txt")
    pl.add_argument("--images", type=Path)
    pl.add_argument("--colmap-txt", type=Path, help="modelo COLMAP em TXT (model_converter)")
    mk = sub.add_parser("markers", help="gera PDF A4 com marcadores ArUco para imprimir")
    mk.add_argument("out", type=Path, nargs="?", default=Path("marcadores.pdf"))
    mk.add_argument("-n", type=int, default=6, help="quantidade (1 a 50)")
    mk.add_argument("--start", type=int, default=0, help="primeiro ID (use IDs diferentes por cômodo)")
    mk.add_argument("--side-cm", type=float, default=15.0)
    ic = sub.add_parser("ifc", help="exporta IFC4 (BIM) de uma cena final (scene.json)")
    ic.add_argument("scene", type=Path, help="scene.json de um levantamento ou da edificação")
    ic.add_argument("out", type=Path, nargs="?", default=Path("modelo.ifc"))
    ic.add_argument("--name", default="3D Your Home", help="nome do projeto/edificação no IFC")
    a = ap.parse_args(argv)
    if a.cmd == "ifc":
        from .exporters.ifc import to_ifc
        from .scenegraph import SceneGraph
        sg = SceneGraph.model_validate_json(a.scene.read_text())
        path, warn = to_ifc(sg, a.out, project_name=a.name, building_name=a.name)
        for w in warn:
            print("aviso:", w)
        print(path)
        return
    if a.cmd == "markers":
        from . import markers
        ids = range(a.start, a.start + a.n)
        if a.n < 1 or ids[-1] > 49:
            ap.error("IDs devem ficar entre 0 e 49")
        print(markers.make_pdf(a.out, ids, a.side_cm * 10))
        return
    if a.cmd == "plan":
        from . import scale as sc
        s, src = a.scale or 1.0, "user_measure" if a.scale else "unknown"
        if a.marker:
            est = sc.scale_from_images(a.images, a.colmap_txt, a.marker)
            s, src = est.factor, "marker"
            print(f"escala por marcador: {s:.5f} (±{est.uncertainty:.1%}, {est.samples} arestas)")
        a.project.mkdir(parents=True, exist_ok=True)
        print(json.dumps(pipeline.make_plan(a.ply, a.project, s, src, colmap_txt=a.colmap_txt, images_dir=a.images), indent=2, ensure_ascii=False))
        return
    if a.cmd == "run":
        rep = pipeline.run(a.src, a.project, a.recipe,
                           progress=lambda s, p, _: print(f"[{p:4.0%}] {s}"))
        print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
