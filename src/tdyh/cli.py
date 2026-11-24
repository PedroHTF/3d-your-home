"""CLI: tdyh run <pasta-ou-video> <pasta-do-projeto> [--recipe auto|classic|neural|hybrid]"""

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
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        from . import scale as sc
        s, src, up = a.scale or 1.0, "user_measure" if a.scale else "unknown", None
        if a.colmap_txt:
            up = sc.camera_up_hint(sc.read_colmap_text(a.colmap_txt))
        if a.marker:
            est = sc.scale_from_images(a.images, a.colmap_txt, a.marker)
            s, src = est.factor, "marker"
            print(f"escala por marcador: {s:.5f} (±{est.uncertainty:.1%}, {est.samples} arestas)")
        a.project.mkdir(parents=True, exist_ok=True)
        print(json.dumps(pipeline.make_plan(a.ply, a.project, s, src, up), indent=2, ensure_ascii=False))
        return
    if a.cmd == "run":
        rep = pipeline.run(a.src, a.project, a.recipe,
                           progress=lambda s, p, _: print(f"[{p:4.0%}] {s}"))
        print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
