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
    ch = sub.add_parser("charuco", help="gera PDF A4 do tabuleiro ChArUco para calibrar lentes")
    ch.add_argument("out", type=Path, nargs="?", default=Path("tabuleiro_charuco.pdf"))
    ca = sub.add_parser("calibrate", help="calibra uma lente com fotos (ou vídeo) do tabuleiro ChArUco")
    ca.add_argument("src", type=Path, help="pasta com fotos ou um vídeo do tabuleiro")
    ca.add_argument("--square-mm", type=float, default=None,
                    help="lado REAL de uma casa medido no papel (padrão 25)")
    ca.add_argument("--fisheye", action="store_true", help="modelo olho-de-peixe (FOV > 130°)")
    ca.add_argument("--profiles", type=Path, default=None, help="arquivo de perfis (padrão <TDYH_DATA>/lenses.json)")
    ls = sub.add_parser("lenses", help="lista perfis de lente calibrados ou as lentes de uma pasta")
    ls.add_argument("folder", type=Path, nargs="?", help="pasta de fotos: mostra lente/modelo de cada grupo")
    a = ap.parse_args(argv)
    if a.cmd == "charuco":
        from . import charuco
        print(charuco.make_pdf(a.out))
        return
    if a.cmd == "calibrate":
        print(json.dumps(_calibrate(a.src, a.square_mm, a.fisheye, a.profiles), indent=2, ensure_ascii=False))
        return
    if a.cmd == "lenses":
        from . import lens
        store = lens.ProfileStore(pipeline.lens_profiles_path())
        if a.folder:
            from . import ingest
            media = [ingest.read_image_meta(p) for p in sorted(a.folder.iterdir())
                     if ingest.classify(p) == "image"]
            print(json.dumps(lens.summarize(lens.plan_cameras(media, a.folder, store)), indent=2, ensure_ascii=False))
        else:
            print(json.dumps([p.model_dump() for p in store.load()], indent=2, ensure_ascii=False))
        return
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


def _calibrate(src: Path, square_mm: float | None, fisheye: bool, profiles: Path | None) -> dict:
    """Calibra a lente das fotos/vídeo do tabuleiro e salva o perfil (identificado pelo EXIF)."""
    import tempfile

    from . import charuco, ingest, lens
    if src.is_file() and ingest.classify(src) == "video":
        meta = ingest.read_video_meta(src)
        key = lens.lens_key(meta.make, meta.model, None, "video")
        tmp = Path(tempfile.mkdtemp())
        images = charuco.frames_from_video(src, tmp)
    else:
        images = sorted(p for p in src.iterdir() if ingest.classify(p) == "image")
        if not images:
            raise SystemExit("nenhuma imagem na pasta")
        keys = {}
        for p in images:
            m = ingest.read_image_meta(p)
            keys.setdefault(lens.lens_key(m.make, m.model, m.focal_mm), []).append(p)
        if len(keys) > 1:
            raise SystemExit("a pasta mistura lentes diferentes: " + "; ".join(f"{k} ({len(v)} fotos)" for k, v in keys.items())
                             + ". Separe uma pasta por lente.")
        key = next(iter(keys))
        tmp = Path(tempfile.mkdtemp())  # orientação EXIF aplicada (igual ao pipeline)
        images = [ingest.normalize_image(p, tmp / f"{k:04d}.jpg") for k, p in enumerate(images)]
    cal = charuco.calibrate(images, key, fisheye=fisheye, measured_square_mm=square_mm)
    lens.ProfileStore(profiles or pipeline.lens_profiles_path()).save(cal.profile)
    return {"perfil": cal.profile.model_dump(), "fotos": cal.per_image}


if __name__ == "__main__":
    main()
