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
    a = ap.parse_args(argv)
    if a.cmd == "run":
        rep = pipeline.run(a.src, a.project, a.recipe,
                           progress=lambda s, p, _: print(f"[{p:4.0%}] {s}"))
        print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
