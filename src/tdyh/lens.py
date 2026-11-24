"""Lentes e modelos de câmera.

Cada lente (ultra-wide 0,6x, principal 1x, tele 3x…) é uma câmera diferente para o COLMAP: focal e
distorção próprias. Este módulo:

1. identifica a lente de cada imagem pelo EXIF (fabricante, modelo, distância focal, resolução);
   frames de vídeo herdam a identidade do vídeo;
2. escolhe o modelo de câmera pelo campo de visão (FOV):
   - tele/normal (FOV < 75°)        -> RADIAL           (f, cx, cy, k1, k2)
   - grande-angular (75–100°)       -> OPENCV           (fx, fy, cx, cy, k1, k2, p1, p2)
   - ultra-wide (100–130°)          -> OPENCV           (o celular já desentorta parte)
   - olho-de-peixe (> 130°)         -> OPENCV_FISHEYE   (fx, fy, cx, cy, k1..k4)
   sem EXIF: OPENCV (autocalibração cobre a maioria dos casos);
3. usa um perfil calibrado (tabuleiro ChArUco) quando existe para aquela lente: os parâmetros
   entram fixos no COLMAP e a reconstrução fica mais estável e mais exata.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

FULL_FRAME_DIAG_MM = math.hypot(36.0, 24.0)  # 43,27 mm


class LensProfile(BaseModel):
    """Calibração de uma lente numa resolução (vale para outras resoluções de mesma proporção)."""

    key: str
    camera_model: str                     # "OPENCV" | "OPENCV_FISHEYE"
    width: int
    height: int
    params: list[float]                   # na ordem do COLMAP
    rms_px: float | None = None
    images_used: int = 0
    coverage: float | None = None         # fração da imagem coberta por cantos detectados
    source: str = "charuco"
    notes: list[str] = Field(default_factory=list)

    def scaled(self, width: int, height: int) -> list[float] | None:
        """Parâmetros para outra resolução (mesma proporção). Distorção é adimensional."""
        if self.width * height != self.height * width:
            return None
        s = width / self.width
        p = list(self.params)
        for i in range(4):  # fx, fy, cx, cy (todos os modelos usados aqui começam assim)
            p[i] *= s
        return p


def lens_key(make: str | None, model: str | None, focal_mm: float | None, kind: str = "image") -> str:
    """Identidade da lente (sem resolução: a mesma lente fotografa em várias)."""
    f = f"{focal_mm:.2f}mm" if focal_mm else "f?"
    return f"{(make or '?').strip()}|{(model or '?').strip()}|{f}|{kind}"


def fov_deg(focal_35mm: float | None) -> float | None:
    """Campo de visão diagonal pela focal equivalente em 35 mm."""
    if not focal_35mm or focal_35mm <= 0:
        return None
    return math.degrees(2 * math.atan(FULL_FRAME_DIAG_MM / (2 * focal_35mm)))


def lens_class(focal_35mm: float | None) -> str:
    fov = fov_deg(focal_35mm)
    if fov is None:
        return "desconhecida"
    if fov > 130:
        return "olho-de-peixe"
    if fov > 100:
        return "ultra-wide"
    if fov > 75:
        return "grande-angular"
    return "normal/tele"


def choose_model(focal_35mm: float | None) -> str:
    c = lens_class(focal_35mm)
    return {"olho-de-peixe": "OPENCV_FISHEYE", "normal/tele": "RADIAL"}.get(c, "OPENCV")


def focal_px(focal_35mm: float | None, width: int, height: int) -> float | None:
    """Focal inicial em pixels (a focal 35 mm é definida pela diagonal)."""
    if not focal_35mm:
        return None
    return focal_35mm * math.hypot(width, height) / FULL_FRAME_DIAG_MM


@dataclass
class CameraGroup:
    """Imagens que o COLMAP deve tratar como UMA câmera."""

    key: str
    width: int
    height: int
    camera_model: str
    images: list[str] = field(default_factory=list)   # nomes de arquivo na pasta de imagens
    params: list[float] | None = None                 # fixos (perfil calibrado) ou chute inicial
    fixed: bool = False                               # True: não refinar no ajuste de feixes
    lens: str = "desconhecida"
    note: str = ""


class ProfileStore:
    """Perfis de lente em JSON (padrão: <TDYH_DATA>/lenses.json)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list[LensProfile]:
        if not self.path.exists():
            return []
        return [LensProfile.model_validate(p) for p in json.loads(self.path.read_text())]

    def save(self, prof: LensProfile) -> None:
        items = [p for p in self.load() if not (p.key == prof.key and p.width * prof.height == p.height * prof.width)]
        items.append(prof)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([p.model_dump() for p in items], indent=2, ensure_ascii=False))

    def delete(self, key: str) -> bool:
        items = self.load()
        keep = [p for p in items if p.key != key]
        if len(keep) == len(items):
            return False
        self.path.write_text(json.dumps([p.model_dump() for p in keep], indent=2, ensure_ascii=False))
        return True

    def find(self, key: str, width: int, height: int) -> tuple[LensProfile, list[float]] | None:
        for p in self.load():
            if p.key == key:
                params = p.scaled(width, height)
                if params is not None:
                    return p, params
        return None


def plan_cameras(media: list, images_dir: Path, profiles: ProfileStore | None = None) -> list[CameraGroup]:
    """Agrupa as imagens da pasta por lente+resolução e escolhe modelo/parâmetros de cada grupo.

    media: lista de ingest.MediaInfo. Fotos são achadas pelo caminho; frames de vídeo pelo prefixo
    (frames.process_video nomeia '<nome-do-vídeo>_...'). Arquivos sem origem conhecida ficam num
    grupo próprio por resolução."""
    from PIL import Image

    by_path = {Path(m.path).name: m for m in media if m.kind == "image"}
    videos = {Path(m.path).stem: m for m in media if m.kind == "video"}
    groups: dict[tuple, CameraGroup] = {}
    files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for f in files:
        with Image.open(f) as im:
            w, h = im.size
        m = by_path.get(f.name)
        if m is None:
            m = next((v for stem, v in videos.items() if f.name.startswith(stem + "_")), None)
        kind = getattr(m, "kind", "image") if m else "unknown"
        key = lens_key(getattr(m, "make", None), getattr(m, "model", None), getattr(m, "focal_mm", None), kind)
        if m is None:
            key = "desconhecida|?|f?|unknown"
        if kind == "video":  # cada vídeo pode ter estabilização/zoom próprio: uma câmera por vídeo
            key += "|" + Path(m.path).stem
        gk = (key, w, h)
        if gk not in groups:
            f35 = getattr(m, "focal_35mm", None)
            g = CameraGroup(key=key, width=w, height=h, camera_model=choose_model(f35), lens=lens_class(f35))
            base_key = key.split("|video|")[0] + "|video" if kind == "video" else key
            hit = profiles.find(base_key, w, h) if profiles else None
            if hit:
                prof, params = hit
                g.camera_model, g.params, g.fixed = prof.camera_model, params, True
                g.note = f"perfil calibrado (RMS {prof.rms_px:.2f} px)" if prof.rms_px is not None else "perfil calibrado"
            else:
                fp = focal_px(f35, w, h)
                if fp:
                    g.params = _initial_params(g.camera_model, fp, w, h)
                    g.note = f"autocalibração; focal inicial pelo EXIF ({f35:.0f} mm eq., FOV {fov_deg(f35):.0f}°)"
                else:
                    g.note = "autocalibração; sem EXIF de focal"
            groups[gk] = g
        groups[gk].images.append(f.name)
    return list(groups.values())


def _initial_params(model: str, f: float, w: int, h: int) -> list[float]:
    cx, cy = w / 2, h / 2
    if model == "RADIAL":
        return [f, cx, cy, 0.0, 0.0]
    return [f, f, cx, cy, 0.0, 0.0, 0.0, 0.0]  # OPENCV e OPENCV_FISHEYE têm 8 parâmetros


def summarize(groups: list[CameraGroup]) -> list[dict]:
    return [{"lente": g.lens, "id": g.key, "resolucao": f"{g.width}x{g.height}", "modelo": g.camera_model,
             "imagens": len(g.images), "calibrada": g.fixed, "nota": g.note} for g in groups]
