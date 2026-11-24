"""Etapa 1 — Ingestão: classificação de mídia, EXIF, GPS e correção de orientação."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import ExifTags, Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


@dataclass
class MediaInfo:
    path: str
    kind: str  # "image" | "video"
    sha256: str
    width: int | None = None
    height: int | None = None
    make: str | None = None
    model: str | None = None
    focal_mm: float | None = None
    focal_35mm: float | None = None
    datetime: str | None = None
    gps: dict | None = None  # {"lat", "lon", "alt"}
    duration_s: float | None = None
    fps: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _rational(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        num, den = v
        return num / den if den else 0.0


def _dms_to_deg(dms, ref: str) -> float:
    d, m, s = (_rational(x) for x in dms)
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def read_image_meta(path: Path) -> MediaInfo:
    info = MediaInfo(path=str(path), kind="image", sha256=sha256_file(path))
    with Image.open(path) as im:
        info.width, info.height = ImageOps.exif_transpose(im).size
        exif = im.getexif()
        if not exif:
            return info
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        info.make = str(tags.get("Make", "")).strip() or None
        info.model = str(tags.get("Model", "")).strip() or None
        info.datetime = str(tags.get("DateTime", "")) or None
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        if ExifTags.Base.FocalLength in sub:
            info.focal_mm = _rational(sub[ExifTags.Base.FocalLength])
        if ExifTags.Base.FocalLengthIn35mmFilm in sub:
            info.focal_35mm = float(sub[ExifTags.Base.FocalLengthIn35mmFilm])
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if gps and 2 in gps and 4 in gps:
            info.gps = {
                "lat": _dms_to_deg(gps[2], gps.get(1, "N")),
                "lon": _dms_to_deg(gps[4], gps.get(3, "E")),
                "alt": _rational(gps[6]) if 6 in gps else None,
            }
    return info


def read_video_meta(path: Path) -> MediaInfo:
    info = MediaInfo(path=str(path), kind="video", sha256=sha256_file(path))
    if not shutil.which("ffprobe"):
        return info
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        return info
    data = json.loads(out.stdout)
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if vs:
        info.width, info.height = vs.get("width"), vs.get("height")
        num, _, den = vs.get("avg_frame_rate", "0/1").partition("/")
        info.fps = float(num) / float(den or 1) if float(den or 1) else None
        rot = (vs.get("tags") or {}).get("rotate")
        for sd in vs.get("side_data_list", []) or []:
            rot = sd.get("rotation", rot)
        if rot is not None:
            info.extra["rotation"] = int(float(rot))
    fmt = data.get("format", {})
    if "duration" in fmt:
        info.duration_s = float(fmt["duration"])
    tags = fmt.get("tags", {})
    loc = tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709")
    if loc:
        info.extra["iso6709"] = loc
    info.make = tags.get("com.android.manufacturer") or tags.get("make")
    info.model = tags.get("com.android.model") or tags.get("model")
    return info


def classify(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    return None


def normalize_image(src: Path, dst: Path) -> Path:
    """Aplica a orientação EXIF nos pixels e salva sem rotação pendente."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        fixed = ImageOps.exif_transpose(im)
        exif = im.getexif()
        exif.pop(ExifTags.Base.Orientation, None)
        params = {"quality": 95} if dst.suffix.lower() in (".jpg", ".jpeg") else {}
        fixed.convert("RGB").save(dst, exif=exif.tobytes(), **params)
    return dst


def ingest_folder(src: Path, project_dir: Path) -> list[MediaInfo]:
    """Organiza mídia em project_dir/{images,videos} e grava media.json."""
    images_dir, videos_dir = project_dir / "images", project_dir / "videos"
    results: list[MediaInfo] = []
    seen: set[str] = set()
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    for p in files:
        kind = classify(p)
        if not kind:
            continue
        digest = sha256_file(p)
        if digest in seen:
            continue
        seen.add(digest)
        if kind == "image":
            dst = normalize_image(p, images_dir / f"{digest[:12]}.jpg")
            info = read_image_meta(p)
            info.path = str(dst)
        else:
            videos_dir.mkdir(parents=True, exist_ok=True)
            dst = videos_dir / f"{digest[:12]}{p.suffix.lower()}"
            shutil.copy2(p, dst)
            info = read_video_meta(dst)
        results.append(info)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "media.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False)
    )
    return results
