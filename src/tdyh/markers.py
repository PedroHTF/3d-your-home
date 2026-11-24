"""Gera PDF imprimível de marcadores ArUco para escala métrica (um por página A4)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scale import ARUCO_DICT

A4_MM = (210.0, 297.0)


_FONTS = ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf", "arial.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
          "/System/Library/Fonts/Supplemental/Arial.ttf", "/system/fonts/Roboto-Regular.ttf"]


def _font(px: int):
    """Fonte com acentos (a padrão do Pillow não tem)."""
    for name in _FONTS:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def marker_page(marker_id: int, side_mm: float = 150.0, dpi: int = 300,
                dictionary: int = ARUCO_DICT) -> Image.Image:
    mm = dpi / 25.4
    W, H = int(round(A4_MM[0] * mm)), int(round(A4_MM[1] * mm))
    page = Image.new("L", (W, H), 255)
    side_px = int(round(side_mm * mm))
    # 4x4 + borda = 6 módulos; gera com resolução exata e amplia sem interpolação
    base = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(dictionary), marker_id, 6)
    mark = Image.fromarray(base).resize((side_px, side_px), Image.NEAREST)
    x0, y0 = (W - side_px) // 2, int(40 * mm)
    page.paste(mark, (x0, y0))
    d = ImageDraw.Draw(page)
    f_big, f_small = _font(int(7 * mm)), _font(int(3.6 * mm))
    d.text((W // 2, int(18 * mm)), f"3D Your Home — marcador ID {marker_id}", fill=0, font=f_big, anchor="mm")
    y = y0 + side_px + int(12 * mm)
    d.text((W // 2, y), f"Lado do quadrado preto: {side_mm / 10:.1f} cm".replace(".", ",") + f"  (dicionário ArUco 4x4_50)",
           fill=0, font=f_small, anchor="mm")
    # régua de 10 cm para conferir a escala da impressão
    ry, rx0 = y + int(14 * mm), (W - int(100 * mm)) // 2
    rx1 = rx0 + int(round(100 * mm))
    d.line([(rx0, ry), (rx1, ry)], fill=0, width=max(2, int(0.4 * mm)))
    for k in range(11):
        x = rx0 + int(round(k * 10 * mm))
        h = int(4 * mm) if k in (0, 5, 10) else int(2 * mm)
        d.line([(x, ry - h), (x, ry)], fill=0, width=max(2, int(0.3 * mm)))
    d.text((W // 2, ry + int(6 * mm)), "Esta régua deve medir exatamente 10 cm", fill=0, font=f_small, anchor="mm")
    tips = [
        "Imprima em tamanho real (100%, sem 'ajustar à página').",
        "Confira com uma régua; se o lado medir diferente, informe a medida real no app.",
        "Cole bem plano na parede (~1,2 m de altura). Um marcador diferente por parede.",
        "Filme o marcador de vários ângulos. Evite reflexo de luz sobre ele.",
    ]
    ty = ry + int(18 * mm)
    for t in tips:
        d.text((int(20 * mm), ty), "• " + t, fill=0, font=f_small)
        ty += int(6.5 * mm)
    return page


def make_pdf(path: Path, ids: list[int] | range = range(6), side_mm: float = 150.0, dpi: int = 300) -> Path:
    if not 30 <= side_mm <= 180:
        raise ValueError("lado deve estar entre 30 e 180 mm para caber numa folha A4")
    pages = [marker_page(i, side_mm, dpi) for i in ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=pages[1:])
    return path
