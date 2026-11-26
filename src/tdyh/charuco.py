"""Tabuleiro ChArUco: impressão (PDF A4) e calibração de lentes.

O tabuleiro é um xadrez em que cada casa branca tem um marcador ArUco. Os cantos do xadrez dão
precisão sub-pixel; os marcadores dizem QUAL canto é qual, então o tabuleiro funciona mesmo
parcialmente visível, cortado na borda da foto ou bem inclinado — essencial para medir a distorção
nas bordas de uma ultra-wide, onde ela é maior.

Usa o dicionário ArUco 5x5 (os marcadores de escala usam 4x4): um nunca é confundido com o outro.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .lens import LensProfile
from .markers import A4_MM, _font

CHARUCO_DICT = cv2.aruco.DICT_5X5_100
SQUARES_X, SQUARES_Y = 7, 10          # cabe em A4 retrato com margem
SQUARE_MM, MARKER_MM = 25.0, 18.0     # marcador = 72% da casa


def board(square_mm: float = SQUARE_MM, marker_mm: float = MARKER_MM) -> cv2.aruco.CharucoBoard:
    """Geometria do tabuleiro em METROS (para a calibração sair em unidades métricas)."""
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), square_mm / 1000, marker_mm / 1000,
                                  cv2.aruco.getPredefinedDictionary(CHARUCO_DICT))


def board_image(px_per_mm: float, square_mm: float = SQUARE_MM, marker_mm: float = MARKER_MM) -> np.ndarray:
    b = board(square_mm, marker_mm)
    size = (int(round(SQUARES_X * square_mm * px_per_mm)), int(round(SQUARES_Y * square_mm * px_per_mm)))
    return b.generateImage(size, marginSize=0, borderBits=1)


def make_pdf(path: Path, dpi: int = 300, square_mm: float = SQUARE_MM) -> Path:
    """Uma página A4: tabuleiro 7x10 de casas de 25 mm (17,5 x 25 cm), régua e instruções."""
    marker_mm = square_mm * MARKER_MM / SQUARE_MM
    if SQUARES_X * square_mm > 190 or SQUARES_Y * square_mm > 255:
        raise ValueError("casa grande demais para A4 (máx. ~25 mm)")
    mm = dpi / 25.4
    W, H = int(round(A4_MM[0] * mm)), int(round(A4_MM[1] * mm))
    page = Image.new("L", (W, H), 255)
    img = Image.fromarray(board_image(mm, square_mm, marker_mm))
    x0, y0 = (W - img.width) // 2, int(14 * mm)
    page.paste(img, (x0, y0))
    d = ImageDraw.Draw(page)
    f, fs = _font(int(4.2 * mm)), _font(int(3.0 * mm))
    d.text((W // 2, int(7 * mm)), "3D Your Home — tabuleiro de calibração ChArUco", fill=0, font=f, anchor="mm")
    y = y0 + img.height + int(5 * mm)
    d.text((W // 2, y), f"{SQUARES_X}x{SQUARES_Y} casas de {square_mm:g} mm · marcadores ArUco 5x5 de "
           f"{marker_mm:g} mm".replace(".", ","), fill=0, font=fs, anchor="mm")
    ry, rx0 = y + int(7 * mm), (W - int(100 * mm)) // 2
    rx1 = rx0 + int(round(100 * mm))
    d.line([(rx0, ry), (rx1, ry)], fill=0, width=max(2, int(0.4 * mm)))
    for k in range(11):
        x = rx0 + int(round(k * 10 * mm))
        d.line([(x, ry - int((3 if k % 5 == 0 else 1.5) * mm)), (x, ry)], fill=0, width=max(2, int(0.3 * mm)))
    d.text((W // 2, ry + int(4 * mm)), "régua de 10 cm — confira a impressão", fill=0, font=fs, anchor="mm")
    d.text((W // 2, ry + int(10 * mm)), "Instruções na página 2", fill=0, font=fs, anchor="mm")
    return _save([page, _instructions(W, H, mm, square_mm)], path, dpi)


def _instructions(W: int, H: int, mm: float, square_mm: float) -> Image.Image:
    page = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(page)
    f, fs = _font(int(5.5 * mm)), _font(int(3.8 * mm))
    d.text((int(18 * mm), int(20 * mm)), "Como calibrar uma lente", fill=0, font=f)
    blocks = [
        ("Para que serve", [
            "Cada lente do celular (0,6x, 1x, 3x) entorta a imagem de um jeito. A calibração mede",
            "a distância focal e a distorção da lente, e a reconstrução 3D fica mais estável e exata.",
            "Basta fazer uma vez por lente (e por modo: foto ou vídeo).",
        ]),
        ("Impressão", [
            "Imprima em 100% (desligue 'ajustar à página') e confira a régua de 10 cm.",
            f"Meça uma casa com a régua; se não der {square_mm:g} mm, informe a medida real.".replace(".", ","),
            "Cole numa superfície rígida e plana (papelão grosso, MDF). Papel ondulado estraga.",
        ]),
        ("Fotos", [
            "Use a MESMA lente e resolução da captura do imóvel; não use zoom digital.",
            "Tire 15 a 30 fotos (ou um vídeo de 30 s) movendo o celular, com o tabuleiro parado.",
            "Varie distância (30 cm a 1,5 m) e inclinação (até ~45°). O tabuleiro pode sair cortado.",
            "Leve o tabuleiro a TODAS as regiões da imagem: centro, bordas e os 4 cantos —",
            "é nas bordas que a ultra-wide mais distorce.",
            "Boa luz, sem tremer, sem reflexo de lâmpada no papel.",
        ]),
        ("No software", [
            "tdyh calibrate <pasta-ou-vídeo> [--square-mm 24.8] [--fisheye]",
            "ou envie as fotos em POST /lenses/calibrate. O perfil é guardado e usado sozinho",
            "nas próximas reconstruções feitas com essa lente (identificada pelo EXIF).",
        ]),
    ]
    y = int(36 * mm)
    for title, lines in blocks:
        d.text((int(18 * mm), y), title, fill=0, font=_font(int(4.4 * mm)))
        y += int(8 * mm)
        for ln in lines:
            d.text((int(22 * mm), y), ln, fill=0, font=fs)
            y += int(6.2 * mm)
        y += int(5 * mm)
    return page



def _save(pages: list[Image.Image], path: Path, dpi: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=pages[1:])
    return path


# ------------------------------------------------------------------ detecção

def detect(gray: np.ndarray, b: cv2.aruco.CharucoBoard) -> tuple[np.ndarray, np.ndarray] | None:
    """Cantos do xadrez (sub-pixel) e seus IDs; None se poucos cantos."""
    det = cv2.aruco.CharucoDetector(b)
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1)


@dataclass
class Calibration:
    profile: LensProfile
    per_image: list[dict]


def calibrate(images: list[Path] | list[np.ndarray], key: str, fisheye: bool = False,
              square_mm: float = SQUARE_MM, measured_square_mm: float | None = None) -> Calibration:
    """Calibra uma lente com fotos do tabuleiro.

    measured_square_mm: lado real da casa medido no papel (impressoras costumam escalar 1–3%).
    A escala do tabuleiro não muda a focal nem a distorção (só as poses), mas usamos a medida
    real para relatar posições corretas."""
    sq = measured_square_mm or square_mm
    b = board(sq, sq * MARKER_MM / SQUARE_MM)
    chess = b.getChessboardCorners()  # (N,3) em metros
    obj, img, info, size = [], [], [], None
    for k, im in enumerate(images):
        name = str(im) if isinstance(im, (str, Path)) else f"imagem {k + 1}"
        g = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE) if isinstance(im, (str, Path)) else \
            (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im)
        if g is None:
            info.append({"imagem": name, "ok": False, "motivo": "não abriu"})
            continue
        if size is None:
            size = g.shape[::-1]
        elif g.shape[::-1] != size:
            info.append({"imagem": name, "ok": False, "motivo": "resolução diferente das demais"})
            continue
        r = detect(g, b)
        if r is None:
            info.append({"imagem": name, "ok": False, "motivo": "tabuleiro não encontrado (ou < 6 cantos)"})
            continue
        c, ids = r
        obj.append(chess[ids].astype(np.float32))
        img.append(c)
        info.append({"imagem": name, "ok": True, "cantos": int(len(ids))})
    if len(obj) < 6:
        raise ValueError(f"só {len(obj)} foto(s) com o tabuleiro detectado; são necessárias pelo menos 6 "
                         "(ideal 15–30, com o tabuleiro em todas as regiões da imagem)")
    w, h = size
    notes = []
    if fisheye:
        # chute inicial (sem ele o ajuste olho-de-peixe diverge): pinhole com distorção racional
        _, K0, _, _, _ = cv2.calibrateCamera(obj, img, (w, h), None, None, flags=cv2.CALIB_RATIONAL_MODEL)
        K = K0.astype(np.float64)
        D = np.zeros((4, 1))
        fe = cv2.fisheye
        flags = getattr(fe, "CALIB_RECOMPUTE_EXTRINSIC", 2) | getattr(fe, "CALIB_FIX_SKEW", 8) | \
            getattr(fe, "CALIB_USE_INTRINSIC_GUESS", 1)
        rms, K, D, _, _ = cv2.fisheye.calibrate([o.reshape(1, -1, 3).astype(np.float64) for o in obj],
                                                [i.reshape(1, -1, 2).astype(np.float64) for i in img],
                                                (w, h), K, D, flags=flags,
                                                criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9))
        params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D.ravel()[:4]]
        model = "OPENCV_FISHEYE"
    else:
        # COLMAP OPENCV = k1, k2, p1, p2 (sem k3)
        rms, K, D, _, _ = cv2.calibrateCamera(obj, img, (w, h), None, None, flags=cv2.CALIB_FIX_K3)
        D = D.ravel()
        params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0], D[1], D[2], D[3]]
        model = "OPENCV"
    cov = coverage(img, w, h)
    if cov < 0.6:
        notes.append(f"Tabuleiro cobriu só {cov:.0%} da imagem: fotografe-o também perto dos cantos e das "
                     "bordas, onde a distorção é maior.")
    if rms > 1.0:
        notes.append(f"Erro de reprojeção alto ({rms:.2f} px): fotos tremidas, tabuleiro curvado "
                     "ou lente com zoom/estabilização variável.")
    if len(obj) < 12:
        notes.append(f"Poucas fotos válidas ({len(obj)}); o ideal é 15–30.")
    prof = LensProfile(key=key, camera_model=model, width=w, height=h, params=[float(x) for x in params],
                       rms_px=round(float(rms), 4), images_used=len(obj), coverage=round(cov, 3), notes=notes)
    return Calibration(prof, info)


def coverage(img_pts: list[np.ndarray], w: int, h: int, grid: int = 8) -> float:
    """Fração de células de uma grade 8x8 sobre a imagem que receberam ao menos um canto."""
    hit = np.zeros((grid, grid), bool)
    for pts in img_pts:
        gx = np.clip((pts[:, 0] / w * grid).astype(int), 0, grid - 1)
        gy = np.clip((pts[:, 1] / h * grid).astype(int), 0, grid - 1)
        hit[gy, gx] = True
    return float(hit.mean())


def frames_from_video(video: Path, out_dir: Path, every_s: float = 0.5) -> list[Path]:
    """Frames para calibração (sem seleção por nitidez agressiva: o tabuleiro decide)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step, k, out = max(1, int(round(fps * every_s))), 0, []
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k % step == 0:
            p = out_dir / f"calib_{k:05d}.jpg"
            cv2.imwrite(str(p), fr)
            out.append(p)
        k += 1
    cap.release()
    return out
