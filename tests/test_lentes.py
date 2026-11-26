"""Lentes: modelo de câmera por FOV, agrupamento por lente/resolução, tabuleiro ChArUco
(detecção e calibração em imagens sintéticas com distorção conhecida) e comandos do COLMAP."""

import functools
import json
import math

import cv2
import numpy as np
import pytest
from PIL import Image

from tdyh import charuco, lens
from tdyh.ingest import MediaInfo
from tdyh.recon import colmap

def distort(x, dist, fisheye=False):
    """Modelo direto (coordenadas normalizadas), iguais ao OpenCV/COLMAP."""
    X, Y = x[:, 0], x[:, 1]
    if fisheye:
        r = np.hypot(X, Y)
        th = np.arctan(r)
        k1, k2, k3, k4 = dist[:4]
        thd = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        return np.c_[X * s, Y * s]
    k1, k2, p1, p2 = dist[:4]
    k3 = dist[4] if len(dist) > 4 else 0.0
    r2 = X * X + Y * Y
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    return np.c_[X * rad + 2 * p1 * X * Y + p2 * (r2 + 2 * X * X),
                 Y * rad + p1 * (r2 + 2 * Y * Y) + 2 * p2 * X * Y]


def undist(uv, K, dist, fisheye=False, iters=12):
    """Inversa exata (Newton) — o undistortPoints padrão do OpenCV erra na distorção forte."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    y = (uv - K[:2, 2]) / [K[0, 0], K[1, 1]]
    x = y.copy()
    e = 1e-7
    for _ in range(iters):
        f = distort(x, dist, fisheye) - y
        jx = (distort(x + [e, 0], dist, fisheye) - distort(x, dist, fisheye)) / e
        jy = (distort(x + [0, e], dist, fisheye) - distort(x, dist, fisheye)) / e
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        dx = (jy[:, 1] * f[:, 0] - jy[:, 0] * f[:, 1]) / det
        dy = (-jx[:, 1] * f[:, 0] + jx[:, 0] * f[:, 1]) / det
        x = x - np.c_[dx, dy]
    return x


def pixel_rays(K, dist, size, fisheye=False):
    w, h = size
    uv = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 2).astype(np.float64)
    return np.c_[undist(uv, K, dist, fisheye), np.ones(w * h)]


def render_view(K, dist, R, t, size, fisheye=False, px_per_mm=8.0, rays=None):
    """Foto sintética do tabuleiro: para cada pixel distorcido, desfaz a distorção, intersecta o raio
    com o plano do tabuleiro e amostra a textura (sem aproximação de homografia)."""
    tex = charuco.board_image(px_per_mm)
    w, h = size
    d = pixel_rays(K, dist, size, fisheye) if rays is None else rays
    n = R[:, 2]
    s = (n @ t) / (d @ n)
    Xb = (s[:, None] * d - t) @ R  # = R^T (s d - t)
    mx = (Xb[:, 0] * 1000 * px_per_mm).reshape(h, w).astype(np.float32)
    my = (Xb[:, 1] * 1000 * px_per_mm).reshape(h, w).astype(np.float32)
    bad = (s <= 0).reshape(h, w)
    mx[bad] = -1
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=200)
    return img


def poses(n, rng, board_wh=(0.175, 0.25)):
    out = []
    for k in range(n):
        rv = rng.uniform(-0.6, 0.6, 3) * [1, 1, 0.4]
        R, _ = cv2.Rodrigues(rv)
        z = rng.uniform(0.25, 0.45)
        # centro do tabuleiro espalhado pela imagem (inclusive bordas e cantos)
        c = np.array([rng.uniform(-0.5, 0.5) * z * 1.6, rng.uniform(-0.5, 0.5) * z * 1.2, z])
        t = c - R @ np.array([board_wh[0] / 2, board_wh[1] / 2, 0])
        out.append((R, t))
    return out


SIZE = (960, 720)


def supersampled(K):
    """Mesma câmera em 2x (centro do pixel u da imagem final = 2u + 0,5 na imagem grande)."""
    K2 = K.copy()
    K2[:2] *= 2
    K2[0, 2], K2[1, 2] = 2 * K[0, 2] + 0.5, 2 * K[1, 2] + 0.5
    return K2


@functools.lru_cache(maxsize=8)
def synth(fisheye=False, n=18, seed=0):
    """Fotos sintéticas renderizadas em 2x e reduzidas (sem serrilhado, que desloca os cantos)."""
    rng = np.random.default_rng(seed)
    f = 330.0 if fisheye else 520.0
    K = np.array([[f, 0, 482.0], [0, f * 1.002, 356.0], [0, 0, 1]])
    dist = np.array([0.05, -0.02, 0.01, -0.004]) if fisheye else np.array([-0.28, 0.09, 0.0008, -0.0005, 0.0])
    K2, big = supersampled(K), (SIZE[0] * 2, SIZE[1] * 2)
    rays = pixel_rays(K2, dist, big, fisheye)
    imgs = [cv2.resize(render_view(K2, dist, R, t, big, fisheye, px_per_mm=16, rays=rays), SIZE,
                       interpolation=cv2.INTER_AREA) for R, t in poses(n, rng)]
    return K, dist, imgs


@pytest.mark.parametrize("seed", [0, 1])
def test_calibration_recovers_strong_ultrawide_distortion(seed):
    K, dist, imgs = synth(seed=seed)
    cal = charuco.calibrate(imgs, "teste|uw|1.8mm|image")
    p = cal.profile
    assert p.camera_model == "OPENCV" and p.images_used >= 12
    fx, fy, cx, cy, k1, k2, p1, p2 = p.params
    assert fx == pytest.approx(K[0, 0], rel=0.01) and fy == pytest.approx(K[1, 1], rel=0.01)
    assert cx == pytest.approx(K[0, 2], abs=1) and cy == pytest.approx(K[1, 2], abs=1)
    assert k1 == pytest.approx(dist[0], abs=0.02)
    assert p.rms_px < 0.5
    # o que importa para o COLMAP: um raio projeta no mesmo pixel nos dois modelos, inclusive
    # perto dos cantos da imagem (onde a ultra-wide mais distorce)
    uv = np.stack(np.meshgrid(np.linspace(20, 940, 24), np.linspace(20, 700, 18)), -1).reshape(-1, 1, 2)
    rays = undist(uv, K, dist)
    X = np.c_[rays, np.ones(len(rays))]
    Kp = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    est, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), Kp, np.array([k1, k2, p1, p2]))
    err = np.linalg.norm(est.reshape(-1, 2) - uv.reshape(-1, 2), axis=1)
    assert np.median(err) < 0.7 and err.max() < 1.5


def test_fisheye_calibration():
    K, dist, imgs = synth(fisheye=True, seed=3)
    p = charuco.calibrate(imgs, "teste|fe|1.0mm|image", fisheye=True).profile
    assert p.camera_model == "OPENCV_FISHEYE"
    fx, fy, cx, cy, k1 = p.params[:5]
    assert fx == pytest.approx(K[0, 0], rel=0.01) and fy == pytest.approx(K[1, 1], rel=0.01)
    assert cx == pytest.approx(K[0, 2], abs=3) and cy == pytest.approx(K[1, 2], abs=3)
    assert k1 == pytest.approx(dist[0], abs=0.01)
    # com FOV tão grande o tabuleiro ocupa pouco da imagem: o aviso de cobertura tem de aparecer
    assert any("cobriu" in n for n in p.notes)


def test_calibration_warns_poor_coverage_and_rejects_too_few():
    rng = np.random.default_rng(5)
    K = np.array([[520.0, 0, 480], [0, 520, 360], [0, 0, 1]])
    dist = np.array([-0.2, 0.05, 0, 0, 0])
    rays = pixel_rays(K, dist, SIZE)
    imgs = [render_view(K, dist, cv2.Rodrigues(rng.uniform(-0.3, 0.3, 3))[0], np.array([-0.087, -0.125, 0.6]),
                        SIZE, rays=rays) for _ in range(8)]  # sempre no centro e longe
    cal = charuco.calibrate(imgs, "k")
    assert cal.profile.coverage < 0.6 and any("cobriu" in n for n in cal.profile.notes)
    with pytest.raises(ValueError, match="pelo menos 6"):
        charuco.calibrate(imgs[:3] + [np.full((720, 960), 128, np.uint8)] * 5, "k")


def test_pdf_true_size_and_detectable(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = charuco.make_pdf(tmp_path / "c.pdf")
    doc = pymupdf.open(path)
    assert len(doc) == 2
    assert doc[0].rect.width == pytest.approx(595.2, abs=1)  # A4 em pontos
    pix = doc[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    c, ids = charuco.detect(g, charuco.board())
    assert len(ids) == (charuco.SQUARES_X - 1) * (charuco.SQUARES_Y - 1)
    # distância entre cantos vizinhos = 25 mm no papel
    px_mm = 150 / 25.4
    order = np.argsort(ids)
    c = c[order]
    assert np.linalg.norm(c[1] - c[0]) / px_mm == pytest.approx(25.0, abs=0.3)
    # não confunde com os marcadores de escala (4x4): o tabuleiro não gera detecções 4x4
    from tdyh.scale import detect_markers
    assert detect_markers(g) == {}


# ------------------------------------------------------------ modelo e agrupamento

def test_fov_and_model_choice():
    assert lens.lens_class(13) == "ultra-wide" and lens.choose_model(13) == "OPENCV"   # 0,6x do S23 FE
    assert lens.lens_class(24) == "grande-angular" and lens.choose_model(24) == "OPENCV"
    assert lens.lens_class(70) == "normal/tele" and lens.choose_model(70) == "RADIAL"
    assert lens.choose_model(8) == "OPENCV_FISHEYE"
    assert lens.choose_model(None) == "OPENCV"
    assert lens.fov_deg(13) == pytest.approx(118.0, abs=1)
    assert lens.focal_px(24, 4000, 3000) == pytest.approx(24 * 5000 / 43.267, rel=1e-3)


def _img(p, w, h):
    Image.new("RGB", (w, h), (128, 128, 128)).save(p)


def test_groups_by_lens_resolution_and_video(tmp_path):
    imgs = tmp_path / "images"
    imgs.mkdir()
    media = []
    for k, (f, f35, w, h) in enumerate([(1.8, 13, 400, 300), (1.8, 13, 400, 300), (5.4, 24, 400, 300),
                                        (5.4, 24, 300, 400)]):
        _img(imgs / f"f{k}.jpg", w, h)
        media.append(MediaInfo(path=str(imgs / f"f{k}.jpg"), kind="image", sha256="x", make="samsung",
                               model="SM-S711B", focal_mm=f, focal_35mm=f35, width=w, height=h))
    media.append(MediaInfo(path=str(tmp_path / "videos" / "abc123.mp4"), kind="video", sha256="y",
                           make="samsung", model="SM-S711B"))
    for k in range(3):
        _img(imgs / f"abc123_{k:04d}.jpg", 320, 180)
    _img(imgs / "solta.jpg", 200, 200)
    gs = lens.plan_cameras(media, imgs)
    by = {(g.key, g.width, g.height): g for g in gs}
    assert len(gs) == 5
    uw = by[("samsung|SM-S711B|1.80mm|image", 400, 300)]
    assert uw.images == ["f0.jpg", "f1.jpg"] and uw.camera_model == "OPENCV" and uw.lens == "ultra-wide"
    assert uw.params[0] == pytest.approx(13 * 500 / 43.267, rel=1e-3) and not uw.fixed
    assert ("samsung|SM-S711B|5.40mm|image", 300, 400) in by  # foto em pé = outra câmera
    vid = [g for g in gs if "video" in g.key][0]
    assert len(vid.images) == 3 and vid.key.endswith("|abc123")
    assert any(g.key.startswith("desconhecida") for g in gs)


def test_profile_is_used_and_scaled(tmp_path):
    store = lens.ProfileStore(tmp_path / "lenses.json")
    store.save(lens.LensProfile(key="samsung|SM-S711B|1.80mm|image", camera_model="OPENCV", width=800,
                                height=600, params=[400, 401, 399, 301, -0.25, 0.07, 0.001, 0.0], rms_px=0.3))
    store.save(lens.LensProfile(key="samsung|SM-S711B|f?|video", camera_model="OPENCV", width=1920,
                                height=1080, params=[1500, 1500, 960, 540, -0.1, 0.0, 0, 0]))
    imgs = tmp_path / "images"
    imgs.mkdir()
    _img(imgs / "a.jpg", 400, 300)
    _img(imgs / "v1_0001.jpg", 960, 540)
    media = [MediaInfo(path=str(imgs / "a.jpg"), kind="image", sha256="1", make="samsung", model="SM-S711B",
                       focal_mm=1.8, focal_35mm=13),
             MediaInfo(path=str(tmp_path / "v1.mp4"), kind="video", sha256="2", make="samsung", model="SM-S711B")]
    gs = {g.images[0]: g for g in lens.plan_cameras(media, imgs, store)}
    a = gs["a.jpg"]
    assert a.fixed and a.params == pytest.approx([200, 200.5, 199.5, 150.5, -0.25, 0.07, 0.001, 0.0])
    v = gs["v1_0001.jpg"]
    assert v.fixed and v.params[:4] == pytest.approx([750, 750, 480, 270])
    # proporção diferente: perfil não se aplica
    _img(imgs / "a.jpg", 300, 300)
    assert not lens.plan_cameras(media, imgs, store)[0].fixed
    # salvar de novo substitui; apagar
    store.save(store.load()[0].model_copy(update={"rms_px": 0.1}))
    assert len(store.load()) == 2 and store.delete("samsung|SM-S711B|f?|video") and len(store.load()) == 1


def test_colmap_commands_one_camera_per_group(tmp_path):
    g1 = lens.CameraGroup(key="a", width=4, height=3, camera_model="OPENCV", images=["x.jpg", "y.jpg"],
                          params=[1, 1, 2, 1.5, 0, 0, 0, 0])
    g2 = lens.CameraGroup(key="b", width=4, height=3, camera_model="RADIAL", images=["z.jpg"])
    cmds = colmap.extractor_commands(tmp_path / "db", tmp_path, [g1, g2], work=tmp_path)
    assert len(cmds) == 2
    c1 = " ".join(cmds[0])
    assert "--ImageReader.camera_model OPENCV" in c1 and "--ImageReader.single_camera 1" in c1
    assert "--ImageReader.camera_params 1,1,2,1.5,0,0,0,0" in c1
    assert (tmp_path / "images_cam0.txt").read_text().split() == ["x.jpg", "y.jpg"]
    assert "camera_params" not in " ".join(cmds[1])
    assert colmap.mapper_fix_flags([g1, g2])["colmap"] == []
    g1.fixed = g2.fixed = True
    assert "--Mapper.ba_refine_extra_params" in colmap.mapper_fix_flags([g1, g2])["colmap"]
    assert len(colmap.extractor_commands(tmp_path / "db", tmp_path, None)) == 1


def test_cli_calibrate_and_api(tmp_path, monkeypatch):
    K, dist, imgs = synth(seed=7, n=14)
    d = tmp_path / "fotos"
    d.mkdir()
    for k, im in enumerate(imgs):
        cv2.imwrite(str(d / f"{k:02d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    from tdyh import cli
    monkeypatch.setenv("TDYH_DATA", str(tmp_path / "data"))
    cli.main(["calibrate", str(d), "--profiles", str(tmp_path / "p.json")])
    prof = json.loads((tmp_path / "p.json").read_text())[0]
    assert prof["camera_model"] == "OPENCV" and prof["params"][0] == pytest.approx(K[0, 0], rel=0.015)

    import importlib

    from fastapi.testclient import TestClient

    import tdyh.api.main as m
    importlib.reload(m)
    c = TestClient(m.app)
    r = c.get("/charuco.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    files = [("files", (f"{k}.jpg", (d / f"{k:02d}.jpg").read_bytes(), "image/jpeg")) for k in range(14)]
    r = c.post("/lenses/calibrate", files=files)
    assert r.status_code == 201, r.text
    assert len(c.get("/lenses").json()) == 1
    r = c.post("/lenses/calibrate", files=files[:3])
    assert r.status_code == 422
