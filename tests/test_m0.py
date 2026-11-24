import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tdyh import frames, ingest, planner
from tdyh.recon.colmap import parse_analyzer
from tdyh.scenegraph import Opening, SceneGraph, Table, Wall


def _textured(seed: int, size=(480, 640)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((*size, 3)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_sharpness_detects_blur():
    img = cv2.cvtColor(_textured(0), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (21, 21), 8)
    assert frames.sharpness(img) > 5 * frames.sharpness(blurred)


def test_phash_dedup():
    a = cv2.cvtColor(_textured(1), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(_textured(2), cv2.COLOR_BGR2GRAY)
    assert frames.hamming(frames.phash(a), frames.phash(a.copy())) == 0
    assert frames.hamming(frames.phash(a), frames.phash(b)) > 10


def test_select_frames_drops_blurry_and_duplicates(tmp_path):
    paths = []
    for i in range(12):
        img = _textured(i // 4)  # 3 cenas distintas, 4 frames cada
        if i % 4 == 1:
            img = cv2.GaussianBlur(img, (31, 31), 12)
        p = tmp_path / f"f_{i:03d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    scored = frames.select_frames(paths, window=2)
    chosen = [s for s in scored if s.selected]
    assert len(chosen) == 3
    assert all("001" not in s.path.name for s in chosen)


def test_ingest_exif_orientation_and_dedup(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    im = Image.fromarray(_textured(3)[:, :, ::-1])  # 640x480
    exif = Image.Exif()
    exif[0x0112] = 6  # rotacionar 90°
    exif[0x010F] = "samsung"
    im.save(src / "a.jpg", exif=exif.tobytes())
    shutil.copy(src / "a.jpg", src / "dup.jpg")
    res = ingest.ingest_folder(src, tmp_path / "proj")
    assert len(res) == 1
    assert (res[0].width, res[0].height) == (480, 640)
    assert res[0].make == "samsung"
    with Image.open(res[0].path) as out:
        assert out.size == (480, 640)
    assert (tmp_path / "proj" / "media.json").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg ausente")
def test_process_video(tmp_path):
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
                    "-t", "3", "-pix_fmt", "yuv420p", str(vid)], check=True)
    out = frames.process_video(vid, tmp_path / "frames", fps=4, window=4)
    assert 1 <= len(out) <= 3


def test_planner_rules(tmp_path):
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), _textured(i))
    q = planner.assess(tmp_path)
    assert q.n_images == 10 and any("Poucas" in n for n in q.notes)
    assert planner.choose(q, has_gpu=False).recipe == "classic"


def test_colmap_analyzer_parse(tmp_path):
    txt = "Cameras: 1\nImages: 80\nRegistered images: 72\nPoints: 15342\nMean reprojection error: 0.61px\n"
    r = parse_analyzer(txt, tmp_path, total=80)
    assert r.registered_images == 72 and r.points3d == 15342
    assert abs(r.registration_ratio - 0.9) < 1e-9 and r.mean_reproj_error == 0.61


def test_scenegraph_roundtrip():
    w = Wall(start=(0, 0), end=(4, 0))
    sg = SceneGraph(entities=[w, Opening(kind="door", host_wall=w.id, offset=1, width=0.8, height=2.1),
                              Table(base="x_base")])
    back = SceneGraph.model_validate_json(sg.model_dump_json())
    assert back.entities[2].base.value == "x_base" and back.get(w.id).thickness == 0.15


def test_api_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("TDYH_DATA", str(tmp_path))
    import importlib

    import tdyh.api.main as m
    importlib.reload(m)
    c = TestClient(m.app)
    assert c.get("/health").json()["ok"]
    pid = c.post("/projects", json={"name": "sala"}).json()["id"]
    ok, buf = cv2.imencode(".jpg", _textured(5))
    r = c.post(f"/projects/{pid}/media", files=[("files", ("a.jpg", buf.tobytes(), "image/jpeg"))])
    assert r.json()["saved"] == ["a.jpg"]
    assert c.get(f"/projects/{pid}/scene").json()["entities"] == []
