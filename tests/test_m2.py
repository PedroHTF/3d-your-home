import numpy as np
import pytest

from tdyh import layout, scale
from tdyh.exporters import plan2d
from tdyh.pointcloud import read_ply_xyz, write_ply_xyz
from tdyh.scenegraph import Opening, Wall


def synthetic_room(W=4.20, D=3.10, H=2.70, density=900, noise=0.004, seed=0):
    """Cômodo retangular com vão de porta (0.80 x 2.10) na parede y=0 e uma mesa no meio."""
    rng = np.random.default_rng(seed)
    pts = []

    def plane(n, f):
        u = rng.random((n, 2))
        pts.append(f(u))

    a = int(W * D * density)
    plane(a, lambda u: np.c_[u[:, 0] * W, u[:, 1] * D, np.zeros(len(u))])        # piso
    plane(a, lambda u: np.c_[u[:, 0] * W, u[:, 1] * D, np.full(len(u), H)])       # teto
    for y in (0.0, D):                                                            # paredes em x
        p = rng.random((int(W * H * density), 2)) * [W, H]
        if y == 0.0:
            p = p[~((p[:, 0] > 1.0) & (p[:, 0] < 1.8) & (p[:, 1] < 2.10))]       # porta
        pts.append(np.c_[p[:, 0], np.full(len(p), y), p[:, 1]])
    for x in (0.0, W):
        p = rng.random((int(D * H * density), 2)) * [D, H]
        pts.append(np.c_[np.full(len(p), x), p[:, 0], p[:, 1]])
    t = rng.random((3000, 2)) * [1.2, 0.8] + [1.5, 1.1]                           # tampo de mesa
    pts.append(np.c_[t, np.full(len(t), 0.75)])
    P = np.vstack(pts)
    return P + rng.normal(0, noise, P.shape)


def random_rotation(seed=1):
    q = np.random.default_rng(seed).normal(size=4)
    q /= np.linalg.norm(q)
    return scale._qvec2rot(q)


@pytest.fixture(scope="module")
def result():
    R = random_rotation()
    s_model = 0.37  # "unidades do SfM" -> a reconstrução é 0.37x o real
    pts = (synthetic_room() @ R.T) * s_model
    up_world = R @ np.array([0, 0, 1.0])
    up_hint = up_world + np.random.default_rng(3).normal(0, 0.05, 3)  # dica ruidosa (câmeras)
    return layout.build_layout(pts, scale=1 / s_model, up_hint=up_hint)


def _wall_lengths(res):
    return sorted(round(np.linalg.norm(np.subtract(w.end, w.start)), 3)
                  for w in res.scene.entities if isinstance(w, Wall))


def test_height(result):
    assert abs((result.ceiling_z - result.floor_z) - 2.70) < 0.03


def test_walls_metric_accuracy(result):
    L = _wall_lengths(result)
    # 4 paredes inteiras + parede com porta dividida em 2 trechos (1.00 e 2.40)
    assert len(L) == 4, L
    for got, want in zip(L, [3.10, 3.10, 4.20, 4.20]):
        assert abs(got - want) < 0.03, (got, want)  # meta M2: erro < 3 cm


def test_floor_area(result):
    floor = next(e for e in result.scene.entities if getattr(e, "role", "") == "floor")
    assert abs(plan2d._area(floor.polygon) - 4.20 * 3.10) < 0.15


def test_exports(result, tmp_path):
    import ezdxf

    sg = result.scene
    dxf = plan2d.to_dxf(sg, tmp_path / "planta.dxf")
    doc = ezdxf.readfile(dxf)
    assert len(doc.modelspace().query('LWPOLYLINE[layer=="PAREDES"]')) == 4
    svg = plan2d.to_svg(sg, tmp_path / "planta.svg").read_text()
    assert svg.startswith("<svg") and "1:50" in svg


def test_marker_scale_triangulation():
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1.0]])
    size = 0.15
    corners = np.array([[0, 0, 2], [size, 0, 2], [size, size, 2], [0, size, 2]], float)
    model_scale = 0.5  # modelo em meia escala
    obs = []
    for tx in (-0.3, 0.0, 0.4):
        P = K @ np.hstack([np.eye(3), [[tx], [0], [0]]])
        pix = []
        for X in corners * model_scale:
            x = P @ np.r_[X, 1]
            pix.append(x[:2] / x[2])
        obs.append((P, np.array(pix)))
    est = scale.from_markers({7: obs}, size)
    assert abs(est.factor - 1 / model_scale) < 1e-6 and est.source == "marker"


def test_reference_scale():
    est = scale.from_reference(np.zeros(3), np.array([0.4, 0, 0]), 0.80)
    assert est.factor == pytest.approx(2.0)


def test_ply_roundtrip(tmp_path):
    p = np.random.default_rng(0).random((100, 3))
    write_ply_xyz(tmp_path / "a.ply", p)
    assert np.allclose(read_ply_xyz(tmp_path / "a.ply"), p, atol=1e-6)


def test_cli_plan(tmp_path):
    from tdyh import cli
    ply = tmp_path / "dense.ply"
    write_ply_xyz(ply, synthetic_room() * 0.5)
    cli.main(["plan", str(ply), str(tmp_path / "proj"), "--scale", "2.0"])
    assert (tmp_path / "proj" / "plan" / "planta.dxf").exists()
    assert (tmp_path / "proj" / "scene.json").exists()
