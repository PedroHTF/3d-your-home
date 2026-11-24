"""I/O de nuvens de pontos (PLY) e utilidades geométricas em numpy puro."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
              "uint": "u4", "float": "f4", "double": "f8", "float32": "f4", "float64": "f8",
              "uint8": "u1", "int32": "i4", "uint32": "u4", "int8": "i1", "int16": "i2", "uint16": "u2"}


def read_ply_xyz(path: Path) -> np.ndarray:
    """Lê os vértices (x, y, z) de um PLY ascii ou binário little-endian."""
    with open(path, "rb") as f:
        header, props, n, fmt = [], [], 0, "ascii"
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                _, name, count = line.split()
                in_vertex = name == "vertex"
                if in_vertex:
                    n = int(count)
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("propriedade de lista em vertex não suportada")
                props.append((parts[2], _PLY_TYPES[parts[1]]))
            elif line == "end_header":
                break
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, usecols=range(len(props)))
            names = [p[0] for p in props]
            return data[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float64)
        endian = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(name, endian + t) for name, t in props])
        arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def write_ply_xyz(path: Path, pts: np.ndarray) -> None:
    pts = np.asarray(pts, dtype="<f4")
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n".encode())
        f.write(pts.tobytes())


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def fit_plane_ransac(pts: np.ndarray, thresh: float, iters: int = 500,
                     normal_hint: np.ndarray | None = None, max_angle_deg: float = 25.0,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """Retorna (normal unitária, d, máscara de inliers) com n·p + d = 0."""
    rng = rng or np.random.default_rng(0)
    best_mask, best = None, None
    cos_lim = np.cos(np.radians(max_angle_deg))
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if normal_hint is not None and abs(n @ normal_hint) < cos_lim:
            continue
        d = -n @ a
        mask = np.abs(pts @ n + d) < thresh
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask, best = mask, (n, d)
    if best is None:
        raise RuntimeError("RANSAC não encontrou plano")
    # refinamento por mínimos quadrados (PCA dos inliers)
    inl = pts[best_mask]
    c = inl.mean(axis=0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
    if n @ best[0] < 0:
        n = -n
    d = -n @ c
    return n, d, np.abs(pts @ n + d) < thresh


def rotation_to_z(n: np.ndarray) -> np.ndarray:
    """Matriz de rotação que leva o vetor n para +Z."""
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), n @ z
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))
