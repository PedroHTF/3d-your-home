// Primitivas da API -> geometria do Three.js (Z para cima, metros).
import * as THREE from "three";
import type { Prim } from "./types";

/** Caixa orientada: canto o, lados horizontais u e v (vetores 2D), altura h. */
export function boxGeometry(o: [number, number, number], u: [number, number], v: [number, number], h: number) {
  const [x, y, z] = o;
  const p = [
    [x, y], [x + u[0], y + u[1]], [x + u[0] + v[0], y + u[1] + v[1]], [x + v[0], y + v[1]],
  ];
  const verts: number[] = [];
  for (const zz of [z, z + h]) for (const [a, b] of p) verts.push(a, b, zz);
  // orientação das faces independe do lado de v (garantida por DoubleSide no material)
  const idx = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7];
  // faces separadas (vértices duplicados) para normais planas e arestas nítidas
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  g.setIndex(idx);
  const flat = g.toNonIndexed();
  flat.computeVertexNormals();
  return flat;
}

/** Polígono 2D extrudado de z0 a z1 (lajes e cômodos). */
export function prismGeometry(poly: [number, number][], z0: number, z1: number) {
  const shape = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
  const g = new THREE.ExtrudeGeometry(shape, { depth: Math.max(z1 - z0, 1e-4), bevelEnabled: false });
  g.translate(0, 0, z0);
  return g;
}

export function primGeometry(p: Prim): THREE.BufferGeometry {
  if (p.t === "box") return boxGeometry(p.o, p.u, p.v, p.h);
  if (p.t === "prism") return prismGeometry(p.poly, p.z0, p.z1);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute([...p.a, ...p.b], 3));
  return g;
}

/** Área de um polígono 2D (para conferência nos testes). */
export function polygonArea(poly: [number, number][]) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    s += x0 * y1 - x1 * y0;
  }
  return Math.abs(s) / 2;
}

/** Volume de uma malha fechada (teorema da divergência). */
export function meshVolume(g: THREE.BufferGeometry) {
  const pos = g.getAttribute("position");
  const idx = g.getIndex();
  const n = idx ? idx.count : pos.count;
  const get = (i: number) => new THREE.Vector3().fromBufferAttribute(pos, idx ? idx.getX(i) : i);
  let v = 0;
  for (let i = 0; i < n; i += 3) v += get(i).dot(get(i + 1).cross(get(i + 2)));
  return Math.abs(v) / 6;
}
