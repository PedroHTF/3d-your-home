import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { boxGeometry, meshVolume, polygonArea, prismGeometry } from "../src/geometry";

describe("geometria", () => {
  it("caixa orientada tem o volume certo para qualquer orientação e lado", () => {
    for (const ang of [0, 0.3, 1.2, 2.7, -2]) {
      for (const side of [1, -1]) {
        const u: [number, number] = [3 * Math.cos(ang), 3 * Math.sin(ang)];
        const v: [number, number] = [-side * 0.15 * Math.sin(ang), side * 0.15 * Math.cos(ang)];
        const g = boxGeometry([1, 2, 0.5], u, v, 2.6);
        expect(meshVolume(g)).toBeCloseTo(3 * 0.15 * 2.6, 5);
        g.computeBoundingBox();
        expect(g.boundingBox!.min.z).toBeCloseTo(0.5);
        expect(g.boundingBox!.max.z).toBeCloseTo(3.1);
      }
    }
  });
  it("prisma: área x altura, inclusive polígono em L e sentido horário", () => {
    const L: [number, number][] = [[0, 0], [5, 0], [5, 2], [2, 2], [2, 4], [0, 4]];
    for (const poly of [L, [...L].reverse()]) {
      const g = prismGeometry(poly, -0.1, 0);
      expect(polygonArea(poly)).toBeCloseTo(14);
      expect(meshVolume(g)).toBeCloseTo(14 * 0.1, 5);
      g.computeBoundingBox();
      expect(g.boundingBox!.min.z).toBeCloseTo(-0.1);
    }
  });
  it("caixa: normais apontam para fora (sem faces invertidas visíveis no corte)", () => {
    const g = boxGeometry([0, 0, 0], [1, 0], [0, 1], 1);
    const pos = g.getAttribute("position"), nor = g.getAttribute("normal");
    const c = new THREE.Vector3(0.5, 0.5, 0.5);
    let consistent = 0;
    for (let i = 0; i < pos.count; i++) {
      const p = new THREE.Vector3().fromBufferAttribute(pos, i).sub(c);
      consistent += Math.sign(p.dot(new THREE.Vector3().fromBufferAttribute(nor, i)));
    }
    expect(Math.abs(consistent)).toBe(pos.count); // todas para o mesmo lado (fora ou dentro)
  });
});
