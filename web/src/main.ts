// Visualizador 3D: lê /projects/{id}/view.json ou /buildings/{id}/view.json (gerados do grafo de cena).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { primGeometry } from "./geometry";
import type { Element, View } from "./types";
import "./style.css";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("c");
const status = $("status");

// ---------------------------------------------------------------- cena
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.localClippingEnabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f2ee);
scene.add(new THREE.HemisphereLight(0xffffff, 0xb8b0a0, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-6, -10, 14);
scene.add(sun);

const persp = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
persp.up.set(0, 0, 1);
const ortho = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.05, 500);
ortho.up.set(0, 1, 0);
let camera: THREE.Camera = persp;
let controls = new OrbitControls(persp, canvas);

const cut = new THREE.Plane(new THREE.Vector3(0, 0, -1), 100); // mostra z <= constante
const COLORS = {
  wall: 0xd9d3c5, door: 0x8a5a33, window: 0x7fbfe6, floor: 0xb9b2a4, ceiling: 0xe8e4dc, space: 0x6fb36f,
  separator: 0x2f6f8f, inferred: 0xe08a3c, user: 0x5b8def,
} as const;
const root = new THREE.Group();
scene.add(root);
const layers = new Map<string, THREE.Object3D[]>();
const pickables: THREE.Mesh[] = [];
let view: View | null = null;
let selected: THREE.Mesh | null = null;
let highlightOrigin = true;

function material(e: Element): THREE.Material {
  const base = COLORS[e.kind];
  let color: number = base;
  if (highlightOrigin && e.kind !== "space" && e.kind !== "separator") {
    if (e.source === "inferida") color = COLORS.inferred;
    else if (e.source === "usuário") color = COLORS.user;
  }
  const transparent = e.kind === "space" || e.kind === "window" || e.kind === "ceiling";
  return new THREE.MeshStandardMaterial({
    color, roughness: 0.85, metalness: 0, side: THREE.DoubleSide, clippingPlanes: [cut],
    transparent, opacity: e.kind === "space" ? 0.12 : e.kind === "window" ? 0.55 : e.kind === "ceiling" ? 0.5 : 1,
    depthWrite: !transparent, polygonOffset: e.kind === "floor", polygonOffsetFactor: 1, polygonOffsetUnits: 1,
  });
}

function addTo(layer: string, o: THREE.Object3D) {
  if (!layers.has(layer)) layers.set(layer, []);
  layers.get(layer)!.push(o);
  root.add(o);
}

function clear() {
  select();
  for (const o of [...root.children]) {
    root.remove(o);
    o.traverse((c) => {
      const m = c as THREE.Mesh;
      m.geometry?.dispose();
      (Array.isArray(m.material) ? m.material : [m.material]).forEach((x) => x?.dispose());
    });
  }
  layers.clear();
  pickables.length = 0;
  $("labels").innerHTML = "";
  selected = null;
}

function build(v: View) {
  clear();
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x4a4a4a, clippingPlanes: [cut] });
  for (const e of v.elements) {
    for (const p of e.prims) {
      const g = primGeometry(p);
      if (p.t === "line") {
        const l = new THREE.Line(g, new THREE.LineDashedMaterial({ color: COLORS.separator, dashSize: 0.15, gapSize: 0.1 }));
        l.computeLineDistances();
        addTo(e.kind, l);
        continue;
      }
      const m = new THREE.Mesh(g, material(e));
      m.userData.el = e;
      if (e.kind === "space") m.renderOrder = 2;
      addTo(e.kind, m);
      pickables.push(m);
      if (e.kind === "wall" || e.kind === "door") {
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g, 30), edgeMat);
        addTo(e.kind, edges);
      }
    }
  }
  // marcadores ArUco
  for (const [id, p] of Object.entries(v.markers)) {
    const s = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.15, 0.01),
      new THREE.MeshBasicMaterial({ color: 0x111111, clippingPlanes: [cut] }));
    s.position.set(p[0], p[1], p[2]);
    s.userData.el = { id: `marcador ${id}`, kind: "marker", name: `Marcador ArUco ${id}`, source: "medida",
      confidence: 1, adjusted: false, params: { x: p[0], y: p[1], altura: p[2] }, prims: [] };
    addTo("markers", s);
    pickables.push(s);
  }
  buildLabels(v);
  frame(v);
  applyLayers();
  const sc = v.scale;
  $("scaleInfo").textContent = sc.source === "marker" ? "escala: marcadores"
    : sc.source === "user" || sc.source === "user_measure" ? "escala: medida do usuário"
    : sc.source === "prior" ? `escala ESTIMADA (±${Math.round((sc.uncertainty ?? 0.1) * 100)}%)` : `escala: ${sc.source}`;
  const top = v.bounds[1][2];
  const cutIn = $<HTMLInputElement>("cut");
  cutIn.max = String(Math.max(top + 0.3, 1));
  cutIn.value = cutIn.max;
  setCut(Number(cutIn.value));
}

async function loadPoints(url: string) {
  const buf = await (await fetch(url)).arrayBuffer();
  const pos = new Float32Array(buf);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  // cor pela altura (azul no piso -> areia no teto)
  g.computeBoundingBox();
  const zmin = g.boundingBox!.min.z, zr = Math.max(g.boundingBox!.max.z - zmin, 1e-3);
  const col = new Float32Array(pos.length);
  const a = new THREE.Color(0x3b6e8f), b = new THREE.Color(0xd8c9a8), c = new THREE.Color();
  for (let i = 0; i < pos.length; i += 3) {
    c.lerpColors(a, b, (pos[i + 2] - zmin) / zr);
    col[i] = c.r; col[i + 1] = c.g; col[i + 2] = c.b;
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const pts = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.012, vertexColors: true, clippingPlanes: [cut] }));
  addTo("points", pts);
  applyLayers();
}

// ---------------------------------------------------------------- rótulos dos cômodos
const roomLabels: { el: HTMLElement; p: THREE.Vector3 }[] = [];
function buildLabels(v: View) {
  roomLabels.length = 0;
  for (const e of v.elements.filter((x) => x.kind === "space")) {
    const prim = e.prims[0];
    if (prim.t !== "prism") continue;
    const c = centroid(prim.poly);
    const el = document.createElement("div");
    el.className = "room";
    el.innerHTML = `<b>${esc(e.name)}</b>${fmt(Number(e.params.area_util))} m²`;
    $("labels").appendChild(el);
    roomLabels.push({ el, p: new THREE.Vector3(c[0], c[1], 0.05) });
  }
}

function centroid(poly: [number, number][]): [number, number] {
  let a = 0, x = 0, y = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    const k = x0 * y1 - x1 * y0;
    a += k; x += (x0 + x1) * k; y += (y0 + y1) * k;
  }
  return Math.abs(a) < 1e-9 ? poly[0] : [x / (3 * a), y / (3 * a)];
}

function updateLabels() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const visible = (document.querySelector('[data-layer="space"]') as HTMLInputElement)?.checked ?? true;
  for (const { el, p } of roomLabels) {
    const q = p.clone().project(camera);
    el.style.display = visible && q.z < 1 ? "" : "none";
    el.style.left = `${((q.x + 1) / 2) * w}px`;
    el.style.top = `${((1 - q.y) / 2) * h}px`;
  }
}

// ---------------------------------------------------------------- câmera e vistas
function frame(v: View) {
  const [lo, hi] = v.bounds;
  const c = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
  const aspect = Math.max(canvas.clientWidth / Math.max(canvas.clientHeight, 1), 0.2);
  const r = (Math.max(hi[0] - lo[0], hi[1] - lo[1], 2) * 0.75) / Math.min(1, aspect * 1.2); // tela vertical: afasta
  persp.position.set(c.x - r * 0.9, c.y - r * 1.4, c.z + r * 1.3);
  ortho.position.set(c.x, c.y, 50);
  ortho.userData.half = r * 0.8;
  controls.target.copy(c);
  controls.update();
  resize();
}

function setMode(plan: boolean) {
  $("v3d").classList.toggle("on", !plan);
  $("vplan").classList.toggle("on", plan);
  const target = controls.target.clone();
  controls.dispose();
  camera = plan ? ortho : persp;
  controls = new OrbitControls(camera as THREE.PerspectiveCamera, canvas);
  controls.target.copy(target);
  if (plan) {
    ortho.position.set(target.x, target.y, 50);
    controls.enableRotate = false; // planta: só arrastar e zoom
    controls.screenSpacePanning = true;
    const cutIn = $<HTMLInputElement>("cut");
    cutIn.value = "1.2"; // corte de planta baixa a 1,20 m
    setCut(1.2);
  }
  controls.update();
  resize();
}

function setCut(z: number) {
  cut.constant = z;
  $("cutv").textContent = z >= Number($<HTMLInputElement>("cut").max) - 1e-6 ? "sem corte" : `${fmt(z)} m`;
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  persp.aspect = w / h;
  persp.updateProjectionMatrix();
  const half = (ortho.userData.half as number) || 5;
  ortho.left = -half * (w / h); ortho.right = half * (w / h); ortho.top = half; ortho.bottom = -half;
  ortho.updateProjectionMatrix();
}

function applyLayers() {
  document.querySelectorAll<HTMLInputElement>("[data-layer]").forEach((i) => {
    for (const o of layers.get(i.dataset.layer!) ?? []) o.visible = i.checked;
  });
  for (const k of ["separator"]) for (const o of layers.get(k) ?? []) o.visible = true;
}

// ---------------------------------------------------------------- seleção e propriedades
const ray = new THREE.Raycaster();
let down: [number, number] | null = null;
canvas.addEventListener("pointerdown", (ev) => (down = [ev.clientX, ev.clientY]));
canvas.addEventListener("pointerup", (ev) => {
  if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 6) return; // foi arrasto
  const r = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hits = ray.intersectObjects(pickables.filter((m) => m.visible), false)
    .filter((h) => h.point.z <= cut.constant + 1e-3);
  // prioridade: aberturas > paredes > marcadores > lajes > cômodos (cômodos são transparentes)
  const rank: Record<string, number> = { door: 0, window: 0, wall: 1, marker: 1, floor: 2, ceiling: 2, space: 3 };
  hits.sort((a, b) => (rank[(a.object.userData.el as Element).kind] ?? 9) - (rank[(b.object.userData.el as Element).kind] ?? 9)
    || a.distance - b.distance);
  select(hits[0]?.object as THREE.Mesh | undefined);
});

let outline: THREE.BoxHelper | null = null;
function select(m?: THREE.Mesh) {
  if (selected) (selected.material as THREE.MeshStandardMaterial).emissive?.setHex(0x000000);
  if (outline) { scene.remove(outline); outline.geometry.dispose(); outline = null; }
  selected = m ?? null;
  if (!m) { $("props").hidden = true; return; }
  const mat = m.material as THREE.MeshStandardMaterial;
  mat.emissive?.setHex(0x224466);
  outline = new THREE.BoxHelper(m, 0xd6336c);
  (outline.material as THREE.LineBasicMaterial).depthTest = false;
  outline.renderOrder = 10;
  scene.add(outline);
  const e = m.userData.el as Element;
  const tagColor = e.source === "inferida" ? "var(--inferred)" : e.source === "usuário" ? "var(--user)" : "#8a8270";
  const rows = Object.entries(e.params).map(([k, v]) => `<tr><td>${esc(label(k))}</td><td>${esc(val(k, v))}</td></tr>`).join("");
  $("propsBody").innerHTML = `<h3>${esc(e.name)}</h3>
    <span class="tag" style="background:${tagColor}">${esc(e.source)}</span>
    <span style="font-size:12px;color:var(--muted)"> confiança ${Math.round(e.confidence * 100)}%${e.adjusted ? " · ajustado pelo usuário" : ""}</span>
    <table>${rows}<tr><td>id</td><td>${esc(e.id)}</td></tr></table>
    ${e.source === "inferida" ? '<p style="font-size:12px;color:var(--muted)">Inferido: não foi observado diretamente. Confira ou informe uma medida.</p>' : ""}`;
  $("props").hidden = false;
}

const LABELS: Record<string, string> = {
  comprimento: "Comprimento", espessura: "Espessura", altura: "Altura", espessura_lado: "Espessura",
  compartilhada_com: "Compartilhada com", largura: "Largura", peitoril: "Peitoril", parede: "Parede",
  espessura_do_vao: "Profundidade do vão", mesma_porta_que: "Mesma porta que", cota: "Cota", uso: "Uso",
  area: "Área", area_util: "Área útil", perimetro: "Perímetro", x: "x", y: "y",
};
const label = (k: string) => LABELS[k] ?? k;
function val(k: string, v: string | number) {
  if (typeof v !== "number") return String(v);
  if (k.startsWith("area")) return `${fmt(v)} m²`;
  return `${fmt(v)} m`;
}
const fmt = (x: number) => x.toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

// ---------------------------------------------------------------- dados
async function loadList() {
  const sel = $<HTMLSelectElement>("source");
  const [ps, bs] = await Promise.all([
    fetch("../projects").then((r) => (r.ok ? r.json() : [])).catch(() => []),
    fetch("../buildings").then((r) => (r.ok ? r.json() : [])).catch(() => []),
  ]);
  sel.innerHTML = "";
  const add = (label: string, items: { id: string; name: string; has_plan: boolean }[], kind: string) => {
    if (!items.length) return;
    const og = document.createElement("optgroup");
    og.label = label;
    for (const it of items) {
      const o = document.createElement("option");
      o.value = `${kind}:${it.id}`;
      o.textContent = it.name + (it.has_plan ? "" : " (sem planta)");
      o.disabled = !it.has_plan;
      og.appendChild(o);
    }
    sel.appendChild(og);
  };
  add("Edificações", bs, "buildings");
  add("Levantamentos", ps, "projects");
  const q = new URLSearchParams(location.search);
  const want = q.get("building") ? `buildings:${q.get("building")}` : q.get("project") ? `projects:${q.get("project")}` : "";
  const first = [...sel.options].find((o) => o.value === want) ?? [...sel.options].find((o) => !o.disabled);
  if (!first) { status.textContent = "Nenhum levantamento com planta ainda. Processe um vídeo ou fotos pela API."; return; }
  sel.value = first.value;
  await open(first.value);
}

async function open(key: string) {
  const [kind, id] = key.split(":");
  status.textContent = "Carregando…";
  const r = await fetch(`../${kind}/${id}/view.json`);
  if (!r.ok) { status.textContent = `Não foi possível abrir (${r.status}).`; return; }
  view = (await r.json()) as View;
  status.textContent = "";
  build(view);
  const q = new URLSearchParams(location.search);
  q.delete("project"); q.delete("building");
  q.set(kind === "buildings" ? "building" : "project", id);
  history.replaceState(null, "", `?${q}`);
  if (view.points) loadPoints(`..${view.points}`).catch(() => {});
  // ?vista=planta e ?sel=<id> (abrir direto numa vista / num elemento)
  if (q.get("vista") === "planta") setMode(true);
  const sel = q.get("sel");
  if (sel) select(pickables.find((m) => (m.userData.el as Element).id === sel));
  if (q.get("camadas") === "1") $("layersBtn").click();
}

// ---------------------------------------------------------------- interface
$("source").addEventListener("change", (e) => open((e.target as HTMLSelectElement).value));
$("v3d").addEventListener("click", () => setMode(false));
$("vplan").addEventListener("click", () => setMode(true));
$("layersBtn").addEventListener("click", () => {
  const p = $("layers");
  p.hidden = !p.hidden;
  $("layersBtn").setAttribute("aria-expanded", String(!p.hidden));
});
$("closeProps").addEventListener("click", () => select());
document.querySelectorAll<HTMLInputElement>("[data-layer]").forEach((i) => i.addEventListener("change", applyLayers));
$("origin").addEventListener("change", (e) => {
  highlightOrigin = (e.target as HTMLInputElement).checked;
  if (view) build(view);
});
$("cut").addEventListener("input", (e) => setCut(Number((e.target as HTMLInputElement).value)));
addEventListener("resize", resize);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
  updateLabels();
});
resize();
loadList();

// para testes automatizados (ver scripts/viewer_smoke.py)
(window as unknown as { __tdyh: object }).__tdyh = {
  ready: () => view !== null, count: () => pickables.length, setMode, setCut,
};
