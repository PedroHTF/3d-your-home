export type Prim =
  | { t: "box"; o: [number, number, number]; u: [number, number]; v: [number, number]; h: number }
  | { t: "prism"; poly: [number, number][]; z0: number; z1: number }
  | { t: "line"; a: [number, number, number]; b: [number, number, number] };

export type Kind = "wall" | "door" | "window" | "floor" | "ceiling" | "space" | "separator";

export interface Element {
  id: string;
  kind: Kind;
  name: string;
  source: string; // "medida" | "inferida" | "usuário"
  confidence: number;
  adjusted: boolean;
  params: Record<string, string | number>;
  prims: Prim[];
}

export interface View {
  title: string;
  units: string;
  bounds: [[number, number, number], [number, number, number]];
  scale: { source: string; uncertainty: number | null };
  markers: Record<string, [number, number, number]>;
  elements: Element[];
  points: string | null;
}
