"""Grafo de cena paramétrico — saída canônica do sistema (CAD/IFC/planta são gerados dele)."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _id() -> str:
    return uuid4().hex[:12]


class Pose(BaseModel):
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)  # metros
    yaw_deg: float = 0.0  # rotação em torno de Z (vertical)


class Entity(BaseModel):
    model_config = ConfigDict(validate_assignment=True)  # ajustes a posteriori são validados

    id: str = Field(default_factory=_id)
    confidence: float = 1.0
    source: Literal["auto", "user", "inferred"] = "auto"


class Wall(Entity):
    type: Literal["wall"] = "wall"
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float = 0.15
    height: float = 2.7


class OpeningKind(str, Enum):
    door = "door"
    window = "window"


class Opening(Entity):
    type: Literal["opening"] = "opening"
    kind: OpeningKind
    host_wall: str
    offset: float  # distância ao longo da parede a partir de start (m)
    width: float
    height: float
    sill: float = 0.0


class Slab(Entity):
    type: Literal["slab"] = "slab"
    polygon: list[tuple[float, float]]
    elevation: float = 0.0
    thickness: float = 0.12
    role: Literal["floor", "ceiling"] = "floor"


class TableBase(str, Enum):
    four_legs = "four_legs"
    central_column = "central_column"
    x_base = "x_base"


class Table(Entity):
    type: Literal["table"] = "table"
    pose: Pose = Pose()
    top_shape: Literal["rect", "round"] = "rect"
    length: float = 1.6
    depth: float = 0.9
    height: float = 0.75
    top_thickness: float = 0.03
    base: TableBase = TableBase.four_legs
    leg_section: float = 0.05


class Box(Entity):
    """Fallback genérico: caixa orientada (sempre editável depois)."""

    type: Literal["box"] = "box"
    label: str = "object"
    pose: Pose = Pose()
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)


class Relation(BaseModel):
    src: str
    dst: str
    kind: Literal["rests_on", "against", "hosted_by", "inside", "aligned_with"]
    params: dict = {}


AnyEntity = Wall | Opening | Slab | Table | Box

ConstraintKind = Literal[
    # medidas (podem calibrar a escala global com mode="scale")
    "length",          # refs=[parede]; value=m
    "distance",        # refs=[parede_a, parede_b] paralelas; value=m (face a face)
    "ceiling_height",  # refs=[]; value=m
    "opening_width", "opening_height", "opening_sill", "opening_offset",  # refs=[abertura]; value=m
    # geometria (sempre locais)
    "parallel",        # refs=[a, b]
    "perpendicular",   # refs=[a, b]
    "angle",           # refs=[a, b]; value=graus (ângulo entre as linhas, 0–180)
    "orientation",     # refs=[a]; value=graus em relação ao eixo X da planta
    "fixed",           # refs=[a]; trava a parede onde está
]


class Constraint(BaseModel):
    """Restrição editável a posteriori. A geometria medida nunca é alterada:
    a cena final = medida -> escala -> ajustes por elemento -> restrições resolvidas."""

    id: str = Field(default_factory=_id)
    kind: ConstraintKind
    refs: list[str] = Field(default_factory=list)
    value: float | None = None
    mode: Literal["local", "scale"] = "local"  # "scale": a medida corrige a escala do modelo inteiro
    enabled: bool = True
    note: str | None = None


class SceneGraph(BaseModel):
    version: int = 1
    units: Literal["m"] = "m"
    scale_source: Literal["arcore", "marker", "user_measure", "prior", "exif", "unknown"] = "unknown"
    scale_factor: float = 1.0          # já aplicado às coordenadas
    scale_uncertainty: float | None = None
    entities: list[AnyEntity] = Field(default_factory=list, discriminator=None)
    relations: list[Relation] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    overrides: dict[str, dict] = Field(default_factory=dict)  # ajustes manuais por elemento

    def get(self, eid: str) -> AnyEntity | None:
        return next((e for e in self.entities if e.id == eid), None)
