# 3D Your Home

Transforma fotos e vídeos de um imóvel em modelo 3D, planta baixa, CAD paramétrico (STEP/FCStd) e BIM (IFC).
A saída canônica é um **grafo de cena paramétrico** (paredes, aberturas, móveis como primitivas com
parâmetros); malhas, CAD, IFC e planta são gerados a partir dele.

Arquitetura completa: ver documento de arquitetura do projeto.

## Estado atual — M0/M1

| Etapa | Módulo | Status |
|---|---|---|
| 1 Ingestão (EXIF, GPS, orientação, dedup) | `tdyh.ingest` | ✅ |
| 2 Frames (FFmpeg, nitidez, pHash) | `tdyh.frames` | ✅ |
| Planejador de receita (classic/neural/hybrid) | `tdyh.planner` | ✅ regras |
| 3 SfM COLMAP/GLOMAP | `tdyh.recon.colmap` | ✅ adaptador |
| 3b MVS + malha OpenMVS | `tdyh.recon.openmvs` | ✅ adaptador |
| Grafo de cena | `tdyh.scenegraph` | ✅ modelo inicial |
| API FastAPI + Celery | `tdyh.api`, `tdyh.worker` | ✅ |
| Escala métrica (ArUco, medida de referência) | `tdyh.scale` | ✅ |
| 9 Layout: piso/teto, paredes, portas/janelas | `tdyh.layout` | ✅ |
| Planta DXF + SVG 1:50 | `tdyh.exporters.plan2d` | ✅ |
| Escala aproximada sem marcador (pé-direito/portas típicos) | `tdyh.scale` | ✅ |
| Edição a posteriori: medidas, ajustes por elemento, restrições | `tdyh.constraints` | ✅ |
| Criar/remover/dividir elementos | `tdyh.constraints`, API | ✅ |
| Cômodos opcionais (nome, área, área útil, perímetro) | `tdyh.rooms` | ✅ |
| Juntar levantamentos diferentes (portas, paredes, pontos; ajuste global) | `tdyh.building` | ✅ |
| 4–8, 10 gsplat, segmentação, móveis, CAD, IFC | — | próximas fases |

## Uso local

Requisitos: Python 3.10+, FFmpeg. Para reconstrução: COLMAP (com CUDA) e OpenMVS no PATH.

```bash
pip install -e .[dev]
tdyh run caminho/do/video.mp4 data/sala        # pipeline completo local
pytest -q                                        # testes
uvicorn tdyh.api.main:app --reload               # API em http://localhost:8000/docs
```

Planta a partir de uma nuvem densa já existente:

```bash
# com medida conhecida (fator modelo->metros)
tdyh plan scene_dense.ply data/sala --scale 2.0
# com marcador ArUco 4x4 (DICT_4X4_50) impresso de 15 cm
colmap model_converter --input_path sparse/0 --output_path sparse_txt --output_type TXT
tdyh plan scene_dense.ply data/sala --marker 0.15 --images images --colmap-txt sparse_txt
```

Com Docker + GPU NVIDIA: `docker compose up --build`.

### Edição a posteriori (API)

A geometria medida nunca é alterada; a cena final é recalculada a cada mudança:
medida → escala → ajustes por elemento → restrições. Apagar/desativar algo restaura o estado anterior.

| Ação | Requisição |
|---|---|
| Listar elementos e medidas | `GET /projects/{id}/scene/elements` |
| Medida que corrige a escala de tudo | `POST .../scene/constraints` `{"kind":"length","refs":["<parede>"],"value":4.20,"mode":"scale"}` |
| Fixar medida só de um elemento | idem com `"mode":"local"` (também `opening_width/height/sill/offset`, `ceiling_height`, `distance`) |
| Paralelismo / perpendicularidade | `{"kind":"parallel"\|"perpendicular","refs":["<a>","<b>"]}` |
| Ângulo entre paredes / orientação | `{"kind":"angle","refs":[a,b],"value":120}` · `{"kind":"orientation","refs":[a],"value":0}` |
| Travar uma parede | `{"kind":"fixed","refs":["<a>"]}` |
| Centralizar abertura na parede | `{"kind":"opening_center","refs":["<abertura>"]}` |
| Alinhar abertura com outra (paredes paralelas) | `{"kind":"opening_align","refs":["<abertura>","<outra>"]}` (centro com centro) |
| Editar / desativar / apagar restrição | `PATCH` / `DELETE .../scene/constraints/{cid}` |
| Criar parede / porta / janela / objeto | `POST .../scene/entities` `{"element":{"type":"wall","start":[x,y],"end":[x,y]}}` (encaixa em cantos e paredes a 10 cm) |
| Remover elemento (aberturas da parede saem junto) | `DELETE .../scene/entities/{eid}` — detectados: reversível com `POST .../{eid}/restore` |
| Dividir parede (ex.: para ligar uma divisória) | `POST .../scene/entities/{eid}/split` `{"at":2.0}` — restrições e aberturas são transferidas |
| Ligar/desligar divisão em cômodos (opcional) | `PUT .../scene/rooms` `{"enabled":true}` · listar: `GET .../scene/rooms` |
| Separar ambiente aberto sem parede | criar `{"type":"separator","start":[x,y],"end":[x,y]}` (linha tracejada) |
| Dar nome a um cômodo | `POST .../scene/rooms/labels` `{"point":[x,y],"name":"Cozinha"}` (o nome acompanha o cômodo após edições) |
| Ajustar um elemento (espessura, tipo, largura…) | `PATCH .../scene/entities/{eid}` `{"thickness":0.25}` (`null` volta ao medido) |

Cada resposta informa se cada restrição foi satisfeita e aponta conflitos e medidas incoerentes.

Elementos criados e ajustes feitos pelo usuário guardam a escala em que foram desenhados: se a
escala mudar depois (nova medida), eles acompanham. Paredes desenhadas pelo usuário fora do contorno
(divisórias) têm a espessura **centralizada** na linha; as medidas continuam com a face na linha e
espessura para fora. Para forçar: `PATCH`/criação com `"alignment":"face"|"center"`.

### Juntar cômodos de levantamentos diferentes

Cada levantamento é um projeto com planta própria (já em metros). Uma *edificação* reúne vários:

1. `POST /buildings` `{"name":"Apto"}` e `POST /buildings/{id}/captures` `{"project_id":...}` para cada levantamento.
2. `GET /buildings/{id}/suggestions` — pares de portas que provavelmente são a mesma, com pontuação,
   motivos e aviso de **ambiguidade** (portas e cômodos parecidos encaixam de mais de um jeito).
3. Confirme criando ligações: `POST /buildings/{id}/links` —
   `door_match` (mesma porta), `wall_contact` (parede encostada), `collinear` (paredes na mesma linha,
   ex.: fachada), `point_match` (mesmo ponto físico, ex.: canto ou marcador compartilhado).
   **Marcadores ArUco** com o mesmo ID vistos em dois levantamentos (ex.: colado no batente da porta)
   viram ligações `point_match` automáticas no `solve`; dois marcadores bastam para posicionar sem
   nenhuma ligação manual. Apagar uma ligação automática fica lembrado.
   Posição manual: `PUT /buildings/{id}/captures/{cid}/pose`.
4. `POST /buildings/{id}/solve` — ajuste global de todas as posições (em ciclos o erro é repartido),
   junção das paredes compartilhadas (espessura medida) e planta unificada em `GET /buildings/{id}/plan/{dxf|svg}`.

A espessura da parede entre cômodos vem da profundidade do batente das portas (medida na detecção);
sem ela, usa 15 cm e avisa. Corrija a escala de cada levantamento antes de juntar.

### Exportação IFC/BIM (M3)

Requer `pip install -e '.[ifc]'` (IfcOpenShell). Gera IFC4 em metros a partir da cena final:
IfcProject/Site/Building/BuildingStorey, IfcWall (espessura para o mesmo lado da planta; paredes
compartilhadas uma vez só), IfcOpeningElement + IfcDoor/IfcWindow (portas duplicadas uma vez só),
IfcSlab de piso e teto (teto inferido do piso, se faltar), IfcSpace por cômodo (nome, uso, área,
área útil). Psets `TDYH_Parametros` e `TDYH_Origem` (medida / inferida / usuário, confiança, escala),
além dos Qto_* padrão.

```bash
tdyh ifc projeto/work/scene.json modelo.ifc --name "Apartamento"
python scripts/render_ifc.py modelo.ifc previa.png   # pré-visualização com os vazios aplicados
```
API: `GET /projects/{id}/ifc` e `GET /buildings/{id}/ifc` (após `POST /buildings/{id}/solve`);
avisos no cabeçalho `X-TDYH-Warnings`.

### Marcadores de escala

```bash
tdyh markers marcadores.pdf -n 6            # 6 marcadores (IDs 0–5), 15 cm, um por página A4
tdyh markers quarto.pdf -n 4 --start 10     # outro cômodo: IDs 10–13
```
Também disponível em `GET /markers.pdf?n=6&side_cm=15`. Use um ID diferente em cada parede.

### Dicas de captura
- Movimento lento, ~70% de sobreposição entre quadros, luz uniforme.
- Para escala métrica, inclua um objeto de medida conhecida (ou um marcador ArUco impresso).

## Licenças

Código deste repositório: **AGPL-3.0-or-later** (texto completo em `LICENSE`).

Dependências externas, instaladas à parte e **não** incluídas aqui:
- OpenMVS, Ultralytics YOLO — AGPL-3.0.
- Gaussian Splatting (Inria) e Instant-NGP (NVIDIA) — opcionais, **licença não comercial**
  (somente pesquisa/uso pessoal).
- COLMAP (BSD), gsplat/Nerfstudio/SAM2/GroundingDINO (Apache-2.0), Open3D (MIT),
  IfcOpenShell (LGPL-3.0), CadQuery (Apache-2.0).

Projeto pessoal, sem uso comercial.
