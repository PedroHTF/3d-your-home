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
| 4–10 gsplat, segmentação, layout, CAD, IFC | — | próximas fases |

## Uso local

Requisitos: Python 3.10+, FFmpeg. Para reconstrução: COLMAP (com CUDA) e OpenMVS no PATH.

```bash
pip install -e .[dev]
tdyh run caminho/do/video.mp4 data/sala        # pipeline completo local
pytest -q                                        # testes
uvicorn tdyh.api.main:app --reload               # API em http://localhost:8000/docs
```

Com Docker + GPU NVIDIA: `docker compose up --build`.

### Dicas de captura
- Movimento lento, ~70% de sobreposição entre quadros, luz uniforme.
- Para escala métrica, inclua um objeto de medida conhecida (ou um marcador ArUco impresso).

## Licenças

Código deste repositório: **AGPL-3.0-or-later** (adicione o texto completo em `LICENSE` pelo GitHub:
*Add file → Create new file → LICENSE → Choose a license template*).

Dependências externas, instaladas à parte e **não** incluídas aqui:
- OpenMVS, Ultralytics YOLO — AGPL-3.0.
- Gaussian Splatting (Inria) e Instant-NGP (NVIDIA) — opcionais, **licença não comercial**
  (somente pesquisa/uso pessoal).
- COLMAP (BSD), gsplat/Nerfstudio/SAM2/GroundingDINO (Apache-2.0), Open3D (MIT),
  IfcOpenShell (LGPL-3.0), CadQuery (Apache-2.0).

Projeto pessoal, sem uso comercial.
