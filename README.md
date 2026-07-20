# ANSI Scaler

ANSI Scaler builds reproducible synthetic training corpora for ANSI-art scaling.
The current pipeline creates isolated cartoon game assets and progressively
simplified image LODs:

```text
authored content -> prompts -> Sana raster -> background cutout -> VTracer LODs
                                                         -> Chuda ANSI pyramids
                                      -> VLM classification -> LLM verification
```

## Setup

The project uses `uv`, a committed dependency lock, and Make installation
stamps. Run:

```bash
make dev
make test
make help
```

`make clean` never removes corpus data.

### System dependencies

The locked Python environment does not replace host compilers or the NVIDIA
driver. On Ubuntu, the baseline package is:

```bash
sudo apt install build-essential
```

The venv deliberately uses uv-managed CPython rather than Ubuntu's system
Python, so its matching development headers are installed automatically and
consistently. Install an NVIDIA driver supported by the locked PyTorch build.
Check the complete host before starting an expensive run:

```bash
make deps
```

This verifies the C/C++ compiler, `Python.h`, NVIDIA driver access, PyTorch
CUDA, and ONNX Runtime's CUDA provider. `make corpus` and `make smoke` run the
same guard automatically.

## Running the corpus

Validate and build prompts without downloading a model:

```bash
make content
make prompts
```

Run ten real records through every stage:

```bash
make smoke
```

This downloads the pinned Sana and background models and requires a CUDA GPU. The
initial 1,200-image run is resumable through compressed ANSI pyramids, VLM
classification, and LLM verification:

```bash
make corpus
```

Infrastructure failures abort a stage immediately and are not recorded as bad
samples. If an older run already filled an error manifest because a dependency
was missing, fix the dependency and explicitly reconsider those records with:

```bash
make corpus RETRY_ERRORS=1
```

Choose another recipe with `RUN_CONFIG=configs/runs/example.yaml`. Generated
manifests, reports, and content-addressed artifacts live under ignored `data/`.
Each stage records failures separately and skips completed output IDs when
resumed.
Ollama classification and verification requests retry transient connection,
timeout, rate-limit, and server failures three times with exponential backoff.
An exhausted request rejects only that item and the pipeline continues.

## Authored content

Canonical object specifications live beneath `content/<theme>/<location>.yaml`.
Theme and location are navigation and balancing metadata only: they are never
implicitly inserted into prompts. Each object has one canonical home and a
complete manually authored semantic prompt. The initial iteration library has
50 specifications expanded to two deterministic seeds, producing 100 requests.

The LOD stage retains the transparent original plus canonical SVGs and small
PNG previews. Pyramid v3 cross-fades aligned, shared-crop LOD rasters before
Chuda chooses categorical ANSI cells. The higher-detail contribution rises in
12.5% steps through fixed four-cell transition windows:

```text
2-5 cells     LOD 3
6-14 cells    LOD 3 → LOD 2, midpoint 10
15-35 cells   LOD 2
36-44 cells   LOD 2 → LOD 1, midpoint 40
45-75 cells   LOD 1
76-84 cells   LOD 1 → LOD 0, midpoint 80
85-120 cells  LOD 0
```

Cross-fades use premultiplied sRGB so transparent pixels cannot leak hidden
colour into object edges. Blend rasters are ephemeral preparation buffers;
archives contain only ANSI levels and their exact weighted LOD provenance.

`make pyramid` renders every integer terminal width from 2 through 120. Each
source image becomes an independently resumable `.tar.zst` containing the ANSI
levels and their hashes. The archive is the compact source corpus; a later
training-set builder will decode it into model-native grids and transition pairs.
Before rendering, the stage measures the original alpha bounds and consistently
crops every aligned LOD to a lightly padded object viewport. LOD SVGs are
rasterized at the original canvas dimensions before the shared integer crop, so
every source presented to Chuda has identical dimensions and alignment; the small
LOD PNGs remain review thumbnails only. The bboxes and rasterizer provenance are
retained in original-canvas coordinates for the training builder.
LOD 0 is a high-fidelity VTracer conversion used at widths 80 and above, so the
entire ANSI pyramid is rendered from one SVG-derived representation family; the
original cutout is retained only for geometry and provenance.
The project pins the `chuda-ansi` Python package, which contains both CUDA and
CPU renderers. Chuda selects CUDA when it is usable and otherwise warns once
before falling back to CPU; it does not need to be installed globally.
Source loading, SVG rasterization, and cropping run in a bounded process pool
while the parent process owns the persistent Chuda renderer and a background
thread packs the preceding archive. Worker count is automatic from logical CPU
count and the configured memory headroom; set `resources.pyramid_workers` to
override it. Progress reports average preparation, rendering, and packing time
so the active bottleneck remains visible.

## Reviewing corpus quality

Start the local review interface without CUDA or Ollama:

```bash
make review
```

Open `http://127.0.0.1:8765`. Six stage tabs cover Generate, Background,
Classify, Verify, LOD, and ANSI. The earliest failed stage is selected in red;
unavailable downstream stages remain visible but disabled. Classify and Verify
show the cutout while outlining the corresponding evidence panel. Use `A` to
accept, `X` to open the optional rejection-note dialog, `?` when unsure, arrow
keys to browse assets, and `Z` to undo. `A` advances in tab order; `X` and `?`
record the selected stage and move to the next asset.

When an ANSI pyramid is available it is the default review surface. The scale
slider selects real stored widths from 2–120; `[` and `]` move one level, and
`P` plays the scale sequence. `Fit` centres each level inside the inspection frame,
while `1:1` preserves 8×16 terminal cells and allows scrolling. The canvas renderer
geometrically synthesizes block, braille, sextant, wedge, and legacy-bar glyphs so coloured cells
remain seamless; ordinary text uses the bundled terminal-symbol font. The
initial width is 40 and browser-local preferences persist between samples.
Raster and SVG-preview stages always use the same fitted display footprint for
direct comparison. The single LOD tab has its own discrete slider and playback;
keys `0`–`3` select levels, `[` and `]` step, and `P` toggles the active player.
All SVG review previews are rasterized once at 512×512 and displayed at 512 CSS
pixels when space permits, avoiding an additional browser enlargement pass.

Pipeline stages keep their manifests append-only. Human actions are appended to
`data/runs/<run>/reviews/annotations.jsonl`; the neighbouring SQLite database is
a disposable index and can be rebuilt. Each judgment records the selected stage
and its causal lineage. It remains valid across later-stage changes and is queued
again when its own output or an ancestor changes. The default queue prioritises
these stale or machine-conflicting judgments, then randomly samples the least-reviewed
kits, concepts, roles, and machine decisions. Set `ANSI_SCALER_REVIEWER` to
override the reviewer name recorded in the log.

Each prompt-and-seed record is one stable reviewable asset, with independent
active judgments per stage. A newer judgment supersedes only the same stage while
retaining append-only history. Undo removes the new decision without resurrecting
the older one. Stage-scoped reviews are calibration evidence and do not yet alter
dataset inclusion; selection continues to use the verifier policy.

## Corpus garbage collection

Safely inspect and compact one run with:

```bash
make gc
make gc RUN_CONFIG=configs/runs/smoke.yaml
make gc GC_CONFIRM=1
```

GC refuses to start while a pipeline command or review server is using the
corpus. It reports removable records, files, true orphans, and on-disk bytes by
stage before asking for confirmation; `GC_CONFIRM=1` supplies that confirmation
for automation. Other runs and active human reviews pin their referenced files.
When a configured chain is only partly rebuilt, GC retains both that partial
chain and the most recent completed pyramid or verification chain until the
replacement reaches the same terminal stage.

Applied collections back up the target manifests, annotations, run config, and
an exact deletion plan beneath `data/runs/<run>/gc/`. Artifact blobs themselves
are permanently deleted and are not included in that metadata backup. The
neighbouring review SQLite index is disposable and rebuilt the next time the
review server starts.
## Compiling a training dataset

Dataset compilation is deliberately separate from corpus generation. The recipe in
`configs/datasets/first.yaml` freezes the selection policy, split seed, pyramid format, shard count, and optional
parent glyph vocabulary.

```console
make dataset-plan
make dataset DATASET_LIMIT=10
make dataset
make dataset-validate DATASET_DIR=data/datasets/ansi-pyramids-v1/<dataset-id>
```

`dataset-plan` is read-only. It reports legacy whole-asset overrides, provisional verifier decisions, prompt-family splits,
cell count, and estimated raw tensor size. `dataset` writes content-addressed `.building` output and resumes complete
shards after interruption. Publication is an atomic rename; an already-published dataset is immutable.

Each safetensors shard stores ragged pyramid levels using offset arrays. Cells contain a `uint16` glyph vocabulary ID,
foreground RGB, background RGB, and a background-present bit. Dataset indexes retain the exact source lineage, source
hashes, selection reason, split, review/verifier provenance, bboxes, and prompt metadata. Glyph vocabularies reserve IDs
0–2 for PAD, MASK, and UNK; a recipe can name an older `vocabulary.json` to preserve every existing ID and append new
Unicode codepoints deterministically.
