# ANSI Scaler

ANSI Scaler builds reproducible synthetic training corpora for ANSI-art scaling.
The current pipeline creates isolated cartoon game assets and progressively
simplified image LODs:

```text
scene-kit catalogue -> prompts -> Sana raster -> rembg cutout -> VTracer LODs
                                                         -> Chuda ANSI pyramids
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

## Running the corpus

Validate and build prompts without downloading a model:

```bash
make catalog
make prompts
```

Run ten real records through every stage:

```bash
make smoke
```

This downloads the pinned Sana and rembg models and requires a CUDA GPU. The
initial 1,200-image run is resumable:

```bash
make corpus
```

Choose another recipe with `RUN_CONFIG=configs/runs/example.yaml`. Generated
manifests, reports, and content-addressed artifacts live under ignored `data/`.
Each stage records failures separately and skips completed output IDs when
resumed.

## Catalogue

Canonical reusable identities live in `catalog/concepts`; scene kits select
and contextualise them in `catalog/scene-kits`. Prompts are disposable derived
data, never catalogue source. The first catalogue contains woodland, village,
city, castle, and spaceport kits with 24 memberships each.

The LOD stage retains the transparent original plus canonical SVGs and small
PNG previews. Intended terminal-width selection is:

```text
0-9 cells    LOD 3
10-39 cells  LOD 2
40-79 cells  LOD 1
80+ cells    original cutout
```

`make pyramid` renders every integer terminal width from 2 through 120. Each
source image becomes an independently resumable `.tar.zst` containing the ANSI
levels and their hashes. The archive is the compact source corpus; a later
training-set builder will decode it into model-native grids and transition pairs.
Before rendering, the stage measures the original alpha bounds and consistently
crops every aligned LOD to a lightly padded object viewport. Both the tight
content bbox and render bbox are retained in original-canvas coordinates so the
training builder can later apply random placement and edge-crossing crops.
The target builds the pinned Chuda 0.1.1 release from crates.io into `.tools/`;
Chuda does not need to be installed globally. A Rust toolchain and `zstd` are
the only system-level requirements for this stage.

## Reviewing corpus quality

Start the local review interface without CUDA or Ollama:

```bash
make review
```

Open `http://127.0.0.1:8765`. The review screen compares the generated raster,
rembg cutout, and LOD previews alongside the VLM observation and verifier
decision. Use `A` to accept, `X` to reject, `?` when unsure, `B` to toggle the
source image, `1`–`3` for LODs, arrow keys to browse, and `Z` to undo.

Pipeline manifests remain immutable. Human actions are appended to
`data/runs/<run>/reviews/annotations.jsonl`; the neighbouring SQLite database is
a disposable index and can be rebuilt. The default queue prioritises conflicts
introduced by new resource versions, then randomly samples the least-reviewed
kits, concepts, roles, and machine decisions. Set `ANSI_SCALER_REVIEWER` to
override the reviewer name recorded in the log.
