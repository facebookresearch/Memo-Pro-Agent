# Download, Prepare And Run

## Data Sources

**Videos come from Ego4D, not Hugging Face.** Visit the official
[Ego4D website](https://ego4d-data.org/), obtain access under its terms, and follow
its current download instructions for the required source videos. We do not
redistribute the original videos or the derived clips.

**User histories are hosted on Hugging Face:**
[facebook/MemProBench](https://huggingface.co/datasets/facebook/MemProBench)
contains exactly the 800 eval800 histories. The dataset is currently private;
downloading requires an account with read access. Sample reference annotations
are already included in this repository's metadata. Retrieval graphs and
embedding matrices are built locally, not downloaded.

The CPU integration test in the main [Quick Start](../README.md#quick-start)
already works with synthetic fixtures. Actual inference requires downloaded
histories, rebuilt clips and either a compatible prebuilt index or a newly built
one. The preparation tool generates annotations from the included metadata.

## Download The Histories

Install the HF CLI and authenticate if accessing the private dataset:

```bash
python -m pip install 'huggingface_hub>=1.0'
hf auth login
export HISTORY_REV=6a662a65693b359bb7a4c5d1712c5a0bb8f58556
```

For the single-sample walkthrough below, download its history and the release
manifest/UID list:

```bash
hf download facebook/MemProBench \
  full_history/01ef2c24_317-451.jsonl history_manifest.json uids.txt \
  --repo-type dataset --revision "$HISTORY_REV" --local-dir data/hf
```

To download all eval800 histories instead (about 6.3 GB):

```bash
hf download facebook/MemProBench --repo-type dataset \
  --revision "$HISTORY_REV" --local-dir data/hf
```

The immutable snapshot contains byte-preserved histories under `full_history`,
plus `uids.txt` and `history_manifest.json` with file sizes, record counts,
SHA-256 values and fingerprints of the corresponding code metadata. No videos,
retrieval indices or checkpoints are included. Use `data/hf/full_history` as the
history input to the preparation tool.

## 1. Download And Cut The Videos

The included `evalsets/clip_specs.jsonl` maps each `sample_id` to its full
`ego4d_video_uid`, source-video timestamps and output `clip_file`. Start/end times
refer to the original source video; `trigger_window` refers to time within the
rebuilt clip. Its `annotations` object also contains the sample's target, open-loop
reference memories, clip duration and visibility observations. These are sample
and evaluation references, not privileged evidence supplied to the GPM policy.

Obtain the specified `source_asset` rendition using Ego4D's official instructions.
The selected samples are listed in `evalsets/eval800.txt`.

Cut each video using its **full source video UID**, `source_start_s` and
`duration_s`. Do not infer source timestamps or full video UIDs from a sample's
filename. For one downloaded source video, use the included metadata directly:

```bash
# Point this to your authorized Ego4D source-video directory.
export EGO4D_VIDEO_DIR="/path/to/ego4d/video_540ss"
export SAMPLE_ID=01ef2c24_317-451
python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

sample_id = os.environ['SAMPLE_ID']
rows = [json.loads(line) for line in Path('evalsets/clip_specs.jsonl').read_text().splitlines()]
row = next(item for item in rows if item['sample_id'] == sample_id)
source = Path(os.environ['EGO4D_VIDEO_DIR']) / (row['ego4d_video_uid'] + '.mp4')
if not source.is_file():
    raise SystemExit('Download the required source video from Ego4D first: ' + str(source))
output = Path('data/clips') / row['clip_file']
output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([
    'ffmpeg', '-n', '-ss', str(row['source_start_s']), '-i', str(source),
    '-t', str(row['duration_s']), '-vf', row['vf'], '-c:v', 'libx264',
    '-preset', 'veryfast', '-crf', '24', '-pix_fmt', 'yuv420p', '-an',
    '-movflags', '+faststart', str(output),
], check=True)
PY
```

FFmpeg is required. `-n` refuses to overwrite an existing clip. Store rebuilt clips
as `<sample_id>.mp4` directly under your local clips directory.
Use the supplied source rendition and settings; matching timestamps does not
necessarily guarantee pixel-identical reconstruction.

## 2. Prepare A Selected Subset

Once the histories are available locally and the clips have been rebuilt, run
the preparation tool. Reference annotations come from the included metadata;
there is no separate annotation package to download. The input is matched by
`sample_id`:

```text
data/hf/full_history/<sample_id>.jsonl
data/clips/<sample_id>.mp4
```

The tool reads local files only: no downloads, video reconstruction or model
calls occur automatically.

```bash
python tools/prepare_propermem.py \
  --history-root data/hf/full_history --clips-root data/clips \
  --clip-specs evalsets/clip_specs.jsonl \
  --output-root data/prepared --uid 01ef2c24_317-451
```

For eval800, omit `--uid` or supply an explicit `--uid-list`. Use a fresh output
directory for a new selection. The resulting **Contract A** layout is:

```text
data/prepared/propermem/
  clips/           rebuilt <uid>.mp4
  full_history/    user history captions
  trigger/         trigger annotations with prepared scoring windows
  memory/          target and open-loop annotations
  visibility/      clip duration and visibility annotations
  uids.txt         selected UID order
  windows.json     matching scoring windows
```

The tool generates `trigger`, `memory` and `visibility` files from metadata and
sets each prepared trigger's `trigger_window` from the selected evaluation
windows. Without external window overrides it uses the metadata window. Readers
do not need to create or edit annotation files manually. Histories and clips are
copied without modifying the downloaded inputs.

## 3. Build The Retrieval Memory Locally

Target/open-loop annotations are not a retrieval index. Build the event and
semantic graphs, passages and embedding matrices (**Contract B**) locally from
the prepared histories. No prebuilt graph or embedding-matrix download is
required or provided for this release.

Install the inference dependencies described in the main Quick Start. Configure
`gate_backend` in `configs/example.yaml` for your running extraction model, and
configure `embedding` for your local embedding model or embedding service. The
example uses the same OpenAI-compatible endpoint for extraction, gate and policy;
start that service before constructing memory. See the main Quick Start for a
serving example.

Begin with the single sample prepared above:

```bash
properagent-build-memory --config configs/example.yaml \
  --uid 01ef2c24_317-451 --workers 4
```

The command segments the history, extracts event/semantic graphs, computes
embeddings, and writes the index plus a checksum manifest under
`data/prepared/memory`. It does not download an index. Construction calls your
configured model and can be expensive for a year-long history, so validate one
sample before building the complete evaluation set. Already-built samples are
skipped unless `--force` is passed.

Verify that the generated files load and that no extraction events failed:

```bash
export SAMPLE_ID=01ef2c24_317-451
python - <<'PY'
import json
import os
from pathlib import Path
from properagent.config import load_config
from properagent.memory import MemoryStore

config = load_config('configs/example.yaml')
uid = os.environ['SAMPLE_ID']
root = Path(config['memory']['root'])
history = Path(config['data']['root']) / 'full_history' / (uid + '.jsonl')
MemoryStore(uid, root, caption_log=history, verify_hashes=True)
manifest = json.loads((root / (uid + '.manifest.json')).read_text())
assert manifest['counts']['events'] > 0
assert manifest['counts']['failed_events'] == 0, 'Inspect failed events before running GPM.'
print({'uid': uid, 'counts': manifest['counts'],
       'embedding_signature': manifest['embedding_signature']})
PY
```

After preparing the full selection and validating your services, use `--all` to
build every selected sample. Inference must use the same embedding model and
dimension as construction; mismatches are rejected when the index is loaded.

For an offline wiring check, run:

```bash
python -m pytest -q tests/test_end_to_end.py -k local_build_cli
```

This CPU test exercises the actual build command, index checksums, loading,
retrieval and GPM control flow with synthetic data and scripted model/embedding
backends. It validates the integration, not real-model extraction quality or
paper metrics.

## 4. Run GPM

Configure the endpoint and served model name in `configs/example.yaml`, then run:

```bash
properagent-run --config configs/example.yaml --paradigm windowed \
  --label first-windowed --no-judge
properagent-run --config configs/example.yaml --paradigm streaming \
  --label first-streaming --no-judge
```

The example uses one endpoint for policy and gate. API keys belong in environment
variables, not configuration files. Outputs are written locally under `runs/`.
Timing-only scores check the pipeline; see [Evaluation](evaluation.md) before
using a content judge or reporting results.
