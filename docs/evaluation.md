# Evaluation And Reproducibility

## Two Causal Paradigms

**Windowed** makes a decision every two seconds from four recent frames, spaced
one second apart. Visual context does not persist between decisions.

**Streaming** appends a new frame every second. It uses chunked context: after
30 turns, the latest four turns seed the next chunk. Prior interventions are
retained explicitly in both paradigms. The schedule is half-open (`t < duration`),
and neither paradigm receives future frames.

## Release Windows

The ordered eval800 UID list and scoring windows live under `evalsets/`. The
preparation tool writes the selected windows into the prepared trigger annotations
and generates the matching `windows.json` used by the example configuration.
Times are relative to the rebuilt clip, not the full Ego4D video. The included
sample metadata supplies target and reference-memory annotations for evaluation.
The preparation tool generates their dataset files; these references are not
passed as privileged evidence to the GPM policy.

For any supplied interval `[start, end]`, an intervention is in-window iff
`start <= t <= end + grace_seconds`, with the default grace of two seconds.
There is no grace before the start. Multiple intervals, when present, are treated
as a union rather than a first-to-last bounding range.

## Timing Versus Content

Use `--no-judge` for a first-run plumbing check:

```bash
properagent-run --config configs/example.yaml --paradigm windowed \
  --label timing-check --no-judge
```

Timing-only outcomes do not assess reminder content. They are upper bounds on
content-judged results, and should not be presented as paper Hit/Perfect numbers.

Content evaluation requires a configured judge service and can incur API charges.
Removing `--no-judge` enables it; make that choice explicitly and provide the
judge key through its configured environment variable. The judge receives four
causal frames around each selected intervention and the sample's target memory.
The default configuration caps selected judgments at eight in-window and five
out-of-window interventions. These configurable caps are part of the protocol;
do not assume they match every historical campaign.

**Perfect** is strict: the run must contain exactly one intervention globally,
that intervention must be in-window, and its judged content must be `GOOD`.
Deduplication or judgment caps do not erase other interventions from the full
fire ledger when deciding Perfect. Inspect the recorded scoring mode, coverage
and evaluator settings before aggregating results.

## Outputs And Trace Reuse

A run writes per-clip traces and scoring outputs under `runs/<label>/`. Those
files are local outputs and are ignored by Git; no intermediate results are
included with this code release.

The CLI can reuse traces for the same UID and paradigm, then rescore them. It
does not fully validate model, prompt, retrieval or sampling changes against an
old trace. Use a new label, or explicitly use `--force`, when changing inference
settings. Preserve the model identity, data/index revisions, release windows,
gate variant and full configuration alongside any reported experiment.

## Available Scope

The CPU integration test validates the inference/scoring plumbing with synthetic
fixtures and scripted backends. It does not measure real model capability.

The basic harness can run with a compatible VLM and a built/imported index. Exact
paper reproduction additionally requires the corresponding frozen indices,
model checkpoints, prompt/configuration identities and judging protocol.
History, index and checkpoint download instructions will be added when ready.
Do not interpret the illustrative Quick Start model as a released paper checkpoint.

## Optional Embedding Service And Slurm

The default uses local sentence-transformer embeddings. For an HTTP service,
`tools/serve_embeddings.py` exposes an OpenAI-compatible endpoint. Replace the
entire embedding mapping in your configuration with the HTTP backend's fields,
including an explicit `dim` and a model name matching the frozen signature.
An incompatible embedding identity or dimension is rejected by the memory loader.

The optional `tools/slurm/run_gpm.sbatch` submits only the Python inference client.
It does not launch model or embedding services. Supply your own checkout path,
Python environment, scheduler partition/resources and service configuration:

```bash
export GPM_REPO="$PWD"
export GPM_CONFIG=configs/example.yaml
export GPM_PARADIGM=windowed
sbatch --time=01:00:00 --cpus-per-task=4 --mem=16G \
  --export=ALL tools/slurm/run_gpm.sbatch
```

This example requests client resources, not resources for serving a VLM. Run
ordinary terminal commands from the Quick Start if you do not use Slurm.
