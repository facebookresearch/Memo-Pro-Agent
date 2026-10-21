# Changelog

All notable changes to MemoProAgent will be documented in this file.

## [Unreleased]

### Changed

- Hard-renamed the software namespace from `properagent` to `memoproagent`
  (package, imports, `memoproagent-run` / `memoproagent-build-memory`, and
  `MEMOPROAGENT_POLICY_KEY` / `MEMOPROAGENT_GATE_KEY`) with no legacy aliases.
  Reinstall and update imports/commands/key names. Stored data is unchanged:
  the `properagent_memory_v1` schema and `propermem/` layout still load as-is.
- Streaming frame alignment: `FrameSource(seek_by_frame=True)` addresses
  frames by index (`int(round(t * fps))` with an fps fallback of 30),
  resizes to 384x288, and JPEG-encodes at quality 85 via OpenCV, mirroring
  the legacy decode recipe. A failed decode raises instead of substituting
  a nearby frame. The default timestamp-seek plus near-end fallback
  (Windowed) behavior is unchanged.
- Streaming compatible degradation: retrieval and final generation still run
  after an incomplete or truncated Observe step (empty evidence becomes
  `"none"`); the fire rule additionally requires
  `final_finish_reason != "length"`; an ineligible step commits `NO_REMINDER` and
  continues rather than dropping the run.
- Streaming exposes `streaming.frame_size` in all three configs.

### Added

- Meta copyright headers, the declared CC-BY-NC-4.0 license text, a Contributor
  Covenant Code of Conduct, and contribution/security-reporting guidance.
- CPU GitHub Actions CI, bug and feature issue forms, and offline checks for
  public release files and configuration.
- MemoProAgent manuscript naming in the release descriptions and citation,
  while retaining existing package, command, repository and dataset identifiers.

No new paper claims are made in this section. Evaluation numbers, if any,
are reported separately with their pinned dataset, index, and model
identities.

## [0.1.0]

Initial release of the GPM inference harness: episodic and semantic memory
organization, retrieval-gated generation, fixed-window and chunked streaming
observation paradigms, synthetic-data CPU tests with scripted backends, and
frozen evaluation windows for the Eval800 set.

Code in this repository is licensed under CC-BY-NC-4.0 (see `LICENSE`).
Ego4D videos and dataset histories accessed through this harness remain
under their own separate terms and are not implicitly relicensed by the
code license.
