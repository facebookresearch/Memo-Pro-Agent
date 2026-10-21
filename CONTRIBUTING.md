# Contributing to MemoProAgent

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

Development happens on GitHub. We actively welcome pull requests against
the `main` branch. This repository is not synchronized with any internal
ShipIt flow; the GitHub `main` branch is the source of truth for external
contributions.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only
need to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Do not upload private keys, credentials, private personal memories, or
restricted videos (including Ego4D footage) to issues. Redact traces and
logs before posting.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

* 4 spaces for indentation rather than tabs.
* 100 character line length, enforced by Ruff (see `pyproject.toml`).
* Format and lint with the repository's pinned Ruff configuration:

```bash
python -m pip install -e '.[dev]'
python -m ruff check memoproagent tests tools --select E9,F63,F7,F82
```

## Testing

Tests are CPU-only and use synthetic data plus scripted backends. No GPU,
live model endpoint, model download, paid judge, or training dependency is
required to run the default suite.

```bash
python -m pip install -e '.[dev,ppr]'
python -m pytest -q
```

## License

By contributing to MemoProAgent, you agree that your contributions will
be licensed under the LICENSE file in the root directory of this source
tree (CC-BY-NC-4.0).

Note: Ego4D videos and dataset histories accessed through this harness
remain under their own separate terms. Contributing code here does not
relicense that data.
