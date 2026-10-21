# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Offline checks for the public release's OSS files and CI configuration."""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COPYRIGHT = "Copyright (c) Meta Platforms, Inc. and affiliates."
LICENSE_ID = "CC-BY-NC-4.0"
LICENSE_SHA = "cc97a2c4ff7723a1f25179dc7c18f676fa84191bbb36ea7d0061acaf3c102559"


@pytest.mark.parametrize("relative", [
    "LICENSE", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "CHANGELOG.md",
    ".github/workflows/ci.yml", ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml", ".github/ISSUE_TEMPLATE/config.yml",
])
def test_required_public_files_exist(relative):
    assert (ROOT / relative).is_file()
    assert (ROOT / relative).stat().st_size > 0


def test_canonical_declared_license_is_preserved():
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == LICENSE_SHA
    assert LICENSE_ID in (ROOT / "pyproject.toml").read_text()


def test_source_copyright_headers_and_python_syntax():
    paths = [p for folder in ("memoproagent", "tests", "tools")
             for p in (ROOT / folder).rglob("*.py")]
    assert paths
    for path in paths:
        source = path.read_text(encoding="utf-8")
        header = "\n".join(source.splitlines()[:16])
        assert COPYRIGHT in header, path
        assert f"SPDX-License-Identifier: {LICENSE_ID}" in header, path
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
    assert (ROOT / "tools/slurm/run_gpm.sbatch").read_text().startswith("#!/")
    for path in [ROOT / "pyproject.toml", ROOT / "tools/slurm/run_gpm.sbatch",
                 *(ROOT / "configs").glob("*.yaml"),
                 *(ROOT / ".github").rglob("*.yml")]:
        header = "\n".join(path.read_text().splitlines()[:16])
        assert COPYRIGHT in header, path
        assert f"SPDX-License-Identifier: {LICENSE_ID}" in header, path


def test_public_branding_and_compatible_identifiers():
    readme = (ROOT / "README.md").read_text()
    project = (ROOT / "pyproject.toml").read_text()
    assert "# MemoProAgent<br>Toward Memory-Driven Proactive Assistance" in readme
    assert "title = {MemoProAgent: Toward Memory-Driven Proactive Assistance}" in readme
    assert "@misc{zeng2026memoproagent," in readme
    assert "# ProPerAgent<br>" not in readme
    assert "[Quick Start]\n" not in readme
    assert "https://github.com/facebookresearch/Memo-Pro-Agent.git" in readme
    assert "https://huggingface.co/datasets/facebook/MemProBench" in readme
    assert 'name = "memoproagent"' in project
    assert 'version = "0.1.0"' in project
    assert 'description = "MemoProAgent:' in project
    assert 'memoproagent-run = "memoproagent.cli.run:main"' in project
    assert 'memoproagent-build-memory = "memoproagent.cli.build_memory:main"' in project


def test_contributor_covenant_and_reporting_routes():
    conduct = (ROOT / "CODE_OF_CONDUCT.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert "Contributor Covenant" in conduct
    assert "opensource-conduct@meta.com" in conduct
    assert "https://www.contributor-covenant.org" in conduct
    assert "MemoProAgent" in contributing
    assert "https://code.facebook.com/cla" in contributing
    assert "https://bugbounty.meta.com/" in contributing
    assert "__________" not in contributing


def test_cpu_ci_configuration():
    config = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert {"push", "pull_request", "workflow_dispatch"} <= set(config["on"])
    assert config["on"]["push"]["branches"] == ["main"]
    assert config["permissions"] == {"contents": "read"}
    job = config["jobs"]["tests"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["strategy"]["matrix"]["python-version"] == ["3.10", "3.11", "3.12"]
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
    steps = job["steps"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", step["uses"])
    commands = "\n".join(step.get("run", "") for step in steps)
    assert ".[dev,ppr]" in commands
    assert "python -m pytest" in commands
    assert "E9,F63,F7,F82" in commands
    assert ".[train]" not in commands
    assert "vllm serve" not in commands


@pytest.mark.parametrize("name", ["bug_report.yml", "feature_request.yml"])
def test_issue_forms_have_valid_structure(name):
    form = yaml.safe_load((ROOT / ".github/ISSUE_TEMPLATE" / name).read_text())
    assert form["name"] and form["description"]
    assert isinstance(form["body"], list) and form["body"]
    ids = []
    for item in form["body"]:
        assert item["type"] in {"markdown", "input", "textarea", "dropdown", "checkboxes"}
        if item["type"] != "markdown":
            ids.append(item["id"])
            assert item["attributes"]["label"]
    assert len(ids) == len(set(ids))


def test_security_reports_are_routed_privately():
    config = yaml.safe_load((ROOT / ".github/ISSUE_TEMPLATE/config.yml").read_text())
    assert config["blank_issues_enabled"] is False
    assert any(link["url"] == "https://bugbounty.meta.com/"
               for link in config["contact_links"])


def test_readme_affiliations_and_local_logo_images():
    from html.parser import HTMLParser
    from PIL import Image, ImageChops

    class ImageReferences(HTMLParser):
        def __init__(self):
            super().__init__()
            self.images = []

        def handle_starttag(self, tag, attrs):
            if tag == "img":
                self.images.append(dict(attrs))

    readme = (ROOT / "README.md").read_text()
    assert "Tianxin Wei<sup>4,6</sup>" in readme
    assert "Meta Recommendation Systems" in readme
    assert "<sup>6</sup> University of Illinois Urbana-Champaign" in readme
    parser = ImageReferences()
    parser.feed(readme)
    banners = [item for item in parser.images
               if item.get("src", "").startswith("assets/affiliations/")]
    assert len(banners) == 1
    attrs = banners[0]
    source = "assets/affiliations/banner.png"
    assert attrs["src"] == source
    assert attrs["width"] == "100%"
    assert "height" not in attrs
    assert attrs["alt"] == (
        "Meta, Michigan State University, University of Washington, "
        "University of Illinois Urbana-Champaign"
    )
    assert readme.index(f'src="{source}"') < readme.index("# MemoProAgent")
    logos = []
    for name in ("meta", "msu", "uw", "uiuc"):
        with Image.open(ROOT / f"assets/affiliations/{name}.png") as image:
            assert image.width > 0 and image.height > 0
            assert image.mode == "RGB"
            bounds = ImageChops.difference(
                image, Image.new("RGB", image.size, "white")
            ).getbbox()
            assert bounds is not None
            left, top, right, bottom = bounds
            assert left > 0 and top > 0
            assert right < image.width and bottom < image.height
            logos.append(image.resize(
                (round(image.width * 96 / image.height), 96),
                Image.Resampling.LANCZOS,
            ))
    with Image.open(ROOT / source) as banner:
        assert banner.mode == "RGB"
        assert banner.size == (1600, 160)
        expected = Image.new("RGB", banner.size, "white")
        x = (banner.width - sum(logo.width for logo in logos) - 80 * 3) // 2
        assert x > 0
        for logo in logos:
            expected.paste(logo, (x, 32))
            x += logo.width + 80
        assert ImageChops.difference(banner, expected).getbbox() is None
    with Image.open(ROOT / source) as image:
        image.verify()