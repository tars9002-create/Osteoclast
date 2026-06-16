#!/usr/bin/env python3
"""Lightweight repository checks for the manuscript code package."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE = 50 * 1024 * 1024

REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "CODE_AVAILABILITY.md",
    "DATA_AVAILABILITY.md",
    "REPRODUCIBILITY.md",
    "LICENSE",
    "CITATION.cff",
    "env/setup.sh",
    "data_discovery/census_osteoclast.py",
    "data_discovery/verify_oc_content.py",
    "data_engineering/build_master.py",
    "data_engineering/integrate_annotate.py",
    "discovery/convergence_discovery_v2.py",
    "discovery/robust_and_driver.py",
    "modeling/insilico_perturb.py",
    "validation/supp_compartment.py",
    "validation/supp_external.py",
    "validation/correction_adult_ref.py",
    "validation/doublet_control.py",
    "validation/reanalysis_harden.py",
    "validation/build_evidence_matrix.py",
    "advanced/run_adv_A.py",
    "advanced/run_adv_B.py",
    "advanced/adv_helpers.py",
    "advanced/make_figs_v2.py",
    "report/pub_style.py",
    "figures/README.md",
    "figures/main/.gitkeep",
    "figures/supplementary/.gitkeep",
]

FORBIDDEN_FILE_PATTERNS = [
    "*.docx",
    "*.zip",
    "*.pdf",
    "*.png",
    "*.h5ad",
    "*.h5",
    "*.pt",
    "*.npy",
    "*.npz",
    ".DS_Store",
]

TEXT_SUFFIXES = {
    ".cff",
    ".gitignore",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yml",
    ".yaml",
}

mac_home_prefix = "/Us" + "ers/"
IDENTITY_PATTERNS = [re.compile(re.escape(mac_home_prefix) + r"[^\\s'\"]+")]
home_name = Path.home().name
if home_name not in {"", "root", "runner"}:
    IDENTITY_PATTERNS.append(re.compile(re.escape(home_name), re.IGNORECASE))


def tracked_or_visible_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        files = [ROOT / line for line in out.splitlines() if line]
        if files:
            return files
    except Exception:
        pass

    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    files: list[Path] = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in names:
            if name == ".DS_Store":
                continue
            files.append(Path(base) / name)
    return files


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    errors: list[str] = []
    files = tracked_or_visible_files()

    for item in REQUIRED_PATHS:
        if not (ROOT / item).exists():
            errors.append(f"missing required path: {item}")

    for path in files:
        r = rel(path)
        if path.is_file() and path.stat().st_size > MAX_FILE_SIZE:
            errors.append(f"file exceeds 50 MB: {r}")
        for pattern in FORBIDDEN_FILE_PATTERNS:
            if fnmatch.fnmatch(path.name, pattern):
                errors.append(f"forbidden file in repository package: {r}")

        if path.suffix in TEXT_SUFFIXES or path.name in {"README.md", "LICENSE"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in IDENTITY_PATTERNS:
                if pattern.search(text):
                    errors.append(f"possible local identity/path leak in {r}: {pattern.pattern}")

    if errors:
        print("Repository check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"Repository check passed ({len(files)} files scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
