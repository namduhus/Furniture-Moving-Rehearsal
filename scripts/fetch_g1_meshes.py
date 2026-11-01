"""Download the G1 visual meshes for the WebGL viewer and verify them against the pinned manifest.

The manifest (web/vendor/g1/meshes.json) lists the STL files referenced by assets/references/g1_wbc.urdf with the
SHA-256 and size of the upstream Git LFS objects at a pinned GR00T-WholeBodyControl commit. Files already present with
the right hash are kept; anything else is downloaded, checked and only then written. The meshes stay Git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "web/vendor/g1/meshes.json"
OUT = ROOT / "web/vendor/g1/meshes"


def download(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def fetch(manifest, out, get=download):
    """Write every manifest file into `out`, downloading the ones missing or not matching; returns (fetched, kept)."""
    out.mkdir(parents=True, exist_ok=True)
    fetched, kept = [], []
    for name, expected in manifest["files"].items():
        if "/" in name or name.startswith("."):
            raise ValueError(f"Unexpected mesh name: {name}")
        path = out / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]:
            kept.append(name)
            continue
        data = get(manifest["download"].format(name=name))
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"{name}: downloaded content does not match the pinned size and SHA-256")
        path.write_bytes(data)
        fetched.append(name)
    return fetched, kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "files": len(manifest["files"]),
                          "megabytes": round(sum(f["bytes"] for f in manifest["files"].values())/1e6, 1),
                          "source": f"{manifest['source']} @ {manifest['commit']}"}, indent=2))
        return 0
    fetched, kept = fetch(manifest, OUT)
    print(json.dumps({"fetched": len(fetched), "kept": len(kept), "out": str(OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
