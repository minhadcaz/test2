"""Download the pinned, official Mini-Dev SQLite questions and schema (no DBs)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "5b256b5a9c17478f06ed6912612e9a14d17065c8"
BASE = f"https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/{REVISION}"
FILES = {
    "mini_dev_sqlite.jsonl": ("data/mini_dev_sqlite.jsonl", "926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606"),
    "schemas.json": ("baseline_inference/schemas/sqlite.json", "5176e324a72e80a8a37a12285596d123e0ba6c310dceaa9f13a1e413ee88ff5b"),
    "manifest.json": ("data/manifest.json", None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/bird_minidev"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"repository": "birdsql/bird_mini_dev", "revision": REVISION,
                  "license": "CC-BY-SA-4.0 (dataset card)", "files": {}}
    for name, (remote, expected) in FILES.items():
        target = args.output / name
        if target.exists() and expected and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            data = target.read_bytes()
        else:
            with urlopen(f"{BASE}/{remote}?download=true", timeout=60) as response:
                data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if expected and digest != expected:
            raise ValueError(f"SHA-256 mismatch for {name}; refusing to write")
        # Validate that a login/error HTML page did not replace JSON.
        if name.endswith("jsonl"):
            rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
            if len(rows) != 500 or len({r['question_id'] for r in rows}) != 500:
                raise ValueError("Expected 500 unique Mini-Dev questions")
        else:
            json.loads(data)
        target.write_bytes(data)
        provenance["files"][name] = {"url": f"{BASE}/{remote}", "sha256": digest, "bytes": len(data)}
        print(f"{name}: {len(data)} bytes, sha256={digest}")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
