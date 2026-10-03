"""Download the public preference datasets used by the diagnostics.

Nothing is fetched without --yes. Every file is recorded in data/MANIFEST.json
with its URL, size, and SHA-256 so reports can cite the exact bytes used.

    python -m hodge.fetch --list
    python -m hodge.fetch --yes helpsteer mt_bench
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"

SOURCES = {
    "helpsteer": dict(
        repo="nvidia/HelpSteer", license="CC-BY-4.0",
        files=[(f"{HF}/datasets/nvidia/HelpSteer/resolve/main/train.jsonl.gz", "train.jsonl.gz", 15.9),
               (f"{HF}/datasets/nvidia/HelpSteer/resolve/main/validation.jsonl.gz", "validation.jsonl.gz", 0.8)]),
    "mt_bench": dict(
        repo="lmsys/mt_bench_human_judgments", license="CC-BY-4.0",
        files=[(f"{HF}/datasets/lmsys/mt_bench_human_judgments/resolve/main/data/"
                "human-00000-of-00001-25f4910818759289.parquet", "human.parquet", 0.7),
               (f"{HF}/datasets/lmsys/mt_bench_human_judgments/resolve/main/data/"
                "gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet", "gpt4_pair.parquet", 0.6)]),
    "ultrafeedback": dict(
        repo="openbmb/UltraFeedback", license="MIT",
        files=[(f"{HF}/api/datasets/openbmb/UltraFeedback/parquet/default/train/{i}.parquet",
                f"train-{i}.parquet", 161.4) for i in (0, 1)]),
    "arena55k": dict(
        repo="lmarena-ai/arena-human-preference-55k", license="Apache-2.0",
        files=[(f"{HF}/api/datasets/lmarena-ai/arena-human-preference-55k/parquet/default/train/0.parquet",
                "train-0.parquet", 101.5)]),
    "helpsteer2_preference": dict(
        repo="nvidia/HelpSteer2", license="CC-BY-4.0",
        files=[(f"{HF}/datasets/nvidia/HelpSteer2/resolve/main/preference/preference.jsonl.gz",
                "preference.jsonl.gz", 15.3)]),
}


def fetch(name, root):
    spec = SOURCES[name]
    target = Path(root) / name
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(root) / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for url, filename, _ in spec["files"]:
        path = target / filename
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "hodge-diagnostics"})
            with urllib.request.urlopen(request, timeout=120) as response, open(path.with_suffix(".part"), "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            path.with_suffix(".part").rename(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[f"{name}/{filename}"] = dict(url=url, repo=spec["repo"], license=spec["license"],
                                              bytes=path.stat().st_size, sha256=digest,
                                              fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        print(f"{name}/{filename}: {path.stat().st_size / 1e6:.1f} MB")
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", choices=list(SOURCES))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm the download")
    args = parser.parse_args(argv)
    if args.list or not args.datasets:
        for name, spec in SOURCES.items():
            size = sum(s or 0 for *_, s in spec["files"])
            print(f"{name:24s} {spec['repo']:42s} {spec['license']:12s} ~{size:.0f} MB")
        return
    if not args.yes:
        sys.exit("Refusing to download without --yes")
    for name in args.datasets:
        fetch(name, args.root)


if __name__ == "__main__":
    main()
