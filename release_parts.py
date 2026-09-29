"""Split/join a large offline ZIP for GitHub Releases using only standard Python."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

def split(source, directory, part_bytes=1024**3):
    source, directory = Path(source), Path(directory)
    if not 1 <= part_bytes <= 1024**3:
        raise ValueError("Part size must be between 1 byte and 1 GiB")
    if shutil.disk_usage(directory.parent).free < source.stat().st_size + 1024**2:
        raise RuntimeError("Insufficient free space to create all release parts")
    directory.mkdir(exist_ok=False)
    total, parts = hashlib.sha256(), []
    with source.open("rb") as incoming:
        while True:
            first = incoming.read(min(part_bytes, 8 * 1024**2))
            if not first:
                break
            name = source.name + f".part{len(parts)+1:04d}"
            size, digest = 0, hashlib.sha256()
            with (directory / name).open("xb") as out:
                block = first
                while block:
                    out.write(block)
                    size += len(block)
                    digest.update(block)
                    total.update(block)
                    if size == part_bytes:
                        break
                    block = incoming.read(min(part_bytes-size, 8 * 1024**2))
            parts.append({"name": name, "bytes": size, "sha256": digest.hexdigest()})
    manifest = {"filename": source.name, "bytes": source.stat().st_size, "sha256": total.hexdigest(), "parts": parts}
    (directory / "parts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def join(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".partial")
    total, total_size = hashlib.sha256(), 0
    with temporary.open("xb") as out:
        for part in manifest["parts"]:
            name = part["name"]
            if not name or Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
                raise ValueError("Unsafe part filename")
            digest, size = hashlib.sha256(), 0
            with (manifest_path.parent / name).open("rb") as incoming:
                while block := incoming.read(8 * 1024**2):
                    out.write(block)
                    digest.update(block)
                    total.update(block)
                    size += len(block)
                    total_size += len(block)
            if size != part["bytes"] or digest.hexdigest() != part["sha256"]:
                raise ValueError("Part checksum mismatch: " + name)
    if total_size != manifest["bytes"] or total.hexdigest() != manifest["sha256"]:
        raise ValueError("Reconstructed archive checksum mismatch")
    temporary.replace(output)
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split")
    s.add_argument("source")
    s.add_argument("directory")
    s.add_argument("--part-mib", type=int, default=1024, choices=range(1, 1025), metavar="1..1024")
    j = sub.add_parser("join")
    j.add_argument("manifest")
    j.add_argument("output")
    args = parser.parse_args()
    if args.command == "split":
        result = split(args.source, args.directory, args.part_mib * 1024**2)
        print(f"Created {len(result['parts'])} parts plus parts.json in {args.directory}")
    else:
        print("Reconstructed and verified", join(args.manifest, args.output))

if __name__ == "__main__":
    main()

