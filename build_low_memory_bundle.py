"""Build the compact, offline ONNX OCR bundle on an internet-connected host."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE / "offline_image_tool"
LOCK = ROOT / "requirements.lowmem.lock"
SOURCES = (
    "image_tool.py", "ocr_engine.py", "install_low_memory.py",
    "check_image_tool_prerequisites.py", "model_provenance.lowmem.json",
    "README.lowmem.md", "requirements.lowmem.lock", "LICENSE",
    "RapidOCR-LICENSE.txt", "THIRD_PARTY.lowmem.md",
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    wheelhouse = ROOT / "wheelhouse-lowmem"
    wheelhouse.mkdir(exist_ok=True)
    command = [sys.executable, "-m", "pip", "download", "--dest", str(wheelhouse),
               "--only-binary=:all:", "--require-hashes", "--no-deps", "--python-version", "3.13",
               "--implementation", "cp", "--abi", "cp313", "--abi", "abi3", "--abi", "none"]
    for platform in ("manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux2014_x86_64", "manylinux_2_17_x86_64"):
        command += ["--platform", platform]
    command += ["-r", str(LOCK)]
    subprocess.run(command, check=True)

    files = [ROOT / name for name in SOURCES]
    files += sorted(wheelhouse.glob("*.whl"))
    if len(files) < len(SOURCES) + 20:
        raise RuntimeError("Wheelhouse is incomplete; expected the complete hash-locked dependency set.")
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    manifest_path = ROOT / "SHA256SUMS.lowmem.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.append(manifest_path)
    archive = BASE / "offline-image-tool-rhel8-lowmem.zip"
    with zipfile.ZipFile(archive, "w", allowZip64=True) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            compress = zipfile.ZIP_STORED if path.suffix == ".whl" else zipfile.ZIP_DEFLATED
            zf.write(path, "offline_image_tool/" + rel, compress_type=compress)
    with zipfile.ZipFile(archive) as zf:
        for name, expected in manifest.items():
            with zf.open("offline_image_tool/" + name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise RuntimeError("Archive verification failed: " + name)
    checksum = sha(archive)
    archive.with_suffix(".zip.sha256").write_text(checksum + "  " + archive.name + "\n", encoding="ascii")
    print(f"Created {archive} ({archive.stat().st_size:,} bytes), SHA-256 {checksum}")


if __name__ == "__main__":
    main()
