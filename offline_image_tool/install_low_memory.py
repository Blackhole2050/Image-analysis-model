"""Install the ONNX OCR profile offline into a local virtual environment."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / "SHA256SUMS.lowmem.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = ROOT / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError("Bundle checksum mismatch: " + name)
    print("Low-memory bundle checksums verified.", flush=True)


def main():
    if sys.version_info[:3] != (3, 13, 12) or platform.python_implementation() != "CPython":
        raise RuntimeError("Run with CPython 3.13.12 exactly.")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("This bundle is for Linux x86_64.")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise RuntimeError("glibc 2.28 or newer is required.")
    verify()
    env = ROOT / "venv-lowmem"
    if env.exists():
        raise RuntimeError("venv-lowmem already exists; use it or move it before reinstalling.")
    print("Creating a private offline Python environment.", flush=True)
    venv.EnvBuilder(with_pip=True).create(env)
    python = env / "bin/python"
    subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-index", "--no-cache-dir", "--disable-pip-version-check", "--no-deps", "--find-links", str(ROOT / "wheelhouse-lowmem"), "--require-hashes", "-r", str(ROOT / "requirements.lowmem.lock")], check=True)
    subprocess.run([str(python), "-c", "import cv2, numpy, onnxruntime, openpyxl, rapidocr"], check=True)
    subprocess.run([str(python), str(ROOT / "image_tool.py"), "--backend", "ocr", "doctor"], check=True)
    config = {"mcp": {"offline_image_low_memory": {"type": "local", "command": [str(python), str(ROOT / "image_tool.py"), "--backend", "ocr", "--threads", "2", "mcp"], "enabled": True, "timeout": 600000}}}
    (ROOT / "opencode.lowmem.generated.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Installed. Run the venv-lowmem Python with image_tool.py; merge opencode.lowmem.generated.json into OpenCode.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("INSTALL FAILED:", exc, file=sys.stderr)
        raise SystemExit(1)
