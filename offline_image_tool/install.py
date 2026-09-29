"""Install entirely offline into ./venv using the existing CPython 3.13.12."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT / "SHA256SUMS.json").read_text())
    for name, expected in manifest.items():
        path = ROOT / name
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while block := f.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError("Corrupt or changed bundle file: " + name)
    print("Bundle checksums verified.", flush=True)

def main():
    if sys.version_info[:3] != (3, 13, 12) or platform.python_implementation() != "CPython":
        raise RuntimeError("Run with CPython 3.13.12 exactly; no alternate Python is bundled.")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("This bundle is for Linux x86_64 only.")
    if sysconfig_free_threaded():
        raise RuntimeError("This bundle requires standard CPython, not the free-threaded build.")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise RuntimeError("glibc 2.28 or newer is required.")
    verify()
    env = ROOT / "venv"
    if env.exists():
        raise RuntimeError("venv already exists. Use its Python, or rename that directory before reinstalling.")
    print("Creating isolated environment; system packages will not be changed.", flush=True)
    venv.EnvBuilder(with_pip=True).create(env)
    python = env / "bin/python"
    subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-index", "--no-cache-dir", "--disable-pip-version-check", "--find-links", str(ROOT / "wheelhouse"), "--require-hashes", "-r", str(ROOT / "requirements.lock")], check=True)
    subprocess.run([str(python), "-m", "pip", "--isolated", "check"], check=True)
    subprocess.run([str(python), str(ROOT / "image_tool.py"), "doctor"], check=True)
    config = {"mcp": {"offline_image": {"type": "local", "command": [str(python), str(ROOT / "image_tool.py"), "mcp"], "enabled": True, "timeout": 1800000}}}
    (ROOT / "opencode.generated.json").write_text(json.dumps(config, indent=2))
    print("Installed. Merge opencode.generated.json into your OpenCode configuration. See README.md.")

def sysconfig_free_threaded():
    import sysconfig
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("INSTALL FAILED:", exc, file=sys.stderr)
        raise SystemExit(1)

