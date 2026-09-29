"""Online source-to-offline-bundle builder. Run on a connected computer only."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE / "offline_image_tool"
EXPECTED = json.loads((BASE / "expected_assets.json").read_text())

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def fetch(name, url):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha(path) == EXPECTED[name]:
        return
    temporary = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, 4 * 1024 * 1024)
    if sha(temporary) != EXPECTED[name]:
        raise RuntimeError("Asset checksum mismatch: " + name)
    temporary.replace(path)
    print("Downloaded", name, flush=True)

def models():
    info = json.loads((ROOT / "model_provenance.json").read_text())
    jobs = []
    for name in EXPECTED:
        if not name.startswith("models/vision/"):
            continue
        filename = name.removeprefix("models/vision/")
        if filename == "LICENSE":
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE / "licenses/Qwen3-VL-2B-Instruct-LICENSE.txt", target)
        else:
            jobs.append((name, f"https://huggingface.co/{info['repository']}/resolve/{info['revision']}/{filename}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(fetch, *job) for job in jobs]:
            future.result()

def wheels():
    command = [sys.executable, "-m", "pip", "download", "--dest", str(ROOT / "wheelhouse"), "--only-binary=:all:", "--require-hashes", "--python-version", "3.13", "--implementation", "cp", "--abi", "cp313", "--abi", "abi3", "--abi", "none"]
    for platform in ("manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux2014_x86_64", "manylinux_2_17_x86_64"):
        command += ["--platform", platform]
    command += ["--extra-index-url", "https://download.pytorch.org/whl/cpu", "-r", str(ROOT / "requirements.lock")]
    subprocess.run(command, check=True)

def main():
    if shutil.disk_usage(BASE).free < 12 * 1024**3:
        raise RuntimeError("At least 12 GiB free disk space is required for a clean build.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(models), pool.submit(wheels)]
        for job in jobs:
            job.result()
    for name, expected in EXPECTED.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Missing or changed asset: " + name)
    excluded = {"__pycache__", "venv", "acceptance-results", "validation_samples"}
    files = [p for p in ROOT.rglob("*") if p.is_file() and not any(x in excluded for x in p.relative_to(ROOT).parts) and p.name not in ("SHA256SUMS.json", "opencode.generated.json") and not p.name.endswith(".partial")]
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    (ROOT / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2))
    files.append(ROOT / "SHA256SUMS.json")
    archive = BASE / "offline-image-tool-rhel8-py313.zip"
    with zipfile.ZipFile(archive, "w", allowZip64=True) as z:
        for path in files:
            z.write(path, "offline_image_tool/" + path.relative_to(ROOT).as_posix(), compress_type=zipfile.ZIP_STORED if path.suffix in (".whl", ".safetensors") else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:
        for name, expected in manifest.items():
            with z.open("offline_image_tool/" + name) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    archive.with_suffix(".zip.sha256").write_text(sha(archive) + "  " + archive.name + "\n")
    print("Created and verified", archive)

if __name__ == "__main__":
    main()

