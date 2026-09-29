# Offline Image Analysis

Experimental local image-to-JSON/XLSX extraction for **CPython 3.13.12**, Linux
x86_64 / RHEL 8.10 (glibc 2.28), CPU execution, and a machine with 64 GB RAM.
Includes a standalone command and a local MCP server for OpenCode.

## Accuracy status

**This is a prototype, not reliable unattended transcription.** Real-image testing
found invented values in blank cells, incorrect numbers, and guesses in cropped rows.
One dense-table test returned only 20 of 23 data rows and misread several values.
Schema validation does not establish that the image was read correctly. All results
are marked `needs_review`; verify important data against the image.

The local Qwen3-VL-2B-Instruct model reads printed text, tables, flowchart nodes/edges,
and chart labels/estimated values. Excel exports contain data sheets only, with no
Read me worksheet. Warnings, model provenance and source hashes remain in JSON.
Model weights are unchanged by this export-layout update.

## Source repository versus offline bundle

**Cloning this repository does not provide a self-contained installation.** Model
weights and Linux wheels are deliberately excluded from Git. A complete offline ZIP
is approximately 4.51 GB. The repository contains source, synthetic test images,
dependency locks, asset checksums and scripts to build/distribute that ZIP.

On an internet-connected computer with Python 3.12 or newer and pip:

```bash
python build_bundle.py
```

This downloads the exact model revision and hash-locked Linux wheels, checks asset
hashes, and creates `offline-image-tool-rhel8-py313.zip`. Allow at least 12 GB of free
disk space for the bundle build, and another 5 GB if you also create release parts.
The build computer's Python is only a packaging utility; the offline tool installer
requires Python 3.13.12 exactly.

Transfer the complete ZIP to the disconnected Linux machine:

```bash
python3.13 -m zipfile -e offline-image-tool-rhel8-py313.zip .
cd offline_image_tool
python3.13 install.py
./venv/bin/python image_tool.py analyze /path/table.png --kind table --output results/table1
```

Installation uses a local virtual environment and needs no administrator privileges,
compiler, Tesseract, or internet access. Keep all bundle directories intact.
The existing Python must include venv and ensurepip, and the system must provide
libstdc++.so.6, libgcc_s.so.1 and libgomp.so.1 as described in the prerequisite check.

See [usage and installation details](offline_image_tool/README.md) and
[validation limits](offline_image_tool/VALIDATION.md). CPU inference can take minutes
per image. Execution was tested on Windows using a separate test environment; native
RHEL execution and default float32 inference still require verification on the target.

## OpenCode

The installer generates a local MCP configuration with absolute paths. Merge its
entry into OpenCode's configuration. The main OpenCode agent also needs an available
local model: a cloud API cannot operate on a completely disconnected computer.
The bundled vision model is used by this tool; it is not a general OpenCode agent server.

## Tests

The tests require Pillow, openpyxl and jsonschema for software checks. Real-model
acceptance tests additionally require the complete model/runtime bundle.

```bash
python -m unittest discover -s offline_image_tool/tests -v
python offline_image_tool/tests/acceptance.py
```

The source contains synthetic fixtures only. Private screenshots, real-image test
outputs, model weights, caches, virtual environments and credentials are not included.
Historical validation evidence referenced in VALIDATION.md lives in the previously
built offline bundle; fresh builds can generate their own acceptance results.

## GitHub Releases

GitHub release files must each be under 2 GiB. Use the included helper to split the
offline ZIP into 1 GiB parts, and upload every part, the JSON manifest, and
`release_parts.py` as release assets:

```bash
python release_parts.py split offline-image-tool-rhel8-py313.zip release-parts
```

Recipients download all assets into one directory and reconstruct the ZIP offline:

```bash
python release_parts.py join release-parts/parts.json offline-image-tool-rhel8-py313.zip
```

Each part and the reconstructed ZIP are checksum-verified. The source ZIP GitHub
automatically offers is not the offline installation bundle. Do not commit model
weights, wheel files, or the large offline ZIP to normal Git history.

## License

Tool source: [MIT](LICENSE). Qwen model: Apache-2.0, with its license and pinned
revision recorded separately. Third-party wheels retain their original license
notices in the offline bundle.

