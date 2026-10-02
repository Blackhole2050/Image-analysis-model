# Offline Image Analysis — low-memory OCR profile

This profile uses RapidOCR 3.9.2 and ONNX Runtime CPU. It contains no PyTorch,
Transformers, or 2B vision-language model. The included RapidOCR wheel also
contains its small detector and recognizer weights, so inference is offline.

## Supported output

- Printed English text is recognized with word-region coordinates and confidence.
- Ruled tables use luminance and individual color channels for border detection.
  For detected grid tables, the dominant interior cell color is recorded in JSON
  and applied as an approximate Excel fill. Merged, wrapped, borderless, or
  irregular cells still need review; subtle colors and gradients can be missed.
- Flowchart text is grouped with detected node boxes where possible. The tool
  saves a `flowchart_N.png` overlay and node/edge XLSX sheets. The overlay marks
  detected text/node regions; it does not reconstruct arrow direction or branch
  connections, which remain empty and must be reviewed.
- Chart text and likely tick labels are extracted. Plotted data values are not
  digitized; no numeric series is invented.

All extracted values are marked `needs_review`. OCR can still misread digits,
punctuation, small labels, and unusual fonts. Suspected standalone `TT`/`π`
readings are flagged in the JSON uncertainty list and as Excel cell comments;
the OCR engine does not return alternate character candidates, so it cannot
reliably decide between them automatically. This is a lightweight OCR and
geometry pipeline rather than a general visual reasoning model.

Color-channel table and flowchart processing is only enabled when the image
contains visible chroma, so ordinary grayscale images keep the lighter path.
The OCR network itself still runs once per image. Do not assume the previous
30-images-in-10-minutes rate until the changed bundle has been timed on the
same host and image set.

## Install on disconnected RHEL 8.10

Extract this bundle with Python 3.13.12 and run:

```bash
cd offline_image_tool
python3.13 install_low_memory.py
```

It creates a private `venv-lowmem` directory, installs only from the bundled
wheelhouse, verifies hashes, and creates an OpenCode MCP configuration file.
It needs no root access, Tesseract, or internet connection. It does not install
PyTorch or Transformers. The OCR weights are included inside RapidOCR's wheel.
No target-machine peak-memory benchmark has been completed; image input is capped
at 40 megapixels and defaults to 1 megapixel before processing. The ONNX runtime
uses two intra-op threads in the generated OpenCode configuration.

## Analyze an image

```bash
./venv-lowmem/bin/python image_tool.py --backend ocr --threads 2 analyze \
  /path/to/image.png --kind table --output results
```

Use `--kind flowchart`, `--kind chart`, or `--kind text` for focused extraction.
The XLSX contains extracted data and approximate table fill colors; flowchart
node regions are also saved as PNG overlays. JSON carries warnings and confidence
details. Color processing uses the existing OCR pass; no second recognition pass
is added. This should preserve the current inference count, but the new image
processing overhead and total batch time have not been benchmarked on your RHEL
machine, so compare a similar 30-image batch before adopting it.
The OpenCode MCP server always uses this OCR profile. The original `--backend qwen`
mode still needs its separate multi-gigabyte model bundle and may use substantially
more memory.

## Build the offline ZIP on a connected system

Use CPython 3.13 or newer with internet access and pip, from the repository root:

```bash
python build_low_memory_bundle.py
```

This downloads only hash-pinned binary Linux wheels into `offline_image_tool/wheelhouse-lowmem/`
and creates `offline-image-tool-rhel8-lowmem.zip`. Transfer that ZIP and its `.sha256`
file to the disconnected RHEL machine. The wheelhouse has been generated for CPython
3.13, Linux x86_64, glibc 2.28 or newer.

Tool source is MIT licensed. RapidOCR and its included model weights are
Apache-2.0. Other package licenses remain with their bundled wheel metadata.
