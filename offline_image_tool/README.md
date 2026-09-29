# Offline image analysis for Python 3.13.12

Target: standard CPython 3.13.12, RHEL 8.10 / glibc 2.28, x86_64, CPU, 64 GB RAM.
No administrator access, internet connection, Tesseract, alternate Python, Docker,
Node installation or compiler is required by this tool. Your existing Python must
provide venv/ensurepip and the native libraries confirmed by the prerequisite check.
Install in a writable directory on a filesystem that allows execution.

## Install on the disconnected machine

Extract the entire ZIP, preserving directories. Python itself can extract it:

```bash
python3.13 -m zipfile -e offline-image-tool-rhel8-py313.zip /path/to/writable/folder
cd /path/to/writable/folder/offline_image_tool
python3.13 install.py
```

The installer checks every bundled file, then installs the exact hashed Linux wheels
into `venv/`. It does not use or modify your existing site-packages. All model weights,
processor/tokenizer files and Python package dependencies are included. Installation
fails on a different Python version or incompatible platform. Keep the folder in
place after installation: virtual environments contain absolute paths.

## Extract an image

```bash
./venv/bin/python image_tool.py analyze /path/table.png --kind table --output results/table1
./venv/bin/python image_tool.py analyze /path/flowchart.png --kind flowchart --verify --output results/flow1
./venv/bin/python image_tool.py analyze /path/chart.png --kind chart --output results/chart1
```

Each output directory must be new. Successful runs contain:

XLSX files contain extracted data worksheets only; there is no Read me worksheet.
Titles, source information, warnings and uncertainty notes remain in result.json.

- `result.xlsx`: separate table, flowchart node/edge and numerical chart sheets.
- `result.json`: structured extraction including axes, uncertainties and provenance.
- `input_used.png`: the actual oriented/cropped/resized image provided to the model.
- `raw_response.txt`: original model output for inspection.
- `verification_response.txt`: optional second reading when `--verify` is used.

Every result has `needs_review` status. Excel cells are literal strings to preserve
leading zeros, units and decimals, and to prevent image text from becoming formulas.
Null values become blank Excel cells. JSON retains nulls. Header rows stay in table
data. Physical merged-cell reconstruction is not implemented; inspect complex tables.
A flat list of column headings returned by the model is normalized to one header row;
cell text is unchanged and the original model response is retained.
In a specifically selected mode, only that category is exported; unrelated model
categories are excluded and flagged. Use auto mode when all categories are wanted.
An omitted uncertainty report is flagged. Automatic mode remains strict about all
categories being present; no table cells, graph connections or point values are inferred
by the parser.
Numeric/logarithmic axes with nonnumeric ticks are flagged for review. The parser
does not silently replace the model's scale classification with a guessed correction.

For dense images, crop the relevant region (left top right bottom in pixels after EXIF
orientation). A crop intentionally excludes all content outside that region:

```bash
./venv/bin/python image_tool.py analyze /path/image.png --crop 0 0 900 700 --output results/crop1
```

Defaults: 8 CPU threads, at most 1,048,576 image pixels, 4096 output tokens. Use
`--threads 16` BEFORE the command to experiment with your CPU. Increasing resolution
or output length costs memory and time. `--max-pixels` accepts up to 4,194,304 and
`--max-tokens` up to 16,384. CPU inference can take minutes or longer; timing depends
on the image and output. There is no established benchmark for your Xeon yet.
`--dtype bfloat16` BEFORE the command reduces model memory, but may be much slower
on CPUs without native BF16 support. The default float32 is intended for your 64 GB host.

```bash
./venv/bin/python image_tool.py --threads 16 analyze /path/image.png --max-tokens 8192 --output results/large1
```

Invalid/truncated model JSON, irregular tables or dangling flowchart connections
produce an error rather than a seemingly successful workbook. The raw response is
preserved. Retry with a smaller crop or higher token limit and a NEW output folder.

## OpenCode integration

The installer writes `opencode.generated.json` with absolute paths. Merge its `mcp`
entry into your existing OpenCode configuration; do not replace unrelated settings.
Restart OpenCode and inspect the local MCP tools. The tool name is `analyze_image`.
No npm downloads are needed: the server speaks MCP over standard input/output.
The model is loaded lazily on the first extraction and retained for subsequent calls.
Only one request is processed at a time. A running extraction cannot be cancelled
through an MCP cancellation notification; stop the process if necessary.

Suggested request: "Use offline_image's analyze_image tool on /absolute/path/table.png,
write to /absolute/path/new-result-folder, and report uncertainties from result.json."
The tool returns paths and counts; OpenCode can read the saved JSON using its file tools.
Some OpenCode versions impose their own tool execution timeout. Long CPU runs may need
the standalone command instead. The configuration's timeout does not guarantee that
every client version will wait for a full inference run.

IMPORTANT: Your cloud-hosted cocoa-cli-pro agent cannot run on a completely disconnected
machine. This ZIP supplies a local image tool, not a replacement general agent model or
OpenAI-compatible inference server. MCP integration is ready, but agent-driven use still
requires configuring OpenCode with an available local agent model. Standalone use works
without OpenCode. No claim is made that the bundled vision model can replace cocoa-cli-pro.

## Quality and scope

The bundled Qwen3-VL-2B-Instruct vision-language model reads images directly; there is no
separate OCR engine. It supports image interpretation, but can omit, misread or invent
content. Structural validation does not establish visual accuracy. `--verify` performs
a second differently prompted reading with the SAME model; agreement is not independent
verification. Differences are flagged, not silently merged. Review all important data.

English printed text is the intended scope. Numerical chart coordinates inferred from
positions must be marked estimated; only printed values may be labeled. The model can
misclassify those too. Arbitrary graph/diagram types, tiny text, crossing arrows, complex
tables and unlabeled continuous curves are not guaranteed. No handwriting guarantee.
The software does not claim to outperform your existing subagent without representative
image testing. See VALIDATION.md for exactly what was tested on the build machine.

## Offline checks

```bash
./venv/bin/python image_tool.py doctor --load-model
./venv/bin/python -m unittest discover -s tests -v
./venv/bin/python tests/acceptance.py
```

The acceptance test runs the actual model against bundled synthetic table, flowchart and
chart images and checks selected known facts. It can take several minutes per image.
Run it on the target before using real work. Failure is a reason to inspect results,
not to treat extraction as reliable. Synthetic tests do not establish general accuracy.

## Licenses and provenance

Tool source: MIT (LICENSE). Model: Apache-2.0; see models/vision/LICENSE and README.md.
The exact model revision is recorded in model_provenance.json. Unmodified Python wheels
retain their upstream license notices. THIRD_PARTY.md lists bundled distributions.
SHA256SUMS.json protects against accidental damage, not a maliciously replaced manifest.

Official references:
- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://opencode.ai/docs/mcp-servers/
- https://download.pytorch.org/whl/cpu/

