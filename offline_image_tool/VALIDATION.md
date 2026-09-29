# Validation scope

This is an offline-capable first version, not a guarantee of transcription accuracy.

Update 2026-09-29: XLSX exports no longer contain a Read me worksheet. Warnings and
provenance remain in JSON. The model weights have not changed. Tests now explicitly
assert that a table workbook contains only Table 1 and opens on that worksheet.

Additional real-image tests exposed unresolved accuracy failures: a blank student age
was invented as 5, and a sales screenshot's 4.99 unit cost was read as 1.99. The model
also filled in incorrect details in a clipped bottom row. Do not treat this tool as
reliable unattended transcription. These failures are not fixed by the worksheet change.
Another dense-table test returned only 20 of 23 visible data rows and misread product
identifiers and names. The source image and private test output are not published.

## Build environment and target

Target dependency resolution: CPython 3.13, standard cp313 ABI, Linux x86_64,
glibc 2.28. The installer requires exactly CPython 3.13.12. The target prerequisite
report confirmed the necessary standard-library modules and native runtimes.

Execution tests here used Windows x86_64, CPython 3.12.14, CPU PyTorch 2.9.1,
Transformers 4.57.6, and bfloat16 model weights. No Windows wheels or alternate
Python runtime are shipped. A full-precision model-load attempt was not completed
on this build machine; float32 on the 64 GB target has not been executed here.
No RHEL environment was available for an end-to-end installation test.

## Dependency and offline checks

- 35 Linux-compatible/pure-Python wheels are included with pinned versions and hashes.
- Dependency markers were audited for Linux x86_64 / CPython 3.13.12, separately
  from the Windows download resolver. No required dependency was missing.
- A hash-enforced pip resolution dry run succeeded with `--no-index` and only
  the bundled wheelhouse. No source builds are needed.
- Static symbol-string inspection of 12 binary wheels found no GLIBC version above
  2.28 or GLIBCXX version above 3.4.25. This is not a substitute for loading them on RHEL.
- The complete model and processor loaded from local files with offline mode enabled.
- Model weights SHA256 matched the upstream Hugging Face LFS object:
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.

## Software tests

The bundled unittest suite covers literal Excel cell values and formula prevention,
leading zeros, null cells, malformed/ragged tables, missing headings, flat heading
normalization, dangling/duplicate graph IDs, partial-output handling, existing-output
protection, disagreement reporting, source-byte provenance, focused-mode handling
and MCP initialization/discovery/error responses. See the bundled test log for count.

## Actual model image tests

Synthetic fixture PNGs are in tests/fixtures. The actual model was run on those images.
Format-handling fixes were subsequently validated by replaying the saved, unmodified
model responses through the final parser and XLSX exporter. This is not a new model
inference run after each parser fix. See validation_samples/checks.json and its raw
responses, images, JSON and workbooks for evidence. Replay elapsed_seconds is not
inference timing. CPU inference took minutes per image on the build machine.

- Table: all nine cells, including headings and exact decimal strings, matched.
- Flowchart: Start -> Check input -> Save result and the two directed edges matched.
- Chart: Jan = 10 and Feb = 20 were read correctly as labeled values. The model also
  invented an unrequested table and classified the month axis as linear. The final
  focused-mode parser excludes the table and flags the axis inconsistency, retaining
  the model's scale for review. This is a known semantic limitation, not a fully
  correct chart extraction. See validation_samples/checks.json for the value check.

The model initially returned flat headings and omitted irrelevant empty categories.
The final parser handles those representation differences without changing cell or
step text. In focused modes an omitted uncertainty report is explicitly flagged.
Automatic mode retains strict requirements for every content category.

These small synthetic checks do not measure accuracy on complex tables, merged cells,
branching/cyclic diagrams, crossing connectors, log plots, dense charts, or real user
images. All outputs therefore retain needs_review status. No comparison benchmark
against OpenCode's original imageAnalysis subagent has been performed.

## Checks to run on your disconnected RHEL machine

After installation, run the doctor, unittest and acceptance commands in README.md.
The acceptance test performs fresh inference. Verify representative real images
visually before relying on results. OpenCode itself still needs an available local
agent model; the cloud-hosted cocoa-cli-pro API is not made offline by this package.

