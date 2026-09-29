# Low-memory profile third-party notices

The offline bundle includes unmodified wheels in `wheelhouse-lowmem/`. Their
package metadata and any bundled license files are retained inside the wheels.
The pinned distribution versions and SHA-256 hashes are in
`requirements.lowmem.lock` and `SHA256SUMS.lowmem.json`.

The low-memory profile uses RapidOCR 3.9.2 and the PP-OCRv6 detector and
recognizer models distributed inside its wheel. RapidOCR is Apache-2.0; the
license text and copyright notices are in `RapidOCR-LICENSE.txt`. ONNX Runtime,
OpenCV, NumPy, Pillow, OpenPyXL and the supporting packages are provided only as
binary wheels; consult each wheel's metadata/notices for its license terms.
