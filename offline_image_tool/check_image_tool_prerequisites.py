"""Read-only prerequisite check. Uses only Python's standard library; no network."""
import importlib
import importlib.metadata
import json
import platform
import shutil
import sys
import sysconfig


def main():
    report = {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "architecture": platform.machine(),
        "platform": sys.platform,
        "libc": platform.libc_ver(),
        "python_abi": sysconfig.get_config_var("SOABI"),
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "free_disk_gib_in_current_directory": round(shutil.disk_usage(".").free / 2**30, 1),
        "standard_library": {},
        "installed_packages": {},
        "native_libraries": {},
    }
    for name in ("venv", "ensurepip", "ssl", "ctypes", "zlib", "bz2", "lzma", "sqlite3"):
        try:
            module = importlib.import_module(name)
            report["standard_library"][name] = "available"
            if name == "ensurepip":
                report["bundled_pip_version"] = module.version()
        except Exception as exc:
            report["standard_library"][name] = type(exc).__name__ + ": " + str(exc)
    for name in ("pip", "numpy", "Pillow", "openpyxl", "onnxruntime", "torch", "transformers"):
        try:
            report["installed_packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["installed_packages"][name] = "missing"
    if sys.platform == "linux":
        try:
            import ctypes
            for name in ("libstdc++.so.6", "libgcc_s.so.1", "libgomp.so.1"):
                try:
                    ctypes.CDLL(name)
                    report["native_libraries"][name] = "loadable"
                except OSError as exc:
                    report["native_libraries"][name] = str(exc)
        except ImportError:
            report["native_libraries"]["status"] = "ctypes unavailable"
    report["required_python_matches"] = (
        sys.version_info[:3] == (3, 13, 12) and platform.python_implementation() == "CPython"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

