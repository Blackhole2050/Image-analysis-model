from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_parts

class ReleaseTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = bytes(range(256)) * 11
            (root / "input.zip").write_bytes(data)
            manifest = release_parts.split(root / "input.zip", root / "parts", 512)
            self.assertEqual(len(manifest["parts"]), 6)
            release_parts.join(root / "parts/parts.json", root / "output.zip")
            self.assertEqual((root / "output.zip").read_bytes(), data)

    def test_corrupt_part_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input.zip").write_bytes(b"original")
            manifest = release_parts.split(root / "input.zip", root / "parts", 512)
            (root / "parts" / manifest["parts"][0]["name"]).write_bytes(b"corrupt!")
            with self.assertRaises(ValueError):
                release_parts.join(root / "parts/parts.json", root / "output.zip")
            self.assertFalse((root / "output.zip").exists())

if __name__ == "__main__":
    unittest.main()

