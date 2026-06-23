from pathlib import Path
import tempfile
import unittest

from alpha_research.manifests import Manifest, iter_manifests, read_manifest, write_manifest


class ManifestTests(unittest.TestCase):
    def test_write_and_read_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = Manifest.create(
                kind="dataset",
                id="normalized.trades.test",
                path=tmp_path / "dataset",
                payload={"row_count": 3},
            )
            path = tmp_path / "dataset.manifest.json"

            write_manifest(manifest, path)
            data = read_manifest(path)

            self.assertEqual(data["kind"], "dataset")
            self.assertEqual(data["id"], "normalized.trades.test")
            self.assertEqual(data["payload"]["row_count"], 3)
            self.assertIn("fingerprint", data)

    def test_iter_manifests_filters_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            write_manifest(
                Manifest.create(kind="source", id="raw.test", path=tmp_path / "raw"),
                tmp_path / "raw.manifest.json",
            )
            write_manifest(
                Manifest.create(kind="run", id="run.test", path=tmp_path / "run"),
                tmp_path / "run" / "manifest.json",
            )

            self.assertEqual(len(iter_manifests(tmp_path)), 2)
            self.assertEqual(len(iter_manifests(tmp_path, kind="run")), 1)
