"""Synthetic, upstream-data-free smoke tests for the reconstruction helpers."""

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load_script("scripts/pathqa_2_kgqa.py", "pathqa_converter")
wrapper = load_script("scripts/reconstruct.py", "pathqa_wrapper")


class ConverterTests(unittest.TestCase):
    def test_original_answer_with_parentheses(self):
        designated, answers = converter.parse_released_answer_field("PG_(USA)(PG_(USA)/)")
        self.assertEqual(designated, "PG_(USA)")
        self.assertEqual(answers, ["PG_(USA)"])

    def test_annotation_end_marker(self):
        path, end = converter.parse_annotated_path("A#r#B#s#C#<end>#C", 2)
        self.assertEqual(path, [["A", "r", "B"], ["B", "s", "C"]])
        self.assertEqual(end, "C")

    def test_enumeration_includes_backtracking(self):
        graph = pd.DataFrame([("A", "r", "B"), ("B", "s", "A")],
                             columns=["head", "relation", "tail"])
        adjacent = converter.build_relation_adjacency(graph)
        self.assertEqual(converter.follow_relation_chain("A", "r->s", adjacent), {"A"})
        self.assertEqual(len(converter.enumerate_relation_chain_paths("A", "r->s", adjacent)), 1)
        self.assertEqual(converter.follow_relation_chain_simple("A", "r->s", adjacent), set())

    def test_family_selection_is_repeatable(self):
        chosen = converter.choose_canonical_question(["Q one", "Q two"], "family_1", 42)
        self.assertEqual(chosen, converter.choose_canonical_question(["Q one", "Q two"], "family_1", 42))


class PreparationTests(unittest.TestCase):
    def test_extracts_expected_pql_member_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "external"
            source.mkdir()
            with zipfile.ZipFile(source / "FB2M_TRI_NEW.txt.zip", "w") as bundle:
                bundle.writestr("data/FB2M_TRI_NEW.txt", "A\tr\tB\n")
                bundle.writestr("../unexpected.txt", "do not extract")
            output = root / "generated"
            extracted = wrapper.prepare_pql_source(source, output)
            self.assertEqual(extracted.read_text(), "A\tr\tB\n")
            self.assertFalse((root / "unexpected.txt").exists())

    def test_archive_missing_expected_member_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.mkdir(exist_ok=True)
            with zipfile.ZipFile(root / "FB2M_TRI_NEW.txt.zip", "w") as bundle:
                bundle.writestr("other.txt", "data")
            with self.assertRaises(ValueError):
                wrapper.prepare_pql_source(root, root / "out")


if __name__ == "__main__":
    unittest.main()
