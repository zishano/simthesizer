"""Cached bridge plans must preserve full reference lowering, not cached times."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rooflang_bridge.execution import export_execution
from rooflang_bridge.llama import LlamaTemplate


def batch(*pairs):
    return [{"q_len": q, "kv_len": kv, "lm_head_len": 1} for q, kv in pairs]


class ExecutionPlanTests(unittest.TestCase):
    def test_repricing_and_workspace_match_uncached_reference(self):
        batches = [
            batch((1, 0)),
            batch((1, 5)),
            batch((1, 8192)),
            batch((5, 0), (3, 8)),
            batch((2, 8192), (6, 32)),
            batch((128, 0)),
            batch((128, 4096)),
        ]
        for gpu in ("rtx6000d", "h200", "b300"):
            for tp in (1, 2, 4, 8):
                template = LlamaTemplate(tp, {"gpu_model": gpu, "gpu_count": tp})
                for requests in batches:
                    with self.subTest(gpu=gpu, tp=tp, batch=requests):
                        actual = export_execution(template, requests)
                        expected = export_execution(template, requests, reference=True)
                        self.assertEqual(actual["operations"], expected["operations"])
                        self.assertEqual(actual["workspace"], expected["workspace"])

    def test_decode_cache_does_not_run_des_or_export_diagnostics(self):
        template = LlamaTemplate(2, {"gpu_model": "h200", "gpu_count": 2})
        first = export_execution(template, batch((1, 5)), details=False)
        with patch.object(
            template, "simulate", side_effect=AssertionError("repeated DES")
        ):
            later = export_execution(template, batch((1, 500)), details=False)
        self.assertIsNone(later["reference"])
        self.assertTrue(all(not op["work"]["details"] for op in later["operations"]))
        self.assertEqual(first["workspace"], later["workspace"])
        self.assertNotEqual(first["operations"], later["operations"])
        with self.assertRaisesRegex(ValueError, "OOM"):
            export_execution(template, batch((1, 10**9)))
