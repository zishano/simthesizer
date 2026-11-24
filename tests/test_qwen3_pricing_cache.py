"""Cached Qwen demands retain Rooflang's exact per-port pricing."""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rooflang.language.tensor import Tensor
from rooflang_bridge.llama import Work
from rooflang_bridge.qwen3 import QwenTemplate

HW = {
    "tflops_fp16": 100,
    "memory_bandwidth_gbs": 1000,
    "memory_capacity_gb": 1000,
    "link_bandwidth_gbs": 100,
    "link_latency_us": 1,
}


class QwenPricingCacheTests(unittest.TestCase):
    def test_cached_shapes_preserve_independent_rooflang_prices(self):
        template = QwenTemplate(2, 8, HW)
        for q, kv in ((1, 0), (3, 17), (1, 1024), (3, 17)):
            with self.subTest(q=q, kv=kv):
                plan = template.export(
                    [{"q_len": q, "kv_len": kv, "lm_head_len": 1}], [3]
                )
                for operation in plan["operations"]:
                    detail = operation["work"]["details"]
                    if detail["stream"] != 0:
                        continue
                    # One complete layer plus embedding/final operations covers
                    # repeated expert shapes, KV growth and partial weight reads.
                    if detail["name"].startswith("layer") and not detail[
                        "name"
                    ].startswith("layer0."):
                        continue
                    tensors = lambda name, detail=detail: {
                        key: Tensor("fp16", shape)
                        for key, shape in detail.get(name, {}).items()
                    }
                    kernel = Work(
                        detail["name"],
                        detail["flops"],
                        tensors("inputs"),
                        tensors("outputs"),
                        tensors("weights"),
                    )
                    if detail["name"] == "embedding":
                        kernel.weight_read_fraction = Fraction(q, 151936)
                    gpu = next(
                        g for g in template.gpus if g.name == detail["devices"][0]
                    )
                    ct, mt, *_ = template.pricing._base_times(kernel, gpu, [])
                    self.assertEqual(
                        (detail["compute_us"], detail["memory_us"]), (ct, mt)
                    )
                    self.assertEqual(detail["input_bytes"], kernel.input_bytes)
                    self.assertEqual(detail["weight_bytes"], kernel.loaded_weight_bytes)
                    self.assertEqual(detail["output_bytes"], kernel.output_bytes)
        self.assertGreater(template.compute_cost.cache_info().hits, 1000)

    def test_reply_mutation_cannot_change_future_demands(self):
        template = QwenTemplate(2, 8, HW)
        batch = [{"q_len": 1, "kv_len": 17, "lm_head_len": 1}]
        before = template.export(batch, [0], details=False)
        damaged = template.export(batch, [0], details=False)
        for operation in damaged["operations"]:
            for demand in operation["work"]["demands"]:
                demand["resources"].clear()
            operation["parents"].clear()
        self.assertEqual(template.export(batch, [0], details=False), before)


if __name__ == "__main__":
    unittest.main()
