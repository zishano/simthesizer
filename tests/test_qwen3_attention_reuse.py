"""Repeated attention work retains each layer's causal and KV identity."""

import json
import unittest

from test_qwen3_compact_work import HW, QwenTemplate, expand
from test_qwen3_operation_rows import expand_rows


class AttentionReuseTests(unittest.TestCase):
    def test_changing_rounds_preserve_costs_dependencies_and_kv_bindings(self):
        rounds = (
            ([{"q_len": 1, "kv_len": 17, "lm_head_len": 1}], [3]),
            (
                [
                    {"q_len": 7, "kv_len": 0, "lm_head_len": 1},
                    {"q_len": 1, "kv_len": 1024, "lm_head_len": 1},
                ],
                [4, 9],
            ),
            (
                [{"q_len": 1, "kv_len": 127 + i, "lm_head_len": 1} for i in range(8)],
                list(range(8)),
            ),
        )
        for ep in (1, 2, 4, 8):
            for tp in (value for value in (1, 2, 4) if ep % value == 0):
                template = QwenTemplate(tp, ep, HW)
                for batch, identities in rounds:
                    with self.subTest(ep=ep, tp=tp, identities=identities):
                        expected = template.export(batch, identities, details=False)
                        for flag, expand_plan in (
                            ("compact_work", expand),
                            ("operation_rows", expand_rows),
                        ):
                            actual = expand_plan(
                                template.export(
                                    batch, identities, details=False, **{flag: True}
                                )
                            )
                            self.assertEqual(json.dumps(actual), json.dumps(expected))
                        attention = [
                            node
                            for node in expected["operations"]
                            if node["layer"] is not None
                        ]
                        self.assertEqual(
                            {node["layer"] for node in attention}, set(range(94))
                        )
                        expected_bindings = {
                            (rank % tp, tuple(indices))
                            for rank in range(ep)
                            if (
                                indices := [
                                    i
                                    for i, identity in enumerate(identities)
                                    if identity % (ep // tp) == rank // tp
                                ]
                            )
                        }
                        for layer in range(94):
                            self.assertEqual(
                                {
                                    (node["kv_rank"], tuple(node["batch_indices"]))
                                    for node in attention
                                    if node["layer"] == layer
                                },
                                expected_bindings,
                            )


if __name__ == "__main__":
    unittest.main()
