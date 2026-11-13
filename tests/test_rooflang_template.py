"""Differential checks for mutable batch bindings and isolated execution state."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rooflang.runtime.simulator import OOMError, Simulator
from rooflang_bridge.llama import LlamaTemplate, build


def batch(*pairs):
    return [{"q_len": q, "kv_len": kv, "lm_head_len": 1} for q, kv in pairs]


def signature(graph, placement):
    tensors = {}

    def ports(values):
        return tuple(
            (
                name,
                tensors.setdefault(t, len(tensors)),
                t.dtype,
                t.shape,
                t.weight_id,
                placement.get_tensor_memory(t).name,
            )
            for name, t in values.items()
        )

    nodes = list(graph._dag.nodes)
    indices = {k: i for i, k in enumerate(nodes)}
    return (
        [
            (
                type(k).__name__,
                k.name,
                k.flops,
                k.weight_read_fraction,
                getattr(k, "total_bytes", None),
                ports(k.inputs),
                ports(k.outputs),
                ports(k.weights),
                (
                    placement.get_kernel_device(k).device.name,
                    placement.get_kernel_device(k).stream,
                )
                if k in placement.placed_kernels
                else None,
            )
            for k in nodes
        ],
        [
            (indices[a], indices[b], attrs["mapping"])
            for a, b, attrs in graph._dag.edges(data=True)
        ],
    )


def trace(result):
    return sorted(
        (
            e.kernel.name,
            e.device.name,
            e.stream,
            e.start_us,
            e.end_us,
            e.compute_time_us,
            e.memory_time_us,
            e.network_time_us,
            e.bound.value,
        )
        for e in result.trace
    )


class TemplateTests(unittest.TestCase):
    def test_rebind_matches_fresh_graph_without_rebuilding_or_revalidation(self):
        batches = [
            batch((3, 0)),
            batch((2, 3)),
            batch((1, 5)),
            batch((4, 0), (1, 16)),
            batch((1, 0), (4, 16)),
            batch((1, 0)),
        ]
        for tp in (1, 2, 4, 8):
            hardware = {"gpu_model": "b300", "gpu_count": tp}
            template = LlamaTemplate(tp, hardware)
            identities = tuple(id(k) for k in template.graph._dag.nodes)
            tensor_ids = {
                id(t)
                for k in template.graph.kernels
                for ports in (k.inputs, k.outputs, k.weights)
                for t in ports.values()
            }
            for requests in batches:
                with self.subTest(tp=tp, batch=requests):
                    fresh, placement, hw, _ = build(requests, tp, hardware)
                    expected = Simulator(fresh, placement, hw).run()
                    with (
                        patch(
                            "rooflang_bridge.llama.build_hardware",
                            side_effect=AssertionError("rebuilt hardware"),
                        ),
                        patch.object(
                            template.graph,
                            "validate",
                            side_effect=AssertionError("revalidated graph"),
                        ),
                        patch.object(
                            template.placement,
                            "validate",
                            side_effect=AssertionError("revalidated placement"),
                        ),
                    ):
                        actual, _ = template.simulate(requests)
                    self.assertEqual(
                        signature(template.graph, template.placement),
                        signature(fresh, placement),
                    )
                    self.assertEqual(trace(actual), trace(expected))
                    self.assertEqual(actual.total_time_us, expected.total_time_us)
                    self.assertEqual(
                        {m.name: n for m, n in actual.peak_memory.items()},
                        {m.name: n for m, n in expected.peak_memory.items()},
                    )
                    self.assertEqual(
                        identities, tuple(id(k) for k in template.graph._dag.nodes)
                    )
                    self.assertEqual(
                        tensor_ids,
                        {
                            id(t)
                            for k in template.graph.kernels
                            for ports in (k.inputs, k.outputs, k.weights)
                            for t in ports.values()
                        },
                    )
                    template.graph.validate()
                    template.placement.validate(template.graph)

    def test_invalid_batch_and_oom_do_not_poison_next_iteration(self):
        template = LlamaTemplate(2, {"gpu_model": "b300", "gpu_count": 2})
        _, baseline = template.simulate(batch((1, 5)))
        before = signature(template.graph, template.placement)
        for invalid in (
            [],
            batch((0, 1)),
            batch((1, -1)),
            [{"q_len": 1, "kv_len": 0, "lm_head_len": 2}],
        ):
            with self.assertRaises(ValueError):
                template.simulate(invalid)
            self.assertEqual(before, signature(template.graph, template.placement))
        with self.assertRaises(OOMError):
            template.simulate(batch((1, 10**9)))
        _, after = template.simulate(batch((1, 5)))
        self.assertEqual(baseline, after)
        self.assertEqual(before, signature(template.graph, template.placement))
