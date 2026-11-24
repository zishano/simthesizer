"""Optimized expert selection retains the original seeded random.sample contract."""

import hashlib
import itertools
import random
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rooflang_bridge import qwen3


def original_experts(request, position, layer, seed, experts=128, top=8):
    digest = hashlib.blake2b(
        struct.pack("<QQQQ", seed, request, position, layer), digest_size=16
    ).digest()
    return random.Random(int.from_bytes(digest, "little")).sample(range(experts), top)


class ExactRoutingTests(unittest.TestCase):
    def test_selection_order_matches_original_across_u64_boundaries_and_layers(self):
        boundaries = (0, 1, 2**32 - 1, 2**32, 2**63, 2**64 - 1)
        identities = list(itertools.product(boundaries, repeat=4))
        rng = random.Random(20261123)
        identities.extend(
            (rng.getrandbits(64), rng.getrandbits(64), layer, rng.getrandbits(64))
            for layer in range(94)
            for _ in range(64)
        )
        for identity in identities:
            self.assertEqual(qwen3.token_experts(*identity), original_experts(*identity))

    def test_rejections_keep_bit_width_draw_count_and_selection_order(self):
        class ScriptedRandom:
            def __init__(self, seed):
                self.draws = iter((255, 128, 9, 9, 200, 3, 127, 0, 1, 2, 4, 5))
                self.widths = []

            def getrandbits(self, width):
                self.widths.append(width)
                return next(self.draws)

        scripted = ScriptedRandom(0)
        with patch.object(qwen3.random, "Random", return_value=scripted):
            self.assertEqual(qwen3.token_experts(3, 15, 93, 19), [9, 3, 127, 0, 1, 2, 4, 5])
        self.assertEqual(scripted.widths, [8] * 12)

    def test_other_expert_dimensions_retain_generic_sampler(self):
        for experts, top in ((8, 8), (64, 8), (128, 4), (256, 16)):
            with (
                self.subTest(experts=experts, top=top),
                patch.object(qwen3, "E", experts),
                patch.object(qwen3, "TOP", top),
            ):
                self.assertEqual(
                    qwen3.token_experts(17, 1025, 93, 19),
                    original_experts(17, 1025, 93, 19, experts, top),
                )

    def test_invalid_u64_identity_still_fails(self):
        for field in range(4):
            for invalid in (-1, 2**64):
                identity = [0] * 4
                identity[field] = invalid
                with self.assertRaises(struct.error):
                    qwen3.token_experts(*identity)


if __name__ == "__main__":
    unittest.main()
