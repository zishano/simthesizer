"""Fixed Qwen3-235B-A22B shared EP graph; seeded routing is an analytical workload.

One synchronized group partitions attention by request ID. Expert assignments
are uniform without replacement, independent of batch order/chunk boundaries.
Compute/HBM use Rooflang pricing; native resources execute the returned DAG.
"""

import hashlib
import random
import struct
import tomllib
from fractions import Fraction
from functools import lru_cache, partial
from pathlib import Path

from rooflang.language.graph import ComputeGraph
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import Simulator

from rooflang_bridge.execution import fabric_resource
from rooflang_bridge.hardware import build_hardware
from rooflang_bridge.llama import Work, batch_shape

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "crates/simthesizer-core/src/models/qwen3_235b_a22b.toml"
)
MODEL = tomllib.loads(CONFIG_PATH.read_text())
H = MODEL["hidden_size"]
Q = MODEL["num_attention_heads"]
K = MODEL["num_key_value_heads"]
D = MODEL["head_dim"]
E = MODEL["num_experts"]
TOP = MODEL["num_experts_per_tok"]
M = MODEL["moe_intermediate_size"]
V = MODEL["vocab_size"]
L = MODEL["num_hidden_layers"]


def validate_parallel(tp, ep, pp=1, expert_tp=1):
    if type(tp) is not int or tp not in (1, 2, 4) or type(pp) is not int or pp != 1:
        raise ValueError("Qwen shared EP requires attention TP=1, 2 or 4 and PP=1")
    if type(ep) is not int or ep not in (1, 2, 4, 8):
        raise ValueError("Qwen shared EP requires EP=1, 2, 4 or 8")
    if ep % tp:
        raise ValueError("Qwen EP must be divisible by attention TP")
    if type(expert_tp) is not int or expert_tp != 1:
        raise ValueError("Qwen shared EP requires expert TP=1")
    if (
        MODEL["decoder_sparse_step"] != 1
        or MODEL["mlp_only_layers"]
        or MODEL["shared_expert_intermediate_size"]
    ):
        raise ValueError("Qwen graph requires the fixed all-sparse checkpoint")


def describe(parallel):
    tp, ep = parallel["tp"], parallel["ep"]
    validate_parallel(tp, ep, parallel["pp"], parallel.get("expert_tp", 1))
    block = [
        {"LayerNorm": {"name": "layernorm", "hidden_size": H}},
        {"Gemm": {"name": "qkv_projection", "m": H, "n": (Q + 2 * K) * D // tp}},
        {"LayerNorm": {"name": "qk_norm", "hidden_size": D}},
        {"Rope": {"name": "rope", "head_dim": D, "num_heads": Q // tp}},
        {
            "Attention": {
                "name": "attention",
                "num_q_heads": Q // tp,
                "num_kv_heads": K // tp,
                "head_dim": D,
            }
        },
        {"Gemm": {"name": "o_projection", "m": Q * D // tp, "n": H}},
        {"LayerNorm": {"name": "layernorm", "hidden_size": H}},
        {
            "Moe": {
                "name": "moe",
                "num_experts": E,
                "num_experts_per_token": TOP,
                "num_shared_experts_per_token": 0,
                "dispatch_bytes_per_token": 2 * (H + E),
                "combine_bytes_per_token": 2 * H,
            }
        },
    ]
    if tp > 1:
        block.insert(6, {"AllReduce": {"name": "all_reduce", "bytes_per_token": 2 * H}})
    attention = (H * (Q + 2 * K) * D + Q * D * H) // tp + 2 * D + 2 * H
    return {
        "version": 1,
        "pre": [
            {
                "Embedding": {
                    "name": "embedding",
                    "num_embeddings": V,
                    "embedding_dim": H,
                }
            }
        ],
        "block": block,
        "post": [
            {"LayerNorm": {"name": "final_layernorm", "hidden_size": H}},
            {"Gemm": {"name": "lm_head", "m": H, "n": V // tp}},
            {"Sampler": {"name": "sampler", "vocab_size": V}},
        ],
        "layers": L,
        "weight_bytes_per_device": 2
        * (V * H + V * H // tp + H + L * (attention + H * E + 3 * H * M * (E // ep))),
        "kv_bytes_per_token_per_device": 4 * (K // tp) * D * L,
    }


def validate_model(model, memory, tp, ep, expert_tp=1):
    expected = describe({"tp": tp, "pp": 1, "ep": ep, "expert_tp": expert_tp})
    if model != {key: expected[key] for key in ("pre", "block", "post", "layers")}:
        raise ValueError(
            "Qwen shared graph requires the exact qwen3_235b_a22b layer plan"
        )
    if memory != {
        key: expected[key]
        for key in ("weight_bytes_per_device", "kv_bytes_per_token_per_device")
    }:
        raise ValueError("Qwen model memory metadata differs from the fixed graph")


def token_experts(request_id, position, layer, seed):
    """Uniform top-k workload, keyed by causal token identity rather than batching."""
    digest = hashlib.blake2b(
        struct.pack("<QQQQ", seed, request_id, position, layer), digest_size=16
    ).digest()
    rng = random.Random(int.from_bytes(digest, "little"))
    if (E, TOP) != (128, 8):
        return rng.sample(range(E), TOP)
    # random.sample(range(128), 8) uses rejection sampling with eight-bit
    # draws. Preserve its draws and selection order without generic dispatch.
    getbits = rng.getrandbits
    selected = set()
    experts = []
    for _ in range(TOP):
        expert = getbits(8)
        while expert >= E or expert in selected:
            expert = getbits(8)
        selected.add(expert)
        experts.append(expert)
    return experts


def route(batch, requests, ep, layer, seed, explicit=None, *, tp=1):
    expert_tokens = [0] * E
    expert_sources = [[0] * ep for _ in range(E)]
    rank_tokens = [[0] * ep for _ in range(ep)]
    assignments = [[0] * ep for _ in range(ep)]
    experts_per_rank = E // ep
    token = 0
    for request, row in zip(requests, batch):
        source = (request % (ep // tp)) * tp
        source_assignments = assignments[source]
        source_tokens = rank_tokens[source]
        for position in range(row["kv_len"], row["kv_len"] + row["q_len"]):
            experts = (
                explicit[token]
                if explicit is not None
                else token_experts(request, position, layer, seed)
            )
            destinations = set()
            for expert in experts:
                owner = expert // experts_per_rank
                expert_tokens[expert] += 1
                expert_sources[expert][source] += 1
                source_assignments[owner] += 1
                destinations.add(owner)
            for owner in destinations:
                source_tokens[owner] += 1
            token += 1
    return {
        "expert_tokens": expert_tokens,
        "expert_sources": expert_sources,
        "rank_tokens": rank_tokens,
        "assignments": assignments,
    }


class QwenTemplate:
    """Fixed hardware with bounded pricing caches; routing and DAG remain per round."""

    def __init__(
        self, tp, ep, hardware, routing_seed=0, *, expert_tp=1, attention_groups=None
    ):
        validate_parallel(tp, ep, expert_tp=expert_tp)
        if type(routing_seed) is not int or not 0 <= routing_seed < 2**64:
            raise ValueError("routing_seed must be a u64 integer")
        self.tp, self.ep, self.seed = tp, ep, routing_seed
        groups = [list(range(start, start + tp)) for start in range(0, ep, tp)]
        if attention_groups is not None and (
            attention_groups != groups
            or any(
                type(rank) is not int for group in attention_groups for rank in group
            )
        ):
            raise ValueError(
                "Qwen attention_groups must be complete contiguous TP groups"
            )
        self.attention_groups = groups
        self.memory = describe({"tp": tp, "ep": ep, "pp": 1})
        self.hw, self.gpus, self.memories, self.hardware = build_hardware(
            hardware, ep, collective=False
        )
        graph = ComputeGraph()
        self.pricing = Simulator(
            graph, Placement(hardware=self.hw, graph=graph), self.hw
        )
        # Prices depend on exact shapes and physical devices, never token identity
        # or layer number. Cache only immutable results, not graph/routing state.
        self.compute_cost = lru_cache(maxsize=8192)(self._compute_cost)
        self.copy_cost = lru_cache(maxsize=8192)(self._copy_cost)
        self.compute_resources = [f"compute:{gpu.name}" for gpu in self.gpus]
        self.memory_resources = [f"hbm:{gpu.name}" for gpu in self.gpus]
        self.streams = [f"stream:{gpu.name}:0" for gpu in self.gpus]
        for memory in self.memories:
            if self.memory["weight_bytes_per_device"] > memory.capacity_gb * 1e9:
                raise ValueError(
                    f"OOM on {memory.name}: Qwen resident weights exceed capacity"
                )

    def _compute_cost(self, rank, flops, inputs, outputs, weights, read_fraction):
        tensors = lambda shapes: {key: Tensor("fp16", shape) for key, shape in shapes}
        kernel = Work(
            "pricing", flops, tensors(inputs), tensors(outputs), tensors(weights)
        )
        kernel.weight_read_fraction = read_fraction
        ct, mt, _, _, _ = self.pricing._base_times(kernel, self.gpus[rank], [])
        return (
            ct,
            mt,
            kernel.input_bytes,
            kernel.loaded_weight_bytes,
            kernel.output_bytes,
        )

    def _copy_cost(self, source, destination, size):
        src, dst = self.gpus[source], self.gpus[destination]
        tensor = Tensor("fp16", (size // 2,))
        read = Work("copy_read", 0, {"x": tensor}, {})
        write = Work("copy_write", 0, {}, {"y": tensor})
        read_us = self.pricing._base_times(read, src, [])[1]
        write_us = self.pricing._base_times(write, dst, [])[1]
        resources = {self.memory_resources[source]: read_us * 1e-6}
        resources[self.memory_resources[destination]] = (
            resources.get(self.memory_resources[destination], 0) + write_us * 1e-6
        )
        latency = 0
        network_seconds = 0
        for edge, direction in self.hw.find_fabric_path_directed(src, dst):
            bandwidth = (
                edge.src_to_dst_bandwidth_gbs
                if direction == "fwd"
                else edge.dst_to_src_bandwidth_gbs
            )
            if bandwidth <= 0:
                raise ValueError("EP route has zero bandwidth")
            service = size / (bandwidth * 1e9)
            resources[fabric_resource(edge, direction)] = service
            network_seconds = max(network_seconds, service)
            latency += edge.alpha_us * 1e-6
        return (
            tuple(resources.items()),
            latency,
            read_us + write_us if src is dst else max(read_us, write_us),
            (latency + network_seconds) * 1e6,
        )

    def export(
        self,
        batch,
        requests,
        *,
        details=True,
        routing=None,
        wall_profile=None,
        compact_work=False,
        operation_rows=False,
    ):
        shape = batch_shape(batch)
        if (
            not isinstance(requests, list)
            or len(requests) != len(batch)
            or any(
                type(value) is not int or not 0 <= value < 2**64 for value in requests
            )
            or len(set(requests)) != len(requests)
        ):
            raise ValueError("requests must contain one distinct u64 ID per batch row")
        if routing is not None and (
            not isinstance(routing, list)
            or len(routing) != shape["tokens"]
            or any(
                not isinstance(row, list)
                or len(row) != TOP
                or any(type(expert) is not int or not 0 <= expert < E for expert in row)
                or len(set(row)) != TOP
                for row in routing
            )
        ):
            raise ValueError(
                "diagnostic routing requires eight distinct expert IDs per token"
            )
        indices = [
            [
                i
                for i, request in enumerate(requests)
                if request % len(self.attention_groups) == rank // self.tp
            ]
            for rank in range(self.ep)
        ]
        tokens = [sum(batch[i]["q_len"] for i in rows) for rows in indices]
        past = [sum(batch[i]["kv_len"] for i in rows) for rows in indices]
        pairs = [
            sum(
                batch[i]["q_len"] * batch[i]["kv_len"]
                + batch[i]["q_len"] * (batch[i]["q_len"] + 1) // 2
                for i in rows
            )
            for rows in indices
        ]
        graph = _Round(
            self,
            details,
            wall_profile,
            compact_work=compact_work,
            operation_rows=operation_rows,
        )
        previous = []
        for group in self.attention_groups:
            rank = group[0]
            count = tokens[rank]
            if count:
                embedded = graph.compute(
                    rank,
                    "embedding",
                    0,
                    {},
                    {"y": (count, H)},
                    {"table": (V, H)},
                    read_fraction=Fraction(count, V),
                )
                previous.extend(
                    graph.broadcast(
                        group, "embedding_broadcast", count * H * 2, embedded, count
                    )
                )
        ready = graph.barrier("embedding_complete", previous)
        q, k = Q // self.tp, K // self.tp
        workspace = [
            2
            * count
            * (
                (3 if self.tp == 1 else 6) * H
                + 2 * (q + 2 * k) * D
                + q * D
                + 2 * k * D
                + ((self.tp - 1) * H if self.tp > 1 and rank % self.tp == 0 else 0)
            )
            for rank, count in enumerate(tokens)
        ]
        routing_summary = []
        for layer in range(L):
            if wall_profile is None:
                routes = route(
                    batch, requests, self.ep, layer, self.seed, routing, tp=self.tp
                )
            else:
                routes = wall_profile.call(
                    "expert_routing",
                    route,
                    batch,
                    requests,
                    self.ep,
                    layer,
                    self.seed,
                    routing,
                    tp=self.tp,
                )
            counts, matrix, assignments = (
                routes["expert_tokens"],
                routes["rank_tokens"],
                routes["assignments"],
            )
            prepared = []
            for group in self.attention_groups:
                leader = group[0]
                count = tokens[leader]
                if not count:
                    continue
                projections = [
                    self._attention(
                        graph,
                        rank,
                        layer,
                        count,
                        past[rank],
                        pairs[rank],
                        indices[rank],
                        ready,
                    )
                    for rank in group
                ]
                reduced = graph.all_reduce(
                    group, f"layer{layer}.tp{leader}", count, projections
                )
                for rank, op in zip(group, reduced):
                    prefix = f"layer{layer}.rank{rank}."
                    op = graph.compute(
                        rank,
                        prefix + "attention_residual",
                        count * H,
                        {"x": (count, H), "residual": (count, H)},
                        {"y": (count, H)},
                        parents=[op],
                    )
                    op = graph.compute(
                        rank,
                        prefix + "post_norm",
                        4 * count * H,
                        {"x": (count, H)},
                        {"y": (count, H)},
                        {"gamma": (H,)},
                        [op],
                    )
                    if rank == leader:
                        op = graph.compute(
                            rank,
                            prefix + "router",
                            2 * count * H * E,
                            {"x": (count, H)},
                            {"logits": (count, E)},
                            {"w": (H, E)},
                            [op],
                        )
                        # Softmax=5 ops/entry; top-k=log2(k) comparisons/entry.
                        # Selected weights use (k-1) additions and k divisions.
                        op = graph.compute(
                            rank,
                            prefix + "topk",
                            count * (E * (5 + (TOP - 1).bit_length()) + 2 * TOP - 1),
                            {"logits": (count, E)},
                            {"scores": (count, E)},
                            parents=[op],
                        )
                    prepared.append(op)
            ready = graph.barrier(f"layer{layer}.router_complete", prepared)
            dispatch = [
                graph.copy(
                    source,
                    owner,
                    f"layer{layer}.dispatch.{source}-{owner}",
                    count * (H + E) * 2,
                    [ready],
                    count,
                )
                for source, row in enumerate(matrix)
                for owner, count in enumerate(row)
                if count
            ]
            dispatched = graph.barrier(f"layer{layer}.dispatch_complete", dispatch)
            experts = [[] for _ in range(self.ep)]
            for expert, count in enumerate(counts):
                if not count:
                    continue
                rank = expert // (E // self.ep)
                if graph.compact:
                    experts[rank].append(graph.expert(rank, count, dispatched))
                    continue
                prefix = f"layer{layer}.expert{expert}."
                extra = {
                    "expert": expert,
                    "expert_tokens": count,
                    "source_rank_tokens": routes["expert_sources"][expert],
                }
                op = graph.compute(
                    rank,
                    prefix + "gate_up",
                    4 * count * H * M,
                    {"x": (count, H)},
                    {"y": (count, 2 * M)},
                    {"w": (H, 2 * M)},
                    [dispatched],
                    extra=extra,
                )
                op = graph.compute(
                    rank,
                    prefix + "silu_mul",
                    6 * count * M,
                    {"x": (count, 2 * M)},
                    {"y": (count, M)},
                    parents=[op],
                    extra=extra,
                )
                op = graph.compute(
                    rank,
                    prefix + "down",
                    2 * count * M * H,
                    {"x": (count, M)},
                    {"y": (count, H)},
                    {"w": (M, H)},
                    [op],
                    extra=extra,
                )
                experts[rank].append(op)
            combined = []
            for owner in range(self.ep):
                for source in range(self.ep):
                    unique, selected = matrix[source][owner], assignments[source][owner]
                    if not unique:
                        continue
                    # Weight each expert output once, then sum on its owner before DMA.
                    op = graph.compute(
                        owner,
                        f"layer{layer}.expert_reduce.{owner}-{source}",
                        (2 * selected - unique) * H,
                        {"experts": (selected, H), "scores": (selected,)},
                        {"partial": (unique, H)},
                        parents=experts[owner],
                    )
                    combined.append(
                        graph.copy(
                            owner,
                            source,
                            f"layer{layer}.combine.{owner}-{source}",
                            unique * H * 2,
                            [op],
                            unique,
                        )
                    )
            ready = graph.barrier(f"layer{layer}.combine_complete", combined)
            residuals = []
            for rank, count in enumerate(tokens):
                if count and rank % self.tp == 0:
                    received = sum(matrix[rank])
                    op = graph.compute(
                        rank,
                        f"layer{layer}.rank{rank}.combine_accumulate",
                        (received - count) * H,
                        {"partials": (received, H)},
                        {"y": (count, H)},
                        parents=[ready],
                    )
                    op = graph.compute(
                        rank,
                        f"layer{layer}.rank{rank}.moe_residual",
                        count * H,
                        {"x": (count, H), "residual": (count, H)},
                        {"y": (count, H)},
                        parents=[op],
                    )
                    residuals.extend(
                        graph.broadcast(
                            self.attention_groups[rank // self.tp],
                            f"layer{layer}.moe_broadcast",
                            count * H * 2,
                            op,
                            count,
                        )
                    )
                incoming = sum(row[rank] for row in matrix)
                local_counts = counts[
                    rank * (E // self.ep) : (rank + 1) * (E // self.ep)
                ]
                # Conservative arena: residual/router, incoming packets, expert outputs,
                # largest gate/up+activation scratch, owner partials and returned partials.
                # Reused across synchronized layers; resident weights/KV are excluded.
                arena = 2 * (
                    count * (2 * H + E)
                    + incoming * (2 * H + E)
                    + sum(local_counts) * (H + 1)
                    + 3 * max(local_counts) * M
                    + sum(matrix[rank]) * H
                )
                workspace[rank] = max(workspace[rank], arena)
            ready = graph.barrier(f"layer{layer}.complete", residuals + [ready])
            if details:
                routing_summary.append(
                    {
                        "layer": layer,
                        "expert_tokens": counts,
                        "rank_tokens": matrix,
                        "dispatch_network_bytes": sum(
                            matrix[s][d]
                            for s in range(self.ep)
                            for d in range(self.ep)
                            if s != d
                        )
                        * (H + E)
                        * 2,
                        "combine_network_bytes": sum(
                            matrix[s][d]
                            for s in range(self.ep)
                            for d in range(self.ep)
                            if s != d
                        )
                        * H
                        * 2,
                    }
                )
        final = []
        for group in self.attention_groups:
            leader = group[0]
            count = tokens[leader]
            if count:
                heads = len(indices[leader])
                logits = []
                for rank in group:
                    op = graph.compute(
                        rank,
                        f"rank{rank}.final_norm",
                        4 * count * H,
                        {"x": (count, H)},
                        {"y": (count, H)},
                        {"gamma": (H,)},
                        [ready],
                    )
                    op = graph.compute(
                        rank,
                        f"rank{rank}.select_last_tokens",
                        0,
                        {"x": (count, H)},
                        {"y": (heads, H)},
                        parents=[op],
                    )
                    logits.append(
                        graph.compute(
                            rank,
                            f"rank{rank}.lm_head",
                            2 * heads * H * (V // self.tp),
                            {"x": (heads, H)},
                            {"y": (heads, V // self.tp)},
                            {"w": (H, V // self.tp)},
                            [op],
                        )
                    )
                    # Leader retains gathered full logits; peers hold their shard.
                    logit_width = V if rank == leader else V // self.tp
                    workspace[rank] = max(
                        workspace[rank],
                        2 * (2 * count * H + heads * (H + logit_width + 1)),
                    )
                gathered = graph.gather(
                    group, "logits_gather", heads * (V // self.tp) * 2, logits, heads
                )
                final.append(
                    graph.compute(
                        leader,
                        f"rank{leader}.sampler",
                        5 * heads * V,
                        {"x": (heads, V)},
                        {"y": (heads,)},
                        parents=gathered,
                    )
                )
        graph.barrier("iteration_complete", final)
        for rank, capacity in enumerate(workspace):
            resident = (
                self.memory["weight_bytes_per_device"]
                + past[rank] * self.memory["kv_bytes_per_token_per_device"]
            )
            if resident + capacity > self.memories[rank].capacity_gb * 1e9:
                raise ValueError(
                    f"OOM on {self.memories[rank].name}: Qwen resident KV and workspace exceed capacity"
                )
        return {
            "operation_rows" if graph.rows else "operations": graph.operations,
            **({"work_templates": graph.work_templates} if graph.compact else {}),
            "workspace": {
                gpu.name: capacity for gpu, capacity in zip(self.gpus, workspace)
            },
            "reference": None,
            "summary": {
                "model_kind": "qwen3_235b_a22b",
                "layers": L,
                "ep": self.ep,
                **(
                    {
                        "tp": self.tp,
                        "expert_tp": 1,
                        "attention_groups": self.attention_groups,
                    }
                    if self.tp > 1
                    else {}
                ),
                "tokens_per_rank": tokens,
                "routing": "explicit_diagnostic"
                if routing is not None
                else "seeded_uniform_top8",
                "routing_seed": self.seed,
                "routing_layers": routing_summary,
            }
            if details
            else {},
        }

    def _attention(self, graph, rank, layer, count, past, pairs, indices, ready):
        if graph.compact and rank in graph.attention_work:
            return graph.repeat_attention(rank, layer, indices, ready)
        first = len(graph.operations)
        q, k = Q // self.tp, K // self.tp
        width = (q + 2 * k) * D
        prefix = f"layer{layer}.rank{rank}."
        op = graph.compute(
            rank,
            prefix + "input_norm",
            4 * count * H,
            {"x": (count, H)},
            {"y": (count, H)},
            {"gamma": (H,)},
            [ready],
        )
        op = graph.compute(
            rank,
            prefix + "qkv",
            2 * count * H * width,
            {"x": (count, H)},
            {"y": (count, width)},
            {"w": (H, width)},
            [op],
        )
        op = graph.compute(
            rank,
            prefix + "qk_norm",
            4 * count * (q + k) * D,
            {"x": (count, width)},
            {"y": (count, width)},
            {"gamma": (2, D)},
            [op],
        )
        op = graph.compute(
            rank,
            prefix + "rope",
            3 * count * (q + k) * D,
            {"x": (count, width)},
            {"y": (count, width)},
            parents=[op],
        )
        op = graph.compute(
            rank,
            prefix + "attention",
            (4 * D + 5) * q * pairs,
            {"qkv": (count, width)},
            {"y": (count, q * D), "kv_append": (count, 2 * k * D)},
            {"kv_prefix": (past, 2 * k * D)},
            [op],
            attention=(layer, indices, rank % self.tp),
        )
        op = graph.compute(
            rank,
            prefix + "o_projection",
            2 * count * q * D * H,
            {"x": (count, q * D)},
            {"y": (count, H)},
            {"w": (q * D, H)},
            [op],
        )
        if graph.compact:
            graph.attention_work[rank] = tuple(
                (node[1], node[2] is not None)
                if graph.rows
                else (node["work_id"], node["layer"] is not None)
                for node in graph.operations[first:]
            )
        return op


class _Round:
    def __init__(
        self,
        template,
        details,
        wall_profile=None,
        *,
        compact_work=False,
        operation_rows=False,
    ):
        self.template, self.details = template, details
        self.rows = operation_rows and not details
        self.compact = (compact_work or operation_rows) and not details
        self.work_templates = []
        self.work_ids = {}
        self.expert_work = {}
        # Batch shapes are fixed within this round, but each layer keeps its
        # own operations, dependencies and KV binding. Never retain this cache
        # on the cross-round template.
        self.attention_work = {}
        self.price = (
            template.compute_cost
            if wall_profile is None
            else partial(wall_profile.call, "operator_pricing", template.compute_cost)
        )
        self.copy_price = (
            template.copy_cost
            if wall_profile is None
            else partial(wall_profile.call, "operator_pricing", template.copy_cost)
        )
        self.operations = []
        self.last = [None] * template.ep

    def intern_work(self, key, demands, slots):
        work_id = len(self.work_templates)
        self.work_ids[key] = work_id
        self.work_templates.append({"demands": demands, "slots": slots, "details": {}})
        return work_id

    def append(
        self,
        name,
        parents,
        demands,
        slots,
        details=None,
        attention=None,
        *,
        work_id=None,
    ):
        parents = sorted({p for p in parents if p is not None})
        if self.compact:
            if work_id is None:
                # Barrier nodes share an empty work item, but retain their IDs
                # and every dependency. This does not coalesce graph events.
                work_id = self.work_ids.get(None)
                if work_id is None:
                    work_id = self.intern_work(None, demands, slots)
            if self.rows:
                node_id = len(self.operations)
                self.operations.append(
                    (
                        parents,
                        work_id,
                        attention[0] if attention else None,
                        attention[2] if attention else None,
                        attention[1] if attention else None,
                    )
                )
                return node_id
            work = {"work_id": work_id}
        else:
            metadata = (
                {
                    "name": name,
                    "devices": [],
                    "stream": None,
                    "compute_us": 0,
                    "memory_us": 0,
                    "network_us": 0,
                    "flops": 0,
                    "input_bytes": 0,
                    "weight_bytes": 0,
                    "output_bytes": 0,
                    "inputs": {},
                    "outputs": {},
                    **(details or {}),
                }
                if self.details
                else {}
            )
            work = {"work": {"demands": demands, "slots": slots, "details": metadata}}
        operation = {
            "id": len(self.operations),
            "parents": parents,
            "layer": attention[0] if attention else None,
            "kv_rank": attention[2] if attention else None,
            **work,
        }
        if attention:
            operation["batch_indices"] = attention[1]
        self.operations.append(operation)
        return operation["id"]

    def barrier(self, name, parents):
        return self.append(name, parents, [], [])

    def repeat_attention(self, rank, layer, indices, ready):
        parents = sorted({ready, self.last[rank]} - {None})
        kv_rank = rank % self.template.tp
        for work_id, attention in self.attention_work[rank]:
            node_id = len(self.operations)
            node_layer = layer if attention else None
            node_kv_rank = kv_rank if attention else None
            if self.rows:
                self.operations.append(
                    (
                        parents,
                        work_id,
                        node_layer,
                        node_kv_rank,
                        indices if attention else None,
                    )
                )
            else:
                node = {
                    "id": node_id,
                    "parents": parents,
                    "layer": node_layer,
                    "kv_rank": node_kv_rank,
                    "work_id": work_id,
                }
                if attention:
                    node["batch_indices"] = indices
                self.operations.append(node)
            parents = [node_id]
        self.last[rank] = node_id
        return node_id

    def expert(self, rank, count, dispatched):
        """Lower the same three expert nodes without rebuilding static shapes."""
        key = (rank, count)
        works = self.expert_work.get(key)
        if works is None:
            gate = self.compute(
                rank,
                "",
                4 * count * H * M,
                {"x": (count, H)},
                {"y": (count, 2 * M)},
                {"w": (H, 2 * M)},
                [dispatched],
            )
            activation = self.compute(
                rank,
                "",
                6 * count * M,
                {"x": (count, 2 * M)},
                {"y": (count, M)},
                parents=[gate],
            )
            down = self.compute(
                rank,
                "",
                2 * count * M * H,
                {"x": (count, M)},
                {"y": (count, H)},
                {"w": (M, H)},
                [activation],
            )
            self.expert_work[key] = tuple(
                self.operations[index][1]
                if self.rows
                else self.operations[index]["work_id"]
                for index in (gate, activation, down)
            )
            return down
        first = len(self.operations)
        parents = sorted(
            {node for node in (dispatched, self.last[rank]) if node is not None}
        )
        if self.rows:
            self.operations.extend(
                (
                    (parents, works[0], None, None, None),
                    ([first], works[1], None, None, None),
                    ([first + 1], works[2], None, None, None),
                )
            )
        else:
            self.operations.extend(
                (
                    {
                        "id": first,
                        "parents": parents,
                        "layer": None,
                        "kv_rank": None,
                        "work_id": works[0],
                    },
                    {
                        "id": first + 1,
                        "parents": [first],
                        "layer": None,
                        "kv_rank": None,
                        "work_id": works[1],
                    },
                    {
                        "id": first + 2,
                        "parents": [first + 1],
                        "layer": None,
                        "kv_rank": None,
                        "work_id": works[2],
                    },
                )
            )
        self.last[rank] = first + 2
        return first + 2

    def broadcast(self, group, name, size, source, tokens):
        leader = group[0]
        return [source] + [
            self.copy(leader, rank, f"{name}.{leader}-{rank}", size, [source], tokens)
            for rank in group[1:]
        ]

    def gather(self, group, name, size, sources, tokens):
        leader = group[0]
        return [sources[0]] + [
            self.copy(rank, leader, f"{name}.{rank}-{leader}", size, [source], tokens)
            for rank, source in zip(group[1:], sources[1:])
        ]

    def all_reduce(self, group, name, tokens, sources):
        if len(group) == 1:
            return sources
        received = self.gather(group, name + ".gather", tokens * H * 2, sources, tokens)
        reduced = self.compute(
            group[0],
            name + ".reduce",
            (len(group) - 1) * tokens * H,
            {"partials": (len(group), tokens, H)},
            {"y": (tokens, H)},
            parents=received,
        )
        return self.broadcast(
            group, name + ".broadcast", tokens * H * 2, reduced, tokens
        )

    def compute(
        self,
        rank,
        name,
        flops,
        inputs,
        outputs,
        weights=None,
        parents=(),
        *,
        attention=None,
        read_fraction=Fraction(1),
        extra=None,
    ):
        template = self.template
        ct, mt, input_bytes, weight_bytes, output_bytes = self.price(
            rank,
            flops,
            tuple(inputs.items()),
            tuple(outputs.items()),
            tuple((weights or {}).items()),
            read_fraction,
        )
        gpu = template.gpus[rank]
        # Only service demand and stream identify an anonymous compute Work.
        # Dynamic node identity, dependencies and KV metadata remain per node.
        key = ("compute", rank, ct, mt)
        work_id = self.work_ids.get(key) if self.compact else None
        demands = slots = None
        if work_id is None:
            demands = []
            if ct:
                demands.append(
                    {"resources": {template.compute_resources[rank]: ct * 1e-6}}
                )
            if mt:
                demands.append(
                    {"resources": {template.memory_resources[rank]: mt * 1e-6}}
                )
            slots = [template.streams[rank]]
            if self.compact:
                work_id = self.intern_work(key, demands, slots)
        operation = self.append(
            name,
            [*parents, self.last[rank]],
            demands,
            slots,
            {
                "devices": [gpu.name],
                "stream": 0,
                "flops": flops,
                "compute_us": ct,
                "memory_us": mt,
                "network_us": 0,
                "input_bytes": input_bytes,
                "weight_bytes": weight_bytes,
                "output_bytes": output_bytes,
                "inputs": inputs,
                "outputs": outputs,
                "weights": weights or {},
                **(extra or {}),
            }
            if self.details
            else None,
            attention,
            work_id=work_id,
        )
        self.last[rank] = operation
        return operation

    def copy(self, source, destination, name, size, parents, tokens):
        template = self.template
        src, dst = template.gpus[source], template.gpus[destination]
        key = ("copy", source, destination, size)
        work_id = self.work_ids.get(key) if self.compact else None
        if work_id is not None:
            return self.append(name, parents, None, None, work_id=work_id)
        resources, latency, memory_us, network_us = self.copy_price(
            source, destination, size
        )
        demands = [{"resources": dict(resources), "latency": latency}]
        if self.compact:
            work_id = self.intern_work(key, demands, [])
        return self.append(
            name,
            parents,
            demands,
            [],
            {
                "devices": [src.name] if src is dst else [src.name, dst.name],
                "stream": None,
                "compute_us": 0,
                "memory_us": memory_us,
                "network_us": network_us,
                "flops": 0,
                "input_bytes": size,
                "weight_bytes": 0,
                "output_bytes": size,
                "communication_bytes": size if src is not dst else 0,
                "token_vectors": tokens,
                "source_rank": source,
                "destination_rank": destination,
            }
            if self.details
            else None,
            work_id=work_id,
        )
