"""FP16 Llama 3.1 8B analytical execution graph, including ragged causal attention.

GEMMs are batched across queries; attention preserves each request's lengths.
Costs are analytical, not calibrated GPU measurements. See architectures/rooflang.md.
"""

import math
from fractions import Fraction

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import AllGather, AllReduce
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import Simulator

H, FF, QH, KVH, D, V, L = 4096, 14336, 32, 8, 128, 128256, 32


class Work(Kernel):
    def __init__(self, name, flops, inputs, outputs, weights=None):
        super().__init__(inputs=inputs, outputs=outputs, weights=weights)
        self.name, self.work, self.dtype_ = name, flops, "fp16"

    @property
    def flops(self):
        return self.work


def layer_plan(tp):
    """Python model definition in the shared, serialized Layer schema."""
    if tp not in (1, 2, 4, 8):
        raise ValueError("TP must be 1, 2, 4 or 8")
    expected_pre = [
        {"Embedding": {"name": "embedding", "num_embeddings": V, "embedding_dim": H}}
    ]
    expected_block = [
        {"LayerNorm": {"name": "layernorm", "hidden_size": H}},
        {"Gemm": {"name": "qkv_projection", "m": H, "n": (H + 2 * KVH * D) // tp}},
        {"Rope": {"name": "rope", "head_dim": D, "num_heads": QH // tp}},
        {
            "Attention": {
                "name": "attention",
                "num_q_heads": QH // tp,
                "num_kv_heads": KVH // tp,
                "head_dim": D,
            }
        },
        {"Gemm": {"name": "o_projection", "m": H // tp, "n": H}},
    ]
    ar = {"AllReduce": {"name": "all_reduce", "bytes_per_token": H * 2}}
    if tp > 1:
        expected_block.append(ar)
    expected_block += [
        {"LayerNorm": {"name": "layernorm", "hidden_size": H}},
        {"Gemm": {"name": "ffn1", "m": H, "n": 2 * (FF // tp)}},
        {"Act": {"name": "act", "width": FF // tp}},
        {"Gemm": {"name": "ffn2", "m": FF // tp, "n": H}},
    ]
    if tp > 1:
        expected_block.append(ar)
    expected_post = [
        {"LayerNorm": {"name": "final_layernorm", "hidden_size": H}},
        {"Gemm": {"name": "lm_head", "m": H, "n": V // tp}},
        {"Sampler": {"name": "sampler", "vocab_size": V}},
    ]
    return {
        "pre": expected_pre,
        "block": expected_block,
        "post": expected_post,
        "layers": L,
    }


def describe(parallel):
    """Scheduler metadata and structure for the Python Model entry point."""
    if parallel["pp"] != 1 or parallel["ep"] != 1:
        raise ValueError("Python Llama supports only PP=1 and EP=1")
    tp = parallel["tp"]
    plan = layer_plan(tp)
    matrices = 2 * V * H + L * (H * (H + 2 * KVH * D) + H * H + 3 * H * FF)
    return {
        "version": 1,
        **plan,
        "weight_bytes_per_device": 2 * (matrices // tp + (2 * L + 1) * H),
        "kv_bytes_per_token_per_device": 2 * 2 * (KVH // tp) * D * L,
    }


def validate_model(model, tp):
    if model != layer_plan(tp):
        raise ValueError("backend supports exactly FP16 Llama 3.1 8B layer plan")


def build(batch, tp, hardware):
    if tp not in (1, 2, 4, 8):
        raise ValueError("TP must be 1, 2, 4 or 8")
    if not batch:
        raise ValueError("empty batch")
    for r in batch:
        if set(r) != {"q_len", "kv_len", "lm_head_len"} or any(
            type(v) is not int for v in r.values()
        ):
            raise ValueError("invalid batch schema")
        if r["q_len"] <= 0 or r["kv_len"] < 0 or r["lm_head_len"] != 1:
            raise ValueError(
                "require q_len>0, kv_len>=0 and one logits row per request"
            )
    required = {
        "tflops_fp16",
        "memory_bandwidth_gbs",
        "memory_capacity_gb",
        "link_bandwidth_gbs",
        "link_latency_us",
    }
    if set(hardware) != required or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in hardware.values()
    ):
        raise ValueError("invalid hardware schema")
    if any(hardware[k] <= 0 for k in required - {"link_latency_us"}):
        raise ValueError("hardware capacities and rates must be positive")
    g = ComputeGraph()
    hw = HardwareGraph()
    p = Placement(hardware=hw, graph=g)
    gpu = []
    mem = []
    for rank in range(tp):
        dev = Compute(name=f"gpu{rank}", tflops={"fp16": hardware["tflops_fp16"]})
        ram = Memory(name=f"hbm{rank}", capacity_gb=hardware["memory_capacity_gb"])
        hw.add_node(dev)
        hw.add_node(ram)
        gpu.append(dev)
        mem.append(ram)
        bw = hardware["memory_bandwidth_gbs"]
        hw.add_edge(
            FabricEdge(
                name=f"hbm{rank}",
                src=dev,
                dst=ram,
                src_to_dst_bandwidth_gbs=bw,
                dst_to_src_bandwidth_gbs=bw,
                is_full_duplex=False,
            )
        )
    for a in range(tp):
        for b in range(a + 1, tp):
            bw = hardware["link_bandwidth_gbs"]
            hw.add_edge(
                FabricEdge(
                    name=f"link{a}-{b}",
                    src=gpu[a],
                    dst=gpu[b],
                    src_to_dst_bandwidth_gbs=bw,
                    dst_to_src_bandwidth_gbs=bw,
                    is_full_duplex=True,
                    alpha_us=hardware["link_latency_us"],
                )
            )

    def tensor(n, wid=None):
        return Tensor("fp16", (n,), weight_id=wid)

    def op(rank, name, flops, sources, nout, weights=None, extra_outputs=None):
        inputs = {
            f"x{i}": source.outputs[port] for i, (source, port) in enumerate(sources)
        }
        outputs = {"y": tensor(nout)}
        outputs.update(extra_outputs or {})
        k = Work(name, flops, inputs, outputs, weights)
        g.add_kernel(k)
        for i, (source, port) in enumerate(sources):
            g.add_data_edge(source, k, {port: f"x{i}"})
        p.set_kernel_device(k, gpu[rank])
        return k

    def fork(rank, source):
        from rooflang.language.kernels.forward import Nop
        from rooflang.language.kernels.identity import Spawn

        fan = Spawn(2)
        fan.name = "residual_fork"
        t = source.outputs["y"]
        fan.inputs = {"x": t}
        fan.outputs = {f"y{i}": tensor(t.n_elements) for i in range(2)}
        g.add_kernel(fan)
        g.add_data_edge(source, fan, {"y": "x"})
        p.set_kernel_device(fan, gpu[rank])
        views = []
        for i in range(2):
            view = Nop(
                inputs={"x": fan.outputs[f"y{i}"]}, outputs={"y": tensor(t.n_elements)}
            )
            view.name = "residual_view"
            g.add_kernel(view)
            g.add_data_edge(fan, view, {f"y{i}": "x"})
            p.set_kernel_device(view, gpu[rank])
            views.append(view)
        return views

    def collective(previous, name, rows, width, cls=AllReduce):
        if tp == 1:
            return previous
        k = cls(
            total_bytes=rows * width * 2,
            world=tp,
            **({"dtype": "fp16"} if cls is AllReduce else {}),
        )
        k.name = name
        k.inputs = {f"x{r}": previous[r].outputs["y"] for r in range(tp)}
        k.outputs = {f"y{r}": tensor(rows * width) for r in range(tp)}
        g.add_kernel(k)
        for r in range(tp):
            g.add_data_edge(previous[r], k, {"y": f"x{r}"})
            p.set_tensor_memory(k.inputs[f"x{r}"], mem[r])
            p.set_tensor_memory(k.outputs[f"y{r}"], mem[r])
        # Zero-work views preserve actual collective output placement and dependency.
        result = []
        from rooflang.language.kernels.forward import Nop

        for r in range(tp):
            view = Nop(
                inputs={"x": k.outputs[f"y{r}"]}, outputs={"y": k.outputs[f"y{r}"]}
            )
            view.name = name + f".rank{r}"
            g.add_kernel(view)
            g.add_data_edge(k, view, {f"y{r}": "x"})
            p.set_kernel_device(view, gpu[r])
            result.append(view)
        return result

    tokens = sum(r["q_len"] for r in batch)
    heads = len(batch)
    pairs = sum(
        r["q_len"] * r["kv_len"] + r["q_len"] * (r["q_len"] + 1) // 2 for r in batch
    )
    previous = []
    kv_appends = []
    for r in range(tp):
        emb = op(
            r,
            "embedding",
            0,
            [],
            tokens * H,
            {"table": tensor(V // tp * H, f"embedding.{r}")},
        )
        # Uniform vocabulary shard hits; gather reads only selected rows, not full table.
        emb.weight_read_fraction = Fraction(tokens, V)
        previous.append(emb)
    previous = collective(previous, "embedding_allreduce", tokens, H)
    for layer in range(L):
        attention = []
        branches = [fork(r, previous[r]) for r in range(tp)]
        previous = [b[0] for b in branches]
        residual = [b[1] for b in branches]
        for r in range(tp):
            prefix = f"layer{layer}.rank{r}."
            norm = op(
                r,
                prefix + "input_norm",
                4 * tokens * H,
                [(previous[r], "y")],
                tokens * H,
                {"gamma": tensor(H, prefix + "norm1")},
            )
            width = (H + 2 * KVH * D) // tp
            qkv = op(
                r,
                prefix + "qkv",
                2 * tokens * H * width,
                [(norm, "y")],
                tokens * width,
                {"w": tensor(H * width, prefix + "qkv")},
            )
            rope = op(
                r,
                prefix + "rope",
                3 * tokens * (QH + KVH) // tp * D,
                [(qkv, "y")],
                tokens * width,
            )
            kv = sum(x["kv_len"] for x in batch) * 2 * (KVH // tp) * D
            attn = op(
                r,
                prefix + "attention",
                (4 * D + 5) * (QH // tp) * pairs,
                [(rope, "y")],
                tokens * H // tp,
                {"kv_prefix": tensor(kv, prefix + "kv")},
                {"kv_append": tensor(tokens * 2 * (KVH // tp) * D)},
            )
            kv_appends.append((attn, "kv_append"))
            out = op(
                r,
                prefix + "o_projection",
                2 * tokens * (H // tp) * H,
                [(attn, "y")],
                tokens * H,
                {"w": tensor(H // tp * H, prefix + "o")},
            )
            attention.append(out)
        attention = collective(
            attention, f"layer{layer}.attention_allreduce", tokens, H
        )
        ff = []
        saved = []
        for r in range(tp):
            prefix = f"layer{layer}.rank{r}."
            add = op(
                r,
                prefix + "attention_residual",
                tokens * H,
                [(attention[r], "y"), (residual[r], "y")],
                tokens * H,
            )
            add, skip = fork(r, add)
            saved.append(skip)
            norm = op(
                r,
                prefix + "post_norm",
                4 * tokens * H,
                [(add, "y")],
                tokens * H,
                {"gamma": tensor(H, prefix + "norm2")},
            )
            up = op(
                r,
                prefix + "gate_up",
                2 * tokens * H * (2 * FF // tp),
                [(norm, "y")],
                tokens * 2 * FF // tp,
                {"w": tensor(H * 2 * FF // tp, prefix + "up")},
            )
            act = op(
                r,
                prefix + "silu_mul",
                6 * tokens * FF // tp,
                [(up, "y")],
                tokens * FF // tp,
            )
            down = op(
                r,
                prefix + "down",
                2 * tokens * (FF // tp) * H,
                [(act, "y")],
                tokens * H,
                {"w": tensor(FF // tp * H, prefix + "down")},
            )
            ff.append(down)
        ff = collective(ff, f"layer{layer}.ffn_allreduce", tokens, H)
        previous = [
            op(
                r,
                f"layer{layer}.rank{r}.ffn_residual",
                tokens * H,
                [(ff[r], "y"), (saved[r], "y")],
                tokens * H,
            )
            for r in range(tp)
        ]
    logits = []
    for r in range(tp):
        norm = op(
            r,
            "final_norm",
            4 * tokens * H,
            [(previous[r], "y")],
            tokens * H,
            {"gamma": tensor(H, f"final_norm.{r}")},
        )
        select = op(r, "select_last_tokens", 0, [(norm, "y")], heads * H)
        logits.append(
            op(
                r,
                "lm_head",
                2 * heads * H * (V // tp),
                [(select, "y")],
                heads * V // tp,
                {"w": tensor(H * (V // tp), f"lm_head.{r}")},
            )
        )
    logits = collective(logits, "logits_allgather", heads, V, AllGather)
    samples = [
        op(r, "sampler", 5 * heads * V, [(logits[r], "y")], heads) for r in range(tp)
    ]
    from rooflang.language.kernels.forward import Nop

    sources = kv_appends + [(k, "y") for k in samples]
    sink = Nop(inputs={f"x{i}": k.outputs[port] for i, (k, port) in enumerate(sources)})
    sink.name = "iteration_complete"
    g.add_kernel(sink)
    for i, (k, port) in enumerate(sources):
        g.add_data_edge(k, sink, {port: f"x{i}"})
    p.set_kernel_device(sink, gpu[0])
    g.validate()
    p.validate(g)
    return (
        g,
        p,
        hw,
        {
            "tokens": tokens,
            "logits_rows": heads,
            "causal_pairs": pairs,
            "tp": tp,
            "layers": L,
        },
    )


def simulate(batch, tp, hardware):
    g, p, hw, summary = build(batch, tp, hardware)
    result = Simulator(g, p, hw).run()
    summary.update(
        duration_us=result.total_time_us,
        kernel_count=len(g.kernels),
        collective_count=sum(isinstance(k, (AllReduce, AllGather)) for k in g.kernels),
        peak_memory_bytes={m.name: n for m, n in result.peak_memory.items()},
    )
    return result, summary
