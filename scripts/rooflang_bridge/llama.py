"""FP16 Llama 3.1 8B analytical execution graph, including ragged causal attention.

GEMMs are batched across queries; attention preserves each request's lengths.
Costs are analytical, not calibrated GPU measurements. See architectures/rooflang.md.
"""

from fractions import Fraction

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.comm import AllGather, AllReduce
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import Simulator

from rooflang_bridge.bindings import BatchBindings
from rooflang_bridge.hardware import build_hardware

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


def batch_shape(batch):
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
    return {
        "tokens": sum(r["q_len"] for r in batch),
        "logits_rows": len(batch),
        "causal_pairs": sum(
            r["q_len"] * r["kv_len"] + r["q_len"] * (r["q_len"] + 1) // 2 for r in batch
        ),
        "kv_tokens": sum(r["kv_len"] for r in batch),
    }


def build(batch, tp, hardware, *, _bindings=None):
    if tp not in (1, 2, 4, 8):
        raise ValueError("TP must be 1, 2, 4 or 8")
    shape = batch_shape(batch)
    bindings = _bindings if _bindings is not None else BatchBindings()
    tokens = bindings.parameter("tokens", shape["tokens"])
    heads = bindings.parameter("logits_rows", shape["logits_rows"])
    pairs = bindings.parameter("causal_pairs", shape["causal_pairs"])
    kv_tokens = bindings.parameter("kv_tokens", shape["kv_tokens"])
    hw, gpu, mem, hardware_info = build_hardware(hardware, tp)
    g = ComputeGraph()
    p = Placement(hardware=hw, graph=g)

    def tensor(n, wid=None):
        t = Tensor("fp16", (0,), weight_id=wid)
        bindings.assign(t, "shape", n, lambda size: (size,))
        bindings.tensor_sizes[t] = n
        return t

    def op(rank, name, flops, sources, nout, weights=None, extra_outputs=None):
        inputs = {
            f"x{i}": source.outputs[port] for i, (source, port) in enumerate(sources)
        }
        outputs = {"y": tensor(nout)}
        outputs.update(extra_outputs or {})
        k = Work(name, 0, inputs, outputs, weights)
        bindings.assign(k, "work", flops)
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
        fan.outputs = {f"y{i}": tensor(bindings.tensor_sizes[t]) for i in range(2)}
        g.add_kernel(fan)
        g.add_data_edge(source, fan, {"y": "x"})
        p.set_kernel_device(fan, gpu[rank])
        views = []
        for i in range(2):
            view = Nop(
                inputs={"x": fan.outputs[f"y{i}"]},
                outputs={"y": tensor(bindings.tensor_sizes[t])},
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
            total_bytes=0,
            world=tp,
            **({"dtype": "fp16"} if cls is AllReduce else {}),
        )
        bindings.assign(k, "total_bytes", rows * width * 2)
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
        bindings.assign(emb, "weight_read_fraction", tokens, lambda n: Fraction(n, V))
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
            kv = kv_tokens * 2 * (KVH // tp) * D
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
            "tokens": shape["tokens"],
            "logits_rows": shape["logits_rows"],
            "causal_pairs": shape["causal_pairs"],
            "tp": tp,
            "layers": L,
            "hardware": hardware_info,
        },
    )


def simulate(batch, tp, hardware):
    g, p, hw, summary = build(batch, tp, hardware)
    return _run(g, p, hw, summary)


def _run(g, p, hw, summary):
    result = Simulator(g, p, hw).run()
    summary.update(
        duration_us=result.total_time_us,
        kernel_count=len(g.kernels),
        collective_count=sum(isinstance(k, (AllReduce, AllGather)) for k in g.kernels),
        peak_memory_bytes={m.name: n for m, n in result.peak_memory.items()},
    )
    return result, summary


class LlamaTemplate:
    """One fixed model/TP/hardware graph, rebound between exclusive iterations.

    Results contain references to mutable kernels. Consume/export a result's
    trace before the next call, as the worker does. Do not share across workers.
    """

    def __init__(self, tp, hardware):
        self.bindings = BatchBindings(record=True)
        self.graph, self.placement, self.hardware, self.summary = build(
            [{"q_len": 1, "kv_len": 0, "lm_head_len": 1}],
            tp,
            hardware,
            _bindings=self.bindings,
        )

    def bind(self, batch):
        shape = batch_shape(batch)
        self.bindings.bind(shape)
        return {
            **self.summary,
            **{k: shape[k] for k in ("tokens", "logits_rows", "causal_pairs")},
        }

    def simulate(self, batch):
        summary = self.bind(batch)
        # A new runtime keeps events, memory accounting and failed runs isolated.
        # Graph, tensor identities, placement and hardware remain unchanged.
        return _run(self.graph, self.placement, self.hardware, summary)
