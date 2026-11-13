"""Export an outer DAG trace and audited Rooflang rounds as an offline explorer.

Inner graphs are reconstructed with the bundled builder and simulator; durations
must match the original audit. All dependency edges and stream order are retained.
"""

import argparse
import json
import math
import re
from itertools import pairwise
from pathlib import Path

import networkx as nx
from draw_graph import load_trace
from rooflang.language.kernels.comm import CommKernel
from rooflang.runtime.simulator import Simulator
from rooflang_bridge.llama import build


def export_round(audit, config):
    graph, placement, hardware, summary = build(
        audit["batch"], config["tp"], config["hardware"]
    )
    result = Simulator(graph, placement, hardware).run()
    if not math.isclose(result.total_time_us, audit["duration_us"], rel_tol=1e-12):
        raise ValueError("reconstructed round duration differs from audit")
    # Pinned upstream has no public iterator for port-level edges.
    dag = graph._dag
    ordered = list(nx.topological_sort(dag))
    ids = {kernel: i for i, kernel in enumerate(ordered)}
    timings = {kernel: [] for kernel in ordered}
    for entry in result.trace:
        timings[entry.kernel].append(
            {
                "device": entry.device.name,
                "stream": entry.stream,
                "start_us": entry.start_us,
                "end_us": entry.end_us,
                "bound": entry.bound.value,
            }
        )

    def tensors(items):
        return {
            name: {
                "dtype": t.dtype,
                "shape": list(t.shape),
                "bytes": t.size_bytes,
                "weight_id": t.weight_id,
            }
            for name, t in items.items()
        }

    nodes = []
    for kernel in ordered:
        name = getattr(kernel, "name", type(kernel).__name__)
        match = re.match(r"layer(\d+)\.", name)
        layer = int(match[1]) if match else -1
        nodes.append(
            {
                "id": ids[kernel],
                "name": name,
                "class": type(kernel).__name__,
                "layer": layer,
                "kind": "communication"
                if isinstance(kernel, CommKernel)
                else "compute",
                "flops": kernel.flops,
                "inputs": tensors(kernel.inputs),
                "outputs": tensors(kernel.outputs),
                "weights": tensors(kernel.weights),
                "timings": timings[kernel],
            }
        )
    # Helper nodes belong to the layer consuming their residual branch.
    for kernel in reversed(ordered):
        node = nodes[ids[kernel]]
        if node["name"] in ("residual_fork", "residual_view"):
            layers = [
                nodes[ids[child]]["layer"]
                for child in dag.successors(kernel)
                if nodes[ids[child]]["layer"] >= 0
            ]
            node["layer"] = min(layers) if layers else -1
    edges = [
        {
            "source": ids[a],
            "target": ids[b],
            "kind": "data" if attrs["mapping"] else "control",
            "ports": attrs["mapping"],
        }
        for a, b, attrs in dag.edges(data=True)
    ]
    # Runtime stream serialization is separate from explicit graph dependencies.
    streams = {}
    for entry in result.trace:
        streams.setdefault((entry.device.name, entry.stream), []).append(entry)
    for (device, stream), entries in streams.items():
        entries.sort(key=lambda e: (e.start_us, e.end_us, ids[e.kernel]))
        for a, b in pairwise(entries):
            if a.kernel is not b.kernel:
                edges.append(
                    {
                        "source": ids[a.kernel],
                        "target": ids[b.kernel],
                        "kind": "stream",
                        "device": device,
                        "stream": stream,
                    }
                )
    fabrics = []
    for _, _, attrs in hardware._graph.edges(data=True):
        for edge in attrs["fabrics"]:
            fabrics.append(
                {
                    "name": edge.name,
                    "source": edge.src.name,
                    "target": edge.dst.name,
                    "bandwidth_gbs": edge.src_to_dst_bandwidth_gbs,
                    "latency_us": edge.alpha_us,
                }
            )
    selected_devices = {d["name"] for d in summary["hardware"]["devices"]}
    hardware_nodes = [
        {
            "name": n.name,
            "kind": n.kind or "component",
            "allocated": n.name in selected_devices,
        }
        for n in sorted(hardware.nodes, key=lambda n: n.name)
    ]
    return {
        "audit": audit,
        "nodes": nodes,
        "edges": edges,
        "hardware": fabrics,
        "hardware_nodes": hardware_nodes,
        "hardware_info": summary["hardware"],
    }


def assemble(trace, audit_dir):
    outer = load_trace(trace)
    config = json.loads((audit_dir / "init.json").read_text())
    iterations = sorted(
        (
            n
            for n in outer["nodes"]
            if n["details"]["label"].startswith("RooflangIteration")
        ),
        key=lambda n: n["dispatches"][0][0],
    )
    paths = sorted(p for p in audit_dir.glob("[0-9]*.json") if ".trace." not in p.name)
    if len(paths) != len(iterations):
        raise ValueError("outer iteration count and audit count differ")
    rounds = []
    for path, node in zip(paths, iterations):
        audit = json.loads(path.read_text())
        if not math.isclose(
            node["finish"] - node["start"],
            audit["duration_us"] * 1e-6,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("outer iteration duration and audit differ")
        inner = export_round(audit, config)
        inner["outer_node"] = node["id"]
        inner["start_s"] = node["start"]
        rounds.append(inner)
    return {"outer": outer, "rounds": rounds, "config": config}


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Simthesizer × Rooflang 全图</title>
<style>
body{font:14px system-ui;margin:24px;background:#f5f7fb;color:#192b42}h1{font-size:25px}h2{font-size:18px}button,select,input{margin:5px;padding:6px}button{cursor:pointer}section{background:white;padding:16px;margin:16px 0;border-radius:10px;border:1px solid #dbe2ec}.view{overflow:auto;height:580px;border:1px solid #dbe2ec;background:#fff}svg text{font:12px monospace}pre{white-space:pre-wrap;word-break:break-word;max-height:400px;overflow:auto}.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);gap:16px}.muted{color:#596a80}.legend span{margin-right:18px}#detail{font-size:12px}a{color:#245fa5}
</style>
<h1>Simthesizer 动态 DAG → Rooflang 完整执行图</h1><h2 id="hardwareTitle"></h2>
<p>真实运行的外层事件记录；内层按该轮批次重建并核对耗时。全部 32 层、残差、KV 和通信节点均保留。</p>
<p class="muted">外层单位：秒。内层单位：微秒；全局时间 = 本轮开始秒数 + 内层时间 / 10⁶。点击外层 RooflangIteration 进入对应轮次。此连接表示封装关系，并非把内层节点插入外层引擎。</p>
<section><h2>1 · 外层动态 DAG</h2><div><button id="prev">上一步</button><button id="next">下一步</button><button id="play">播放</button><button id="last">完整图</button><input id="step" type="range" style="width:45%"><span id="event"></span></div>
<div class="grid"><div class="view" id="outer"></div><div><p>紫色：逻辑节点；蓝色：整轮 Rooflang。节点按创建顺序展示，连线来自真实 edge_add。虚线表示已移除的历史依赖。</p><pre id="outerInfo">点击节点查看创建者、创建事件、执行时间。</pre></div></div></section>
<section><h2>2 · 本轮完整计算与通信图</h2>
<label>轮次 <select id="round"></select></label><label>显示层 <select id="layer"><option value="all">全部 32 层</option><option value="-1">Embedding / 输出</option></select></label>
<label><input id="streams" type="checkbox">显示 stream 执行顺序</label><label>缩放 <input id="zoom" type="range" min="15" max="150" value="80"></label>
<p id="summary"></p><p class="legend"><span>蓝：计算</span><span>绿：集合通信</span><span>灰：零成本辅助节点</span><span>实线：数据/控制依赖</span><span>橙色虚线：同设备 stream 顺序</span></p>
<div class="grid"><div class="view" id="inner"></div><div><div id="neighbors"></div><pre id="detail">点击算子查看形状、FLOPs、权重、KV、设备和时间。选择“全部 32 层”可滚动查看完整图。</pre></div></div>
<p class="muted">按层显示时仅隐藏其他层；选中节点的跨层前驱/后继仍可点击跳转。权重和 KV 前缀是驻留张量描述，不是独立算子节点；详见节点属性。无计时记录的辅助节点不表示缺失算子。</p></section>
<section><h2>3 · 硬件拓扑</h2><div id="hardware"></div><pre id="hwinfo"></pre></section>
<script>
const data=__DATA__, $=id=>document.getElementById(id), esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const outer=data.outer, step=$('step');step.min=0;step.max=outer.changes.length-1;step.value=step.max;
let selected=null,timer=null;
function outerRender(){const change=outer.changes[+step.value],seq=change[0],ns=outer.nodes.filter(n=>n.created<=seq),pos=new Map(ns.map((n,i)=>[n.id,45+i*65]));$('event').textContent=`事件 ${seq} · ${change[1].toFixed(6)} s · ${change[2]}`;let s=`<svg width="970" height="${Math.max(400,ns.length*65+60)}">`;
for(const e of outer.edges){if(e.added>seq||!pos.has(e.parent)||!pos.has(e.child))continue;let a=pos.get(e.parent),b=pos.get(e.child),x=660+Math.min(260,Math.abs(a-b)/12);s+=`<path d="M650 ${a} C${x} ${a},${x} ${b},650 ${b}" fill="none" stroke="#a3afc0" ${e.removed!==null&&e.removed<=seq?'stroke-dasharray="4 4"':''}/><circle cx="650" cy="${b}" r="3" fill="#687a91"/>`;}
for(const n of ns){let label=n.details.label.split('::').pop(),isRound=label.startsWith('RooflangIteration'),done=n.resolved<=seq;s+=`<g data-id="${n.id}" style="cursor:pointer"><rect x="15" y="${pos.get(n.id)-21}" width="635" height="44" rx="6" fill="${done?'#e9edf2':isRound?'#dbeaff':'#eee4fb'}" stroke="${isRound?'#2871bb':'#8a5abb'}"/><text x="26" y="${pos.get(n.id)+4}">${esc(('#'+n.id+' '+label).slice(0,83))}</text><title>${esc(label)}</title></g>`;}
$('outer').innerHTML=s+'</svg>';$('outer').querySelectorAll('[data-id]').forEach(el=>el.onclick=()=>{let id=+el.dataset.id;$('outerInfo').textContent=JSON.stringify(outer.nodes.find(n=>n.id===id),null,2);let r=data.rounds.findIndex(r=>r.outer_node===id);if(r>=0){$('round').value=r;selected=null;innerRender();}});}
for(let i=0;i<32;i++)$('layer').add(new Option('Layer '+i,i));data.rounds.forEach((r,i)=>$('round').add(new Option(`第 ${i+1} 轮 · 外层 #${r.outer_node}`,i)));
function current(){return data.rounds[+$('round').value];}
function jump(id){selected=id;const n=current().nodes.find(n=>n.id===id);if($('layer').value!=='all'&&+ $('layer').value!==n.layer)$('layer').value=String(n.layer);innerRender();$('inner').querySelector(`[data-id="${id}"]`)?.scrollIntoView({block:'center',inline:'center'});}
function innerRender(){const r=current(),filter=$('layer').value,ns=r.nodes.filter(n=>filter==='all'||n.layer===+filter),ids=new Set(ns.map(n=>n.id)),es=r.edges.filter(e=>e.kind!=='stream'&&ids.has(e.source)&&ids.has(e.target)),levels=new Map();for(const n of ns){let prev=es.filter(e=>e.target===n.id).map(e=>levels.get(e.source)??0);levels.set(n.id,prev.length?1+Math.max(...prev):0);}const groups=new Map();for(const n of ns){let l=levels.get(n.id);if(!groups.has(l))groups.set(l,[]);groups.get(l).push(n);}const pos=new Map();let width=900;for(const [l,items] of groups){width=Math.max(width,items.length*330+30);items.forEach((n,i)=>pos.set(n.id,[20+i*330,30+l*78]));}let height=Math.max(300,groups.size*78+70),z=+$('zoom').value/100,s=`<svg width="${width*z}" height="${height*z}" viewBox="0 0 ${width} ${height}"><defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#91a0b3"/></marker></defs>`;
for(const e of r.edges){if(!ids.has(e.source)||!ids.has(e.target)||(e.kind==='stream'&&!$('streams').checked))continue;const [x,y]=pos.get(e.source),[a,b]=pos.get(e.target);s+=`<path d="M${x+145} ${y+44} C${x+145} ${y+60},${a+145} ${b-15},${a+145} ${b}" fill="none" stroke="${e.kind==='stream'?'#d6932e':'#a9b7c7'}" ${e.kind==='stream'?'stroke-dasharray="5 5"':''} marker-end="url(#arrow)"/>`;}
for(const n of ns){const [x,y]=pos.get(n.id),color=n.kind==='communication'?'#d7f0e7':n.class==='Work'?'#dfecfc':'#eef0f2';s+=`<g data-id="${n.id}" style="cursor:pointer"><rect x="${x}" y="${y}" width="300" height="44" rx="5" fill="${color}" stroke="${selected===n.id?'#f06428':'#8399af'}" stroke-width="${selected===n.id?3:1}"/><text x="${x+8}" y="${y+17}">${esc(('#'+n.id+' '+n.name).slice(0,39))}</text><text x="${x+8}" y="${y+34}">${esc(n.timings.map(t=>t.device).join(',')||'helper')} · ${n.flops} FLOPs</text><title>${esc(n.name)}</title></g>`;}
$('inner').innerHTML=s+'</svg>';$('inner').querySelectorAll('[data-id]').forEach(el=>el.onclick=()=>{selected=+el.dataset.id;details();});
$('summary').textContent=`批次 ${JSON.stringify(r.audit.batch)}；${r.audit.duration_us.toFixed(6)} μs；${r.nodes.length} 个节点，${r.edges.filter(e=>e.kind!=='stream').length} 条图依赖，${r.audit.collective_count} 个集合通信；本轮开始 ${r.start_s.toFixed(9)} s。当前显示 ${ns.length} 个节点。`;
details();hardwareRender(r);}
function details(){const r=current(),n=r.nodes.find(n=>n.id===selected);$('neighbors').replaceChildren();if(!n){$('detail').textContent='点击节点查看完整属性';return;}$('detail').textContent=JSON.stringify({...n,global_times_s:n.timings.map(t=>({device:t.device,start:r.start_s+t.start_us/1e6,end:r.start_s+t.end_us/1e6}))},null,2);for(const [title,edges,field] of [['前驱',r.edges.filter(e=>e.target===n.id),'source'],['后继',r.edges.filter(e=>e.source===n.id),'target']]){let p=document.createElement('div');p.textContent=title+': ';for(const id of new Set(edges.filter(e=>e.kind!=='stream').map(e=>e[field]))){let b=document.createElement('button');b.textContent='#'+id;b.onclick=()=>jump(id);p.append(b);}$('neighbors').append(p);}}
function hardwareRender(r){
const info=r.hardware_info;$('hardwareTitle').textContent=`${info.gpu_count} × ${info.gpu_model} · TP=${data.config.tp}`;
const groups=new Map();for(const n of r.hardware_nodes){if(!groups.has(n.kind))groups.set(n.kind,[]);groups.get(n.kind).push(n);}const positions=new Map();let row=0,width=1000;for(const items of groups.values()){width=Math.max(width,items.length*240+40);items.forEach((n,i)=>positions.set(n.name,[130+i*240,50+row*120]));row++;}
let s=`<svg width="${width}" height="${row*120+40}">`;for(const e of r.hardware){let [x,y]=positions.get(e.source),[a,b]=positions.get(e.target);s+=`<path d="M${x} ${y} Q${(x+a)/2+25} ${(y+b)/2-35},${a} ${b}" fill="none" stroke="#a6b7c8"><title>${esc(e.name)} · ${e.bandwidth_gbs} GB/s · ${e.latency_us} μs</title></path>`;}
for(const n of r.hardware_nodes){let [x,y]=positions.get(n.name);s+=`<g><rect x="${x-110}" y="${y-23}" width="220" height="46" rx="6" fill="${n.allocated?'#c4e5fc':n.kind==='gpu'?'#eee':'#e4eee7'}" stroke="${n.allocated?'#1976b7':'#bcc9ce'}"/><text x="${x-103}" y="${y-3}">${esc(n.name)}</text><text x="${x-103}" y="${y+14}">${esc(n.kind)}${n.kind==='gpu'?(n.allocated?' · 本实例使用':' · 未分配'):''}</text></g>`;}
$('hardware').style.overflow='auto';$('hardware').innerHTML=s+'</svg>';$('hwinfo').textContent=`本实例分配 ${info.gpu_count} 张卡；上游原始节点拓扑共 ${info.topology_gpu_count} 张卡。未分配 GPU 不执行本实例算子。\n`+JSON.stringify({resolved:info,selection:data.config.hardware,links:r.hardware},null,2);}

$('prev').onclick=()=>{step.value=Math.max(0,+step.value-1);outerRender();};$('next').onclick=()=>{step.value=Math.min(+step.max,+step.value+1);outerRender();};$('last').onclick=()=>{step.value=step.max;outerRender();};step.oninput=outerRender;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';return;}if(step.value===step.max)step.value=0;$('play').textContent='暂停';timer=setInterval(()=>{if(+step.value>=+step.max){$('play').click();return;}$('next').click();},200);};$('round').onchange=()=>{selected=null;innerRender();};$('layer').onchange=innerRender;$('streams').onchange=innerRender;$('zoom').oninput=innerRender;outerRender();innerRender();
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = assemble(args.trace, args.audit_dir)
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", encoded))
    args.output.with_suffix(".json").write_text(encoded)
    print(
        json.dumps(
            {
                "html": str(args.output),
                "outer_nodes": len(data["outer"]["nodes"]),
                "rounds": [
                    {
                        "outer_node": r["outer_node"],
                        "nodes": len(r["nodes"]),
                        "edges": len(r["edges"]),
                        "duration_us": r["audit"]["duration_us"],
                    }
                    for r in data["rounds"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
