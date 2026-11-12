"""Render --output-graph-trace JSONL as a self-contained, paginated HTML replay."""

import argparse
import json
from pathlib import Path


def load_trace(path):
    nodes, edges, changes = {}, [], []
    edge_stacks = {}
    last_seq = -1
    complete = False
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            seq, time = row["seq"], row["time_s"]
            if seq <= last_seq:
                raise ValueError("trace sequence must strictly increase")
            last_seq = seq
            event = row["event"]
            if seq == 0 and (event != "header" or row.get("version") != 1):
                raise ValueError("expected graph trace version 1")
            if event == "node_create":
                node = row["node"]
                if node in nodes:
                    raise ValueError(f"duplicate node {node}")
                nodes[node] = {
                    "id": node,
                    "created": seq,
                    "time": time,
                    "cause": row["cause"],
                    "release": row["release_s"],
                    "details": row["details"],
                    "dispatches": [],
                }
            elif event == "edge_add":
                parent, child = row["parent"], row["child"]
                if parent not in nodes or child not in nodes:
                    raise ValueError("edge references unknown node")
                edge_stacks.setdefault((parent, child), []).append(len(edges))
                edges.append(
                    {"parent": parent, "child": child, "added": seq, "removed": None}
                )
            elif event == "edge_remove":
                key = row["parent"], row["child"]
                if not edge_stacks.get(key):
                    raise ValueError("removing unknown edge")
                edges[edge_stacks[key].pop()]["removed"] = seq
            elif event == "dispatch":
                nodes[row["node"]]["dispatches"].append([seq, time])
            elif event == "node_complete":
                nodes[row["node"]].update(
                    resolved=seq, start=row["start_s"], finish=row["finish_s"]
                )
            elif event == "run_complete":
                complete = True
            changes.append([seq, time, event, row.get("node", row.get("parent"))])
    if not complete:
        raise ValueError("trace has no successful run_complete event")
    if any("finish" not in node for node in nodes.values()):
        raise ValueError("successful run contains unresolved nodes")
    return {"nodes": list(nodes.values()), "edges": edges, "changes": changes}


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Simthesizer 动态图</title>
<style>
body{font:14px system-ui;margin:20px;color:#192333;background:#f5f7fb}button,input,select{font:inherit;margin:5px;padding:5px}header{position:sticky;top:0;background:#f5f7fb;z-index:2;padding:8px;border-bottom:1px solid #ccd3df}#step{width:65%}#status{font-family:monospace}main{display:grid;grid-template-columns:minmax(700px,1fr) 360px;gap:18px}#diagram{background:white;overflow:auto;max-height:78vh}svg text{font:12px monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:14px}small{display:block;color:#556;line-height:1.5}.selected{stroke:#e64b22;stroke-width:3}
</style>
<header><b>Simthesizer · 一次运行的完整图历史</b><small>蓝：计算，紫：逻辑，绿：通信。浅色：尚未执行，深色：执行中，灰色：已完成。虚线：已移除依赖；橙色：所选节点。FastForward 保留引擎原始表示。</small>
<div><button id="play">播放</button><button id="back">上一步</button><button id="next">下一步</button><button id="end">全部历史</button><label><input id="follow" type="checkbox" checked>跟随事件</label><label><input id="live" type="checkbox">仅当前引擎图</label></div>
<input id="step" type="range" min="0"><span id="status"></span>
<div><input id="query" placeholder="搜索节点 ID、请求 ID 或算子"><select id="kind"><option value="">全部类型</option><option>logical</option><option>compute</option><option>network</option></select><button id="prevPage">上一页</button><span id="page"></span><button id="nextPage">下一页</button></div>
</header>
<main><div id="diagram"></div><aside><b>节点详情</b><pre id="detail">点击节点查看完整参数与前驱、后继。</pre><small>全量节点按创建 ID 分页，每页 80 个，未抽样。“仅当前引擎图”按节点解析删除时刻过滤，运行结束为空；关闭后查看全部历史。跨页依赖显示在详情中，可点击跳转。滑块按引擎操作序号回放，同一模拟时间可能有多个操作。完成记录表示引擎已计算出结束时间，节点是否完成按模拟时间判断；运行结束视图显示完整已解析结果。组件状态时间线请同时导出 --output-engine-trace。</small><div id="links"></div></aside></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const nodes=new Map(data.nodes.map(n=>[n.id,n])), size=80;
let page=0, selected=null, timer=null;
const step=$('step');step.max=data.changes.length-1;step.value=step.max;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function visible(){let seq=data.changes[+step.value][0],q=$('query').value.toLowerCase();return data.nodes.filter(n=>n.created<=seq&&(!$('live').checked||!(n.resolved<=seq))&&(!$('kind').value||n.details.kind===$('kind').value)&&(!q||(n.id+' '+JSON.stringify(n.details)).toLowerCase().includes(q)));}
function jump(id){selected=id;$('query').value='';$('kind').value='';if(nodes.get(id).created>data.changes[+step.value][0])step.value=nodes.get(id).created;page=Math.floor(visible().findIndex(n=>n.id===id)/size);render();}
function render(){const [seq,time,event,id]=data.changes[+step.value], all=+step.value===+step.max;
let list=visible();page=Math.max(0,Math.min(page,Math.ceil(list.length/size)-1));let slice=list.slice(page*size,(page+1)*size),positions=new Map(slice.map((n,i)=>[n.id,50+i*68]));
$('page').textContent=`${page+1}/${Math.max(1,Math.ceil(list.length/size))} · ${list.length}/${data.nodes.length} 节点 · ${data.edges.length} 条历史依赖`;
$('status').textContent=`#${seq} t=${time.toFixed(6)}s ${event}`;
let svg=`<svg width="850" height="${Math.max(200,slice.length*68+50)}" viewBox="0 0 850 ${Math.max(200,slice.length*68+50)}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8a97ac"/></marker></defs>`;
for(const e of data.edges){if(e.added>seq||!positions.has(e.parent)||!positions.has(e.child)||($('live').checked&&e.removed!==null&&e.removed<=seq))continue;let y=positions.get(e.parent)+16,z=positions.get(e.child)-16,removed=e.removed!==null&&e.removed<=seq;svg+=`<path d="M 650 ${y} C 790 ${y} 790 ${z} 650 ${z}" fill="none" stroke="${removed?'#c19a75':'#8a97ac'}" ${removed?'stroke-dasharray="4 4"':''} marker-end="url(#arrow)"/>`;}
for(const n of slice){const y=positions.get(n.id),begun=n.dispatches.some(d=>d[0]<=seq),done=n.resolved<=seq&&(all||n.finish<=time),base={compute:'#226fc9',logical:'#8150b5',network:'#18886b'}[n.details.kind];const fill=done?'#e2e7ef':begun?base:'#f0f4ff',text=begun&&!done?'white':'#25324b';const label=n.details.label.split('::').pop();svg+=`<g data-node="${n.id}" style="cursor:pointer"><rect x="24" y="${y-17}" width="626" height="36" rx="7" fill="${fill}" stroke="${base}" class="${selected===n.id?'selected':''}"/><text x="35" y="${y+5}" fill="${text}">${esc(('#'+n.id+' '+label).slice(0,78))}</text><title>${esc(n.details.label)}</title></g>`;}
$('diagram').innerHTML=svg+'</svg>';
$('diagram').querySelectorAll('[data-node]').forEach(el=>el.onclick=()=>{selected=+el.dataset.node;render();});
if(selected!==null){let n=nodes.get(selected),incoming=data.edges.filter(e=>e.child===selected&&e.added<=seq),outgoing=data.edges.filter(e=>e.parent===selected&&e.added<=seq);$('detail').textContent=JSON.stringify(n,null,2);$('links').replaceChildren();for(const [label,ids] of [['创建者',n.cause===null?[]:[n.cause]],['前驱',incoming.map(e=>e.parent)],['后继',outgoing.map(e=>e.child)]]){let div=document.createElement('div');div.textContent=label+': ';for(const id of new Set(ids)){let b=document.createElement('button');b.textContent='#'+id;b.onclick=()=>jump(id);div.appendChild(b);}$('links').appendChild(div);}}
}
function move(delta){step.value=Math.max(0,Math.min(+step.max,+step.value+delta));const row=data.changes[+step.value];if($('follow').checked&&row[3]!=null&&nodes.has(row[3])){selected=row[3];$('query').value='';$('kind').value='';page=Math.floor(visible().findIndex(n=>n.id===selected)/size);}render();}
step.oninput=()=>move(0);$('back').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('end').onclick=()=>{step.value=step.max;render();};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';}else{if(+step.value===+step.max)step.value=0;timer=setInterval(()=>{move(1);if(+step.value===+step.max)$('play').click();},120);$('play').textContent='暂停';}};
$('prevPage').onclick=()=>{page--;render();};$('nextPage').onclick=()=>{page++;render();};for(const id of ['query','kind','live'])$(id).oninput=()=>{page=0;render();};render();
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_trace(args.trace)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(HTML.replace("__DATA__", encoded))
    print(
        f"{args.output}: {len(data['nodes'])} nodes, {len(data['edges'])} historical edges"
    )


if __name__ == "__main__":
    main()
