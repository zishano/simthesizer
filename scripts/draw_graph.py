"""Render graph history as a self-contained module timeline and node replay."""

import argparse
import hashlib
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


def workload_info(path):
    raw = path.read_bytes()
    requests = [json.loads(line) for line in raw.splitlines() if line.strip()]
    calls = [
        node
        for request in requests
        for node in request["sub_requests"]
        if node["kind"] == "llm"
    ]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "requests": len(requests),
        "llm_calls": len(calls),
        "input_tokens": sum(node["input_tokens"] for node in calls),
        "output_tokens": sum(node["output_tokens"] for node in calls),
        "source": "viewer_argument",
    }


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Simthesizer 动态图</title>
<style>
body{font:14px system-ui;margin:20px;color:#192333;background:#f5f7fb}
button,input,select{font:inherit;padding:3px 6px;margin:3px}
header{position:sticky;top:0;background:#f5f7fb;z-index:2;padding:8px;border-bottom:1px solid #ccd3df}
#step{width:65%}#status{font-family:monospace}
main{display:grid;grid-template-columns:minmax(0,max-content) minmax(260px,340px);gap:12px}
#diagram{background:white;overflow:auto;max-height:78vh}#diagram svg text{font:12px monospace}
aside{max-height:78vh;overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:10px}
small{display:block;color:#556;line-height:1.5}.selected{stroke:#e64b22;stroke-width:3}
#overview{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}#guide{margin:4px 0;color:#556;font-size:12px}
#meaning{background:white;padding:10px;line-height:1.6;white-space:pre-line}#meaning h3{font:13px monospace;margin:0 0 8px}
details{margin:10px 0}#links button{text-align:left;font:12px monospace}#links{line-height:1.8;font-size:12px}
#legend table{border-collapse:collapse;font-size:12px;width:100%;margin-top:6px}#legend td{padding:4px 6px;border-bottom:1px solid #e2e7ef}#legend code{white-space:nowrap}
#key{margin:4px 0;font-size:12px}#key code{margin-right:12px}
@media(max-width:700px){main{grid-template-columns:1fr}aside{max-height:none}}
[hidden]{display:none!important}
#modes{display:inline-flex;margin-left:16px;gap:4px}#modes button[aria-pressed=true]{background:#253b62;color:white;border-color:#253b62}
#moduleTime{width:min(720px,65vw)}#moduleStatus{font:12px monospace}
#moduleView{display:grid;grid-template-columns:950px minmax(650px,1fr);column-gap:16px;align-items:start}#moduleExecution{min-width:0;grid-column:1;grid-row:1}#moduleLayout{display:grid;grid-template-columns:minmax(0,620px) minmax(250px,300px);gap:12px;margin-top:8px}
#moduleGraph{background:white;overflow:auto;border:1px solid #e2e7ef;border-radius:8px}#moduleGraph svg{display:block;width:auto;height:auto;font:12px system-ui}
#moduleDetails{background:white;padding:10px;border-radius:8px;font-size:12px;line-height:1.6;max-height:370px;overflow:auto}#moduleDetails h3{margin:0 0 8px}#moduleDetails table{border-collapse:collapse;width:100%;font-size:12px}#moduleDetails th,#moduleDetails td{padding:4px;border-bottom:1px solid #e2e7ef;text-align:right}#moduleDetails td:first-child,#moduleDetails th:first-child{text-align:left}
#moduleTimeline{background:white;border:1px solid #e2e7ef;border-radius:8px;margin-top:12px;overflow:auto}#moduleTimeline svg{display:block;width:100%;min-width:620px;font:12px system-ui}
#moduleHelp{margin:8px 0;color:#556;font-size:12px;line-height:1.6}#moduleLatest{font-size:13px;min-height:22px;margin-top:6px}#moduleDetails button{font:12px monospace}
#workloadInfo{font-size:12px;color:#46546b;margin:6px 0;overflow-wrap:anywhere}
#computePanel{background:white;padding:12px;border:1px solid #e2e7ef;border-radius:8px;margin-top:8px;min-width:0;overflow:auto;grid-column:2;grid-row:1/3}#moduleHistory{grid-column:1;grid-row:2}
#computePanel h3{margin:0 0 6px;font-size:14px}#computeRound{max-width:310px}#computeSummary,#computeFootnote{font-size:12px;line-height:1.6;margin:6px 0}
#computeTimes{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}#computeTimes th,#computeTimes td{padding:5px 7px;border-bottom:1px solid #e2e7ef;text-align:right;white-space:nowrap}#computeTimes th:nth-child(2),#computeTimes td:nth-child(2){text-align:left}#computeTimes button{font:12px monospace;padding:0;border:0;background:none;color:#226fc9;text-decoration:underline;cursor:pointer}
#computeTimes tr[data-state=active]{background:#d7f5e3}#computeTimes tr[data-state=instant]{background:#fff2d5}#computeTimes tr[data-state=future]{color:#788396}
@media(max-width:900px){#moduleLayout{grid-template-columns:1fr}}
</style>
</head><body>
<header><b>Simthesizer · 动态图</b><span id="modes"><button id="modeModules" aria-pressed="true">模块总览</button><button id="modeNodes" aria-pressed="false">节点详情</button></span>
<div id="workloadInfo"></div>
<div id="moduleControls"><div><button id="modulePlay">播放</button><button id="moduleStart">起点</button><button id="modulePrev">上一时刻</button><button id="moduleNext">下一时刻</button><button id="moduleEnd">终点</button><select id="moduleSpeed" aria-label="播放速度"><option value="0.1">0.1×</option><option value="1" selected>1×</option><option value="4">4×</option></select></div><input id="moduleTime" aria-label="模拟时间" type="range" min="0" step="any" value="0"><span id="moduleStatus"></span><div id="moduleLatest"></div></div>
<div id="nodeControls" hidden><div id="key"><code># 节点 ID</code><code style="color:#226fc9">C 计算</code><code style="color:#8150b5">L 逻辑</code><code style="color:#18886b">N 通信</code><code>R 请求 ID</code><code>S 子调用索引</code></div>
<div id="overview"></div><p id="guide"></p>
<div><button id="play">播放</button><button id="back">上一步</button><button id="next">下一步</button><button id="end">全部历史</button><label><input id="follow" type="checkbox" checked>跟随事件</label><label><input id="live" type="checkbox">仅当前引擎图</label></div>
<input id="step" type="range" min="0"><span id="status"></span>
<div><input id="query" placeholder="搜索 ID、中文名称或算子"><select id="kind"><option value="">全部类型</option><option value="logical">逻辑：请求与调度</option><option value="compute">计算：模型算子</option><option value="network">通信：设备间传输</option></select><button id="prevPage">上一页</button><span id="page"></span><button id="nextPage">下一页</button></div>
</div></header>
<section id="moduleView"><div id="moduleExecution"><p id="moduleHelp">箭头表示执行流向（依赖），不是张量传输量。绿色高亮实际执行中的模块；零耗时事件只在对应时刻以橙色标出。模块会反复执行，累计完成数不表示只流转一次。</p><div id="moduleLayout"><div id="moduleGraph"></div><div id="moduleDetails"></div></div></div>
<section id="computePanel" hidden><h3>计算模块 · 算子时间明细</h3><div><label><input id="computeFollow" type="checkbox" checked>跟随模拟时间</label><button id="computePrev">上一轮</button><select id="computeRound" aria-label="计算轮次"></select><button id="computeNext">下一轮</button><button id="computeSeek">定位本轮开始</button></div><div id="computeSummary"></div><table id="computeTimes"><thead><tr><th>节点 ID</th><th>算子</th><th>开始 (s)</th><th>结束 (s)</th><th>耗时 (ms)</th><th>状态</th></tr></thead><tbody></tbody></table><div id="computeFootnote"></div><small>时间均为完整 trace 中的模拟时间；尚未执行的行显示后续记录。点击节点 ID 查看算子形状、batch 和 KV 原始参数。同名算子按节点 ID 分开列出。</small></section>
<div id="moduleHistory"><div id="moduleTimeline"></div><small>时间概览：每行一个模块，色深表示该时间段内启动的事件数，不表示耗时或利用率。点击时间格或拖动上方游标查看进度。</small></div></section>
<div id="nodeView" hidden>
<main><div id="diagram"></div><aside><div id="meaning">点击节点查看参数。</div><div id="links"></div><details id="legend" open><summary>节点图例</summary><div id="legendContent"></div></details><details><summary>原始记录</summary><pre id="detail"></pre></details><details><summary>阅读说明</summary><small>箭头：后继等待前驱。虚线：已移除依赖。创建者：动态生成该节点的事件，不一定是前驱。浅色／深色／灰色：未执行／执行中／已完成；橙边：选中。编号沿用 trace。分页或筛选不显示跨页连线，可点编号跳转；每页 80 个节点。“仅当前引擎图”在运行结束为空。滑块按图操作序号回放，详情是完整运行记录。计算节点对应整个 batch，FastForward 保留合并耗时。</small></details></aside></main>
</div><script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const nodes=new Map(data.nodes.map(n=>[n.id,n])), size=80;
let page=0, selected=null, timer=null;
const step=$('step');step.max=data.changes.length-1;step.value=step.max;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const logicalInfo = {
  RequestArrival: ['请求到达', '激活请求工作流。', 'RA'],
  SubRequestArrival: ['子请求就绪', '接入本次 LLM 或工具调用。', 'SA'],
  SubRequestComplete: ['子请求完成', '释放后续调用。', 'SC'],
  LlmEnqueue: ['LLM 入队', '加入调度队列。', 'EQ'],
  LlmSchedule: ['组建本轮 batch', '选择 batch，生成计算节点。', 'SCH'],
  LlmBatchComplete: ['本轮 batch 完成', '提交 token/KV 进度，触发下一轮。', 'BC'],
  BlockStart: ['block 开始', '记录 block 起始时间。', 'BS'],
  FastForward: ['其余 block 耗时', '复用一个 block 的耗时；×n 为剩余 block 数。', 'FF'],
  Barrier: ['通信同步点', '等待前置传输。', 'BAR'],
  ToolCallRun: ['工具调用', '计入工具耗时。', 'TOOL'],
};
const computeInfo = {
  embedding: ['词嵌入', '将 token 映射为隐藏向量。'],
  layernorm: ['层归一化', '对隐藏向量进行归一化。一个 block 中可出现多次。'],
  qkv_projection: ['Q/K/V 投影', '通过矩阵乘法生成注意力的 Query、Key 和 Value。'],
  rope: ['旋转位置编码', '向注意力向量注入位置信息。'],
  attention: ['注意力计算', '本轮 Query 与可见的 Key/Value 进行注意力计算。'],
  o_projection: ['注意力输出投影', '将注意力结果投影回隐藏维度。'],
  ffn1: ['前馈网络输入投影', '将隐藏向量投影到前馈网络的中间维度。'],
  act: ['激活函数', '执行前馈网络中的非线性变换。'],
  ffn2: ['前馈网络输出投影', '将前馈网络结果投影回隐藏维度。'],
  final_layernorm: ['最终归一化', '模型输出头之前的归一化。'],
  lm_head: ['词表投影', '将隐藏向量投影为词表上的分数。'],
  sampler: ['token 采样', '模拟根据词表分数选择输出 token 的开销。'],
};
const kindNames = {logical:'逻辑', compute:'计算', network:'通信'};
const kindCodes = {logical:'L', compute:'C', network:'N'};
function nodeName(n) {
  const raw=n.details.label.split('::').pop(), event=raw.split(/[ {]/)[0];
  return n.details.kind==='logical' ? (logicalInfo[event]?.[2] || event) : raw;
}
function nodeText(n) {
  const raw=n.details.label, request=raw.match(/\brequest_id: (\d+)/), sub=raw.match(/\bsubrequest_index: (\d+)/), remaining=raw.match(/remaining_blocks: (\d+)/);
  const identity=request?` R${request[1]}${sub?'/S'+sub[1]:''}`:'';
  return `#${n.id} ${kindCodes[n.details.kind]||n.details.kind} ${nodeName(n)}${identity}${remaining?' ×'+remaining[1]:''}`;
}
function explain(n) {
  const raw=n.details.label.split('::').pop(), event=raw.split(/[ {]/)[0];
  if(n.details.kind==='compute') return computeInfo[raw] || [raw, '对本轮 batch 执行该模型算子。具体维度见原始记录。'];
  if(n.details.kind==='network') return ['设备间传输', `设备 ${n.details.src} → ${n.details.dst}，传输 ${n.details.bytes} 字节。`];
  return logicalInfo[event] || [raw, '组件控制事件；负责状态更新、依赖衔接或动态生成工作。具体含义见事件名称和原始参数。'];
}
function caption(n) {
  const raw=n.details.label.split('::').pop(), request=raw.match(/\brequest_id: (\d+)/), sub=raw.match(/\bsubrequest_index: (\d+)/);
  const identity=request?` · 请求 ${request[1]}${sub?' / 调用 '+sub[1]:''}`:'';
  return `${explain(n)[0]}${identity}`;
}
function showMeaning(n) {
  const lines=[];
  if(n.details.batch){
    const batch=n.details.batch, sum=field=>batch.reduce((total,r)=>total+r[field],0);
    lines.push(`Batch ${batch.length} · Token ${sum('q_len')}`);
  }
  const start=n.dispatches[0]?.[1];
  if(start!==undefined) lines.push(`${start.toFixed(6)} → ${n.finish.toFixed(6)} s\n耗时 ${((n.finish-start)*1000).toFixed(4)} ms`);
  $('meaning').replaceChildren();
  const heading=document.createElement('h3');heading.textContent=nodeText(n);
  const description=document.createElement('div');description.textContent=lines.join('\n');$('meaning').append(heading,description);
}
function legend() {
  const entries=new Map();
  for(const n of data.nodes){
    const name=nodeName(n), key=`${kindCodes[n.details.kind]} ${name}`;
    if(!entries.has(key)) entries.set(key, explain(n));
  }
  const table=document.createElement('table');
  for(const [key,[title,meaning]] of entries){
    const row=table.insertRow(), code=document.createElement('code');code.textContent=key;
    row.insertCell().append(code);row.insertCell().textContent=title;row.title=meaning;
  }
  $('legendContent').append(table);
}
function overview() {
  for(const kind of ['', 'logical','compute','network']){
    const count=data.nodes.filter(n=>!kind||n.details.kind===kind).length;
    const button=document.createElement('button');button.dataset.kind=kind;
    button.textContent=`${kindNames[kind]||'全部'} ${count.toLocaleString()}`;
    button.onclick=()=>{step.value=step.max;$('live').checked=false;$('query').value='';$('kind').value=kind;page=0;selected=visible()[0]?.id??null;render();};
    $('overview').appendChild(button);
  }
  const firstDispatch=data.changes.find(c=>c[2]==='dispatch');
  const seeded=data.nodes.filter(n=>firstDispatch&&n.created<firstDispatch[0]).length;
  $('guide').textContent=`前 ${seeded} 个节点为预建节点；计算节点随后生成。点击类型筛选，点击节点查看参数。`;
}
function visible(){let seq=data.changes[+step.value][0],q=$('query').value.toLowerCase();return data.nodes.filter(n=>n.created<=seq&&(!$('live').checked||!(n.resolved<=seq))&&(!$('kind').value||n.details.kind===$('kind').value)&&(!q||(nodeText(n)+' '+caption(n)+' '+JSON.stringify(n.details)).toLowerCase().includes(q)));}
function jump(id){selected=id;$('query').value='';$('kind').value='';$('live').checked=false;if(nodes.get(id).created>data.changes[+step.value][0])step.value=data.changes.findIndex(c=>c[0]>=nodes.get(id).created);page=Math.floor(visible().findIndex(n=>n.id===id)/size);render();}
const textMeasure=document.createElement('canvas').getContext('2d');
textMeasure.font='12px monospace';
function render(){const [seq,time,event,id]=data.changes[+step.value], all=+step.value===+step.max;
let list=visible();page=Math.max(0,Math.min(page,Math.ceil(list.length/size)-1));let slice=list.slice(page*size,(page+1)*size),positions=new Map(slice.map((n,i)=>[n.id,24+i*36]));
const widths=new Map(slice.map(n=>[n.id,Math.ceil(textMeasure.measureText(nodeText(n)).width)+20]));
const svgWidth=Math.max(320,...Array.from(widths.values(),w=>w+110)),svgHeight=Math.max(120,slice.length*36+12);
$('page').textContent=`${page+1}/${Math.max(1,Math.ceil(list.length/size))} · ${list.length}/${data.nodes.length} 节点 · ${data.edges.length} 条历史依赖`;
$('status').textContent=`#${seq} t=${time.toFixed(6)}s ${event}`;
let svg=`<svg width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8a97ac"/></marker></defs>`;
for(const e of data.edges){if(e.added>seq||!positions.has(e.parent)||!positions.has(e.child)||($('live').checked&&e.removed!==null&&e.removed<=seq))continue;const y=positions.get(e.parent),z=positions.get(e.child),x=16+widths.get(e.parent),u=16+widths.get(e.child),bend=Math.max(x,u)+30+Math.min(40,Math.abs(y-z)/8),removed=e.removed!==null&&e.removed<=seq;svg+=`<path d="M ${x} ${y} C ${bend} ${y} ${bend} ${z} ${u} ${z}" fill="none" stroke="${removed?'#c19a75':'#8a97ac'}" ${removed?'stroke-dasharray="4 4"':''} marker-end="url(#arrow)"/>`;}
for(const n of slice){const y=positions.get(n.id),begun=n.dispatches.some(d=>d[0]<=seq),done=n.resolved<=seq&&(all||n.finish<=time),base={compute:'#226fc9',logical:'#8150b5',network:'#18886b'}[n.details.kind];const fill=done?'#e2e7ef':begun?base:'#f0f4ff',text=begun&&!done?'white':'#25324b';svg+=`<g data-node="${n.id}" style="cursor:pointer"><rect x="16" y="${y-13}" width="${widths.get(n.id)}" height="26" rx="4" fill="${fill}" stroke="${base}" class="${selected===n.id?'selected':''}"/><text x="26" y="${y+4}" fill="${text}">${esc(nodeText(n))}</text><title>${esc(explain(n)[0]+'：'+explain(n)[1]+'\n'+n.details.label)}</title></g>`;}
$('diagram').innerHTML=svg+'</svg>';
$('diagram').querySelectorAll('[data-node]').forEach(el=>el.onclick=()=>{selected=+el.dataset.node;render();});
$('links').replaceChildren();
if(selected!==null){let n=nodes.get(selected),incoming=data.edges.filter(e=>e.child===selected&&e.added<=seq),outgoing=data.edges.filter(e=>e.parent===selected&&e.added<=seq);showMeaning(n);$('detail').textContent=JSON.stringify(n,null,2);for(const [label,ids] of [['创建者',n.cause===null?[]:[n.cause]],['前驱（历史）',incoming.map(e=>e.parent)],['后继（历史）',outgoing.map(e=>e.child)]]){let div=document.createElement('div');div.textContent=label+': ';for(const id of new Set(ids)){let b=document.createElement('button');b.textContent='#'+id;b.title=caption(nodes.get(id));b.onclick=()=>jump(id);div.appendChild(b);}$('links').appendChild(div);}}
else{$('meaning').textContent=list.length?'点击节点查看它的作用。':'当前筛选下没有节点。';$('detail').textContent='';}
}
function move(delta){step.value=Math.max(0,Math.min(+step.max,+step.value+delta));const row=data.changes[+step.value];if($('follow').checked&&row[3]!=null&&nodes.has(row[3])){selected=row[3];$('query').value='';$('kind').value='';page=Math.floor(visible().findIndex(n=>n.id===selected)/size);}render();}
step.oninput=()=>move(0);$('back').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('end').onclick=()=>{step.value=step.max;render();};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';}else{if(+step.value===+step.max)step.value=0;timer=setInterval(()=>{move(1);if(+step.value===+step.max)$('play').click();},120);$('play').textContent='暂停';}};
$('prevPage').onclick=()=>{page--;render();};$('nextPage').onclick=()=>{page++;render();};for(const id of ['query','kind','live'])$(id).oninput=()=>{page=0;selected=visible()[0]?.id??null;render();};overview();legend();selected=visible()[0]?.id??null;render();
</script><script>__MODULE_SCRIPT__</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", type=Path, help="Workload used for this run (older traces do not store its path)")
    args = parser.parse_args()
    data = load_trace(args.trace)
    if args.workload:
        data["workload"] = workload_info(args.workload)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        modules = Path(__file__).with_name("graph_modules.js").read_text()
        stream.write(HTML.replace("__MODULE_SCRIPT__", modules).replace("__DATA__", encoded))
    print(
        f"{args.output}: {len(data['nodes'])} nodes, {len(data['edges'])} historical edges"
    )


if __name__ == "__main__":
    main()
