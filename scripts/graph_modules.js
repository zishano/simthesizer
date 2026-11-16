// Embedded by draw_graph.py; uses the original trace and node viewer helpers.
(() => {
  const definitions = {
    request: {title:'请求流程', color:'#287c70', x:20, y:30, help:'RA 到达 → SA 就绪 → EQ 入队；SC 在调用完成后释放。模块内节点总数不是请求数。'},
    schedule: {title:'调度 SCH', color:'#8150b5', x:370, y:30, help:'一次 SCH 是一次调度尝试。选到非空 batch 才生成本轮计算；没有选到时直接结束。'},
    compute: {title:'模型计算 C', color:'#226fc9', x:370, y:235, help:'相同算子跨 batch 的执行合并在此。每个原始 C 节点处理整个 batch；表中保留各算子次数。'},
    control: {title:'轮次控制', color:'#a56323', x:20, y:235, help:'BS 记录 block 起点；FF 计入其余 block 耗时；BC 提交进度并触发下一轮。它们穿插在计算中，并非一个连续执行步骤。'},
    network: {title:'通信 N', color:'#18886b', x:720, y:30, help:'汇总设备间传输。开始取首次 dispatch，结束取传输完成时间。'},
    other: {title:'其他逻辑', color:'#667085', x:720, y:235, help:'未归入上述流程的逻辑事件，保留其事件名称。'},
  };
  function groupKey(n) {
    if(n.details.kind==='compute') return 'compute';
    if(n.details.kind==='network') return 'network';
    const name=nodeName(n);
    if(['RA','SA','EQ','SC','TOOL'].includes(name)) return 'request';
    if(name==='SCH') return 'schedule';
    if(['BS','FF','BC','BAR'].includes(name)) return 'control';
    return 'other';
  }
  const groups = new Map(), membership = new Map();
  for(const n of data.nodes) {
    const key=groupKey(n);
    if(!groups.has(key)) groups.set(key, {...definitions[key], key, nodes:[], types:new Map()});
    const g=groups.get(key), name=nodeName(n);
    g.nodes.push(n);
    if(!g.types.has(name)) g.types.set(name, []);
    g.types.get(name).push(n);
    membership.set(n.id,key);
  }
  const starts=n=>n.dispatches[0][1];
  const sorted=values=>values.sort((a,b)=>a-b);
  for(const g of groups.values()) {
    if(g.key==='request') {
      const order=['RA','SA','EQ','SC','TOOL'];
      g.types=new Map([...g.types].sort((a,b)=>order.indexOf(a[0])-order.indexOf(b[0])));
    }
    g.starts=sorted(g.nodes.map(starts));
    g.finishes=sorted(g.nodes.map(n=>n.finish));
    g.zeroTimes=sorted(g.nodes.filter(n=>starts(n)===n.finish).map(starts));
    g.ordered=g.nodes.slice().sort((a,b)=>starts(a)-starts(b)||a.dispatches[0][0]-b.dispatches[0][0]);
  }
  // Upper/lower bounds make zero-duration events complete at their timestamp.
  function bound(values, value, inclusive=true) {
    let lo=0, hi=values.length;
    while(lo<hi) { const mid=(lo+hi)>>>1; if(values[mid]<value||(inclusive&&values[mid]===value)) lo=mid+1; else hi=mid; }
    return lo;
  }
  function stats(g,t) {
    const begun=bound(g.starts,t), done=bound(g.finishes,t);
    return {begun, done, active:begun-done, instant:bound(g.zeroTimes,t)-bound(g.zeroTimes,t,false)};
  }
  const transitions = new Map();
  for(const edge of data.edges) {
    // Withdrawn edges were rewiring placeholders, not execution constraints.
    if(edge.removed!==null) continue;
    const from=membership.get(edge.parent), to=membership.get(edge.child);
    const key=from+'>'+to;
    if(!transitions.has(key)) transitions.set(key,{from,to,count:0,starts:[],finishes:[],zeroTimes:[]});
    const transition=transitions.get(key), child=nodes.get(edge.child);
    transition.count++;transition.starts.push(starts(child));transition.finishes.push(child.finish);
    if(starts(child)===child.finish)transition.zeroTimes.push(starts(child));
  }
  for(const e of transitions.values()){sorted(e.starts);sorted(e.finishes);sorted(e.zeroTimes);}
  const dispatches=data.changes.filter(c=>c[2]==='dispatch');
  const dispatchTimes=dispatches.map(c=>c[1]);
  const boundaries=sorted([...new Set([0,...data.nodes.flatMap(n=>[starts(n),n.finish])])]);
  const end=boundaries.at(-1)||0;
  // Normalize the range so the browser cannot round down an exact end time.
  const slider=$('moduleTime'), sliderMax=1000000;
  slider.max=String(sliderMax); slider.disabled=end===0;
  let now=0, selectedGroup=groups.has('compute')?'compute':groups.keys().next().value, animation=null, previousWall=0;
  const workload=data.workload;
  $('workloadInfo').textContent=workload?`Workload：${workload.path} · ${workload.requests} 个请求 · ${workload.llm_calls} 次 LLM 调用 · 输入 ${workload.input_tokens} / 输出 ${workload.output_tokens} tokens`:'Workload：未提供（旧 trace 不含输入路径）';
  $('workloadInfo').title=workload?`来源：绘图时指定的 workload；旧 trace 未记录输入路径。SHA-256：${workload.sha256}`:'';
  function owner(n) {
    const visited=new Set([n.id]);let ancestor=nodes.get(n.cause);
    while(ancestor&&!visited.has(ancestor.id)) {
      if(nodeName(ancestor)==='SCH') return ancestor;
      visited.add(ancestor.id);ancestor=nodes.get(ancestor.cause);
    }
    return nodes.get(n.cause)||n;
  }
  const roundMap=new Map();
  for(const n of groups.get('compute')?.ordered||[]) {
    const parent=owner(n);
    if(!roundMap.has(parent.id))roundMap.set(parent.id,{parent,nodes:[],fast:[],start:starts(n)});
    roundMap.get(parent.id).nodes.push(n);
  }
  for(const n of data.nodes)if(nodeName(n)==='FF')roundMap.get(owner(n).id)?.fast.push(n);
  const rounds=[...roundMap.values()].sort((a,b)=>a.start-b.start||a.parent.id-b.parent.id);
  const roundStarts=rounds.map(r=>r.start);let roundIndex=0;
  rounds.forEach((r,i)=>{
    const option=document.createElement('option');option.value=i;
    option.textContent=`第 ${i+1} / ${rounds.length} 轮 · ${nodeName(r.parent)} #${r.parent.id}`;
    $('computeRound').append(option);
  });
  function renderCompute() {
    $('computePanel').hidden=selectedGroup!=='compute'||!rounds.length;
    if($('computePanel').hidden)return;
    if($('computeFollow').checked)roundIndex=Math.max(0,bound(roundStarts,now)-1);
    const r=rounds[roundIndex], batch=r.nodes[0].details.batch||[];
    $('computeRound').value=roundIndex;
    $('computePrev').disabled=roundIndex===0;$('computeNext').disabled=roundIndex===rounds.length-1;
    $('computeSummary').textContent=`${nodeName(r.parent)} #${r.parent.id} · Batch ${batch.length} · 本轮输入 token ${batch.reduce((s,b)=>s+b.q_len,0)} · 已有 KV token ${batch.reduce((s,b)=>s+b.kv_len,0)} · ${r.nodes.length} 次算子执行`;
    const body=$('computeTimes').tBodies[0];body.replaceChildren();
    for(const n of r.nodes) {
      const start=starts(n), duration=(n.finish-start)*1000;
      const state=now<start?'future':now<n.finish?'active':now===start&&start===n.finish?'instant':'done';
      const row=body.insertRow();row.dataset.computeNode=n.id;row.dataset.state=state;
      row.dataset.start=start;row.dataset.finish=n.finish;row.dataset.duration=duration;
      const button=document.createElement('button');button.textContent='#'+n.id;button.onclick=()=>goNode(n.id);row.insertCell().append(button);
      for(const value of [nodeName(n),start.toFixed(9),n.finish.toFixed(9),duration.toFixed(6),{future:'未开始',active:'执行中',instant:'瞬时完成',done:'已完成'}[state]])row.insertCell().textContent=value;
      row.title=n.details.layer||'';
    }
    const sum=r.nodes.reduce((s,n)=>s+(n.finish-starts(n))*1000,0);
    $('computeFootnote').textContent=`C 算子耗时合计 ${sum.toFixed(6)} ms。`+r.fast.map(n=>` FF ${nodeText(n)}：${starts(n).toFixed(9)} → ${n.finish.toFixed(9)} s，耗时 ${((n.finish-starts(n))*1000).toFixed(6)} ms（其余 block 的快进时间，不计入 C 合计）。`).join('');
  }
  function selectRound(index){$('computeFollow').checked=false;roundIndex=index;renderCompute();}
  $('computeRound').onchange=()=>selectRound(Number($('computeRound').value));
  $('computePrev').onclick=()=>selectRound(Math.max(0,roundIndex-1));
  $('computeNext').onclick=()=>selectRound(Math.min(rounds.length-1,roundIndex+1));
  $('computeFollow').onchange=()=>renderCompute();
  $('computeSeek').onclick=()=>{stop();$('computeFollow').checked=false;seek(rounds[roundIndex].start);};
  const boxWidth=210, boxHeight=106;
  const graphWidth=groups.has('network')||groups.has('other')?950:620;
  const number=n=>n.toLocaleString();
  function goNode(id) {
    setMode(false);
    step.value=step.max;
    jump(id);
  }
  function dependencyRoute(from,to) {
    const a=groups.get(from), b=groups.get(to);
    // Route the four main flows around the module boxes, without crossing them.
    if(from==='request'&&to==='schedule') return {path:'M 230 65 L 370 65',x:300,y:49,lines:['EQ → SCH','请求入队']};
    if(from==='schedule'&&to==='compute') return {path:'M 475 136 L 475 235',x:525,y:178,lines:['SCH → C','非空 batch']};
    if(from==='compute'&&to==='control') return {path:'M 370 275 L 230 275',x:300,y:259,lines:['C → BS / FF / BC']};
    if(from==='control'&&to==='compute') return {path:'M 230 315 L 370 315',x:300,y:334,lines:['BS / FF → C','继续本轮计算']};
    if(from==='control'&&to==='schedule') return {path:'M 230 245 C 295 245 295 108 370 108',x:300,y:174,lines:['BC → SCH','下一轮调度']};
    if(from==='control'&&to==='request') return {path:'M 125 235 L 125 136',x:65,y:177,lines:['BC → SC','请求完成']};
    const x=a.x+boxWidth/2, y=a.y+boxHeight/2, u=b.x+boxWidth/2, v=b.y+boxHeight/2;
    if(a.x===b.x) return {path:`M ${x} ${a.y+(y<v?boxHeight:0)} L ${u} ${b.y+(y<v?0:boxHeight)}`,x:x+45,y:(y+v)/2,lines:[a.title+' → '+b.title]};
    const startX=a.x+(x<u?boxWidth:0), finishX=b.x+(x<u?0:boxWidth), mid=(startX+finishX)/2;
    return {path:`M ${startX} ${y} C ${mid} ${y} ${mid} ${v} ${finishX} ${v}`,x:mid,y:(y+v)/2-10,lines:[a.title+' → '+b.title]};
  }
  function renderGraph() {
    let svg=`<svg width="${graphWidth}" height="390" viewBox="0 0 ${graphWidth} 390" role="img" aria-label="合并后的模块依赖图"><defs>`;
    for(const [name,color] of [['idle','#93a1b5'],['active','#10934d'],['instant','#bf7300']])svg+=`<marker id="moduleArrow-${name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="${color}"/></marker>`;
    svg+='</defs>';
    for(const e of transitions.values()) {
      if(e.from===e.to) continue;
      const s=stats(e,now), state=s.active?'active':s.instant?'instant':'idle', color={active:'#10934d',instant:'#bf7300',idle:'#93a1b5'}[state], route=dependencyRoute(e.from,e.to);
      svg+=`<path data-module-edge="${e.from}>${e.to}" data-flow-state="${state}" d="${route.path}" fill="none" stroke="${color}" stroke-width="${state==='idle'?1.5:3}" marker-end="url(#moduleArrow-${state})"><title>${esc(groups.get(e.from).title+' → '+groups.get(e.to).title+'：'+number(e.count)+' 条历史依赖')}</title></path>`;
      for(const [i,line] of route.lines.entries())svg+=`<text data-flow-label="${e.from}>${e.to}" x="${route.x}" y="${route.y+i*14}" text-anchor="middle" font-size="10" fill="${color}" stroke="white" stroke-width="4" paint-order="stroke">${esc(line)}</text>`;
    }
    for(const g of groups.values()) {
      const s=stats(g,now), state=s.active?'active':s.instant?'instant':'idle';
      const fill={active:'#d7f5e3',instant:'#fff2d5',idle:'#ffffff'}[state], stroke={active:'#10934d',instant:'#bf7300',idle:g.color}[state];
      const subtype=g.key==='compute'?`${g.types.size} 种算子 · ${number(g.nodes.length)} 次执行`:
        [...g.types].map(([name,ns])=>`${name} ${number(ns.length)}`).join(' · ');
      const status=s.active?`▶ 执行中 ${s.active}`:s.instant?`◆ 本时刻瞬时事件 ${s.instant}`:s.done===g.nodes.length?'已全部完成':s.begun?'等待后续执行':'尚未开始';
      svg+=`<g data-module="${g.key}" data-total="${g.nodes.length}" data-done="${s.done}" data-active="${s.active}" data-state="${state}" style="cursor:pointer" role="button" tabindex="0" aria-label="${esc(g.title)}">
        ${g.key===selectedGroup?`<rect x="${g.x-4}" y="${g.y-4}" width="${boxWidth+8}" height="${boxHeight+8}" rx="8" fill="none" stroke="#4267a0" stroke-dasharray="3 3"/>`:''}
        <rect data-module-card="${g.key}" x="${g.x}" y="${g.y}" width="${boxWidth}" height="${boxHeight}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="${state==='idle'?1.3:3}"/>
        <text x="${g.x+10}" y="${g.y+19}" fill="${g.color}" font-size="13" font-weight="600">${esc(g.title)}</text>
        <text x="${g.x+10}" y="${g.y+38}" font-size="10">${esc(subtype)}</text>
        <text x="${g.x+10}" y="${g.y+59}" font-size="11">完成 ${number(s.done)} / ${number(g.nodes.length)}</text>
        <text x="${g.x+10}" y="${g.y+79}" fill="${state==='idle'?'#667085':stroke}" font-size="11" font-weight="${state==='idle'?400:700}">${status}</text>
        <rect x="${g.x+10}" y="${g.y+92}" width="190" height="4" rx="2" fill="#e8edf4"/>
        <rect x="${g.x+10}" y="${g.y+92}" width="${190*s.done/g.nodes.length}" height="4" rx="2" fill="${g.color}"/>
        <title>${esc(g.help)}</title></g>`;
    }
    svg+='<text x="20" y="380" fill="#667085" font-size="11">绿色：执行中及其进入路径 · 橙色：本时刻瞬时事件 · 虚线框：查看详情</text></svg>';
    $('moduleGraph').innerHTML=svg;
    for(const element of $('moduleGraph').querySelectorAll('[data-module]')) {
      const select=()=>{selectedGroup=element.dataset.module;renderModules();};
      element.onclick=select; element.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
    }
  }
  function renderDetails() {
    const g=groups.get(selectedGroup), panel=$('moduleDetails'); panel.replaceChildren();
    if(!g) return;
    const heading=document.createElement('h3');heading.textContent=g.title;
    const help=document.createElement('p');help.textContent=g.help;
    const table=document.createElement('table');
    table.innerHTML='<thead><tr><th>事件 / 算子</th><th>总数</th><th>完成</th><th>执行中</th></tr></thead>';
    const body=table.createTBody();
    for(const [name,ns] of g.types) {
      const row=body.insertRow(), done=ns.filter(n=>n.finish<=now).length, active=ns.filter(n=>starts(n)<=now&&now<n.finish).length;
      for(const value of [name,number(ns.length),number(done),number(active)]) row.insertCell().textContent=value;
    }
    const internal=document.createElement('p'); internal.textContent=`模块内依赖 ${number(transitions.get(g.key+'>'+g.key)?.count||0)} 条（图中省略）`;
    panel.append(heading,help,table,internal);
    const edgeInfo=document.createElement('small');
    edgeInfo.textContent=[...transitions.values()].filter(e=>e.from===g.key&&e.to!==g.key).map(e=>`→ ${groups.get(e.to).title}：${number(e.count)}`).join('；');panel.append(edgeInfo);
    const label=document.createElement('p');label.textContent='最近启动的事件（点击查看原始节点）';panel.append(label);
    const recent=g.ordered.slice(0,bound(g.starts,now)).slice(-4).reverse();
    if(!recent.length){const empty=document.createElement('small');empty.textContent='尚未启动';panel.append(empty);}
    for(const n of recent) {
      const button=document.createElement('button');button.textContent=nodeText(n);button.title=`${starts(n).toFixed(6)} → ${n.finish.toFixed(6)} s`;
      button.onclick=()=>goNode(n.id);panel.append(button);
    }
  }
  const bins=160, timelineLeft=110, timelineWidth=800;
  function buildTimeline() {
    let svg=`<svg viewBox="0 0 950 ${groups.size*35+65}" aria-label="模块启动次数时间分布">`;
    let row=0;
    for(const g of groups.values()) {
      const counts=Array(bins).fill(0);
      for(const start of g.starts) counts[Math.min(bins-1,end?Math.floor(start/end*bins):0)]++;
      const max=Math.max(1,...counts), y=25+row*35;
      svg+=`<text x="10" y="${y+15}" fill="${g.color}">${esc(g.title)}</text>`;
      counts.forEach((count,i)=>{
        svg+=`<rect data-time-bin="${i}" x="${timelineLeft+i*timelineWidth/bins}" y="${y}" width="${timelineWidth/bins}" height="22" fill="${g.color}" opacity="${count?0.18+0.82*count/max:0.04}" style="cursor:pointer"><title>${(i/bins*end).toFixed(5)}–${((i+1)/bins*end).toFixed(5)} s：启动 ${count} 次</title></rect>`;
      });row++;
    }
    const axisY=groups.size*35+40;
    for(let i=0;i<=4;i++) svg+=`<text x="${timelineLeft+i*timelineWidth/4}" y="${axisY}" text-anchor="middle" fill="#667085">${(end*i/4).toFixed(2)} s</text>`;
    svg+=`<line id="moduleCursor" x1="${timelineLeft}" x2="${timelineLeft}" y1="16" y2="${axisY-15}" stroke="#e64b22" stroke-width="2" pointer-events="none"/></svg>`;
    $('moduleTimeline').innerHTML=svg;
    for(const cell of $('moduleTimeline').querySelectorAll('[data-time-bin]')) cell.onclick=()=>seek((Number(cell.dataset.timeBin)+0.5)/bins*end);
  }
  function renderModules() {
    const latest=dispatches[bound(dispatchTimes,now)-1];
    const complete=[...groups.values()].reduce((sum,g)=>sum+stats(g,now).done,0);
    $('moduleStatus').textContent=`${now.toFixed(6)} / ${end.toFixed(6)} s · 完成 ${number(complete)} / ${number(data.nodes.length)}`;
    const active=[];
    for(const g of groups.values())if(stats(g,now).active){
      for(let i=bound(g.starts,now)-1;i>=0;i--)if(g.ordered[i].finish>now){active.push(`${g.title} · ${nodeText(g.ordered[i])}`);break;}
    }
    $('moduleLatest').textContent=active.length?'执行中：'+active.join('；'):now>=end?'运行结束':latest?`本时刻无持续执行 · 最近事件 ${nodeText(nodes.get(latest[3]))}`:'等待首个请求到达';
    renderGraph();renderDetails();renderCompute();
    const x=timelineLeft+(end?now/end:0)*timelineWidth;
    $('moduleCursor').setAttribute('x1',x);$('moduleCursor').setAttribute('x2',x);
  }
  function stop(){if(animation!==null)cancelAnimationFrame(animation);animation=null;$('modulePlay').textContent='播放';}
  function seek(time) {now=Math.max(0,Math.min(end,time));slider.value=String(end?now/end*sliderMax:0);renderModules();}
  function tick(wall) {
    const elapsed=(wall-previousWall)/1000;previousWall=wall;
    seek(now+elapsed*Number($('moduleSpeed').value));
    if(now>=end) stop(); else animation=requestAnimationFrame(tick);
  }
  function setMode(modules) {
    stop();if(timer!==null){clearInterval(timer);timer=null;$('play').textContent='播放';}
    $('moduleView').hidden=!modules;$('moduleControls').hidden=!modules;
    $('nodeView').hidden=modules;$('nodeControls').hidden=modules;
    $('modeModules').setAttribute('aria-pressed',String(modules));$('modeNodes').setAttribute('aria-pressed',String(!modules));
  }
  $('modeModules').onclick=()=>setMode(true);$('modeNodes').onclick=()=>setMode(false);
  slider.oninput=()=>{stop();seek(Number(slider.value)/sliderMax*end);};
  $('modulePlay').onclick=()=>{if(animation!==null){stop();return;}if(now>=end)seek(0);previousWall=performance.now();$('modulePlay').textContent='暂停';animation=requestAnimationFrame(tick);};
  $('moduleStart').onclick=()=>{stop();seek(0);};$('moduleEnd').onclick=()=>{stop();seek(end);};
  $('modulePrev').onclick=()=>{stop();seek(boundaries[Math.max(0,bound(boundaries,now,false)-1)]);};
  $('moduleNext').onclick=()=>{stop();seek(boundaries[Math.min(boundaries.length-1,bound(boundaries,now))]);};
  buildTimeline();setMode(true);seek(0);
})();
