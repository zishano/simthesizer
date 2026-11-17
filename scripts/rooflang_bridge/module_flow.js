// Indexed replay of existing observations. No simulated cost is recomputed here.
const FLOW_LABELS = {
  arrivals:'请求到达', p_schedule:'P · 排队 / 调度', p_control:'P · 轮次控制',
  p_compute:'P · 计算 / TP', p_kv:'P 端 KV', transfer:'KV 分片传输',
  d_schedule:'D · 排队 / 调度', d_control:'D · 轮次控制',
  d_compute:'D · 计算 / TP', d_kv:'D 端 KV', complete:'完成 / 释放',
  unassigned:'未归属逻辑'
};

function flowBound(list, t, key, inclusive=true) {
  let lo=0,hi=list.length;
  while(lo<hi){const mid=(lo+hi)>>1,v=key(list[mid]);if(v<t||(inclusive&&v===t))lo=mid+1;else hi=mid;}
  return lo;
}

class FlowIntervalIndex {
  constructor(items) {
    const tie=(a,b)=>(a.id>b.id)-(a.id<b.id);
    this.starts=[...items].sort((a,b)=>a.start-b.start||tie(a,b));
    this.ends=[...items].sort((a,b)=>a.finish-b.finish||tie(a,b));
    this.instants=this.starts.filter(n=>n.start===n.finish);
    this.active=new Set();this.startCursor=0;this.endCursor=0;
  }
  at(t) {
    const start=flowBound(this.starts,t,n=>n.start),end=flowBound(this.ends,t,n=>n.finish);
    while(this.startCursor>start)this.active.delete(this.starts[--this.startCursor]);
    while(this.endCursor>end){const n=this.ends[--this.endCursor];if(n.start<=t&&n.finish>t)this.active.add(n);}
    while(this.startCursor<start){const n=this.starts[this.startCursor++];if(n.finish>t)this.active.add(n);}
    while(this.endCursor<end)this.active.delete(this.ends[this.endCursor++]);
    const a=flowBound(this.instants,t,n=>n.start,false),b=flowBound(this.instants,t,n=>n.start);
    return {active:this.active,done:end,total:this.ends.length,pulse:this.instants.slice(a,b),latest:this.ends[end-1]};
  }
}

class ModuleFlowModel {
  constructor(data) {
    this.data=data;this.groups={};this.rounds={};this.admission={};
    const groups=Object.fromEntries(Object.keys(FLOW_LABELS).map(k=>[k,[]]));
    for(const n of data.nodes)(groups[n.view.flow]||groups.unassigned).push(n);
    for(const [key,items] of Object.entries(groups))this.groups[key]=new FlowIntervalIndex(items);
    for(const side of ['p','d']){
      const rows=data.rounds.filter(r=>r.side===side).map(r=>({...r,id:r.key}));
      this.rounds[side]=new FlowIntervalIndex(rows);
      this.admission[side]=new FlowIntervalIndex(rows.filter(r=>r.start>r.queued).map(r=>({...r,start:r.queued,finish:r.start})));
    }
    this.kvWait=new FlowIntervalIndex(data.nodes.filter(n=>n.view.kv_wait).map(n=>({id:n.id,start:n.view.kv_wait[0],finish:n.view.kv_wait[1],node:n})));
    this.arrivals=[...data.requests].sort((a,b)=>a.arrival-b.arrival);
    this.finishes=[...data.requests].sort((a,b)=>a.finish-b.finish);
    this.copyBytes=groups.transfer.reduce((sum,n)=>sum+(n.details.work.details.bytes||0),0);
    this.kvEvents=[...data.kv].sort((a,b)=>a.time_s-b.time_s||a.seq-b.seq);
    this.kvSeries=[];
    // Each row is a small aggregate, never a snapshot of all live blocks.
    const live=new Map(),s={p:0,d:0,flight:0,pBlocks:0,dBlocks:0,flightBlocks:0,received:0,receivedCopies:0,pReleased:0,dReleased:0,appended:0};
    const account=(block,sign)=>{
      if(!block)return;
      const key=block.state==='in_flight'?'flight':block.prefill?'p':'d';
      s[key]+=sign*block.bytes;s[key+'Blocks']+=sign;
    };
    for(const e of this.kvEvents){
      const old=live.get(e.block.id);account(old,-1);
      if(e.transition==='release'){live.delete(e.block.id);s[e.block.prefill?'pReleased':'dReleased']++;}
      else{live.set(e.block.id,e.block);account(e.block,1);}
      if(e.transition==='receive'){s.received+=e.block.bytes;s.receivedCopies++;}
      if(e.transition==='extend')s.appended+=e.block.bytes-(old?.bytes||0);
      const row={time:e.time_s,...s};
      if(this.kvSeries.at(-1)?.time===e.time_s)this.kvSeries[this.kvSeries.length-1]=row;else this.kvSeries.push(row);
    }
    this.emptyKv={time:0,p:0,d:0,flight:0,pBlocks:0,dBlocks:0,flightBlocks:0,received:0,receivedCopies:0,pReleased:0,dReleased:0,appended:0};
  }
  at(t) {
    const groups=Object.fromEntries(Object.entries(this.groups).map(([k,index])=>[k,index.at(t)]));
    const rounds={},admission={},queues={};
    const snapshot=this.data.states[flowBound(this.data.states,t,s=>s.time_s)-1]?.state;
    for(const side of ['p','d']){
      rounds[side]=this.rounds[side].at(t);admission[side]=this.admission[side].at(t);
      const pool=snapshot?.[side==='p'?'prefill_instances':'decode_instances'];
      queues[side]=pool?.every(i=>i.scheduler)?{
        waiting:pool.reduce((sum,i)=>sum+i.scheduler.waiting_requests,0),
        admitted:pool.reduce((sum,i)=>sum+i.scheduler.running_requests,0)
      }:null;
    }
    const a=flowBound(this.kvEvents,t,e=>e.time_s,false),b=flowBound(this.kvEvents,t,e=>e.time_s);
    return {
      t,groups,rounds,admission,queues,kvWait:this.kvWait.at(t),
      kv:this.kvSeries[flowBound(this.kvSeries,t,s=>s.time)-1]||this.emptyKv,
      kvPulse:this.kvEvents.slice(a,b),
      arrived:flowBound(this.arrivals,t,r=>r.arrival),completed:flowBound(this.finishes,t,r=>r.finish),
      arrivalsNow:this.arrivals.slice(flowBound(this.arrivals,t,r=>r.arrival,false),flowBound(this.arrivals,t,r=>r.arrival)),
      finishesNow:this.finishes.slice(flowBound(this.finishes,t,r=>r.finish,false),flowBound(this.finishes,t,r=>r.finish))
    };
  }
}

function toggleModuleSize() {
  const on=$('modules-panel').classList.toggle('module-expanded');
  $('moduleExpand').textContent=on?'收起':'展开';
  $('moduleExpand').setAttribute('aria-pressed',String(on));
}

function renderModules() {
  if($('modules-panel').hidden)return;
  const s=flowModel.at(time()),g=s.groups,kv=s.kv;
  const positions={arrivals:[6,8],p_schedule:[180,8],p_control:[354,8],p_compute:[528,8],p_kv:[702,8],
    unassigned:[528,91],transfer:[702,91],complete:[6,174],d_schedule:[180,174],d_control:[354,174],d_compute:[528,174],d_kv:[702,174]};
  const cards={},pulse=(key,event)=>g[key].pulse.some(n=>!event||n.view.flow_event===event);
  const nextRound=side=>g[side+'_schedule'].pulse.some(n=>n.view.flow_event==='LlmSchedule'&&n.view.round!==null&&nodes.get(n.cause)?.view.flow_event==='LlmBatchComplete');
  const summarizeRound=side=>{
    const running=[...s.rounds[side].active];
    return running.length?running.map(r=>`${r.instance} 轮${r.round} B${r.batch.length}`).join(' / '):s.admission[side].active.size?'等待显存准入':'当前无轮次';
  };
  for(const key of Object.keys(positions)){
    const group=g[key],side=key[0],on=group.active.size>0;
    const card={key,title:FLOW_LABELS[key],lines:[],state:on?'running':group.pulse.length?'pulse':group.done?'idle':'future',pulse:group.pulse.length>0,done:group.done,total:group.total};
    if(key.endsWith('_schedule')){
      const q=s.queues[side];
      card.lines=[`队列 ${q?.waiting??'—'} · 已接纳 ${q?.admitted??'—'}`,`逻辑 ${group.done} / ${group.total}`];
      if(q?.waiting&&!card.pulse)card.state='waiting';
      card.extra=data.instances.filter(i=>i.side===side).map(i=>`${i.id} · ${i.gpu_model} · TP=${i.tp}`).join('；');
    }else if(key.endsWith('_control')){
      const r=s.rounds[side],wait=s.admission[side].active.size;
      card.lines=[summarizeRound(side),`完成 ${r.done}/${r.total} 轮 · 准入等待 ${wait}`];
      card.done=r.done;card.total=r.total;
      card.state=wait?'waiting':card.pulse?'pulse':r.active.size?'in_round':card.state;
    }else if(key.endsWith('_compute')){
      const running=[...group.active],tp=running.filter(n=>n.view.category==='tp').length;
      const layers=[...new Set(running.map(n=>n.view.layer).filter(l=>l!==null))];
      card.lines=[`运行 ${running.length} · TP ${tp}${layers.length?' · L'+layers.join('/'):''}`,`完成 ${group.done}/${group.total}${side==='d'&&s.kvWait.active.size?' · 等KV '+s.kvWait.active.size:''}`];
      card.extra=running.slice(0,8).map(n=>`#${n.id} ${n.view.instance} ${n.view.name}`).join('\n');
      if(!on&&side==='d'&&s.kvWait.active.size)card.state='waiting';
    }else if(key.endsWith('_kv')){
      const events=s.kvPulse.filter(e=>(e.block.prefill?'p':'d')===side);
      const written=events.filter(e=>['produce','extend'].includes(e.transition)).length,released=events.filter(e=>e.transition==='release').length;
      card.lines=[`就绪 ${bytes(kv[side])} · ${kv[side+'Blocks']} 块`,`本刻写入 ${written} · 释放 ${released}`];
      card.total=0;card.pulse=card.pulse||events.length>0;
      card.state=card.pulse?'pulse':kv[side]>0?'ready':'idle';
    }else if(key==='arrivals'){
      card.lines=[`已到达 ${s.arrived} / ${data.requests.length}`,`服务中 ${s.arrived-s.completed}`];
      card.done=s.arrived;card.total=data.requests.length;card.pulse=s.arrivalsNow.length>0;card.state=card.pulse?'pulse':'idle';
    }else if(key==='complete'){
      card.lines=[`已完成 ${s.completed} / ${data.requests.length}`,`已释放块 P ${kv.pReleased} · D ${kv.dReleased}`];
      card.done=s.completed;card.total=data.requests.length;card.pulse=card.pulse||s.finishesNow.length>0;card.state=card.pulse?'pulse':'idle';
    }else if(key==='transfer'){
      card.lines=[`在传 ${group.active.size} · 确认 ${kv.receivedCopies}/${group.total}`,`${bytes(kv.received)} / ${bytes(flowModel.copyBytes)}`];
      card.done=kv.receivedCopies;card.total=group.total;
      card.pulse=s.kvPulse.some(e=>e.transition==='receive');
      card.extra=`已提交未确认 ${bytes(kv.flight)}；按接收确认计数，不按时间估算进度。\n`+[...group.active].slice(0,8).map(n=>{const d=n.details.work.details;return `#${n.id} ${gpu(d.src)} → ${gpu(d.dst)} · L${d.layer} · ${bytes(d.bytes)}`;}).join('\n');
      if(!on&&kv.flightBlocks)card.state='waiting';
      else if(!on&&card.pulse)card.state='pulse';
    }else card.lines=[`完成 ${group.done} / ${group.total}`,`归属不明确，保留原始节点`];
    if((key.startsWith('d_')||key==='transfer')&&!data.requests.some(r=>r.output>1)){
      card.state='unused';card.pulse=false;card.lines=['本次无需 Decode','请求在 P 阶段完成'];
    }
    cards[key]=card;
  }
  const edges=[
    ['arrival','M154 35 H180','control',s.arrivalsNow.length?'pulse':''],
    ['p_batch','M328 35 H354','control',s.admission.p.active.size?'waiting':pulse('p_control','RooflangRoundBegin')?'pulse':''],
    ['p_run','M502 35 H528','dependency',g.p_compute.active.size?'running':''],
    ['p_return','M547 62 V77 H447 V62','control',pulse('p_control','LlmBatchComplete')?'pulse':'','轮次完成',495,87],
    ['p_next','M382 62 V94 H254 V62','control',nextRound('p')?'pulse':'','继续输入',316,106],
    ['p_write','M676 35 H702','data',s.kvPulse.some(e=>e.block.prefill&&e.transition==='produce')?'pulse':''],
    ['copy','M776 62 V91','data',g.transfer.active.size?'running':'',data.meta.mode==='layer'?'逐层发起':'P完成后发起',810,81],
    ['receive','M776 145 V174','data',s.kvPulse.some(e=>e.transition==='receive')?'pulse':'','接收确认',810,164],
    ['handoff','M428 62 V125 H254 V174','control',pulse('d_schedule','RequestArrival')?'pulse':'','首 token 后调度 D',450,127],
    ['d_batch','M328 201 H354','control',s.admission.d.active.size?'waiting':pulse('d_control','RooflangRoundBegin')?'pulse':''],
    ['d_run','M502 201 H528','dependency',g.d_compute.active.size?'running':''],
    ['kv_read','M702 190 H676','dependency',g.d_compute.active.size&&[...g.d_compute.active].some(n=>/attention/i.test(n.view.name))?'running':s.kvWait.active.size?'waiting':'','按层可读',689,169],
    ['kv_append','M676 215 H702','data',s.kvPulse.some(e=>!e.block.prefill&&['produce','extend'].includes(e.transition))?'pulse':''],
    ['d_return','M576 174 V151 H475 V174','control',pulse('d_control','LlmBatchComplete')?'pulse':'','轮次完成',530,148],
    ['d_next','M382 174 V144 H288 V174','control',nextRound('d')?'pulse':'','下一 token',335,141],
    ['done','M354 215 H154','control',g.complete.pulse.some(n=>n.view.module==='d')?'pulse':'','输出完成',254,235],
    ['no_decode','M399 62 V113 H80 V174','control',g.complete.pulse.some(n=>n.view.module==='p')?'pulse':'','无 D 阶段则直接完成',183,122]
  ];
  let svg='<svg viewBox="0 0 860 242" role="img" aria-label="P/D 调度、计算与 KV 数据流"><defs>';
  for(const [key,c] of Object.entries({idle:'#a6b3c3',data:'#1d8e91',running:'#16804b',pulse:'#d68a10',waiting:'#c47b15'}))svg+=`<marker id="moduleArrow-${key}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="${c}"/></marker>`;
  svg+='</defs>';
  for(const [id,path,type,state,label,x,y] of edges){
    const style=state||(type==='data'?'data':'idle');
    svg+=`<g data-flow-edge="${id}" data-state="${state||'idle'}" class="module-edge ${type} ${state}"><path d="${path}" marker-end="url(#moduleArrow-${style})"/>${label?`<text x="${x}" y="${y}" text-anchor="middle">${esc(label)}</text>`:''}</g>`;
  }
  const selectedModule=nodes.get(selected)?.view.flow;
  const short=text=>text.length>23?text.slice(0,22)+'…':text;
  for(const [key,[x,y]] of Object.entries(positions)){
    const c=cards[key];if(key==='unassigned'&&!c.total)continue;
    const stateText={running:'执行中',in_round:'轮次中',pulse:'本刻事件',waiting:'等待',ready:'已就绪',idle:'空闲',future:'未执行',unused:'未使用'}[c.state];
    svg+=`<g class="module-card" data-flow-module="${key}" data-state="${c.state}" data-active="${c.state==='running'}" data-pulse="${c.pulse}" data-done="${c.done}" data-total="${c.total}" tabindex="0" role="button" aria-label="${esc(c.title+'，'+stateText+'，'+c.lines.join('，'))}" transform="translate(${x} ${y})"><title>${esc([c.title,stateText,...c.lines,c.extra||'','点击查看相关节点；时间位置保持不变。'].join('\n'))}</title><rect class="card-bg" width="148" height="54" rx="5"/>${selectedModule===key?'<rect class="module-selected" x="-3" y="-3" width="154" height="60" rx="7"/>':''}<text class="module-title" x="7" y="14">${esc(c.title)}</text><text class="module-status" x="142" y="13" text-anchor="end">${esc(stateText)}</text><text x="7" y="28">${esc(short(c.lines[0]))}</text><text x="7" y="41">${esc(short(c.lines[1]))}</text>`;
    if(c.total)svg+=`<rect x="7" y="47" width="134" height="3" rx="1" fill="#e3eaf3"/><rect x="7" y="47" width="${134*Math.min(1,c.done/c.total)}" height="3" rx="1" fill="${c.state==='running'?'#16804b':'#8ea9c6'}"/>`;
    svg+='</g>';
  }
  svg+='</svg>';
  $('moduleFlow').innerHTML=svg;
  $('moduleClock').textContent=`+${ms(s.t)} ms · ${data.meta.mode==='layer'?'逐层交接':'整轮交接'}`;
  $('moduleFlow').querySelectorAll('[data-flow-module]').forEach(el=>{
    const chooseModule=()=>{
      const key=el.dataset.flowModule;
      if($('modules-panel').classList.contains('module-expanded'))toggleModuleSize();
      for(const id of ['query','kind','instance','request','layer'])$(id).value='';
      $('moduleFilter').value=key;page=0;
      const list=filtered(),id=currentNode(list)??list[0]?.id??null;
      selectNode(id,list);render();
    };
    el.onclick=chooseModule;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();chooseModule();}};
  });
}

if(typeof module!=='undefined'&&module.exports)module.exports={FlowIntervalIndex,ModuleFlowModel};
