// Run with Node; optional HTML arguments check indexed replay against native data.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {FlowIntervalIndex,ModuleFlowModel}=require('../scripts/rooflang_bridge/module_flow.js');

const intervals=[{id:1,start:1,finish:4},{id:2,start:2,finish:3},{id:3,start:2,finish:2},{id:4,start:4,finish:6}];
const index=new FlowIntervalIndex(intervals);
for(const t of [0,2,4,6,3,1,5,2,6,0]){
  const state=index.at(t);
  assert.deepEqual([...state.active].map(n=>n.id).sort(),intervals.filter(n=>n.start<=t&&t<n.finish).map(n=>n.id).sort());
  assert.equal(state.done,intervals.filter(n=>n.finish<=t).length);
  assert.deepEqual(state.pulse.map(n=>n.id),intervals.filter(n=>n.start===t&&n.finish===t).map(n=>n.id));
}
const block=(id,prefill,bytes,state='ready')=>({id,prefill,bytes,state});
const kv=[
  [1,'produce',block(0,true,20)],
  [2,'send',block(1,false,20,'in_flight')],
  [4,'receive',block(1,false,20)],
  [4,'release',block(0,true,20)],
  [5,'produce',block(2,false,4)],
  [6,'extend',block(2,false,8)],
  [7,'release',block(1,false,20)],
  [7,'release',block(2,false,8)]
].map(([time_s,transition,block],seq)=>({time_s,transition,block,seq}));
const fixture={nodes:[],rounds:[],states:[],kv,requests:[{id:0,arrival:1,finish:7}]};
const model=new ModuleFlowModel(fixture);
assert.equal(model.at(3).kv.received,0);
assert.equal(model.at(3).kv.flight,20);
assert.equal(model.at(3).kv.p,20);
assert.equal(model.at(4).kv.p,0);
assert.equal(model.at(6).kv.d,28);
assert.equal(model.at(6).kv.appended,4);
assert.equal(model.at(7).kv.d,0);
assert.equal(model.at(2).kv.d,0);
assert.equal(model.at(6).kv.d,28);
assert.equal(model.at(1).queues.p,null);

const reports=[];
for(const file of process.argv.slice(2)){
  const html=fs.readFileSync(file,'utf8');
  const data=JSON.parse(html.split('<script id="data" type="application/json">')[1].split('</script>')[0]);
  const model=new ModuleFlowModel(data);
  const groups=new Map();
  for(const n of data.nodes){const key=n.view.flow;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(n);}
  const selected=data.times.filter((_,i)=>i%Math.max(1,Math.floor(data.times.length/120))===0);
  const times=[...selected,data.meta.finish,...selected.toReversed(),...selected.filter((_,i)=>i%2===0)];
  for(const t of times){
    const state=model.at(t);
    for(const [key,items] of groups){
      assert.equal(state.groups[key].done,items.filter(n=>n.finish<=t).length,`${file} ${key} completion at ${t}`);
      assert.deepEqual([...state.groups[key].active].map(n=>n.id).sort((a,b)=>a-b),items.filter(n=>n.start<=t&&n.finish>t).map(n=>n.id).sort((a,b)=>a-b),`${file} ${key} active at ${t}`);
    }
    const live=new Map();let received=0;
    for(const e of data.kv){if(e.time_s>t)continue;if(e.transition==='release')live.delete(e.block.id);else live.set(e.block.id,e.block);if(e.transition==='receive')received+=e.block.bytes;}
    const sum=predicate=>[...live.values()].filter(predicate).reduce((s,b)=>s+b.bytes,0);
    assert.equal(state.kv.p,sum(b=>b.prefill&&b.state==='ready'));
    assert.equal(state.kv.d,sum(b=>!b.prefill&&b.state==='ready'));
    assert.equal(state.kv.flight,sum(b=>b.state==='in_flight'));
    assert.equal(state.kv.received,received);
    assert.equal(state.completed,data.requests.filter(r=>r.finish<=t).length);
  }
  const end=model.at(data.meta.finish);
  assert.equal(end.kv.p+end.kv.d+end.kv.flight,0);
  assert.equal(end.kv.received,model.copyBytes);
  assert.equal([...groups.values()].reduce((s,n)=>s+n.length,0),data.nodes.length);
  reports.push({file,nodes:data.nodes.length,sampledSeeks:times.length,unassigned:groups.get('unassigned')?.length||0,receivedBytes:end.kv.received});
}
console.log(JSON.stringify({synthetic:'interval boundaries, reverse seeks, KV ACK, release and extension passed',native:reports},null,2));
