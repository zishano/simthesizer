//! Persistent KV ownership and causal P/D copies on the shared resource domain.
use crate::resources::{Allocation, Demand, Work};
use crate::types::NodeID;
use crate::{Graph, LogicalHandler, Module};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BatchRequest {
    pub request_id: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Clone)]
pub struct Route {
    pub src: String,
    pub dst: String,
    pub head_start: u32,
    pub head_end: u32,
    pub seconds_per_byte: BTreeMap<String, f64>,
    pub latency: f64,
}

pub struct Plan {
    pub source: Vec<String>,
    pub target: Vec<String>,
    pub routes: Vec<Route>,
    pub layers: u32,
    pub heads: u32,
    pub bytes_per_token_head_layer: u64,
    pub input_tokens: u32,
    pub max_decode_tokens: u32,
    pub layer_handoff: bool,
    pub prefill_done: bool,
    pub copies: BTreeMap<(u32, usize), BTreeSet<NodeID>>,
}

#[derive(Clone, Serialize)]
struct Block {
    id: u64,
    source: Option<u64>,
    request: u64,
    prefill: bool,
    layer: u32,
    token_start: u32,
    token_end: u32,
    head_start: u32,
    head_end: u32,
    device: String,
    state: &'static str,
    refs: usize,
    bytes: u64,
}

#[derive(Debug)]
pub enum Event {
    Read {
        request: u64,
        prefill: bool,
        layer: u32,
        rank: usize,
        tokens: u32,
    },
    Produce {
        request: u64,
        prefill: bool,
        layer: u32,
        rank: usize,
        start: u32,
        end: u32,
    },
    Ack {
        request: u64,
        source: u64,
        target: u64,
    },
    PrefillDone(u64),
    DecodeDone(u64),
}

/// Disjoint, merged ready intervals. A later arrival must never bridge a gap.
#[derive(Default)]
struct Intervals(BTreeMap<u32, u32>);

impl Intervals {
    fn insert(&mut self, mut start: u32, mut end: u32) {
        if let Some((&left, &right)) = self.0.range(..=start).next_back() {
            if right >= start {
                self.0.remove(&left);
                start = left;
                end = end.max(right);
            }
        }
        while let Some((&left, &right)) = self.0.range(start..).next() {
            if left > end {
                break;
            }
            self.0.remove(&left);
            end = end.max(right);
        }
        self.0.insert(start, end);
    }

    fn prefix(&self) -> u32 {
        self.0.get(&0).copied().unwrap_or(0)
    }
}

#[derive(Default)]
struct RequestKv {
    sources: BTreeMap<u64, Block>,
    received: BTreeMap<u64, Block>,
    appends: BTreeMap<(u32, usize), Block>,
    // Per head because a target TP rank may receive shards from several sources.
    ready: BTreeMap<(bool, u32, usize), Vec<Intervals>>,
}

#[derive(Default)]
pub struct KvRuntime {
    pub plans: BTreeMap<u64, Plan>,
    requests: BTreeMap<u64, RequestKv>,
    next: u64,
    pub hbm_seconds_per_byte: BTreeMap<String, f64>,
}

impl KvRuntime {
    pub fn reservations(&self, request: u64, prefill: bool) -> Result<Vec<Allocation>> {
        let p = self
            .plans
            .get(&request)
            .context("shared execution requires a P/D KV plan")?;
        let full = p.bytes_per_token_head_layer * u64::from(p.layers) * u64::from(p.heads);
        let mut result = Vec::new();
        for (stage, devices, tokens) in [
            ("p", &p.source, p.input_tokens),
            ("d", &p.target, p.max_decode_tokens),
        ] {
            if (stage == "p" && !prefill) || tokens == 0 {
                continue;
            }
            ensure!(!devices.is_empty(), "missing KV devices");
            result.push(Allocation {
                id: format!("kv:{request}:{stage}"),
                bytes: devices
                    .iter()
                    .map(|d| (d.clone(), u64::from(tokens) * full / devices.len() as u64))
                    .collect(),
            });
        }
        Ok(result)
    }

    pub fn read_dependencies(&self, request: u64, layer: u32, rank: usize) -> Vec<NodeID> {
        self.plans[&request]
            .copies
            .get(&(layer, rank))
            .into_iter()
            .flatten()
            .copied()
            .collect()
    }

    fn emit(&self, graph: &mut Graph, event: &str, block: &Block) {
        if graph.tracing_enabled() {
            graph.record(json!({"event":"kv", "transition":event, "block":block}));
        }
    }

    fn mark_ready(&mut self, block: &Block) {
        let p = &self.plans[&block.request];
        let devices = if block.prefill { &p.source } else { &p.target };
        let rank = devices.iter().position(|d| *d == block.device).unwrap();
        let width = p.heads / devices.len() as u32;
        let heads = self
            .requests
            .entry(block.request)
            .or_default()
            .ready
            .entry((block.prefill, block.layer, rank))
            .or_insert_with(|| (0..width).map(|_| Intervals::default()).collect());
        for head in block.head_start..block.head_end {
            heads[(head - rank as u32 * width) as usize].insert(block.token_start, block.token_end);
        }
    }

    fn copy(
        &mut self,
        request: u64,
        source: u64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let block = self.requests[&request].sources[&source].clone();
        let plan = &self.plans[&block.request];
        let bytes_per_token_head_layer = plan.bytes_per_token_head_layer;
        let routes: Vec<_> = plan
            .routes
            .iter()
            .filter(|r| r.src == block.device)
            .cloned()
            .collect();
        for route in routes {
            let bytes = u64::from(block.token_end - block.token_start)
                * u64::from(route.head_end - route.head_start)
                * bytes_per_token_head_layer;
            let target = self.next;
            self.next += 1;
            let received = Block {
                id: target,
                source: Some(source),
                prefill: false,
                device: route.dst.clone(),
                head_start: route.head_start,
                head_end: route.head_end,
                state: "in_flight",
                refs: 1,
                bytes,
                ..block.clone()
            };
            self.requests
                .get_mut(&request)
                .unwrap()
                .sources
                .get_mut(&source)
                .unwrap()
                .refs += 1;
            self.emit(graph, "send", &received);
            self.requests
                .get_mut(&request)
                .unwrap()
                .received
                .insert(target, received);
            let mut resources: BTreeMap<_, _> = route
                .seconds_per_byte
                .iter()
                .map(|(k, v)| (k.clone(), v * bytes as f64))
                .collect();
            for device in [&route.src, &route.dst] {
                *resources.entry(format!("hbm:{device}")).or_default() +=
                    self.hbm_seconds_per_byte[device] * bytes as f64;
            }
            let transfer = graph.add_resource_node(Work {
                demands: vec![Demand { resources, latency: route.latency }],
                details: json!({"name":"kv_transfer", "request_id":block.request,"layer":block.layer,"token_start":block.token_start,"token_end":block.token_end,"head_start":route.head_start,"head_end":route.head_end,"src":route.src,"dst":route.dst,"bytes":bytes}),
                ..Default::default()
            });
            let ack = graph.add_logical_node(
                self as *mut dyn LogicalHandler,
                Event::Ack {
                    request,
                    source,
                    target,
                },
            );
            graph.add_edge(current, transfer)?;
            graph.add_edge(transfer, ack)?;
            let p = self.plans.get_mut(&request).unwrap();
            let rank = p.target.iter().position(|d| *d == route.dst).unwrap();
            p.copies.entry((block.layer, rank)).or_default().insert(ack);
        }
        Ok(())
    }

    fn release_source(&mut self, request: u64, source: u64, graph: &mut Graph) -> Result<()> {
        let state = self.requests.get_mut(&request).unwrap();
        if state.sources[&source].refs == 0 {
            let block = state.sources.remove(&source).unwrap();
            self.emit(graph, "release", &block);
        }
        if self.plans[&request].prefill_done && self.requests[&request].sources.is_empty() {
            let id = format!("kv:{request}:p");
            if graph.resources.allocated(&id) {
                graph.resources.release(&id)?;
                graph.record_memory();
                graph.record(json!({"event":"kv_reservation_release","allocation":id}));
            }
        }
        Ok(())
    }

    pub fn apply(&mut self, event: Event, current: NodeID, graph: &mut Graph) -> Result<()> {
        match event {
            Event::Read {
                request,
                prefill,
                layer,
                rank,
                tokens,
            } => {
                let end = self
                    .requests
                    .get(&request)
                    .and_then(|s| s.ready.get(&(prefill, layer, rank)))
                    .map(|heads| heads.iter().map(Intervals::prefix).min().unwrap_or(0))
                    .unwrap_or(0);
                ensure!(end >= tokens, "KV not ready: request {request} layer {layer} rank {rank}, need {tokens}, have {end}");
            }
            Event::Produce {
                request,
                prefill,
                layer,
                rank,
                start,
                end,
            } => {
                ensure!(end > start, "empty KV production");
                let p = &self.plans[&request];
                let devices = if prefill { &p.source } else { &p.target };
                let width = p.heads / devices.len() as u32;
                ensure!(
                    end <= if prefill {
                        p.input_tokens
                    } else {
                        p.max_decode_tokens
                    },
                    "KV exceeds reservation"
                );
                let block = Block {
                    id: self.next,
                    source: None,
                    request,
                    prefill,
                    layer,
                    token_start: start,
                    token_end: end,
                    head_start: rank as u32 * width,
                    head_end: (rank as u32 + 1) * width,
                    device: devices[rank].clone(),
                    state: "ready",
                    refs: 1,
                    bytes: u64::from(end - start) * u64::from(width) * p.bytes_per_token_head_layer,
                };
                let copy = prefill && p.layer_handoff && !p.target.is_empty();
                self.next += 1;
                self.mark_ready(&block);
                let id = block.id;
                if prefill {
                    self.emit(graph, "produce", &block);
                    self.requests
                        .get_mut(&request)
                        .unwrap()
                        .sources
                        .insert(id, block);
                } else {
                    let appends = &mut self.requests.get_mut(&request).unwrap().appends;
                    if let Some(previous) = appends.get_mut(&(layer, rank)) {
                        ensure!(
                            previous.token_end == start,
                            "non-contiguous decode KV append"
                        );
                        previous.token_end = end;
                        previous.bytes += block.bytes;
                        let extended = previous.clone();
                        self.emit(graph, "extend", &extended);
                    } else {
                        appends.insert((layer, rank), block.clone());
                        self.emit(graph, "produce", &block);
                    }
                }
                if copy {
                    self.copy(request, id, current, graph)?;
                }
            }
            Event::Ack {
                request,
                source,
                target,
            } => {
                let b = self
                    .requests
                    .get_mut(&request)
                    .context("missing KV request")?
                    .received
                    .get_mut(&target)
                    .context("missing transfer destination")?;
                ensure!(b.state == "in_flight", "duplicate KV acknowledgement");
                b.state = "ready";
                let received = b.clone();
                self.mark_ready(&received);
                self.emit(graph, "receive", &received);
                self.requests
                    .get_mut(&request)
                    .unwrap()
                    .sources
                    .get_mut(&source)
                    .context("premature KV source release")?
                    .refs -= 1;
                let p = self.plans.get_mut(&request).unwrap();
                let rank = p.target.iter().position(|d| *d == received.device).unwrap();
                let key = (received.layer, rank);
                let copies = p
                    .copies
                    .get_mut(&key)
                    .context("missing KV copy dependency")?;
                ensure!(
                    copies.remove(&current),
                    "missing KV acknowledgement dependency"
                );
                if copies.is_empty() {
                    p.copies.remove(&key);
                }
                self.release_source(request, source, graph)?;
            }
            Event::PrefillDone(request) => {
                graph.record(json!({"event":"kv_prefill_done","request_id":request}));
                let p = &self.plans[&request];
                let copy = !p.layer_handoff && !p.target.is_empty();
                let ids: Vec<_> = self.requests[&request].sources.keys().copied().collect();
                for &id in &ids {
                    if copy {
                        self.copy(request, id, current, graph)?;
                    }
                    self.requests
                        .get_mut(&request)
                        .unwrap()
                        .sources
                        .get_mut(&id)
                        .unwrap()
                        .refs -= 1;
                }
                self.plans.get_mut(&request).unwrap().prefill_done = true;
                self.requests
                    .get_mut(&request)
                    .unwrap()
                    .ready
                    .retain(|(prefill, _, _), _| !prefill);
                for id in ids {
                    self.release_source(request, id, graph)?;
                }
                if self.plans[&request].target.is_empty() {
                    self.plans.remove(&request);
                    self.requests.remove(&request);
                }
            }
            Event::DecodeDone(request) => {
                let state = self
                    .requests
                    .remove(&request)
                    .context("missing decode KV")?;
                ensure!(
                    state.sources.is_empty() && self.plans[&request].copies.is_empty(),
                    "request finished with pending KV transfer"
                );
                for b in state.received.values().chain(state.appends.values()) {
                    ensure!(
                        !b.prefill && b.state == "ready",
                        "request finished with pending KV transfer"
                    );
                    self.emit(graph, "release", b);
                }
                graph.resources.release(&format!("kv:{request}:d"))?;
                graph.record_memory();
                graph.record(json!({"event":"kv_reservation_release","allocation":format!("kv:{request}:d")}));
                self.plans.remove(&request);
            }
        }
        Ok(())
    }

    pub fn check_drained(&self) -> Result<()> {
        ensure!(
            self.requests.is_empty() && self.plans.is_empty(),
            "KV state did not drain"
        );
        Ok(())
    }
}
impl Module for KvRuntime {}
impl LogicalHandler for KvRuntime {
    fn describe(&self, payload: &dyn Any) -> String {
        format!("KV {:?}", payload.downcast_ref::<Event>())
    }
    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        _: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        let event = *payload
            .downcast::<Event>()
            .map_err(|_| anyhow::anyhow!("invalid KV event"))?;
        self.apply(event, current, graph)?;
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> KvRuntime {
        let mut kv = KvRuntime::default();
        kv.plans.insert(
            7,
            Plan {
                source: vec!["p0".into(), "p1".into()],
                target: vec!["d0".into()],
                routes: vec![],
                layers: 2,
                heads: 4,
                bytes_per_token_head_layer: 16,
                input_tokens: 20,
                max_decode_tokens: 4096,
                layer_handoff: true,
                prefill_done: false,
                copies: Default::default(),
            },
        );
        kv
    }

    #[test]
    fn intervals_merge_without_hiding_gaps() {
        let mut ranges = Intervals::default();
        ranges.insert(10, 20);
        ranges.insert(0, 5);
        ranges.insert(6, 10);
        assert_eq!(ranges.prefix(), 5);
        ranges.insert(5, 6);
        assert_eq!(ranges.prefix(), 20);
        ranges.insert(3, 15);
        assert_eq!(ranges.0.len(), 1);
    }

    #[test]
    fn ready_index_checks_all_heads_and_scopes() {
        let mut kv = runtime();
        let mut graph = Graph::new();
        let mut block = Block {
            id: 0,
            source: Some(0),
            request: 7,
            prefill: false,
            layer: 0,
            token_start: 0,
            token_end: 20,
            head_start: 0,
            head_end: 2,
            device: "d0".into(),
            state: "ready",
            refs: 1,
            bytes: 640,
        };
        kv.mark_ready(&block);
        let read = |kv: &mut KvRuntime, graph: &mut Graph, request, layer, tokens| {
            kv.apply(
                Event::Read {
                    request,
                    prefill: false,
                    layer,
                    rank: 0,
                    tokens,
                },
                0,
                graph,
            )
        };
        assert!(read(&mut kv, &mut graph, 7, 0, 1).is_err());
        block.head_start = 2;
        block.head_end = 4;
        block.token_end = 5;
        kv.mark_ready(&block);
        block.token_start = 10;
        block.token_end = 20;
        kv.mark_ready(&block);
        assert!(read(&mut kv, &mut graph, 7, 0, 6).is_err());
        block.token_start = 5;
        block.token_end = 10;
        kv.mark_ready(&block);
        assert!(read(&mut kv, &mut graph, 7, 0, 20).is_ok());
        assert!(read(&mut kv, &mut graph, 8, 0, 1).is_err());
        assert!(read(&mut kv, &mut graph, 7, 1, 1).is_err());
    }

    #[test]
    fn decode_metadata_is_bounded_by_layer_and_rank() {
        let mut kv = runtime();
        let mut graph = Graph::new();
        for start in 0..2000 {
            kv.apply(
                Event::Produce {
                    request: 7,
                    prefill: false,
                    layer: 0,
                    rank: 0,
                    start,
                    end: start + 1,
                },
                0,
                &mut graph,
            )
            .unwrap();
        }
        let state = &kv.requests[&7];
        assert_eq!(state.appends.len(), 1);
        assert_eq!(state.appends[&(0, 0)].bytes, 2000 * 4 * 16);
        for head in &state.ready[&(false, 0, 0)] {
            assert_eq!(head.prefix(), 2000);
            assert_eq!(head.0.len(), 1);
        }
    }
}
