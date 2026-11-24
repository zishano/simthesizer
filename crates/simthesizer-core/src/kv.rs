//! Persistent KV ownership and causal P/D copies on the shared resource domain.
use crate::policies::{KvObservation, PendingKv, SharedPolicy};
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
    produced_at: f64,
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
    PolicyWake,
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
    gates: BTreeMap<(u32, usize), BTreeMap<u32, NodeID>>,
}

struct PendingTransfer {
    observation: PendingKv,
    source: u64,
    route: Route,
}

/// An unmet read registered by an expanded batch, not necessarily its next
/// executable layer or a current critical-path stall.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KvReadRequirement {
    pub request_id: u64,
    pub layer: u32,
    pub target_rank: usize,
    pub required_tokens: u32,
    /// Contiguous prefix present for every KV head of the target rank.
    pub ready_tokens: u32,
}

#[derive(Default)]
pub struct KvRuntime {
    pub plans: BTreeMap<u64, Plan>,
    requests: BTreeMap<u64, RequestKv>,
    next: u64,
    pub hbm_seconds_per_byte: BTreeMap<String, f64>,
    policy: Option<SharedPolicy>,
    pending: BTreeMap<u64, PendingTransfer>,
    inflight: usize,
    wake: Option<NodeID>,
}

impl KvRuntime {
    /// Read only registered requirements; do not predict future decode work.
    pub fn policy_reads(&self) -> Vec<KvReadRequirement> {
        self.policy_read_requirements().collect()
    }

    /// Count all registered unmet reads while allocating only the requested prefix.
    pub fn policy_read_sample(&self, limit: usize) -> (usize, Vec<KvReadRequirement>) {
        let mut total = 0;
        let mut reads = Vec::new();
        for read in self.policy_read_requirements() {
            total += 1;
            if reads.len() < limit {
                reads.push(read);
            }
        }
        (total, reads)
    }

    fn policy_read_requirements(&self) -> impl Iterator<Item = KvReadRequirement> + '_ {
        self.requests.iter().flat_map(move |(&request_id, state)| {
            state.gates.iter().flat_map(move |(&(layer, rank), gates)| {
                let ready_tokens = self.ready_tokens(request_id, false, layer, rank);
                gates
                    .keys()
                    .copied()
                    .filter(move |&required_tokens| required_tokens > ready_tokens)
                    .map(move |required_tokens| KvReadRequirement {
                        request_id,
                        layer,
                        target_rank: rank,
                        required_tokens,
                        ready_tokens,
                    })
            })
        })
    }

    pub fn configure_policy(&mut self, policy: SharedPolicy) -> Result<()> {
        // The reference keeps the existing eager transfer path.
        if !policy.borrow().is_reference() {
            self.policy = Some(policy);
        }
        Ok(())
    }

    pub fn controlled(&self) -> bool {
        self.policy.is_some()
    }

    pub fn ready_tokens(&self, request: u64, prefill: bool, layer: u32, rank: usize) -> u32 {
        self.requests
            .get(&request)
            .and_then(|s| s.ready.get(&(prefill, layer, rank)))
            .map(|heads| heads.iter().map(Intervals::prefix).min().unwrap_or(0))
            .unwrap_or(0)
    }

    pub fn read_gate(&self, request: u64, layer: u32, rank: usize, tokens: u32) -> Option<NodeID> {
        self.requests
            .get(&request)?
            .gates
            .get(&(layer, rank))?
            .get(&tokens)
            .copied()
    }

    pub fn register_read_gate(
        &mut self,
        request: u64,
        layer: u32,
        rank: usize,
        tokens: u32,
        node: NodeID,
    ) {
        self.requests
            .entry(request)
            .or_default()
            .gates
            .entry((layer, rank))
            .or_default()
            .insert(tokens, node);
    }

    fn release_read_gates(
        &mut self,
        request: u64,
        layer: u32,
        rank: usize,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let end = self.ready_tokens(request, false, layer, rank);
        let state = self
            .requests
            .get_mut(&request)
            .context("missing KV request")?;
        if let Some(gates) = state.gates.get_mut(&(layer, rank)) {
            while gates
                .first_key_value()
                .is_some_and(|(&tokens, _)| tokens <= end)
            {
                let (_, node) = gates.pop_first().unwrap();
                graph.release_condition(node, current)?;
            }
            if gates.is_empty() {
                state.gates.remove(&(layer, rank));
            }
        }
        Ok(())
    }

    fn drive_policy(&mut self, current: NodeID, graph: &mut Graph) -> Result<()> {
        let Some(policy) = self.policy.clone() else {
            return Ok(());
        };
        if self.pending.is_empty()
            || self.wake.is_some()
            || self.inflight >= policy.borrow().limits().kv_inflight_bounds[1]
        {
            return Ok(());
        }
        let pending: Vec<_> = self.pending.values().map(|p| p.observation).collect();
        let action = policy.borrow_mut().kv(&KvObservation {
            now: graph.now(),
            pending: &pending,
            inflight: self.inflight,
            environment: Some(crate::policies::EnvironmentObservation::new(
                &graph.resources,
                self,
            )),
        })?;
        for id in action.submissions {
            let transfer = self
                .pending
                .remove(&id)
                .context("missing pending KV submission")?;
            self.send_transfer(
                transfer.observation.request_id,
                transfer.source,
                id,
                transfer.route,
                current,
                graph,
            )?;
        }
        if !self.pending.is_empty() {
            if let Some(until) = action.wait_until {
                let wake = graph.add_logical_node_at(
                    self as *mut dyn LogicalHandler,
                    Event::PolicyWake,
                    until,
                );
                graph.add_edge(current, wake)?;
                self.wake = Some(wake);
            }
        }
        Ok(())
    }

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
                state: if self.controlled() {
                    "pending_send"
                } else {
                    "in_flight"
                },
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
            if self.controlled() {
                self.emit(graph, "pending_send", &received);
            }
            self.requests
                .get_mut(&request)
                .unwrap()
                .received
                .insert(target, received);
            if self.controlled() {
                let p = &self.plans[&request];
                let source_rank = p.source.iter().position(|d| *d == route.src).unwrap() as u32;
                let target_rank = p.target.iter().position(|d| *d == route.dst).unwrap() as u32;
                self.pending.insert(
                    target,
                    PendingTransfer {
                        observation: PendingKv {
                            id: target,
                            request_id: request,
                            layer: block.layer,
                            source_rank,
                            target_rank,
                            token_start: block.token_start,
                            token_end: block.token_end,
                            bytes,
                            produced_at: block.produced_at,
                        },
                        source,
                        route,
                    },
                );
            } else {
                self.send_transfer(request, source, target, route, current, graph)?;
            }
        }
        Ok(())
    }

    fn send_transfer(
        &mut self,
        request: u64,
        source: u64,
        target: u64,
        route: Route,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let received = self
            .requests
            .get_mut(&request)
            .unwrap()
            .received
            .get_mut(&target)
            .unwrap();
        received.state = "in_flight";
        let block = received.clone();
        let bytes = block.bytes;
        self.emit(graph, "send", &block);
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
        if self.controlled() {
            self.inflight += 1;
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
                let end = self.ready_tokens(request, prefill, layer, rank);
                ensure!(
                    end >= tokens,
                    "KV not ready: request {request} layer {layer} rank {rank}, need {tokens}, have {end}"
                );
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
                    produced_at: graph.now(),
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
                        if graph.tracing_enabled() {
                            let extended = previous.clone();
                            self.emit(graph, "extend", &extended);
                        }
                    } else {
                        appends.insert((layer, rank), block.clone());
                        self.emit(graph, "produce", &block);
                    }
                }
                if copy {
                    self.copy(request, id, current, graph)?;
                    self.drive_policy(current, graph)?;
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
                if self.controlled() {
                    self.inflight = self
                        .inflight
                        .checked_sub(1)
                        .context("KV inflight underflow")?;
                    self.release_read_gates(request, received.layer, rank, current, graph)?;
                    self.drive_policy(current, graph)?;
                }
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
                self.drive_policy(current, graph)?;
            }
            Event::DecodeDone(request) => {
                let state = self
                    .requests
                    .remove(&request)
                    .context("missing decode KV")?;
                ensure!(
                    state.sources.is_empty()
                        && state.gates.is_empty()
                        && self.plans[&request].copies.is_empty(),
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
            Event::PolicyWake => {
                ensure!(self.wake == Some(current), "unexpected KV policy wakeup");
                self.wake = None;
                self.drive_policy(current, graph)?;
            }
        }
        Ok(())
    }

    pub fn check_drained(&self) -> Result<()> {
        ensure!(
            self.requests.is_empty()
                && self.plans.is_empty()
                && self.pending.is_empty()
                && self.inflight == 0
                && self.wake.is_none(),
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
    fn deferred_copy_pins_sources_until_all_heads_arrive_and_wakes_only_ready_reads() {
        use crate::policies::{PolicyConfig, PolicyKind, PolicyLimits, PolicyRuntime, PolicySlo};
        let policy = std::rc::Rc::new(std::cell::RefCell::new(
            PolicyRuntime::new(PolicyConfig {
                kind: PolicyKind::Candidate,
                seed: 0,
                diagnostics: None,
                limits: PolicyLimits {
                    p_token_budget_bounds: [1, 128],
                    d_token_budget_bounds: [1, 32],
                    p_batch_bounds: [1, 32],
                    d_batch_bounds: [1, 32],
                    kv_inflight_bounds: [1, 1],
                    max_wait_s: 0.05,
                    state_limit_bytes: 64,
                },
                slo: PolicySlo {
                    ttft_limit_s: 5.0,
                    tpot_limit_s: 0.05,
                },
            })
            .unwrap(),
        ));
        let mut kv = Box::new(runtime());
        kv.configure_policy(policy).unwrap();
        let p = kv.plans.get_mut(&7).unwrap();
        p.layer_handoff = false;
        p.routes = (0..2)
            .map(|rank| Route {
                src: format!("p{rank}"),
                dst: "d0".into(),
                head_start: rank * 2,
                head_end: rank * 2 + 2,
                seconds_per_byte: [("link".into(), 0.01)].into(),
                latency: 0.0,
            })
            .collect();
        kv.hbm_seconds_per_byte = [
            ("p0".into(), 0.01),
            ("p1".into(), 0.01),
            ("d0".into(), 0.01),
        ]
        .into();
        let mut graph = Graph::new();
        let current =
            graph.add_logical_node(&mut *kv as *mut dyn LogicalHandler, Event::PrefillDone(7));
        for rank in 0..2 {
            kv.apply(
                Event::Produce {
                    request: 7,
                    prefill: true,
                    layer: 0,
                    rank,
                    start: 0,
                    end: 20,
                },
                current,
                &mut graph,
            )
            .unwrap();
        }
        let gate = graph.add_condition();
        kv.register_read_gate(7, 0, 0, 20, gate);
        kv.apply(Event::PrefillDone(7), current, &mut graph)
            .unwrap();
        assert_eq!(kv.pending.len(), 1);
        assert_eq!(kv.inflight, 1);
        assert!(kv.requests[&7].sources.values().all(|b| b.refs == 1));
        assert_eq!(kv.ready_tokens(7, false, 0, 0), 0);
        assert_eq!(
            kv.policy_reads(),
            vec![KvReadRequirement {
                request_id: 7,
                layer: 0,
                target_rank: 0,
                required_tokens: 20,
                ready_tokens: 0,
            }]
        );
        for index in 0..2 {
            let b = kv.requests[&7]
                .received
                .values()
                .find(|b| b.state == "in_flight")
                .unwrap()
                .clone();
            let ack = *kv.plans[&7].copies[&(0, 0)].first().unwrap();
            kv.apply(
                Event::Ack {
                    request: 7,
                    source: b.source.unwrap(),
                    target: b.id,
                },
                ack,
                &mut graph,
            )
            .unwrap();
            if index == 0 {
                assert_eq!(kv.ready_tokens(7, false, 0, 0), 0);
                assert_eq!(kv.read_gate(7, 0, 0, 20), Some(gate));
                assert_eq!(kv.requests[&7].sources.len(), 1);
                assert_eq!(kv.policy_reads()[0].ready_tokens, 0);
            }
        }
        assert_eq!(kv.ready_tokens(7, false, 0, 0), 20);
        assert_eq!(kv.read_gate(7, 0, 0, 20), None);
        assert!(kv.pending.is_empty());
        assert!(kv.requests[&7].sources.is_empty());
        assert_eq!(kv.inflight, 0);
        assert!(kv.policy_reads().is_empty());
    }

    #[test]
    fn policy_reads_preserve_gaps_and_include_only_registered_requirements() {
        let mut kv = runtime();
        let mut graph = Graph::new();
        let mut block = Block {
            id: 0,
            source: Some(0),
            request: 7,
            prefill: false,
            layer: 0,
            token_start: 0,
            token_end: 2,
            head_start: 0,
            head_end: 4,
            device: "d0".into(),
            state: "ready",
            refs: 1,
            bytes: 128,
            produced_at: 0.0,
        };
        kv.mark_ready(&block);
        block.token_start = 4;
        block.token_end = 6;
        kv.mark_ready(&block);
        // Neither known request length nor incomplete KV creates a requirement.
        assert!(kv.policy_reads().is_empty());
        let gate = graph.add_condition();
        kv.register_read_gate(7, 0, 0, 6, gate);
        let later = graph.add_condition();
        kv.register_read_gate(7, 1, 0, 6, later);
        let expected = vec![
            KvReadRequirement {
                request_id: 7,
                layer: 0,
                target_rank: 0,
                required_tokens: 6,
                ready_tokens: 2,
            },
            KvReadRequirement {
                request_id: 7,
                layer: 1,
                target_rank: 0,
                required_tokens: 6,
                ready_tokens: 0,
            },
        ];
        for _ in 0..3 {
            assert_eq!(kv.policy_reads(), expected);
            assert_eq!(kv.read_gate(7, 0, 0, 6), Some(gate));
            assert_eq!(kv.ready_tokens(7, false, 0, 0), 2);
        }
        block.token_start = 2;
        block.token_end = 4;
        kv.mark_ready(&block);
        // Sampling never releases the gate itself; the ACK owner does that.
        assert_eq!(kv.policy_reads(), expected[1..]);
        assert_eq!(kv.read_gate(7, 0, 0, 6), Some(gate));
        let cause = graph.add_resource_node(Work::default());
        kv.release_read_gates(7, 0, 0, cause, &mut graph).unwrap();
        assert_eq!(kv.read_gate(7, 0, 0, 6), None);
        assert_eq!(kv.read_gate(7, 1, 0, 6), Some(later));
        assert_eq!(kv.policy_reads(), expected[1..]);
    }

    #[test]
    fn policy_read_sample_retains_only_its_bound_and_counts_all_requirements() {
        let mut kv = runtime();
        let mut graph = Graph::new();
        for tokens in 1..=100 {
            kv.register_read_gate(7, 0, 0, tokens, graph.add_condition());
        }
        let (total, reads) = kv.policy_read_sample(64);
        assert_eq!(total, 100);
        assert_eq!(reads.len(), 64);
        assert!(reads.capacity() <= 64);
        assert_eq!(reads[0].required_tokens, 1);
        assert_eq!(reads[63].required_tokens, 64);
        assert_eq!(reads, kv.policy_reads()[..64]);
        let (total, none) = kv.policy_read_sample(0);
        assert_eq!(total, 100);
        assert_eq!(none.capacity(), 0);
        assert_eq!(kv.requests[&7].gates[&(0, 0)].len(), 100);
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
            produced_at: 0.0,
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
