use serde_json::{json, Value};
use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::Write;

use anyhow::{anyhow, bail, Context, Result};

use crate::compute_sim::ComputeSimulator;
use crate::module::LogicalHandler;
use crate::network_sim::NetworkSimulator;
use crate::system::System;
use crate::types::{
    ComputeContext, Layer, NetworkDeviceID, NodeID, PollResult, RequestResult, RequestState,
};

pub struct Graph {
    pub resources: crate::resources::Resources,
    pub kv: Box<crate::kv::KvRuntime>,
    next_node_id: NodeID,
    nodes: HashMap<NodeID, Node>,
    trace: Option<GraphTrace>,
    clock: f64,
    current: Option<NodeID>,
}

struct GraphTrace {
    writer: Box<dyn Write>,
    error: Option<std::io::Error>,
    sequence: u64,
}

pub struct Engine {
    graph: Graph,
    system: Box<dyn System>,
}

struct Node {
    kind: NodeKind,
    children: Children,
    indegree: usize,
    start_time: f64,
}

// Most execution nodes release at most three children. Keep those edges beside
// the node; larger fan-outs retain an ordinary vector and insertion order.
#[derive(Default)]
enum Children {
    #[default]
    Empty,
    One(NodeID),
    Two([NodeID; 2]),
    Three([NodeID; 3]),
    Many(Vec<NodeID>),
}

impl Children {
    fn push(&mut self, child: NodeID) {
        *self = match self {
            Self::Empty => Self::One(child),
            Self::One(a) => Self::Two([*a, child]),
            Self::Two([a, b]) => Self::Three([*a, *b, child]),
            Self::Three([a, b, c]) => Self::Many(vec![*a, *b, *c, child]),
            Self::Many(children) => {
                children.push(child);
                return;
            }
        };
    }

    fn as_slice(&self) -> &[NodeID] {
        match self {
            Self::Empty => &[],
            Self::One(child) => std::slice::from_ref(child),
            Self::Two(children) => children,
            Self::Three(children) => children,
            Self::Many(children) => children,
        }
    }

    fn into_vec(self) -> Vec<NodeID> {
        match self {
            Self::Empty => Vec::new(),
            Self::One(child) => vec![child],
            Self::Two(children) => Vec::from(children),
            Self::Three(children) => Vec::from(children),
            Self::Many(children) => children,
        }
    }
}

enum NodeKind {
    /// A real external condition, released only by its owning component.
    Condition { held: bool },
    Resource {
        work: Option<crate::resources::SharedWork>,
        reserves_memory: bool,
        releases_memory: bool,
    },
    Compute {
        simulator: *mut dyn ComputeSimulator,
        payload: Box<ComputePayload>,
    },
    Network {
        simulator: *mut dyn NetworkSimulator,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
        state: Option<Box<dyn Any>>,
    },
    Logical {
        handler: *mut dyn LogicalHandler,
        payload: Option<Box<dyn Any>>,
    },
}

struct ComputePayload {
    layer: Layer,
    batch: Vec<RequestState>,
    context: ComputeContext,
}

#[derive(Clone, Copy, Debug)]
struct QueueEntry {
    time: f64,
    node_id: NodeID,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            resources: Default::default(),
            kv: Default::default(),
            next_node_id: 0,
            nodes: HashMap::new(),
            trace: None,
            clock: 0.0,
            current: None,
        }
    }

    /// Attach before seeding requests to capture the complete graph history.
    pub fn with_trace(writer: impl Write + 'static) -> Self {
        let mut graph = Self::new();
        graph.trace = Some(GraphTrace {
            writer: Box::new(writer),
            error: None,
            sequence: 0,
        });
        graph.record(json!({"event":"header", "version":1, "time_unit":"seconds"}));
        graph
    }

    pub fn tracing_enabled(&self) -> bool {
        self.trace.is_some()
    }

    pub fn record(&mut self, mut event: Value) {
        if let Some(trace) = &mut self.trace {
            if trace.error.is_some() {
                return;
            }
            event["seq"] = json!(trace.sequence);
            event["time_s"] = json!(self.clock);
            event["cause"] = json!(self.current);
            trace.sequence += 1;
            let result = serde_json::to_writer(&mut trace.writer, &event)
                .map_err(std::io::Error::other)
                .and_then(|_| trace.writer.write_all(b"\n"));
            if let Err(error) = result {
                trace.error = Some(error);
            }
        }
    }

    pub fn record_memory(&mut self) {
        if self.trace.is_some() {
            self.record(json!({"event":"memory_state","state":self.resources.snapshot()}));
        }
    }

    fn check_trace(&mut self, flush: bool) -> Result<()> {
        if let Some(trace) = &mut self.trace {
            if let Some(error) = trace.error.take() {
                return Err(error.into());
            }
            if flush {
                trace.writer.flush()?;
            }
        }
        Ok(())
    }

    pub fn add_compute_node(
        &mut self,
        simulator: *mut dyn ComputeSimulator,
        layer: Layer,
        batch: Vec<RequestState>,
        context: ComputeContext,
    ) -> NodeID {
        self.insert_node(
            NodeKind::Compute {
                simulator,
                payload: Box::new(ComputePayload {
                    layer,
                    batch,
                    context,
                }),
            },
            0.0,
        )
    }

    pub fn add_resource_node(&mut self, work: crate::resources::Work) -> NodeID {
        self.add_shared_resource_node(crate::resources::SharedWork::new(work))
    }

    pub(crate) fn add_shared_resource_node(
        &mut self,
        work: crate::resources::SharedWork,
    ) -> NodeID {
        self.insert_node(
            NodeKind::Resource {
                reserves_memory: !work.reserve.is_empty(),
                releases_memory: !work.release.is_empty(),
                work: Some(work),
            },
            0.0,
        )
    }

    pub fn is_pending(&self, node: NodeID) -> bool {
        self.nodes.contains_key(&node)
    }

    pub fn now(&self) -> f64 {
        self.clock
    }

    /// The extra indegree is a condition token, not a runnable placeholder.
    /// An unreleased token remains in the graph and fails the drain check.
    pub fn add_condition(&mut self) -> NodeID {
        let id = self.insert_node(NodeKind::Condition { held: true }, 0.0);
        self.nodes.get_mut(&id).unwrap().indegree = 1;
        id
    }

    pub fn release_condition(&mut self, condition: NodeID, cause: NodeID) -> Result<()> {
        let node = self
            .nodes
            .get_mut(&condition)
            .context("missing condition node")?;
        let NodeKind::Condition { held } = &mut node.kind else {
            bail!("node {condition} is not a condition");
        };
        anyhow::ensure!(*held, "condition {condition} released twice");
        *held = false;
        node.indegree -= 1;
        self.add_edge(cause, condition)?;
        if self.tracing_enabled() {
            self.record(json!({"event":"condition_release","node":condition,"cause_node":cause}));
        }
        Ok(())
    }

    pub fn kv_event(&mut self, event: crate::kv::Event, current: NodeID) -> Result<()> {
        let mut kv = std::mem::take(&mut self.kv);
        let result = kv.apply(event, current, self);
        self.kv = kv;
        result
    }

    pub fn add_kv_node(&mut self, event: crate::kv::Event) -> NodeID {
        let handler = self.kv.as_mut() as *mut dyn LogicalHandler;
        self.add_logical_node(handler, event)
    }

    pub fn add_network_node(
        &mut self,
        simulator: *mut dyn NetworkSimulator,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
    ) -> NodeID {
        self.insert_node(
            NodeKind::Network {
                simulator,
                src,
                dst,
                bytes,
                state: None,
            },
            0.0,
        )
    }

    pub fn add_logical_node<T: Any>(
        &mut self,
        handler: *mut dyn LogicalHandler,
        payload: T,
    ) -> NodeID {
        self.add_logical_node_at(handler, payload, 0.0)
    }

    pub fn add_logical_node_at<T: Any>(
        &mut self,
        handler: *mut dyn LogicalHandler,
        payload: T,
        start_time: f64,
    ) -> NodeID {
        self.insert_node(
            NodeKind::Logical {
                handler,
                payload: Some(Box::new(payload)),
            },
            start_time,
        )
    }

    pub fn add_edge(&mut self, parent: NodeID, child: NodeID) -> Result<()> {
        if parent == child {
            let node = self
                .nodes
                .get_mut(&parent)
                .ok_or_else(|| anyhow!("node {parent} does not exist"))?;
            node.children.push(child);
            node.indegree += 1;
        } else {
            let [parent_node, child_node] = self.nodes.get_disjoint_mut([&parent, &child]);
            let parent_node = parent_node.ok_or_else(|| anyhow!("node {parent} does not exist"))?;
            let child_node = child_node.ok_or_else(|| anyhow!("node {child} does not exist"))?;
            parent_node.children.push(child);
            child_node.indegree += 1;
        }
        if self.trace.is_some() {
            self.record(json!({"event":"edge_add", "parent":parent, "child":child}));
        }
        Ok(())
    }

    pub fn pop_edges(&mut self, parent: NodeID) -> Result<Vec<NodeID>> {
        let parent_node = self
            .nodes
            .get_mut(&parent)
            .ok_or_else(|| anyhow!("node {parent} does not exist"))?;
        let children = std::mem::take(&mut parent_node.children).into_vec();

        for &child_id in &children {
            let child = self
                .nodes
                .get_mut(&child_id)
                .ok_or_else(|| anyhow!("child node {child_id} is missing during edge removal"))?;
            child.indegree = child
                .indegree
                .checked_sub(1)
                .ok_or_else(|| anyhow!("child node {child_id} indegree underflow"))?;
        }

        if self.trace.is_some() {
            for &child in &children {
                self.record(json!({"event":"edge_remove", "parent":parent, "child":child}));
            }
        }
        Ok(children)
    }

    fn take_completed_node(&mut self, node_id: NodeID) -> Result<Node> {
        let std::collections::hash_map::Entry::Occupied(entry) = self.nodes.entry(node_id) else {
            bail!("node {node_id} does not exist");
        };
        let indegree = entry.get().indegree;
        if indegree != 0 {
            bail!("node {node_id} is not completed: indegree is {indegree}");
        }
        Ok(entry.remove())
    }

    fn insert_node(&mut self, kind: NodeKind, start_time: f64) -> NodeID {
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        if self.trace.is_some() {
            let details = match &kind {
                NodeKind::Condition { held } => {
                    json!({"kind":"logical", "label":"KV ready condition", "held":held})
                }
                NodeKind::Resource { work, .. } => {
                    let work = work.as_ref().unwrap();
                    json!({"kind":"resource", "label":work.details.get("name").and_then(Value::as_str).unwrap_or("resource"), "work":work})
                }
                NodeKind::Compute { payload, .. } => {
                    let ComputePayload {
                        layer,
                        batch,
                        context,
                    } = payload.as_ref();
                    json!({
                    "kind":"compute", "label":layer.name(), "layer":format!("{layer:?}"),
                    "batch":batch.iter().map(|r| json!({"q_len":r.q_len,"kv_len":r.kv_len,"lm_head_len":r.lm_head_len})).collect::<Vec<_>>(),
                    "context":format!("{context:?}")})
                }
                NodeKind::Network {
                    src, dst, bytes, ..
                } => {
                    json!({"kind":"network", "label":"transfer", "src":src,"dst":dst,"bytes":bytes})
                }
                NodeKind::Logical { handler, payload } => {
                    // Same lifetime guarantee as event dispatch; description is read-only.
                    let label =
                        unsafe { (&**handler).describe(payload.as_ref().unwrap().as_ref()) };
                    json!({"kind":"logical", "label":label})
                }
            };
            self.record(json!({"event":"node_create", "node":node_id, "release_s":start_time, "details":details}));
        }
        self.nodes.insert(
            node_id,
            Node {
                kind,
                children: Children::default(),
                indegree: 0,
                start_time,
            },
        );
        node_id
    }
}

impl Engine {
    pub fn new(graph: Graph, system: Box<dyn System>) -> Self {
        Self { graph, system }
    }

    pub fn run(self) -> Result<Vec<RequestResult>> {
        self.run_with_observer(|_, _| Ok(()))
    }

    /// Observe after all events at a timestamp, including zero-duration descendants.
    /// The callback also receives time zero for an empty or delayed workload.
    pub fn run_with_observer(
        self,
        observe: impl FnMut(f64, &dyn System) -> Result<()>,
    ) -> Result<Vec<RequestResult>> {
        self.run_with_summary(observe).map(|(results, _)| results)
    }

    /// Read the aggregate policy counters once, after all work has drained.
    pub fn run_with_summary(
        mut self,
        mut observe: impl FnMut(f64, &dyn System) -> Result<()>,
    ) -> Result<(
        Vec<RequestResult>,
        Option<crate::policies::PolicyStatistics>,
    )> {
        use crate::wall_profile::{scope, Section};
        let _wall = scope(Section::EngineOther);
        self.graph.check_trace(false)?;
        let mut queue = self.ready_queue()?;
        let mut last_observed = None;
        loop {
            let admitted = {
                let _wall = scope(Section::ResourceAdmit);
                self.graph.resources.admit(self.graph.clock)?
            };
            if self.graph.tracing_enabled() {
                for id in admitted {
                    self.graph
                        .record(json!({"event":"resource_start", "node":id}));
                    if let NodeKind::Resource {
                        reserves_memory: true,
                        ..
                    } = &self.graph.nodes[&id].kind
                    {
                        self.graph.record_memory();
                    }
                }
            }
            let next_resource = {
                let _wall = scope(Section::ResourceRates);
                self.graph.resources.next_time()
            };
            let next_queue = queue.peek().map(|entry| entry.time);
            let time = match (next_resource, next_queue) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => break,
            };
            if time > self.graph.clock && last_observed != Some(self.graph.clock) {
                observe(self.graph.clock, self.system.as_ref())?;
                last_observed = Some(self.graph.clock);
            }
            self.graph.clock = time;
            let finished = {
                let _wall = scope(Section::ResourceAdvance);
                self.graph.resources.advance(time)?
            };
            for (id, start) in finished {
                if self.graph.tracing_enabled()
                    && matches!(
                        self.graph.nodes[&id].kind,
                        NodeKind::Resource {
                            releases_memory: true,
                            ..
                        }
                    )
                {
                    self.graph.record_memory();
                }
                self.complete_node(id, start, time - start, &mut queue)?;
            }
            // Resource completions release ordinary descendants at the same timestamp.
            if queue.peek().is_none_or(|entry| entry.time > time) {
                continue;
            }
            let entry = queue.pop().unwrap();
            self.graph.clock = entry.time;
            self.graph.current = Some(entry.node_id);
            if self.graph.trace.is_some() {
                self.graph
                    .record(json!({"event":"dispatch", "node":entry.node_id}));
            }
            {
                let _wall = scope(Section::EventDispatch);
                self.process_entry(entry, &mut queue)?;
            }
            self.graph.current = None;
            self.graph.check_trace(false)?;
        }
        self.graph.resources.check_drained()?;
        self.graph.kv.check_drained()?;
        if last_observed != Some(self.graph.clock) {
            observe(self.graph.clock, self.system.as_ref())?;
        }
        if !self.graph.nodes.is_empty() {
            bail!(
                "graph contains a cycle or blocked nodes: {} nodes remain",
                self.graph.nodes.len()
            );
        }
        let mut statistics = self.system.policy_statistics();
        if let Some(statistics) = &mut statistics {
            statistics.simulation_end_s = Some(self.graph.clock);
        }
        let results = self.system.into_results()?;
        if self.graph.trace.is_some() {
            self.graph.record(json!({"event":"run_complete"}));
        }
        self.graph.check_trace(true)?;
        Ok((results, statistics))
    }

    fn ready_queue(&self) -> Result<BinaryHeap<QueueEntry>> {
        let mut queue = BinaryHeap::new();
        for (&node_id, node) in &self.graph.nodes {
            if !node.start_time.is_finite() || node.start_time < 0.0 {
                bail!(
                    "node {node_id} has invalid release time {}",
                    node.start_time
                );
            }
            if node.indegree == 0 {
                queue.push(QueueEntry {
                    time: node.start_time,
                    node_id,
                });
            }
        }
        Ok(queue)
    }

    fn process_entry(
        &mut self,
        entry: QueueEntry,
        queue: &mut BinaryHeap<QueueEntry>,
    ) -> Result<()> {
        let node = self
            .graph
            .nodes
            .get_mut(&entry.node_id)
            .ok_or_else(|| anyhow!("queued node {} is missing", entry.node_id))?;
        match &mut node.kind {
            NodeKind::Condition { held } => {
                anyhow::ensure!(!*held, "unreleased condition was scheduled");
                self.complete_node(entry.node_id, entry.time, 0.0, queue)
            }
            NodeKind::Resource { work, .. } => {
                let work = work.take().context("resource work submitted twice")?;
                self.graph.resources.submit_shared(entry.node_id, work)
            }
            NodeKind::Compute { .. } => self.process_compute_node(entry, queue),
            NodeKind::Network { .. } => self.process_network_node(entry, queue),
            NodeKind::Logical { .. } => self.process_logical_node(entry, queue),
        }
    }

    fn process_compute_node(
        &mut self,
        entry: QueueEntry,
        queue: &mut BinaryHeap<QueueEntry>,
    ) -> Result<()> {
        let (start_time, latency) = {
            let node = self
                .graph
                .nodes
                .get(&entry.node_id)
                .ok_or_else(|| anyhow!("queued node {} is missing", entry.node_id))?;
            let NodeKind::Compute { simulator, payload } = &node.kind else {
                bail!("node {} is not a compute node", entry.node_id);
            };
            let ComputePayload {
                layer,
                batch,
                context,
            } = payload.as_ref();
            let latency = unsafe {
                // NOTE: Safety: compute nodes store simulator pointers to stable system-owned
                // allocations. Engine execution is single-threaded, so these shared calls do not
                // race with any concurrent simulator access.
                (&*(*simulator)).simulate(layer, batch, *context)?
            };
            (node.start_time, latency)
        };
        self.complete_node(entry.node_id, start_time, latency, queue)
    }

    fn process_network_node(
        &mut self,
        entry: QueueEntry,
        queue: &mut BinaryHeap<QueueEntry>,
    ) -> Result<()> {
        let (start_time, poll_result) = {
            let node = self
                .graph
                .nodes
                .get_mut(&entry.node_id)
                .ok_or_else(|| anyhow!("queued node {} is missing", entry.node_id))?;
            let start_time = node.start_time;
            match &mut node.kind {
                NodeKind::Network {
                    simulator,
                    src,
                    dst,
                    bytes,
                    state,
                } => {
                    let simulator = *simulator;
                    let src = *src;
                    let dst = *dst;
                    let bytes = *bytes;
                    let poll_result = unsafe {
                        // NOTE: Safety: network nodes store simulator pointers to stable
                        // system-owned allocations. Engine execution is single-threaded, so
                        // mutable simulator calls never overlap.
                        (&mut *simulator).simulate(src, dst, bytes, start_time, state)?
                    };
                    (start_time, poll_result)
                }
                _ => bail!("node {} is not a network node", entry.node_id),
            }
        };
        match poll_result {
            PollResult::Pending { latency } => {
                let next_time = start_time + latency;
                if !latency.is_finite()
                    || latency <= 0.0
                    || !next_time.is_finite()
                    || next_time <= start_time
                {
                    bail!("network poll must advance finite time, got delay {latency} at {start_time}");
                }
                if self.graph.trace.is_some() {
                    self.graph.record(json!({"event":"network_pending", "node":entry.node_id, "next_poll_s":next_time}));
                }
                let node = self.graph.nodes.get_mut(&entry.node_id).ok_or_else(|| {
                    anyhow!("queued node {} disappeared during poll", entry.node_id)
                })?;
                node.start_time = next_time;
                queue.push(QueueEntry {
                    time: node.start_time,
                    node_id: entry.node_id,
                });
                Ok(())
            }
            PollResult::Complete { latency } => {
                self.complete_node(entry.node_id, start_time, latency, queue)
            }
        }
    }

    fn process_logical_node(
        &mut self,
        entry: QueueEntry,
        queue: &mut BinaryHeap<QueueEntry>,
    ) -> Result<()> {
        let (start_time, handler, payload) = {
            let node = self
                .graph
                .nodes
                .get_mut(&entry.node_id)
                .ok_or_else(|| anyhow!("queued node {} is missing", entry.node_id))?;
            let start_time = node.start_time;
            match &mut node.kind {
                NodeKind::Logical { handler, payload } => (
                    start_time,
                    *handler,
                    payload
                        .take()
                        .ok_or_else(|| anyhow!("logical node {} has no payload", entry.node_id))?,
                ),
                _ => bail!("node {} is not a logical node", entry.node_id),
            }
        };
        let latency = unsafe {
            // NOTE: Safety: logical nodes only store handler pointers to stable allocations.
            // Engine dispatch is single-threaded and completes one logical node at a time,
            // so these mutable handle calls never overlap.
            (&mut *handler).handle(payload, start_time, entry.node_id, &mut self.graph)?
        };
        self.complete_node(entry.node_id, start_time, latency, queue)
    }

    fn complete_node(
        &mut self,
        node_id: NodeID,
        start_time: f64,
        latency: f64,
        queue: &mut BinaryHeap<QueueEntry>,
    ) -> Result<()> {
        let finish_time = start_time + latency;
        if !latency.is_finite() || latency < 0.0 || !finish_time.is_finite() {
            bail!("node {node_id} has invalid latency {latency} at {start_time}");
        }
        if self.graph.trace.is_some() {
            self.graph.record(json!({"event":"node_complete", "node":node_id, "start_s":start_time, "finish_s":finish_time}));
        }
        let mut node = self.graph.take_completed_node(node_id)?;
        let children = std::mem::take(&mut node.children);
        for &child_id in children.as_slice() {
            let child = self
                .graph
                .nodes
                .get_mut(&child_id)
                .ok_or_else(|| anyhow!("child node {child_id} is missing during run"))?;
            if !child.start_time.is_finite() || child.start_time < 0.0 {
                bail!(
                    "node {child_id} has invalid release time {}",
                    child.start_time
                );
            }
            child.start_time = child.start_time.max(finish_time);
            child.indegree = child
                .indegree
                .checked_sub(1)
                .ok_or_else(|| anyhow!("child node {child_id} indegree underflow"))?;
            if child.indegree == 0 {
                queue.push(QueueEntry {
                    time: child.start_time,
                    node_id: child_id,
                });
            }
        }
        drop(node);
        Ok(())
    }
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.time.to_bits() == other.time.to_bits() && self.node_id == other.node_id
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.time.total_cmp(&self.time) {
            Ordering::Equal => other.node_id.cmp(&self.node_id),
            ordering => ordering,
        }
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn missing_edge_endpoint_does_not_mutate_the_other_endpoint() {
        let mut graph = Graph::new();
        let node = graph.add_resource_node(Default::default());
        assert!(graph.add_edge(node, 99).is_err());
        assert!(graph.add_edge(99, node).is_err());
        assert!(graph.add_edge(99, 99).is_err());
        assert!(graph.nodes[&node].children.as_slice().is_empty());
        assert_eq!(graph.nodes[&node].indegree, 0);
    }

    #[test]
    fn self_edges_retain_cycle_and_pop_semantics() {
        let mut graph = Graph::new();
        let node = graph.add_resource_node(Default::default());
        graph.add_edge(node, node).unwrap();
        assert_eq!(graph.nodes[&node].children.as_slice(), &[node]);
        assert_eq!(graph.nodes[&node].indegree, 1);
        assert!(graph.take_completed_node(node).is_err());
        assert!(graph.is_pending(node));
        assert_eq!(graph.pop_edges(node).unwrap(), vec![node]);
        assert_eq!(graph.nodes[&node].indegree, 0);
        assert!(graph.nodes[&node].children.as_slice().is_empty());
    }

    #[test]
    fn inline_and_spilled_edges_preserve_duplicates_and_pop_order() {
        for count in 0..10 {
            let mut graph = Graph::new();
            let parent = graph.add_resource_node(Default::default());
            let children = [
                graph.add_resource_node(Default::default()),
                graph.add_resource_node(Default::default()),
            ];
            let expected: Vec<_> = (0..count).map(|i| children[i % 2]).collect();
            for &child in &expected {
                graph.add_edge(parent, child).unwrap();
            }
            assert_eq!(graph.nodes[&parent].children.as_slice(), expected);
            assert_eq!(
                matches!(graph.nodes[&parent].children, Children::Many(_)),
                count > 3
            );
            assert_eq!(graph.nodes[&children[0]].indegree, count.div_ceil(2));
            assert_eq!(graph.nodes[&children[1]].indegree, count / 2);
            assert_eq!(graph.pop_edges(parent).unwrap(), expected);
            assert!(graph.nodes[&parent].children.as_slice().is_empty());
            for child in children {
                assert_eq!(graph.nodes[&child].indegree, 0);
            }
            graph.add_edge(parent, children[1]).unwrap();
            assert_eq!(graph.pop_edges(parent).unwrap(), vec![children[1]]);
        }
    }

    #[test]
    fn missing_endpoint_preserves_inline_and_spilled_edges() {
        for count in 0..6 {
            let mut graph = Graph::new();
            let parent = graph.add_resource_node(Default::default());
            let child = graph.add_resource_node(Default::default());
            for _ in 0..count {
                graph.add_edge(parent, child).unwrap();
            }
            assert!(graph.add_edge(parent, 99).is_err());
            assert!(graph.add_edge(99, child).is_err());
            assert_eq!(graph.nodes[&parent].children.as_slice(), vec![child; count]);
            assert_eq!(graph.nodes[&child].indegree, count);
        }
    }
}
