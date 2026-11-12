use serde_json::{json, Value};
use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::Write;

use anyhow::{anyhow, bail, Result};

use crate::compute_sim::ComputeSimulator;
use crate::module::LogicalHandler;
use crate::network_sim::NetworkSimulator;
use crate::system::System;
use crate::types::{
    ComputeContext, Layer, NetworkDeviceID, NodeID, PollResult, RequestResult, RequestState,
};

pub struct Graph {
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
    children: Vec<NodeID>,
    indegree: usize,
    start_time: f64,
}

enum NodeKind {
    Compute {
        simulator: *mut dyn ComputeSimulator,
        layer: Layer,
        batch: Vec<RequestState>,
        context: ComputeContext,
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

#[derive(Clone, Copy, Debug)]
struct QueueEntry {
    time: f64,
    node_id: NodeID,
}

#[derive(Clone, Copy)]
enum EntryKind {
    Compute,
    Network,
    Logical,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
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

    fn record(&mut self, mut event: Value) {
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
                layer,
                batch,
                context,
            },
            0.0,
        )
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
        self.ensure_node_exists(parent)?;
        self.ensure_node_exists(child)?;
        self.nodes
            .get_mut(&parent)
            .ok_or_else(|| anyhow!("node {parent} disappeared during edge insertion"))?
            .children
            .push(child);
        self.nodes
            .get_mut(&child)
            .ok_or_else(|| anyhow!("node {child} disappeared during edge insertion"))?
            .indegree += 1;
        if self.trace.is_some() {
            self.record(json!({"event":"edge_add", "parent":parent, "child":child}));
        }
        Ok(())
    }

    pub fn pop_edges(&mut self, parent: NodeID) -> Result<Vec<NodeID>> {
        self.ensure_node_exists(parent)?;
        let parent_node = self
            .nodes
            .get_mut(&parent)
            .ok_or_else(|| anyhow!("node {parent} disappeared during edge removal"))?;
        let children = std::mem::take(&mut parent_node.children);

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

    fn remove_completed_node(&mut self, node_id: NodeID) -> Result<()> {
        let indegree = self
            .nodes
            .get(&node_id)
            .ok_or_else(|| anyhow!("node {node_id} does not exist"))?
            .indegree;
        if indegree != 0 {
            bail!("node {node_id} is not completed: indegree is {indegree}");
        }
        self.nodes
            .remove(&node_id)
            .ok_or_else(|| anyhow!("node {node_id} disappeared during removal"))?;
        Ok(())
    }

    fn insert_node(&mut self, kind: NodeKind, start_time: f64) -> NodeID {
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        if self.trace.is_some() {
            let details = match &kind {
                NodeKind::Compute {
                    layer,
                    batch,
                    context,
                    ..
                } => json!({
                    "kind":"compute", "label":layer.name(), "layer":format!("{layer:?}"),
                    "batch":batch.iter().map(|r| json!({"q_len":r.q_len,"kv_len":r.kv_len,"lm_head_len":r.lm_head_len})).collect::<Vec<_>>(),
                    "context":format!("{context:?}")}),
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
                children: Vec::new(),
                indegree: 0,
                start_time,
            },
        );
        node_id
    }

    fn ensure_node_exists(&self, node_id: NodeID) -> Result<()> {
        if self.nodes.contains_key(&node_id) {
            return Ok(());
        }
        bail!("node {node_id} does not exist")
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
        mut self,
        mut observe: impl FnMut(f64, &dyn System) -> Result<()>,
    ) -> Result<Vec<RequestResult>> {
        self.graph.check_trace(false)?;
        let mut queue = self.ready_queue()?;
        if queue.peek().is_none_or(|entry| entry.time > 0.0) {
            observe(0.0, self.system.as_ref())?;
        }
        while let Some(entry) = queue.pop() {
            self.graph.clock = entry.time;
            self.graph.current = Some(entry.node_id);
            if self.graph.trace.is_some() {
                self.graph
                    .record(json!({"event":"dispatch", "node":entry.node_id}));
            }
            self.process_entry(entry, &mut queue)?;
            self.graph.current = None;
            self.graph.check_trace(false)?;
            if queue.peek().is_none_or(|next| next.time != entry.time) {
                observe(entry.time, self.system.as_ref())?;
            }
        }
        if !self.graph.nodes.is_empty() {
            bail!(
                "graph contains a cycle or blocked nodes: {} nodes remain",
                self.graph.nodes.len()
            );
        }
        let results = self.system.into_results()?;
        if self.graph.trace.is_some() {
            self.graph.record(json!({"event":"run_complete"}));
        }
        self.graph.check_trace(true)?;
        Ok(results)
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
        match self.entry_kind(entry.node_id)? {
            EntryKind::Compute => self.process_compute_node(entry, queue),
            EntryKind::Network => self.process_network_node(entry, queue),
            EntryKind::Logical => self.process_logical_node(entry, queue),
        }
    }

    fn entry_kind(&self, node_id: NodeID) -> Result<EntryKind> {
        let node = self
            .graph
            .nodes
            .get(&node_id)
            .ok_or_else(|| anyhow!("queued node {node_id} is missing"))?;
        match &node.kind {
            NodeKind::Compute { .. } => Ok(EntryKind::Compute),
            NodeKind::Network { .. } => Ok(EntryKind::Network),
            NodeKind::Logical { .. } => Ok(EntryKind::Logical),
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
            let NodeKind::Compute {
                simulator,
                layer,
                batch,
                context,
            } = &node.kind
            else {
                bail!("node {} is not a compute node", entry.node_id);
            };
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
        let children = self
            .graph
            .nodes
            .get(&node_id)
            .ok_or_else(|| anyhow!("completed node {node_id} is missing"))?
            .children
            .to_vec();
        for child_id in children {
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
        self.graph.remove_completed_node(node_id)?;
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
