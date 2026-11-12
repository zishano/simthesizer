mod common;

use std::any::Any;

use anyhow::{bail, Result};
use common::{close, norm, FixedCompute, Recorder};
use simthesizer_core::network_sims::flow::FlowNetworkSimulator;
use simthesizer_core::types::{ComputeContext, NodeID, PollResult};
use simthesizer_core::{Engine, Graph, LogicalHandler, Module, NetworkSimulator};

#[test]
fn fork_join_waits_for_all_parents_and_the_childs_release_time() {
    let mut owner = Box::<Recorder>::default();
    let events = owner.events.clone();
    let ptr = &mut *owner as *mut dyn LogicalHandler;
    let mut graph = Graph::new();
    let a = graph.add_logical_node_at(ptr, ("a".to_string(), 3.0), 2.0);
    let b = graph.add_logical_node_at(ptr, ("b".to_string(), 5.0), 1.0);
    let join = graph.add_logical_node_at(ptr, ("join".to_string(), 2.0), 4.0);
    graph.add_edge(a, join).unwrap();
    graph.add_edge(b, join).unwrap();
    let late = graph.add_logical_node_at(ptr, ("late".to_string(), 0.0), 10.0);
    graph.add_edge(join, late).unwrap();
    Engine::new(graph, owner).run().unwrap();
    assert_eq!(
        *events.borrow(),
        vec![
            ("b".to_string(), 1.0),
            ("a".to_string(), 2.0),
            ("join".to_string(), 6.0),
            ("late".to_string(), 10.0)
        ]
    );
}

#[test]
fn simultaneous_nodes_run_in_creation_order() {
    for _ in 0..8 {
        let mut owner = Box::<Recorder>::default();
        let events = owner.events.clone();
        let ptr = &mut *owner as *mut dyn LogicalHandler;
        let mut graph = Graph::new();
        for label in ["c", "b", "a"] {
            graph.add_logical_node(ptr, (label.to_string(), 0.0));
        }
        Engine::new(graph, owner).run().unwrap();
        assert_eq!(
            *events.borrow(),
            vec![
                ("c".to_string(), 0.0),
                ("b".to_string(), 0.0),
                ("a".to_string(), 0.0)
            ]
        );
    }
}

struct Splice;
impl Module for Splice {}
impl LogicalHandler for Splice {
    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        _: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        if *payload.downcast::<bool>().unwrap() {
            let children = graph.pop_edges(current)?;
            let work = graph.add_logical_node(self as *mut dyn LogicalHandler, false);
            graph.add_edge(current, work)?;
            for child in children {
                graph.add_edge(work, child)?;
            }
            Ok(2.0)
        } else {
            Ok(3.0)
        }
    }
}

#[test]
fn dynamic_splicing_preserves_static_dependencies() {
    let mut owner = Box::<Recorder>::default();
    let events = owner.events.clone();
    let mut splice = Box::new(Splice);
    let mut graph = Graph::new();
    let start = graph.add_logical_node(&mut *splice as *mut dyn LogicalHandler, true);
    let finish = graph.add_logical_node(
        &mut *owner as *mut dyn LogicalHandler,
        ("finish".to_string(), 0.0),
    );
    graph.add_edge(start, finish).unwrap();
    Engine::new(graph, owner).run().unwrap();
    assert_eq!(*events.borrow(), vec![("finish".to_string(), 5.0)]);
}

#[test]
fn cycles_and_missing_node_ids_fail() {
    let mut owner = Box::<Recorder>::default();
    let ptr = &mut *owner as *mut dyn LogicalHandler;
    let mut graph = Graph::new();
    let a = graph.add_logical_node(ptr, ("a".to_string(), 0.0));
    let b = graph.add_logical_node(ptr, ("b".to_string(), 0.0));
    assert!(graph.add_edge(a, 99).is_err());
    assert!(graph.pop_edges(99).is_err());
    graph.add_edge(a, b).unwrap();
    graph.add_edge(b, a).unwrap();
    assert!(Engine::new(graph, owner).run().is_err());
}

#[test]
fn invalid_release_times_are_rejected() {
    for time in [-1.0, f64::NAN, f64::INFINITY] {
        let mut owner = Box::<Recorder>::default();
        let mut graph = Graph::new();
        graph.add_logical_node_at(
            &mut *owner as *mut dyn LogicalHandler,
            ("bad".to_string(), 0.0),
            time,
        );
        assert!(Engine::new(graph, owner).run().is_err(), "accepted {time}");
    }
}

#[test]
fn invalid_logical_costs_and_finish_time_overflow_are_rejected() {
    for (start, duration) in [
        (0.0, -1.0),
        (0.0, f64::NAN),
        (0.0, f64::INFINITY),
        (f64::MAX, f64::MAX),
    ] {
        let mut owner = Box::<Recorder>::default();
        let mut graph = Graph::new();
        graph.add_logical_node_at(
            &mut *owner as *mut dyn LogicalHandler,
            ("bad".to_string(), duration),
            start,
        );
        assert!(Engine::new(graph, owner).run().is_err());
    }
}

#[test]
fn invalid_compute_costs_are_rejected_even_for_terminal_nodes() {
    for cost in [-1.0, f64::NAN, f64::INFINITY] {
        let mut compute = Box::new(FixedCompute {
            costs: [("work".to_string(), cost)].into(),
            per_token: false,
        });
        let mut graph = Graph::new();
        graph.add_compute_node(
            &mut *compute,
            norm("work"),
            common::batch(1),
            ComputeContext::default(),
        );
        assert!(Engine::new(graph, Box::<Recorder>::default())
            .run()
            .is_err());
    }
}

struct BadPoll(f64);
impl Module for BadPoll {}
impl NetworkSimulator for BadPoll {
    fn add_device(&mut self) -> u64 {
        0
    }
    fn add_link(&mut self, _: u64, _: u64, _: f64, _: f64) -> Result<()> {
        Ok(())
    }
    fn simulate(
        &mut self,
        _: u64,
        _: u64,
        _: u64,
        _: f64,
        state: &mut Option<Box<dyn Any>>,
    ) -> Result<PollResult> {
        if state.is_some() {
            return Ok(PollResult::Complete { latency: 0.0 });
        }
        *state = Some(Box::new(()));
        Ok(PollResult::Pending { latency: self.0 })
    }
}

#[test]
fn pending_network_polls_must_advance_finite_time() {
    for delay in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut net = Box::new(BadPoll(delay));
        let mut graph = Graph::new();
        graph.add_network_node(&mut *net, 0, 1, 1);
        assert!(Engine::new(graph, Box::<Recorder>::default())
            .run()
            .is_err());
    }
}

#[test]
fn observer_errors_stop_the_run() {
    let result = Engine::new(Graph::new(), Box::<Recorder>::default())
        .run_with_observer(|_, _| bail!("observer failed"));
    assert_eq!(result.unwrap_err().to_string(), "observer failed");
}

#[test]
fn network_completion_releases_children_at_the_revised_finish_time() {
    let mut owner = Box::<Recorder>::default();
    let events = owner.events.clone();
    let ptr = &mut *owner as *mut dyn LogicalHandler;
    let mut net = Box::<FlowNetworkSimulator>::default();
    let src = net.add_device();
    let dst = net.add_device();
    net.add_link(src, dst, 10.0, 0.0).unwrap();
    let mut graph = Graph::new();
    for (name, bytes) in [("short", 10), ("long", 100)] {
        let flow = graph.add_network_node(&mut *net, src, dst, bytes);
        let done = graph.add_logical_node(ptr, (name.to_string(), 0.0));
        graph.add_edge(flow, done).unwrap();
    }
    Engine::new(graph, owner).run().unwrap();
    let events = events.borrow();
    close(events[0].1, 2.0);
    close(events[1].1, 11.0);
}

#[derive(Clone, Default)]
struct TraceBuffer(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);
impl std::io::Write for TraceBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn graph_trace_records_network_repolls_and_future_completion() {
    let trace = TraceBuffer::default();
    let mut graph = Graph::with_trace(trace.clone());
    let mut net = Box::new(BadPoll(2.0));
    graph.add_network_node(&mut *net, 0, 1, 1024);
    Engine::new(graph, Box::<Recorder>::default())
        .run()
        .unwrap();
    let bytes = trace.0.borrow();
    let rows: Vec<serde_json::Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let pending = rows
        .iter()
        .find(|r| r["event"] == "network_pending")
        .unwrap();
    assert_eq!(pending["next_poll_s"], 2.0);
    assert_eq!(rows.iter().filter(|r| r["event"] == "dispatch").count(), 2);
    let done = rows.iter().find(|r| r["event"] == "node_complete").unwrap();
    assert_eq!(done["finish_s"], 2.0);
    assert_eq!(rows[1]["details"]["bytes"], 1024);
}

#[test]
fn graph_trace_write_errors_fail_the_run() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("trace disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let error = Engine::new(Graph::with_trace(Broken), Box::<Recorder>::default())
        .run()
        .unwrap_err();
    assert!(error.to_string().contains("trace disk full"));
}
