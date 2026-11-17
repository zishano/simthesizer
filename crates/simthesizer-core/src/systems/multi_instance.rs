use std::any::Any;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::engine::Graph;
use crate::module::{LogicalHandler, Module};
use crate::router::Router;
use crate::system::{pending_llm_load, System};
use crate::types::{InstanceLoad, NodeID, Request, RequestResult};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiInstanceSystem {
    instances: Vec<Box<dyn System>>,
    #[serde(default = "default_router")]
    router: Box<dyn Router>,

    #[serde(skip)]
    pending_requests: HashMap<u64, PendingRequest>,
}

struct PendingRequest {
    request: Request,
    sink: Option<*mut dyn LogicalHandler>,
}

crate::register_system!(MultiInstanceSystem, "multi_instance");

fn default_router() -> Box<dyn Router> {
    Box::new(crate::routers::round_robin::RoundRobinRouter::default())
}

impl LogicalHandler for MultiInstanceSystem {
    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        _now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        let request_id = *payload
            .downcast::<u64>()
            .map_err(|_| anyhow!("multi_instance received an unexpected payload"))?;
        let pending = self
            .pending_requests
            .remove(&request_id)
            .ok_or_else(|| anyhow!("multi_instance is missing request {request_id}"))?;
        let loads = self
            .instances
            .iter()
            .map(|instance| instance.instance_load())
            .collect::<Vec<_>>();
        let index = self.router.select_instance(&pending.request, &loads)?;
        let child = self
            .instances
            .get_mut(index)
            .ok_or_else(|| anyhow!("router selected invalid instance index {index}"))?;
        child.add_request_arrival(pending.request, graph, Some(current), pending.sink)?;
        Ok(0.0)
    }
}

impl Module for MultiInstanceSystem {
    fn observe(&self) -> serde_json::Value {
        serde_json::json!({
            "instances": self.instances.iter().map(|s| s.observe()).collect::<Vec<_>>()
        })
    }

    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        self.router.resolve_path(base_dir)?;
        for instance in &mut self.instances {
            instance.resolve_path(base_dir)?;
        }
        Ok(())
    }
}

impl System for MultiInstanceSystem {
    fn completed_requests(&self) -> Option<usize> {
        self.instances.iter().try_fold(0, |total, system| {
            Some(total + system.completed_requests()?)
        })
    }

    fn instance_load(&self) -> InstanceLoad {
        self.instances
            .iter()
            .map(|s| s.instance_load())
            .chain(self.pending_requests.values().map(|pending| {
                pending_llm_load(&pending.request, pending.request.initial.iter().copied())
            }))
            .sum()
    }

    fn init(&mut self) -> Result<()> {
        if self.instances.is_empty() {
            bail!("multi_instance requires at least one child instance");
        }

        for instance in &mut self.instances {
            instance.init()?;
        }
        Ok(())
    }

    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()> {
        request.validate()?;
        if self.instances.is_empty() {
            bail!("multi_instance requires at least one child instance");
        }
        let request_id = request.request_id;
        if self.pending_requests.contains_key(&request_id) {
            bail!("multi_instance received duplicate request id {request_id}");
        }
        let node = graph.add_logical_node_at(
            self as *mut Self as *mut dyn LogicalHandler,
            request_id,
            request.arrival_time,
        );
        if let Some(parent) = parent {
            graph.add_edge(parent, node)?;
        }
        self.pending_requests
            .insert(request_id, PendingRequest { request, sink });
        Ok(())
    }

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>> {
        let MultiInstanceSystem {
            instances,
            pending_requests,
            ..
        } = *self;
        if !pending_requests.is_empty() {
            bail!(
                "multi_instance finished with {} requests still awaiting arrival",
                pending_requests.len()
            );
        }
        let mut results = Vec::new();
        for instance in instances {
            results.extend(instance.into_results()?);
        }
        Ok(results)
    }
}
