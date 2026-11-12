use std::any::Any;
use std::path::Path;

use anyhow::Result;

use crate::engine::Graph;
use crate::types::NodeID;

pub trait Module {
    /// Read-only state in the component's own schema. Null means unavailable.
    fn observe(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    fn resolve_path(&mut self, _base_dir: &Path) -> Result<()> {
        Ok(())
    }
}

pub trait LogicalHandler {
    /// Optional human-readable payload description for graph tracing.
    fn describe(&self, _payload: &dyn Any) -> String {
        std::any::type_name::<Self>().to_owned()
    }

    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64>;
}
