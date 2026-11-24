use anyhow::{ensure, Context, Result};
use simthesizer_core::System;

pub struct Progress {
    enabled: bool,
    total: usize,
    previous: Option<usize>,
}

impl Progress {
    pub fn new(enabled: bool, total: usize) -> Self {
        Self {
            enabled,
            total,
            previous: None,
        }
    }

    pub fn observe(&mut self, time_s: f64, system: &dyn System) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let completed = system
            .completed_requests()
            .context("system does not support request progress")?;
        ensure!(
            completed <= self.total && self.previous.is_none_or(|last| completed >= last),
            "invalid completed-request count"
        );
        if self.previous != Some(completed) {
            let mut event =
                serde_json::json!({"completed": completed, "total": self.total, "time_s": time_s});
            if let Some(profile) = simthesizer_core::wall_profile::snapshot() {
                event["wall_profile"] = serde_json::to_value(profile)?;
            }
            eprintln!("SIMTHESIZER_PROGRESS {event}");
            self.previous = Some(completed);
        }
        Ok(())
    }
}
