//! Causal policy decisions. The execution owners validate and commit all work.
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use crate::types::{InstanceLoad, NodeID};

pub mod candidate;

pub const API_VERSION: &str = "serving-policy-v1";
pub type SharedPolicy = Rc<RefCell<PolicyRuntime>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct InstanceId {
    pub stage: Stage,
    pub index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    Reference,
    Candidate,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyLimits {
    pub p_token_budget_bounds: [u32; 2],
    pub d_token_budget_bounds: [u32; 2],
    pub p_batch_bounds: [usize; 2],
    pub d_batch_bounds: [usize; 2],
    pub kv_inflight_bounds: [usize; 2],
    pub max_wait_s: f64,
    pub state_limit_bytes: usize,
}

impl PolicyLimits {
    pub fn batch(&self, stage: Stage) -> ([u32; 2], [usize; 2]) {
        match stage {
            Stage::Prefill => (self.p_token_budget_bounds, self.p_batch_bounds),
            Stage::Decode => (self.d_token_budget_bounds, self.d_batch_bounds),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySlo {
    pub ttft_limit_s: f64,
    pub tpot_limit_s: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    pub kind: PolicyKind,
    pub seed: u64,
    pub limits: PolicyLimits,
    pub slo: PolicySlo,
}

impl PolicyConfig {
    pub fn validate(&self) -> Result<()> {
        for [lo, hi] in [
            self.limits.p_token_budget_bounds,
            self.limits.d_token_budget_bounds,
        ] {
            ensure!(lo > 0 && lo <= hi, "invalid policy token budget bounds");
        }
        for [lo, hi] in [
            self.limits.p_batch_bounds,
            self.limits.d_batch_bounds,
            self.limits.kv_inflight_bounds,
        ] {
            ensure!(
                lo > 0 && lo <= hi,
                "invalid policy batch/concurrency bounds"
            );
        }
        ensure!(
            self.limits.state_limit_bytes > 0,
            "policy state_limit_bytes must be positive"
        );
        for value in [
            self.limits.max_wait_s,
            self.slo.ttft_limit_s,
            self.slo.tpot_limit_s,
        ] {
            ensure!(
                value.is_finite() && value > 0.0,
                "policy time limits must be finite and positive"
            );
        }
        Ok(())
    }
}

/// Only currently available work; no requested output length or future token IDs.
#[derive(Debug, Clone, Copy)]
pub struct QueuedRequest {
    pub completion_node_id: NodeID,
    pub request_id: u64,
    pub subrequest_index: usize,
    pub arrival_time: f64,
    pub input_tokens: u32,
    pub committed_tokens: u32,
    pub kv_tokens: u32,
    pub max_q_len: u32,
    pub last_token_time: Option<f64>,
}

pub struct RouteObservation<'a> {
    pub now: f64,
    pub request_id: u64,
    pub input_tokens: u32,
    pub prefill: &'a [InstanceLoad],
    pub decode: &'a [InstanceLoad],
}

pub struct BatchObservation<'a> {
    pub now: f64,
    pub instance: InstanceId,
    pub requests: &'a [QueuedRequest],
    pub default_token_budget: u32,
    pub default_max_batch: usize,
    pub memory_limit_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct PendingKv {
    pub id: u64,
    pub request_id: u64,
    pub layer: u32,
    pub source_rank: u32,
    pub target_rank: u32,
    pub token_start: u32,
    pub token_end: u32,
    pub bytes: u64,
    pub produced_at: f64,
}

pub struct KvObservation<'a> {
    pub now: f64,
    pub pending: &'a [PendingKv],
    pub inflight: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RouteAction {
    pub prefill: usize,
    pub decode: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BatchItem {
    pub completion_node_id: NodeID,
    pub q_len: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BatchPlan {
    pub items: Vec<BatchItem>,
    pub token_budget: u32,
    pub max_batch: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum BatchAction {
    Run(BatchPlan),
    Wait { until: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KvAction {
    pub submissions: Vec<u64>,
    pub wait_until: Option<f64>,
}

pub struct PolicyContext<'a> {
    pub config: &'a PolicyConfig,
    pub route_sequence: u64,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct PolicyStatistics {
    pub route_decisions: u64,
    pub batch_decisions: u64,
    pub kv_decisions: u64,
    pub waits: u64,
    pub decision_wall_seconds: f64,
    pub simulation_end_s: Option<f64>,
}

type RouteFn = fn(&RouteObservation<'_>, &PolicyContext<'_>, &mut [u8]) -> RouteAction;
type BatchFn = fn(&BatchObservation<'_>, &PolicyContext<'_>, &mut [u8]) -> BatchAction;
type KvFn = fn(&KvObservation<'_>, &PolicyContext<'_>, &mut [u8]) -> KvAction;

pub struct PolicyRuntime {
    config: PolicyConfig,
    memory: Box<[u8]>,
    statistics: PolicyStatistics,
    route_fn: RouteFn,
    batch_fn: BatchFn,
    kv_fn: KvFn,
}

impl PolicyRuntime {
    pub fn new(config: PolicyConfig) -> Result<Self> {
        config.validate()?;
        // This is the only retained mutable memory passed to candidate functions.
        let memory = vec![0; config.limits.state_limit_bytes].into_boxed_slice();
        let (route_fn, batch_fn, kv_fn): (RouteFn, BatchFn, KvFn) = match config.kind {
            PolicyKind::Reference => (reference_route, reference_batch, reference_kv),
            PolicyKind::Candidate => (candidate::route, candidate::batch, candidate::kv),
        };
        Ok(Self {
            config,
            memory,
            statistics: PolicyStatistics::default(),
            route_fn,
            batch_fn,
            kv_fn,
        })
    }

    pub fn is_reference(&self) -> bool {
        self.config.kind == PolicyKind::Reference
    }
    pub fn limits(&self) -> &PolicyLimits {
        &self.config.limits
    }
    pub fn statistics(&self) -> PolicyStatistics {
        self.statistics.clone()
    }

    pub fn record_reference_batch(&mut self) {
        self.statistics.batch_decisions += 1;
    }
    pub fn record_reference_kv(&mut self) {
        self.statistics.kv_decisions += 1;
    }
    pub fn record_reference_route(&mut self) {
        self.statistics.route_decisions += 1;
    }

    pub fn route(&mut self, observation: &RouteObservation<'_>) -> Result<RouteAction> {
        ensure!(
            observation.now.is_finite()
                && !observation.prefill.is_empty()
                && !observation.decode.is_empty(),
            "route requires current time and nonempty P/D pools"
        );
        let started = Instant::now();
        let context = PolicyContext {
            config: &self.config,
            route_sequence: self.statistics.route_decisions,
        };
        let action = (self.route_fn)(observation, &context, &mut self.memory);
        self.statistics.decision_wall_seconds += started.elapsed().as_secs_f64();
        self.statistics.route_decisions += 1;
        ensure!(
            action.prefill < observation.prefill.len() && action.decode < observation.decode.len(),
            "policy route selects unknown instance"
        );
        Ok(action)
    }

    pub fn batch(&mut self, observation: &BatchObservation<'_>) -> Result<BatchAction> {
        let started = Instant::now();
        let context = PolicyContext {
            config: &self.config,
            route_sequence: self.statistics.route_decisions,
        };
        let action = (self.batch_fn)(observation, &context, &mut self.memory);
        self.statistics.decision_wall_seconds += started.elapsed().as_secs_f64();
        self.statistics.batch_decisions += 1;
        let (tokens, batches) = self.config.limits.batch(observation.instance.stage);
        match &action {
            BatchAction::Wait { until } => {
                validate_wait(observation.now, *until, self.config.limits.max_wait_s)?;
                self.statistics.waits += 1;
            }
            BatchAction::Run(plan) => {
                let items = &plan.items;
                ensure!(
                    (tokens[0]..=tokens[1]).contains(&plan.token_budget),
                    "policy token budget outside bounds"
                );
                ensure!(
                    (batches[0]..=batches[1]).contains(&plan.max_batch),
                    "policy batch capacity outside bounds"
                );
                ensure!(
                    !items.is_empty(),
                    "empty policy batch requires a future wakeup"
                );
                ensure!(
                    items.len() <= plan.max_batch,
                    "policy batch exceeds batch capacity"
                );
                let available: HashMap<_, _> = observation
                    .requests
                    .iter()
                    .map(|r| (r.completion_node_id, r))
                    .collect();
                let mut selected = HashSet::with_capacity(items.len());
                let mut total = 0_u64;
                for item in items {
                    ensure!(
                        selected.insert(item.completion_node_id),
                        "duplicate policy batch request"
                    );
                    let request = available.get(&item.completion_node_id).ok_or_else(|| {
                        anyhow::anyhow!("policy selects unknown or not-yet-arrived request")
                    })?;
                    ensure!(
                        item.q_len > 0 && item.q_len <= request.max_q_len,
                        "policy q_len exceeds available known tokens"
                    );
                    total += u64::from(item.q_len);
                }
                ensure!(
                    total <= u64::from(plan.token_budget),
                    "policy batch exceeds token budget"
                );
            }
        }
        Ok(action)
    }

    pub fn kv(&mut self, observation: &KvObservation<'_>) -> Result<KvAction> {
        let started = Instant::now();
        let context = PolicyContext {
            config: &self.config,
            route_sequence: self.statistics.route_decisions,
        };
        let action = (self.kv_fn)(observation, &context, &mut self.memory);
        self.statistics.decision_wall_seconds += started.elapsed().as_secs_f64();
        self.statistics.kv_decisions += 1;
        let mut ids = HashSet::with_capacity(action.submissions.len());
        let pending: HashSet<_> = observation.pending.iter().map(|p| p.id).collect();
        for id in &action.submissions {
            ensure!(ids.insert(*id), "duplicate KV submission");
            ensure!(pending.contains(id), "KV submission is not pending");
        }
        ensure!(
            action.submissions.len()
                <= self.config.limits.kv_inflight_bounds[1].saturating_sub(observation.inflight),
            "KV submission exceeds concurrency bound"
        );
        if let Some(until) = action.wait_until {
            validate_wait(observation.now, until, self.config.limits.max_wait_s)?;
            self.statistics.waits += 1;
        }
        ensure!(
            observation.pending.is_empty()
                || !action.submissions.is_empty()
                || observation.inflight > 0
                || action.wait_until.is_some(),
            "pending KV requires submission, in-flight ACK, or future wakeup"
        );
        Ok(action)
    }
}

pub fn validate_wait(now: f64, until: f64, max_wait_s: f64) -> Result<()> {
    ensure!(
        now.is_finite() && until.is_finite() && until > now && until <= now + max_wait_s,
        "policy wakeup must be finite, future, and within max_wait_s"
    );
    Ok(())
}

pub fn reference_route(
    observation: &RouteObservation<'_>,
    context: &PolicyContext<'_>,
    _: &mut [u8],
) -> RouteAction {
    RouteAction {
        prefill: (context.route_sequence % observation.prefill.len() as u64) as usize,
        decode: (context.route_sequence % observation.decode.len() as u64) as usize,
    }
}

pub fn reference_batch(
    observation: &BatchObservation<'_>,
    _: &PolicyContext<'_>,
    _: &mut [u8],
) -> BatchAction {
    let mut remaining = observation.default_token_budget;
    let mut selected = Vec::new();
    for request in observation
        .requests
        .iter()
        .take(observation.default_max_batch)
    {
        if remaining == 0 {
            break;
        }
        let q_len = request.max_q_len.min(remaining);
        selected.push(BatchItem {
            completion_node_id: request.completion_node_id,
            q_len,
        });
        remaining -= q_len;
    }
    BatchAction::Run(BatchPlan {
        items: selected,
        token_budget: observation.default_token_budget,
        max_batch: observation.default_max_batch,
    })
}

pub fn reference_kv(
    observation: &KvObservation<'_>,
    context: &PolicyContext<'_>,
    _: &mut [u8],
) -> KvAction {
    let available =
        context.config.limits.kv_inflight_bounds[1].saturating_sub(observation.inflight);
    KvAction {
        submissions: observation
            .pending
            .iter()
            .take(available)
            .map(|p| p.id)
            .collect(),
        wait_until: None,
    }
}

#[cfg(test)]
impl PolicyRuntime {
    pub(crate) fn set_batch_for_test(&mut self, callback: BatchFn) {
        self.batch_fn = callback;
    }
    pub(crate) fn set_route_for_test(&mut self, callback: RouteFn) {
        self.route_fn = callback;
    }
}

#[cfg(test)]
pub(crate) fn test_config() -> PolicyConfig {
    PolicyConfig {
        kind: PolicyKind::Candidate,
        seed: 0,
        limits: PolicyLimits {
            p_token_budget_bounds: [1, 16],
            d_token_budget_bounds: [1, 16],
            p_batch_bounds: [1, 4],
            d_batch_bounds: [1, 4],
            kv_inflight_bounds: [1, 2],
            max_wait_s: 1.0,
            state_limit_bytes: 8,
        },
        slo: PolicySlo {
            ttft_limit_s: 1.0,
            tpot_limit_s: 0.1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue() -> Vec<QueuedRequest> {
        (0..3)
            .map(|id| QueuedRequest {
                completion_node_id: id,
                request_id: id,
                subrequest_index: 0,
                arrival_time: 0.0,
                input_tokens: 4,
                committed_tokens: 0,
                kv_tokens: 0,
                max_q_len: 4,
                last_token_time: None,
            })
            .collect()
    }

    fn observation(requests: &[QueuedRequest]) -> BatchObservation<'_> {
        BatchObservation {
            now: 1.0,
            instance: InstanceId {
                stage: Stage::Prefill,
                index: 0,
            },
            requests,
            default_token_budget: 8,
            default_max_batch: 4,
            memory_limit_bytes: Some(1024),
        }
    }

    #[test]
    fn invalid_waits_cannot_create_a_zero_time_loop() {
        for until in [0.0, 1.0, 2.1, f64::INFINITY, f64::NAN] {
            assert!(validate_wait(1.0, until, 1.0).is_err());
        }
        assert!(validate_wait(1.0, 1.5, 1.0).is_ok());
        assert!(validate_wait(0.1, 0.1 + 0.05, 0.05).is_ok());
    }

    #[test]
    fn arbitrary_ids_are_validated_without_requiring_a_prefix() {
        let queue = queue();
        let mut policy = PolicyRuntime::new(test_config()).unwrap();
        policy.batch_fn = |_, _, _| {
            BatchAction::Run(BatchPlan {
                items: vec![
                    BatchItem {
                        completion_node_id: 0,
                        q_len: 2,
                    },
                    BatchItem {
                        completion_node_id: 2,
                        q_len: 4,
                    },
                ],
                token_budget: 8,
                max_batch: 4,
            })
        };
        assert!(policy.batch(&observation(&queue)).is_ok());
        assert_eq!(policy.statistics().batch_decisions, 1);
    }

    #[test]
    fn malformed_batches_are_rejected_before_commit() {
        let queue = queue();
        let mut policy = PolicyRuntime::new(test_config()).unwrap();
        for callback in [
            (|_: &BatchObservation<'_>, _: &PolicyContext<'_>, _: &mut [u8]| {
                BatchAction::Run(BatchPlan {
                    items: vec![],
                    token_budget: 8,
                    max_batch: 4,
                })
            }) as BatchFn,
            |_, _, _| {
                BatchAction::Run(BatchPlan {
                    items: vec![BatchItem {
                        completion_node_id: 7,
                        q_len: 1,
                    }],
                    token_budget: 8,
                    max_batch: 4,
                })
            },
            |_, _, _| {
                BatchAction::Run(BatchPlan {
                    items: vec![BatchItem {
                        completion_node_id: 0,
                        q_len: 0,
                    }],
                    token_budget: 8,
                    max_batch: 4,
                })
            },
            |_, _, _| {
                BatchAction::Run(BatchPlan {
                    items: vec![BatchItem {
                        completion_node_id: 0,
                        q_len: 5,
                    }],
                    token_budget: 8,
                    max_batch: 4,
                })
            },
            |_, _, _| {
                BatchAction::Run(BatchPlan {
                    items: vec![
                        BatchItem {
                            completion_node_id: 0,
                            q_len: 2
                        };
                        2
                    ],
                    token_budget: 8,
                    max_batch: 4,
                })
            },
            |_, _, _| {
                BatchAction::Run(BatchPlan {
                    items: vec![BatchItem {
                        completion_node_id: 0,
                        q_len: 2,
                    }],
                    token_budget: 1,
                    max_batch: 4,
                })
            },
            |_, _, _| {
                BatchAction::Run(BatchPlan {
                    items: vec![BatchItem {
                        completion_node_id: 0,
                        q_len: 2,
                    }],
                    token_budget: 17,
                    max_batch: 4,
                })
            },
            |_, _, _| {
                BatchAction::Run(BatchPlan {
                    items: vec![BatchItem {
                        completion_node_id: 0,
                        q_len: 2,
                    }],
                    token_budget: 8,
                    max_batch: 5,
                })
            },
        ] {
            policy.batch_fn = callback;
            assert!(policy.batch(&observation(&queue)).is_err());
        }
    }

    #[test]
    fn pending_kv_cannot_be_abandoned_and_duplicate_submission_is_invalid() {
        let mut policy = PolicyRuntime::new(test_config()).unwrap();
        let pending = [PendingKv {
            id: 1,
            request_id: 0,
            layer: 0,
            source_rank: 0,
            target_rank: 0,
            token_start: 0,
            token_end: 4,
            bytes: 8,
            produced_at: 0.0,
        }];
        policy.kv_fn = |_, _, _| KvAction {
            submissions: vec![],
            wait_until: None,
        };
        assert!(policy
            .kv(&KvObservation {
                now: 1.0,
                pending: &pending,
                inflight: 0
            })
            .is_err());
        assert!(policy
            .kv(&KvObservation {
                now: 1.0,
                pending: &pending,
                inflight: 1
            })
            .is_ok());
        policy.kv_fn = |_, _, _| KvAction {
            submissions: vec![1, 1],
            wait_until: None,
        };
        assert!(policy
            .kv(&KvObservation {
                now: 1.0,
                pending: &pending,
                inflight: 0
            })
            .is_err());
        policy.kv_fn = |_, _, _| KvAction {
            submissions: vec![2],
            wait_until: None,
        };
        assert!(policy
            .kv(&KvObservation {
                now: 1.0,
                pending: &pending,
                inflight: 0
            })
            .is_err());
    }

    #[test]
    fn new_trials_reset_bounded_state_and_route_sequences() {
        fn stateful(
            _: &RouteObservation<'_>,
            _: &PolicyContext<'_>,
            memory: &mut [u8],
        ) -> RouteAction {
            let prefill = memory[0] as usize;
            memory[0] = (memory[0] + 1) % 2;
            RouteAction { prefill, decode: 0 }
        }
        let loads = [InstanceLoad::default(); 2];
        let observation = RouteObservation {
            now: 0.0,
            request_id: 0,
            input_tokens: 4,
            prefill: &loads,
            decode: &loads,
        };
        for _ in 0..2 {
            let mut runtime = PolicyRuntime::new(test_config()).unwrap();
            assert_eq!(runtime.memory.len(), 8);
            runtime.route_fn = stateful;
            assert_eq!(runtime.route(&observation).unwrap().prefill, 0);
            assert_eq!(runtime.route(&observation).unwrap().prefill, 1);
        }
    }
}
