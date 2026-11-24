//! Opt-in search components. Disabled components preserve the reference actions.
//! Queue age is stage-local; pressure is remaining work, not utilization.
use crate::policies::{
    reference_batch, reference_kv, reference_route, BatchAction, BatchItem, BatchObservation,
    BatchPlan, InstanceId, KvAction, KvObservation, PolicyContext, RouteAction, RouteObservation,
    Stage,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub routing_load: bool,
    pub aged_short_prefill: bool,
    pub decode_fair: bool,
    pub microbatch_wait: bool,
    pub pressure_chunks: bool,
    pub kv_demand_aging: bool,
    pub hysteresis: bool,
    pub wait_s: f64,
    pub age_s: f64,
    pub pressure_low_s: f64,
    pub pressure_high_s: f64,
    pub dwell_s: f64,
    pub small_chunk_tokens: u32,
    pub large_chunk_tokens: u32,
    pub wait_batch_threshold: usize,
}

impl Options {
    pub const BASELINE: Self = Self {
        routing_load: false,
        aged_short_prefill: false,
        decode_fair: false,
        microbatch_wait: false,
        pressure_chunks: false,
        kv_demand_aging: false,
        hysteresis: false,
        wait_s: 0.0001,
        age_s: 0.01,
        pressure_low_s: 0.0001,
        pressure_high_s: 0.001,
        dwell_s: 0.001,
        small_chunk_tokens: 32,
        large_chunk_tokens: 512,
        wait_batch_threshold: 4,
    };
}

impl Default for Options {
    fn default() -> Self {
        Self::BASELINE
    }
}

// KV owns bytes 0..32. Each (instance index, stage) has a separate 8-byte
// wait slot thereafter: index * 2 + (P=0, D=1). Missing storage disables waiting.
const KV_BYTES: usize = 32;
const WAIT_BYTES: usize = 8;
const KV_MAGIC: u64 = 0x4b56_4859_5330_3031;

fn read_word(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn write_word(bytes: &mut [u8], offset: usize, word: u64) {
    bytes[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
}

fn wait_slot(state: &mut [u8], instance: InstanceId) -> Option<&mut [u8]> {
    let stage = usize::from(instance.stage == Stage::Decode);
    let slot = instance.index.checked_mul(2)?.checked_add(stage)?;
    let start = KV_BYTES.checked_add(slot.checked_mul(WAIT_BYTES)?)?;
    state.get_mut(start..start.checked_add(WAIT_BYTES)?)
}

fn microbatch_deadline(
    o: &BatchObservation<'_>,
    c: &PolicyContext<'_>,
    state: &mut [u8],
    options: &Options,
) -> Option<f64> {
    let slot = wait_slot(state, o.instance)?;
    if read_word(slot, 0) == 1 {
        // A timer wake must make progress even if the queue is still small.
        write_word(slot, 0, 0);
        return None;
    }
    if o.requests.is_empty() || o.requests.len() >= options.wait_batch_threshold {
        return None;
    }
    let oldest_age = o
        .requests
        .iter()
        .map(|r| {
            let origin = if o.instance.stage == Stage::Decode {
                r.last_token_time.unwrap_or(r.arrival_time)
            } else {
                r.arrival_time
            };
            (o.now - origin).max(0.0)
        })
        .fold(0.0_f64, f64::max);
    let delay = options
        .wait_s
        .min(c.config.limits.max_wait_s)
        .min(options.age_s - oldest_age);
    let until = o.now + delay;
    if !delay.is_finite()
        || delay <= 0.0
        || !until.is_finite()
        || until <= o.now
        || until > o.now + c.config.limits.max_wait_s
    {
        return None;
    }
    write_word(slot, 0, 1);
    Some(until)
}

pub fn route_with(
    o: &RouteObservation<'_>,
    c: &PolicyContext<'_>,
    state: &mut [u8],
    options: &Options,
) -> RouteAction {
    if !options.routing_load {
        return reference_route(o, c, state);
    }
    let choose = |loads: &[crate::types::InstanceLoad]| {
        loads
            .iter()
            .enumerate()
            .min_by_key(|(index, load)| (load.total_tokens(), load.total_requests(), *index))
            .map(|(index, _)| index)
            .unwrap_or(0)
    };
    RouteAction {
        prefill: choose(o.prefill),
        decode: choose(o.decode),
    }
}

pub fn batch_with(
    o: &BatchObservation<'_>,
    c: &PolicyContext<'_>,
    state: &mut [u8],
    options: &Options,
) -> BatchAction {
    let prefill = o.instance.stage == Stage::Prefill;
    if !options.microbatch_wait
        && !(prefill && (options.aged_short_prefill || options.pressure_chunks))
        && !(!prefill && options.decode_fair)
    {
        return reference_batch(o, c, state);
    }
    if options.microbatch_wait {
        if let Some(until) = microbatch_deadline(o, c, state, options) {
            return BatchAction::Wait { until };
        }
    }
    let (tokens, members) = c.config.limits.batch(o.instance.stage);
    let mut token_budget = o.default_token_budget.clamp(tokens[0], tokens[1]);
    if prefill && options.pressure_chunks {
        if options.small_chunk_tokens == options.large_chunk_tokens {
            token_budget = options.small_chunk_tokens;
        } else if let Some(environment) = o.environment {
            let pressure = environment
                .pressure()
                .into_iter()
                .filter(|p| p.resource.starts_with("fabric:") || p.resource.starts_with("hbm:"))
                .map(|p| p.active_remaining_service_s + p.waiting_service_s)
                .fold(0.0_f64, f64::max);
            if pressure >= options.pressure_high_s {
                token_budget = options.small_chunk_tokens;
            } else if pressure <= options.pressure_low_s {
                token_budget = options.large_chunk_tokens;
            }
        }
        token_budget = token_budget.clamp(tokens[0], tokens[1]);
    }
    let max_batch = o.default_max_batch.clamp(members[0], members[1]);
    let mut requests: Vec<_> = o.requests.iter().collect();
    if prefill && options.aged_short_prefill {
        requests.sort_by(|a, b| {
            let a_old = o.now - a.arrival_time >= options.age_s;
            let b_old = o.now - b.arrival_time >= options.age_s;
            b_old
                .cmp(&a_old)
                .then_with(|| {
                    if a_old && b_old {
                        a.arrival_time.total_cmp(&b.arrival_time)
                    } else {
                        a.max_q_len.cmp(&b.max_q_len)
                    }
                })
                .then(a.completion_node_id.cmp(&b.completion_node_id))
        });
    } else if !prefill && options.decode_fair {
        requests.sort_by(|a, b| {
            a.last_token_time
                .unwrap_or(a.arrival_time)
                .total_cmp(&b.last_token_time.unwrap_or(b.arrival_time))
                .then(a.committed_tokens.cmp(&b.committed_tokens))
                .then(a.completion_node_id.cmp(&b.completion_node_id))
        });
    }
    let mut remaining = token_budget;
    let mut items = Vec::new();
    for request in requests.into_iter().take(max_batch) {
        if remaining == 0 {
            break;
        }
        let q_len = request.max_q_len.min(remaining);
        items.push(BatchItem {
            completion_node_id: request.completion_node_id,
            q_len,
        });
        remaining -= q_len;
    }
    BatchAction::Run(BatchPlan {
        items,
        token_budget,
        max_batch,
    })
}

fn hysteretic_window(
    now: f64,
    pressure: f64,
    state: &mut [u8],
    bounds: [usize; 2],
    options: &Options,
) -> usize {
    let Some(slot) = state.get_mut(..KV_BYTES) else {
        return bounds[1];
    };
    let changed = f64::from_bits(read_word(slot, 16));
    let previous = f64::from_bits(read_word(slot, 24));
    let valid = read_word(slot, 0) == KV_MAGIC
        && read_word(slot, 8) <= 1
        && changed.is_finite()
        && previous.is_finite()
        && changed <= now
        && previous <= now;
    let congested = if !valid {
        let congested = pressure >= options.pressure_high_s;
        write_word(slot, 0, KV_MAGIC);
        write_word(slot, 8, u64::from(congested));
        write_word(slot, 16, now.to_bits());
        congested
    } else {
        let mut congested = read_word(slot, 8) == 1;
        let proposed = if pressure >= options.pressure_high_s {
            true
        } else if pressure <= options.pressure_low_s {
            false
        } else {
            congested
        };
        if proposed != congested && now - changed >= options.dwell_s {
            congested = proposed;
            write_word(slot, 8, u64::from(congested));
            write_word(slot, 16, now.to_bits());
        }
        congested
    };
    write_word(slot, 24, now.to_bits());
    if congested {
        bounds[0]
    } else {
        bounds[1]
    }
}

pub fn kv_with(
    o: &KvObservation<'_>,
    c: &PolicyContext<'_>,
    state: &mut [u8],
    options: &Options,
) -> KvAction {
    if !options.kv_demand_aging && !options.hysteresis {
        return reference_kv(o, c, state);
    }
    let bounds = c.config.limits.kv_inflight_bounds;
    let mut window = bounds[1];
    if options.hysteresis {
        if let Some(environment) = o.environment {
            let pressure = environment
                .pressure()
                .into_iter()
                .filter(|p| p.resource.starts_with("fabric:"))
                .map(|p| p.active_remaining_service_s + p.waiting_service_s)
                .fold(0.0_f64, f64::max);
            window = hysteretic_window(o.now, pressure, state, bounds, options);
        }
    }
    let mut pending: Vec<_> = o.pending.iter().collect();
    if options.kv_demand_aging {
        let mut requirements = BTreeMap::<(u64, u32, usize), (u32, u32)>::new();
        if let Some(environment) = o.environment {
            for read in environment.registered_reads() {
                let entry = requirements
                    .entry((read.request_id, read.layer, read.target_rank))
                    .or_insert((read.ready_tokens, read.required_tokens));
                entry.1 = entry.1.max(read.required_tokens);
            }
        }
        // Compute each task's class once; sorting never rescans resource/read state.
        let mut ranked: Vec<_> = pending
            .into_iter()
            .map(|p| {
                let needed = requirements
                    .get(&(p.request_id, p.layer, p.target_rank as usize))
                    .is_some_and(|&(ready, required)| {
                        ready < p.token_end && p.token_start < required
                    });
                let class = if o.now - p.produced_at >= options.age_s {
                    0
                } else if needed {
                    1
                } else {
                    2
                };
                (class, p)
            })
            .collect();
        ranked.sort_by(|(a_class, a), (b_class, b)| {
            a_class
                .cmp(b_class)
                .then(a.produced_at.total_cmp(&b.produced_at))
                .then(a.id.cmp(&b.id))
        });
        pending = ranked.into_iter().map(|(_, p)| p).collect();
    }
    KvAction {
        submissions: pending
            .into_iter()
            .take(window.saturating_sub(o.inflight))
            .map(|p| p.id)
            .collect(),
        wait_until: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::KvRuntime;
    use crate::policies::{
        test_config, EnvironmentObservation, PendingKv, PolicyRuntime, QueuedRequest,
    };
    use crate::resources::{Demand, Resources, Work};
    use crate::types::InstanceLoad;

    fn queue() -> Vec<QueuedRequest> {
        [12, 8, 1]
            .into_iter()
            .enumerate()
            .map(|(id, max_q_len)| QueuedRequest {
                completion_node_id: id as u64,
                request_id: id as u64,
                subrequest_index: 0,
                arrival_time: 0.999,
                input_tokens: max_q_len,
                committed_tokens: 0,
                kv_tokens: 0,
                max_q_len,
                last_token_time: None,
            })
            .collect()
    }

    fn batch_input(requests: &[QueuedRequest]) -> BatchObservation<'_> {
        BatchObservation {
            now: 1.0,
            instance: InstanceId {
                stage: Stage::Prefill,
                index: 0,
            },
            requests,
            default_token_budget: 16,
            default_max_batch: 4,
            memory_limit_bytes: None,
            environment: None,
        }
    }

    fn runtime() -> PolicyRuntime {
        let mut config = test_config();
        config.limits.state_limit_bytes = 128;
        PolicyRuntime::new(config).unwrap()
    }

    fn pending() -> Vec<PendingKv> {
        (0..3)
            .map(|id| PendingKv {
                id,
                request_id: id,
                layer: 0,
                source_rank: 0,
                target_rank: 0,
                token_start: 0,
                token_end: 4,
                bytes: 16,
                produced_at: 0.999,
            })
            .collect()
    }

    fn add_pressure(resources: &mut Resources, id: u64, now: f64, seconds: f64) {
        resources.advance(now).unwrap();
        resources
            .submit(
                id,
                Work {
                    demands: vec![Demand {
                        resources: [("fabric:test".into(), seconds)].into(),
                        latency: 0.0,
                    }],
                    ..Default::default()
                },
            )
            .unwrap();
        resources.admit(now).unwrap();
    }

    #[test]
    fn disabled_components_match_reference_actions_and_leave_state_untouched() {
        let config = test_config();
        let context = PolicyContext {
            config: &config,
            route_sequence: 3,
        };
        let mut state = vec![0xa5; 128];
        let queue = queue();
        let loads = [InstanceLoad::default(); 2];
        let route = RouteObservation {
            now: 1.0,
            request_id: 0,
            input_tokens: 4,
            prefill: &loads,
            decode: &loads,
            environment: None,
        };
        assert_eq!(
            route_with(&route, &context, &mut state, &Options::BASELINE),
            reference_route(&route, &context, &mut [])
        );
        assert_eq!(
            batch_with(
                &batch_input(&queue),
                &context,
                &mut state,
                &Options::BASELINE
            ),
            reference_batch(&batch_input(&queue), &context, &mut [])
        );
        let pending = pending();
        let kv = KvObservation {
            now: 1.0,
            pending: &pending,
            inflight: 1,
            environment: None,
        };
        assert_eq!(
            kv_with(&kv, &context, &mut state, &Options::BASELINE),
            reference_kv(&kv, &context, &mut [])
        );
        assert_eq!(state, vec![0xa5; 128]);
    }

    #[test]
    fn aged_prefill_and_decode_gap_fairness_change_actual_members() {
        let mut runtime = runtime();
        runtime.batch_fn = |o, c, s| {
            batch_with(
                o,
                c,
                s,
                &Options {
                    aged_short_prefill: true,
                    decode_fair: true,
                    ..Options::BASELINE
                },
            )
        };
        let mut queue = queue();
        let BatchAction::Run(short) = runtime.batch(&batch_input(&queue)).unwrap() else {
            panic!()
        };
        assert_eq!(
            short
                .items
                .iter()
                .map(|i| i.completion_node_id)
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
        queue[0].arrival_time = 0.0;
        let BatchAction::Run(aged) = runtime.batch(&batch_input(&queue)).unwrap() else {
            panic!()
        };
        assert_eq!(
            aged.items
                .iter()
                .map(|i| i.completion_node_id)
                .collect::<Vec<_>>(),
            vec![0, 2, 1]
        );
        for (request, time) in queue.iter_mut().zip([0.9, 0.7, 0.8]) {
            request.max_q_len = 1;
            request.last_token_time = Some(time);
        }
        let mut input = batch_input(&queue);
        input.instance.stage = Stage::Decode;
        input.default_max_batch = 1;
        let BatchAction::Run(fair) = runtime.batch(&input).unwrap() else {
            panic!()
        };
        assert_eq!(fair.items[0].completion_node_id, 1);
        assert_eq!(fair.items[0].q_len, 1);
    }

    #[test]
    fn pressure_changes_executed_prefill_tokens_and_equal_caps_are_static() {
        const CHUNKS: Options = Options {
            pressure_chunks: true,
            small_chunk_tokens: 2,
            large_chunk_tokens: 12,
            pressure_low_s: 0.1,
            pressure_high_s: 0.5,
            ..Options::BASELINE
        };
        let mut runtime = runtime();
        runtime.batch_fn = |o, c, s| batch_with(o, c, s, &CHUNKS);
        let queue = queue();
        let mut resources = Resources::default();
        let kv = KvRuntime::default();
        let run = |runtime: &mut PolicyRuntime, resources: &Resources| {
            let mut input = batch_input(&queue[..1]);
            input.environment = Some(EnvironmentObservation::new(resources, &kv));
            let BatchAction::Run(plan) = runtime.batch(&input).unwrap() else {
                panic!()
            };
            plan.items[0].q_len
        };
        assert_eq!(run(&mut runtime, &resources), 12);
        add_pressure(&mut resources, 0, 1.0, 2.0);
        assert_eq!(run(&mut runtime, &resources), 2);
        runtime.batch_fn = |o, c, s| {
            batch_with(
                o,
                c,
                s,
                &Options {
                    large_chunk_tokens: 2,
                    ..CHUNKS
                },
            )
        };
        assert_eq!(run(&mut runtime, &resources), 2);
        let BatchAction::Run(plan) = runtime.batch(&batch_input(&queue[..1])).unwrap() else {
            panic!()
        };
        assert_eq!(plan.items[0].q_len, 2);
    }

    #[test]
    fn combined_batch_components_obey_hard_bounds_and_age_guard() {
        const OPTIONS: Options = Options {
            aged_short_prefill: true,
            decode_fair: true,
            microbatch_wait: true,
            pressure_chunks: true,
            wait_s: 10.0,
            age_s: 0.01,
            small_chunk_tokens: 32,
            large_chunk_tokens: 32,
            ..Options::BASELINE
        };
        let mut runtime = runtime();
        runtime.config.limits.p_token_budget_bounds = [3, 5];
        runtime.config.limits.p_batch_bounds = [1, 2];
        runtime.config.limits.max_wait_s = 0.0001;
        runtime.batch_fn = |o, c, s| batch_with(o, c, s, &OPTIONS);
        let mut queue = queue();
        let input = batch_input(&queue);
        let BatchAction::Wait { until } = runtime.batch(&input).unwrap() else {
            panic!()
        };
        assert_eq!(until, input.now + runtime.config.limits.max_wait_s);
        let BatchAction::Run(plan) = runtime
            .batch(&BatchObservation {
                now: until,
                ..input
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(plan.token_budget, 5);
        assert_eq!(plan.max_batch, 2);
        assert_eq!(
            plan.items
                .iter()
                .map(|i| (i.completion_node_id, i.q_len))
                .collect::<Vec<_>>(),
            vec![(2, 1), (1, 4)]
        );

        queue[0].arrival_time = 0.0;
        let BatchAction::Run(plan) = runtime.batch(&batch_input(&queue)).unwrap() else {
            panic!()
        };
        assert_eq!(plan.items[0].completion_node_id, 0);
        assert_eq!(plan.items[0].q_len, 5);

        runtime.batch_fn = |o, c, s| {
            batch_with(
                o,
                c,
                s,
                &Options {
                    small_chunk_tokens: 1,
                    large_chunk_tokens: 1,
                    microbatch_wait: false,
                    ..OPTIONS
                },
            )
        };
        let BatchAction::Run(plan) = runtime.batch(&batch_input(&queue)).unwrap() else {
            panic!()
        };
        assert_eq!(plan.token_budget, 3);
        assert_eq!(plan.items[0].q_len, 3);
    }

    #[test]
    fn microbatch_wait_is_bounded_and_isolated_by_stage_and_instance() {
        let mut runtime = runtime();
        runtime.batch_fn = |o, c, s| {
            batch_with(
                o,
                c,
                s,
                &Options {
                    microbatch_wait: true,
                    wait_s: 0.001,
                    age_s: 10.0,
                    wait_batch_threshold: 2,
                    ..Options::BASELINE
                },
            )
        };
        let queue = queue();
        let mut input = batch_input(&queue[..1]);
        assert!(matches!(
            runtime.batch(&input).unwrap(),
            BatchAction::Wait { .. }
        ));
        input.instance.stage = Stage::Decode;
        assert!(matches!(
            runtime.batch(&input).unwrap(),
            BatchAction::Wait { .. }
        ));
        input.instance.index = 1;
        assert!(matches!(
            runtime.batch(&input).unwrap(),
            BatchAction::Wait { .. }
        ));
        input.instance = InstanceId {
            stage: Stage::Prefill,
            index: 0,
        };
        input.now += 0.001;
        assert!(matches!(
            runtime.batch(&input).unwrap(),
            BatchAction::Run(_)
        ));
        input.instance.stage = Stage::Decode;
        assert!(matches!(
            runtime.batch(&input).unwrap(),
            BatchAction::Run(_)
        ));
        input.instance.index = 1;
        assert!(matches!(
            runtime.batch(&input).unwrap(),
            BatchAction::Run(_)
        ));
    }

    #[test]
    fn missing_state_and_unrepresentable_waits_fall_back_to_run() {
        let config = test_config();
        let context = PolicyContext {
            config: &config,
            route_sequence: 0,
        };
        let queue = queue();
        let mut input = batch_input(&queue[..1]);
        let options = Options {
            microbatch_wait: true,
            age_s: 1e30,
            ..Options::BASELINE
        };
        for size in [0, 1, 31, 32, 39] {
            assert!(matches!(
                batch_with(&input, &context, &mut vec![0; size], &options),
                BatchAction::Run(_)
            ));
        }
        let mut state = vec![0; 128];
        assert!(matches!(
            batch_with(&input, &context, &mut state, &options),
            BatchAction::Wait { .. }
        ));
        assert!(matches!(
            batch_with(&input, &context, &mut state[..39], &options),
            BatchAction::Run(_)
        ));
        assert!(matches!(
            batch_with(&input, &context, &mut state, &options),
            BatchAction::Run(_)
        ));
        input.now = 1e20;
        assert!(matches!(
            batch_with(&input, &context, &mut state, &options),
            BatchAction::Run(_)
        ));
        input.now = 1.0;
        input.instance.index = usize::MAX;
        assert!(matches!(
            batch_with(&input, &context, &mut state, &options),
            BatchAction::Run(_)
        ));
    }

    #[test]
    fn kv_prioritizes_aged_tasks_then_registered_demand() {
        let mut runtime = runtime();
        runtime.kv_fn = |o, c, s| {
            kv_with(
                o,
                c,
                s,
                &Options {
                    kv_demand_aging: true,
                    ..Options::BASELINE
                },
            )
        };
        let mut pending = pending();
        pending[2].produced_at = 0.0;
        let resources = Resources::default();
        let mut kv = KvRuntime::default();
        kv.register_read_gate(1, 0, 0, 4, 0);
        let action = runtime
            .kv(&KvObservation {
                now: 1.0,
                pending: &pending,
                inflight: 0,
                environment: Some(EnvironmentObservation::new(&resources, &kv)),
            })
            .unwrap();
        assert_eq!(action.submissions, vec![2, 1]);
    }

    #[test]
    fn hysteresis_respects_dwell_and_preserves_wait_state_and_progress() {
        const OPTIONS: Options = Options {
            hysteresis: true,
            microbatch_wait: true,
            age_s: 10.0,
            pressure_low_s: 0.01,
            pressure_high_s: 0.05,
            dwell_s: 1.0,
            ..Options::BASELINE
        };
        let mut runtime = runtime();
        runtime.batch_fn = |o, c, s| batch_with(o, c, s, &OPTIONS);
        runtime.kv_fn = |o, c, s| kv_with(o, c, s, &OPTIONS);
        let queue = queue();
        assert!(matches!(
            runtime.batch(&batch_input(&queue[..1])).unwrap(),
            BatchAction::Wait { .. }
        ));
        let wait_state = runtime.memory[32..].to_vec();
        let pending = pending();
        let kv = KvRuntime::default();
        let mut resources = Resources::default();
        add_pressure(&mut resources, 0, 1.0, 0.2);
        let run = |runtime: &mut PolicyRuntime, resources: &Resources, now, inflight| {
            runtime
                .kv(&KvObservation {
                    now,
                    pending: &pending,
                    inflight,
                    environment: Some(EnvironmentObservation::new(resources, &kv)),
                })
                .unwrap()
        };
        assert_eq!(run(&mut runtime, &resources, 1.0, 0).submissions.len(), 1);
        assert!(run(&mut runtime, &resources, 1.0, 1).submissions.is_empty());
        assert_eq!(&runtime.memory[32..], wait_state);
        resources.advance(1.2).unwrap();
        assert_eq!(run(&mut runtime, &resources, 1.2, 0).submissions.len(), 1);
        resources.advance(2.1).unwrap();
        assert_eq!(run(&mut runtime, &resources, 2.1, 0).submissions.len(), 2);
        add_pressure(&mut resources, 1, 2.2, 2.0);
        assert_eq!(run(&mut runtime, &resources, 2.2, 0).submissions.len(), 2);
        resources.advance(3.2).unwrap();
        assert_eq!(run(&mut runtime, &resources, 3.2, 0).submissions.len(), 1);
        let context = PolicyContext {
            config: &runtime.config,
            route_sequence: 0,
        };
        let input = KvObservation {
            now: 3.2,
            pending: &pending,
            inflight: 0,
            environment: Some(EnvironmentObservation::new(&resources, &kv)),
        };
        assert_eq!(
            kv_with(&input, &context, &mut [0; 31], &OPTIONS)
                .submissions
                .len(),
            2
        );
    }

    #[test]
    fn load_routing_selects_current_pool_loads() {
        let mut runtime = runtime();
        runtime.route_fn = |o, c, s| {
            route_with(
                o,
                c,
                s,
                &Options {
                    routing_load: true,
                    ..Options::BASELINE
                },
            )
        };
        let busy = InstanceLoad {
            waiting_tokens: 9,
            ..Default::default()
        };
        assert_eq!(
            runtime
                .route(&RouteObservation {
                    now: 1.0,
                    request_id: 0,
                    input_tokens: 4,
                    prefill: &[busy, InstanceLoad::default()],
                    decode: &[InstanceLoad::default(), busy],
                    environment: None,
                })
                .unwrap(),
            RouteAction {
                prefill: 1,
                decode: 0
            }
        );
    }
}
