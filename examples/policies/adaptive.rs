//! Example search seed, not a promoted policy. Copy into the candidate directory.
//! Thresholds and combined rules require paired evaluation and ablation.
use super::{
    BatchAction, BatchItem, BatchObservation, BatchPlan, KvAction, KvObservation, PolicyContext,
    RouteAction, RouteObservation, Stage,
};

pub fn route(o: &RouteObservation<'_>, _: &PolicyContext<'_>, _: &mut [u8]) -> RouteAction {
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

pub fn batch(o: &BatchObservation<'_>, c: &PolicyContext<'_>, _: &mut [u8]) -> BatchAction {
    let (tokens, members) = c.config.limits.batch(o.instance.stage);
    let token_budget = if o.instance.stage == Stage::Prefill && o.requests.len() <= 2 {
        tokens[1]
    } else {
        o.default_token_budget.clamp(tokens[0], tokens[1])
    };
    let max_batch = o.default_max_batch.clamp(members[0], members[1]);
    let mut requests: Vec<_> = o.requests.iter().collect();
    requests.sort_by(|a, b| {
        a.last_token_time
            .unwrap_or(a.arrival_time)
            .total_cmp(&b.last_token_time.unwrap_or(b.arrival_time))
            .then(a.completion_node_id.cmp(&b.completion_node_id))
    });
    let share = (token_budget / (requests.len().min(max_batch).max(1) as u32)).max(1);
    let mut remaining = token_budget;
    let mut items = Vec::new();
    for request in requests.into_iter().take(max_batch) {
        if remaining == 0 {
            break;
        }
        let q_len = request.max_q_len.min(share).min(remaining);
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

pub fn kv(o: &KvObservation<'_>, c: &PolicyContext<'_>, _: &mut [u8]) -> KvAction {
    let reads = o
        .environment
        .map(|e| e.registered_reads())
        .unwrap_or_default();
    let fabric_busy = o.environment.is_some_and(|e| {
        e.pressure()
            .iter()
            .any(|p| p.resource.starts_with("fabric:") && p.active_demands > 0)
    });
    let bounds = c.config.limits.kv_inflight_bounds;
    let window = if fabric_busy { bounds[0] } else { bounds[1] };
    let mut pending: Vec<_> = o.pending.iter().collect();
    pending.sort_by(|a, b| {
        let needed = |p: &super::PendingKv| {
            reads.iter().any(|r| {
                r.request_id == p.request_id
                    && r.layer == p.layer
                    && r.target_rank == p.target_rank as usize
                    && r.ready_tokens < p.token_end
                    && p.token_start < r.required_tokens
            })
        };
        needed(b)
            .cmp(&needed(a))
            .then(a.produced_at.total_cmp(&b.produced_at))
            .then(a.id.cmp(&b.id))
    });
    KvAction {
        submissions: pending
            .into_iter()
            .take(window.saturating_sub(o.inflight))
            .map(|p| p.id)
            .collect(),
        wait_until: None,
    }
}
