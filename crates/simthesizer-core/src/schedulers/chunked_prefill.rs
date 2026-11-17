use std::collections::{HashSet, LinkedList};

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::module::Module;
use crate::scheduler::Scheduler;
use crate::schedulers::prefix_cache::{hash_block, PrefixCache};
use crate::types::{
    InstanceLoad, NodeID, ParallelStrategy, ReadySubRequest, RequestState, SubRequestCompletion,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkedPrefillScheduler {
    #[serde(skip)]
    shared_memory: bool,
    block_size: u32,
    max_batch: usize,
    max_num_batched_tokens: u32,
    device_mem: u64,
    device_mem_utilization: f64,
    enable_prefix_caching: bool,

    #[serde(skip_deserializing, default)]
    queue: LinkedList<RunningRequest>,

    #[serde(skip_deserializing, default)]
    running: bool,

    #[serde(skip_deserializing, default)]
    current_batch: Vec<RequestState>,

    #[serde(skip_deserializing, default)]
    admitted: HashSet<NodeID>,

    #[serde(skip_deserializing, default)]
    generation_tokens_total: u64,

    #[serde(skip_deserializing, default)]
    model_weight_bytes_per_device: u64,

    #[serde(skip_deserializing, default)]
    kv_bytes_per_token_per_device: u64,

    #[serde(skip_deserializing, default)]
    kv_bytes_per_request_per_device: u64,

    #[serde(skip_deserializing, default)]
    usable_device_mem: u64,

    #[serde(skip_deserializing, default)]
    prefix_cache: PrefixCache,

    #[serde(skip_deserializing, default)]
    effective_max_batch: usize,

    #[serde(skip_deserializing, default)]
    effective_max_num_batched_tokens: u32,
}

crate::register_scheduler!(ChunkedPrefillScheduler, "chunked_prefill");

impl Module for ChunkedPrefillScheduler {
    fn observe(&self) -> serde_json::Value {
        serde_json::json!({
            "running_requests": self.admitted.len(),
            "waiting_requests": self.queue.len() - self.admitted.len(),
            "generation_tokens_total": self.generation_tokens_total,
        })
    }
}

#[derive(Debug, Clone)]
struct RunningRequest {
    request_id: u64,
    subrequest_index: usize,
    completion_node_id: NodeID,
    arrival_time: f64,
    input_tokens: u32,
    output_tokens: u32,

    known_tokens: u32,
    shared_kv_tokens: u32,
    private_kv_tokens: u32,
    last_time: f64,
    token_latency: Vec<f64>,

    input_token_ids: Option<Vec<u32>>,
    output_token_ids: Option<Vec<u32>>,
    block_hashes: Vec<u64>,
}

fn block_count(num_tokens: u32, block_size: u32) -> u32 {
    num_tokens.div_ceil(block_size)
}

impl RunningRequest {
    fn new(request: ReadySubRequest) -> Self {
        Self {
            request_id: request.request_id,
            subrequest_index: request.subrequest_index,
            completion_node_id: request.completion_node_id,
            arrival_time: request.arrival_time,
            input_tokens: request.input_tokens,
            output_tokens: request.output_tokens,
            known_tokens: request.known_tokens,
            shared_kv_tokens: 0,
            private_kv_tokens: request.kv_tokens,
            last_time: request.arrival_time,
            token_latency: Vec::new(),
            input_token_ids: request.input_token_ids,
            output_token_ids: request.output_token_ids,
            block_hashes: Vec::new(),
        }
    }

    fn effective_kv_tokens(&self) -> u32 {
        self.shared_kv_tokens + self.private_kv_tokens
    }

    fn private_kv_bytes(
        &self,
        block_size: u32,
        kv_bytes_per_request_per_device: u64,
        kv_bytes_per_token_per_device: u64,
    ) -> u64 {
        let token_bytes = u64::from(block_count(self.private_kv_tokens, block_size))
            * u64::from(block_size)
            * kv_bytes_per_token_per_device;
        let request_bytes =
            u64::from(self.effective_kv_tokens() > 0) * kv_bytes_per_request_per_device;
        token_bytes + request_bytes
    }

    fn private_kv_bytes_after(
        &self,
        q_len: u32,
        block_size: u32,
        kv_bytes_per_request_per_device: u64,
        kv_bytes_per_token_per_device: u64,
    ) -> u64 {
        let total_kv_tokens = self.private_kv_tokens + q_len;
        let token_bytes = u64::from(block_count(total_kv_tokens, block_size))
            * u64::from(block_size)
            * kv_bytes_per_token_per_device;
        let request_bytes =
            u64::from(self.effective_kv_tokens() + q_len > 0) * kv_bytes_per_request_per_device;
        token_bytes + request_bytes
    }

    fn ensure_block_hashes(&mut self, block_size: u32, max_tokens: u32) {
        let Some(input) = self.input_token_ids.as_deref() else {
            return;
        };
        let output = self.output_token_ids.as_deref().unwrap_or_default();
        let bs = block_size as usize;
        let count = (max_tokens as usize).min(input.len() + output.len()) / bs;
        for index in self.block_hashes.len()..count {
            let start = index * bs;
            let end = start + bs;
            let first = &input[start.min(input.len())..end.min(input.len())];
            let second =
                &output[start.saturating_sub(input.len())..end.saturating_sub(input.len())];
            let parent = self.block_hashes.last().copied().unwrap_or(0);
            self.block_hashes.push(hash_block(parent, first, second));
        }
    }

    fn longest_cache_hit(&mut self, prefix_cache: &mut PrefixCache, block_size: u32) -> u32 {
        if self.effective_kv_tokens() != 0 {
            return self.shared_kv_tokens;
        }

        // A cached prefix has KV, but no logits for the next token.
        let max_tokens = self.known_tokens.saturating_sub(1);
        self.ensure_block_hashes(block_size, max_tokens);
        let count = self
            .block_hashes
            .len()
            .min((max_tokens / block_size) as usize);
        prefix_cache.longest_hit(&self.block_hashes[..count]) as u32 * block_size
    }

    fn publish_shared_prefix(&mut self, prefix_cache: &mut PrefixCache, block_size: u32) {
        let total = self.effective_kv_tokens();
        self.ensure_block_hashes(block_size, total);
        let start = (self.shared_kv_tokens / block_size) as usize;
        let end = self.block_hashes.len().min((total / block_size) as usize);
        prefix_cache.insert(&self.block_hashes[start..end]);
        self.relock_shared_prefix(prefix_cache, block_size, end as u32 * block_size);
        self.private_kv_tokens = total - self.shared_kv_tokens;
    }

    fn relock_shared_prefix(
        &mut self,
        prefix_cache: &mut PrefixCache,
        block_size: u32,
        new_shared_kv_tokens: u32,
    ) {
        let old = (self.shared_kv_tokens / block_size) as usize;
        let new = (new_shared_kv_tokens / block_size) as usize;
        if new > old {
            prefix_cache.lock(&self.block_hashes[old..new]);
        } else if new < old {
            prefix_cache.unlock(&self.block_hashes[new..old]);
        }
        self.shared_kv_tokens = new_shared_kv_tokens;
    }

    fn preempt(&mut self, prefix_cache: &mut PrefixCache, block_size: u32) {
        self.relock_shared_prefix(prefix_cache, block_size, 0);
        self.private_kv_tokens = 0;
    }
}

impl ChunkedPrefillScheduler {
    fn cache_block_bytes(&self) -> u64 {
        u64::from(self.block_size) * self.kv_bytes_per_token_per_device
    }
}

impl Scheduler for ChunkedPrefillScheduler {
    fn use_shared_memory(&mut self) -> Result<()> {
        if self.enable_prefix_caching {
            bail!("shared Rooflang KV does not yet support prefix caching");
        }
        self.shared_memory = true;
        Ok(())
    }
    fn memory_limit(&self) -> Option<u64> {
        Some(self.usable_device_mem)
    }
    fn batch_requests(&self) -> Vec<crate::kv::BatchRequest> {
        self.queue
            .iter()
            .take(self.current_batch.len())
            .map(|r| crate::kv::BatchRequest {
                request_id: r.request_id,
                input_tokens: r.input_tokens,
                output_tokens: r.output_tokens,
            })
            .collect()
    }
    fn init(
        &mut self,
        model_weight_bytes_per_device: u64,
        kv_bytes_per_request_per_device: u64,
        kv_bytes_per_token_per_device: u64,
        parallel: ParallelStrategy,
    ) -> Result<()> {
        if self.block_size == 0 {
            bail!("block_size must be positive");
        }
        if !self.device_mem_utilization.is_finite()
            || self.device_mem_utilization <= 0.0
            || self.device_mem_utilization > 1.0
        {
            bail!(
                "device_mem_utilization must be in (0, 1], got {}",
                self.device_mem_utilization
            );
        }

        self.model_weight_bytes_per_device = model_weight_bytes_per_device;
        self.kv_bytes_per_request_per_device = kv_bytes_per_request_per_device;
        self.kv_bytes_per_token_per_device = kv_bytes_per_token_per_device;
        self.usable_device_mem =
            (self.device_mem as f64 * self.device_mem_utilization).floor() as u64;
        self.effective_max_batch = self
            .max_batch
            .checked_mul(parallel.ep as usize)
            .ok_or_else(|| anyhow!("effective max_batch overflowed"))?;
        self.effective_max_num_batched_tokens = self
            .max_num_batched_tokens
            .checked_mul(parallel.ep)
            .ok_or_else(|| anyhow!("effective max_num_batched_tokens overflowed"))?;
        if self.usable_device_mem == 0 {
            bail!(
                "device_mem_utilization leaves no usable device memory from device_mem={}",
                self.device_mem
            );
        }
        Ok(())
    }

    fn schedule(&mut self) -> Result<Vec<RequestState>> {
        if self.running || self.queue.is_empty() {
            return Ok(Vec::new());
        }

        let mut batch_requests = Vec::new();
        let mut total_q_len = 0_u32;
        let block_size = self.block_size;
        let kv_bytes_per_request_per_device = self.kv_bytes_per_request_per_device;
        let kv_bytes_per_token_per_device = self.kv_bytes_per_token_per_device;
        let usable_device_mem = if self.shared_memory {
            u64::MAX
        } else {
            self.usable_device_mem
        };
        let enable_prefix_caching = self.enable_prefix_caching;
        let cache_block_bytes = self.cache_block_bytes();
        let prefix_cache = &mut self.prefix_cache;
        let mut gpu_bytes = self.model_weight_bytes_per_device
            + prefix_cache.bytes_used(block_size, kv_bytes_per_token_per_device);

        for request in self.queue.iter_mut() {
            if batch_requests.len() >= self.effective_max_batch {
                break;
            }

            let remaining_batch_tokens = self
                .effective_max_num_batched_tokens
                .saturating_sub(total_q_len);
            if remaining_batch_tokens == 0 {
                break;
            }

            let original_shared = request.shared_kv_tokens;
            let matched_shared = if enable_prefix_caching {
                request.longest_cache_hit(prefix_cache, block_size)
            } else {
                0
            };
            if matched_shared > original_shared {
                request.relock_shared_prefix(prefix_cache, block_size, matched_shared);
            }

            let effective_kv_tokens = request.effective_kv_tokens();
            let num_tokens = request
                .known_tokens
                .saturating_sub(effective_kv_tokens)
                .clamp(1, remaining_batch_tokens);
            let needed_private_bytes = request.private_kv_bytes_after(
                num_tokens,
                block_size,
                kv_bytes_per_request_per_device,
                kv_bytes_per_token_per_device,
            );

            if enable_prefix_caching {
                while gpu_bytes.saturating_add(needed_private_bytes) > usable_device_mem {
                    if cache_block_bytes == 0 || !prefix_cache.evict_one() {
                        request.relock_shared_prefix(prefix_cache, block_size, original_shared);
                        break;
                    }
                    gpu_bytes = gpu_bytes.saturating_sub(cache_block_bytes);
                }
                if gpu_bytes.saturating_add(needed_private_bytes) > usable_device_mem {
                    break;
                }
            } else if gpu_bytes.saturating_add(needed_private_bytes) > usable_device_mem {
                request.relock_shared_prefix(prefix_cache, block_size, original_shared);
                break;
            }

            total_q_len += num_tokens;
            gpu_bytes += needed_private_bytes;
            self.admitted.insert(request.completion_node_id);

            batch_requests.push(RequestState {
                q_len: num_tokens,
                kv_len: effective_kv_tokens,
                lm_head_len: 1,
            });
        }

        if batch_requests.is_empty() {
            bail!("system cannot handle a request!");
        }

        for request in self.queue.iter_mut().skip(batch_requests.len()) {
            if request.effective_kv_tokens() == 0 {
                break;
            }

            let private_bytes = request.private_kv_bytes(
                block_size,
                kv_bytes_per_request_per_device,
                kv_bytes_per_token_per_device,
            );
            if gpu_bytes.saturating_add(private_bytes) > usable_device_mem {
                request.preempt(prefix_cache, block_size);
                self.admitted.remove(&request.completion_node_id);
            } else {
                gpu_bytes += private_bytes;
            }
        }

        self.running = true;
        self.current_batch = batch_requests.clone();
        Ok(batch_requests)
    }

    fn enqueue_sub_request(&mut self, request: ReadySubRequest) {
        self.queue.push_back(RunningRequest::new(request));
    }

    fn done_iteration(&mut self, now: f64) -> Vec<SubRequestCompletion> {
        let current_batch = std::mem::take(&mut self.current_batch);

        let mut tail = self.queue.split_off(current_batch.len());
        for (request, scheduled) in self.queue.iter_mut().zip(&current_batch) {
            request.private_kv_tokens += scheduled.q_len;
            if request.known_tokens <= request.effective_kv_tokens()
                && request.known_tokens < request.input_tokens + request.output_tokens
            {
                let latency = if request.token_latency.is_empty() {
                    now - request.arrival_time
                } else {
                    now - request.last_time
                };
                request.token_latency.push(latency);
                self.generation_tokens_total += 1;
                request.last_time = now;
                request.known_tokens += 1;
            }

            if self.enable_prefix_caching {
                request.publish_shared_prefix(&mut self.prefix_cache, self.block_size);
            }
        }

        let mut completed = Vec::new();
        let mut next_queue = LinkedList::new();
        while let Some(mut request) = self.queue.pop_front() {
            let total_tokens = request.input_tokens + request.output_tokens;
            if request.known_tokens >= total_tokens {
                self.admitted.remove(&request.completion_node_id);
                if self.enable_prefix_caching {
                    request.relock_shared_prefix(&mut self.prefix_cache, self.block_size, 0);
                }
                completed.push(SubRequestCompletion {
                    request_id: request.request_id,
                    subrequest_index: request.subrequest_index,
                    completion_node_id: request.completion_node_id,
                    token_latency: request.token_latency,
                });
            } else {
                next_queue.push_back(request);
            }
        }

        next_queue.append(&mut tail);
        self.queue = next_queue;
        self.running = false;
        completed
    }

    fn instance_load(&self) -> InstanceLoad {
        let running_requests = self.current_batch.len();
        let running_tokens = self
            .current_batch
            .iter()
            .map(|request| u64::from(request.q_len))
            .sum();
        let mut waiting_requests = 0usize;
        let mut waiting_tokens = 0u64;
        for request in self.queue.iter().skip(running_requests) {
            waiting_requests += 1;
            waiting_tokens += u64::from(
                request
                    .known_tokens
                    .saturating_sub(request.effective_kv_tokens())
                    .max(1),
            );
        }
        InstanceLoad {
            waiting_requests,
            running_requests,
            waiting_tokens,
            running_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler(budget: u32, prefix_caching: bool) -> ChunkedPrefillScheduler {
        let mut scheduler: ChunkedPrefillScheduler = toml::from_str(&format!(
            "block_size = 4\nmax_batch = 32\nmax_num_batched_tokens = {budget}\n\
             device_mem = 1048576\ndevice_mem_utilization = 1.0\n\
             enable_prefix_caching = {prefix_caching}"
        ))
        .unwrap();
        scheduler
            .init(
                0,
                0,
                1,
                ParallelStrategy {
                    tp: 1,
                    pp: 1,
                    ep: 1,
                },
            )
            .unwrap();
        scheduler
    }

    fn request(input: u32, output: u32) -> ReadySubRequest {
        ReadySubRequest {
            request_id: 0,
            subrequest_index: 0,
            completion_node_id: 0,
            arrival_time: 0.0,
            input_tokens: input,
            output_tokens: output,
            known_tokens: input,
            kv_tokens: 0,
            input_token_ids: Some((0..input).collect()),
            output_token_ids: None,
        }
    }

    fn state(scheduler: &ChunkedPrefillScheduler, running: u64, waiting: u64, generated: u64) {
        assert_eq!(
            scheduler.observe(),
            serde_json::json!({
                "running_requests": running,
                "waiting_requests": waiting,
                "generation_tokens_total": generated,
            })
        );
    }

    #[test]
    fn observations_follow_admission_and_emission_not_scheduled_work() {
        let mut scheduler = scheduler(4, false);
        scheduler.enqueue_sub_request(request(8, 2));
        state(&scheduler, 0, 1, 0);
        scheduler.schedule().unwrap();
        state(&scheduler, 1, 0, 0);
        scheduler.done_iteration(1.0);
        state(&scheduler, 1, 0, 0);
        scheduler.schedule().unwrap();
        scheduler.done_iteration(2.0);
        state(&scheduler, 1, 0, 1);
        scheduler.schedule().unwrap();
        scheduler.done_iteration(3.0);
        state(&scheduler, 0, 0, 2);
    }

    #[test]
    fn retained_admission_and_preemption_have_different_observations() {
        for memory in [12, 8] {
            let mut scheduler = scheduler(8, false);
            scheduler.usable_device_mem = memory;
            for id in 0..2 {
                let mut request = request(4, 3);
                request.completion_node_id = id;
                scheduler.enqueue_sub_request(request);
            }
            assert_eq!(scheduler.schedule().unwrap().len(), 2);
            scheduler.done_iteration(1.0);
            state(&scheduler, 2, 0, 2);
            scheduler.effective_max_num_batched_tokens = 1;
            assert_eq!(scheduler.schedule().unwrap().len(), 1);
            assert_eq!(scheduler.instance_load().running_requests, 1);
            if memory == 12 {
                state(&scheduler, 2, 0, 2);
            } else {
                state(&scheduler, 1, 1, 2);
            }
        }
    }

    #[test]
    fn seeded_kv_waits_for_admission_and_zero_output_emits_nothing() {
        let mut scheduler = scheduler(4, false);
        let mut seeded = request(8, 1);
        seeded.kv_tokens = 7;
        scheduler.enqueue_sub_request(seeded);
        state(&scheduler, 0, 1, 0);
        scheduler.schedule().unwrap();
        state(&scheduler, 1, 0, 0);
        scheduler.done_iteration(1.0);
        state(&scheduler, 0, 0, 1);
        scheduler.enqueue_sub_request(request(4, 0));
        scheduler.schedule().unwrap();
        scheduler.done_iteration(2.0);
        state(&scheduler, 0, 0, 1);
    }

    #[test]
    fn prefill_produces_the_first_token_without_an_extra_decode() {
        let mut scheduler = scheduler(16, false);
        scheduler.enqueue_sub_request(request(8, 1));
        let batch = scheduler.schedule().unwrap();
        assert_eq!(batch[0].q_len, 8);
        assert_eq!(batch[0].kv_len, 0);
        let results = scheduler.done_iteration(0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].token_latency, vec![0.5]);
        assert!(scheduler.schedule().unwrap().is_empty());
    }

    #[test]
    fn chunked_prefill_emits_only_after_the_last_chunk() {
        let mut scheduler = scheduler(4, false);
        scheduler.enqueue_sub_request(request(8, 3));
        assert_eq!(scheduler.schedule().unwrap()[0].q_len, 4);
        assert!(scheduler.done_iteration(1.0).is_empty());
        assert!(scheduler.queue.front().unwrap().token_latency.is_empty());
        let batch = scheduler.schedule().unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 4));
        assert!(scheduler.done_iteration(2.0).is_empty());
        assert_eq!(scheduler.queue.front().unwrap().token_latency, vec![2.0]);
        for kv in 8..10 {
            let batch = scheduler.schedule().unwrap();
            assert_eq!((batch[0].q_len, batch[0].kv_len), (1, kv));
            let results = scheduler.done_iteration(f64::from(kv - 5));
            if kv == 9 {
                assert_eq!(results[0].token_latency, vec![2.0, 1.0, 1.0]);
            } else {
                assert!(results.is_empty());
            }
        }
        assert!(scheduler.schedule().unwrap().is_empty());
    }

    #[test]
    fn full_prefix_hit_recomputes_the_last_block_for_logits() {
        let mut scheduler = scheduler(16, true);
        scheduler.enqueue_sub_request(request(8, 1));
        scheduler.schedule().unwrap();
        assert_eq!(scheduler.done_iteration(1.0).len(), 1);
        scheduler.enqueue_sub_request(request(8, 1));
        let batch = scheduler.schedule().unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 4));
        assert_eq!(scheduler.done_iteration(2.0).len(), 1);
    }

    #[test]
    fn eviction_after_completion_keeps_the_earlier_prefix_reusable() {
        let mut scheduler = scheduler(16, true);
        scheduler.enqueue_sub_request(request(12, 1));
        scheduler.schedule().unwrap();
        assert_eq!(scheduler.done_iteration(1.0).len(), 1);
        assert!(scheduler.prefix_cache.evict_one());
        scheduler.enqueue_sub_request(request(12, 1));
        let batch = scheduler.schedule().unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 8));
        assert_eq!(scheduler.done_iteration(2.0).len(), 1);
    }

    #[test]
    fn hashing_future_blocks_does_not_publish_uncomputed_kv() {
        let mut scheduler = scheduler(4, true);
        scheduler.enqueue_sub_request(request(20, 1));
        scheduler.schedule().unwrap();
        scheduler.done_iteration(1.0);
        let request = scheduler.queue.front().unwrap();
        assert_eq!(request.block_hashes.len(), 4);
        assert_eq!(scheduler.prefix_cache.bytes_used(4, 1), 4);
        assert_eq!(scheduler.prefix_cache.longest_hit(&request.block_hashes), 1);
        assert!(!scheduler.prefix_cache.evict_one());
    }

    #[test]
    fn output_blocks_cross_the_prompt_boundary_and_publish_once_computed() {
        let mut scheduler = scheduler(16, true);
        let mut input = request(6, 7);
        input.output_token_ids = Some((6..13).collect());
        scheduler.enqueue_sub_request(input);
        for time in 1_u32..=7 {
            scheduler.schedule().unwrap();
            let completed = scheduler.done_iteration(f64::from(time));
            let computed = 6 + time - 1;
            assert_eq!(
                scheduler.prefix_cache.bytes_used(4, 1),
                u64::from(computed / 4 * 4)
            );
            if time < 7 {
                let request = scheduler.queue.front().unwrap();
                assert_eq!(request.block_hashes.len(), (computed / 4) as usize);
                assert!(!scheduler.prefix_cache.evict_one());
            } else {
                assert_eq!(completed.len(), 1);
            }
        }
        scheduler.enqueue_sub_request(request(12, 1));
        let batch = scheduler.schedule().unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 8));
    }

    #[test]
    fn preemption_keeps_hashes_but_rechecks_cache_residency() {
        let mut request = RunningRequest::new(request(12, 3));
        let mut cache = PrefixCache::default();
        request.private_kv_tokens = 12;
        request.publish_shared_prefix(&mut cache, 4);
        let hashes = request.block_hashes.clone();
        request.preempt(&mut cache, 4);
        assert_eq!(request.effective_kv_tokens(), 0);
        assert_eq!(request.block_hashes, hashes);
        assert!(cache.evict_one());
        assert_eq!(request.longest_cache_hit(&mut cache, 4), 8);
        request.relock_shared_prefix(&mut cache, 4, 8);
        request.private_kv_tokens = 4;
        request.publish_shared_prefix(&mut cache, 4);
        assert_eq!(request.block_hashes, hashes);
        assert!(!cache.evict_one());
        request.preempt(&mut cache, 4);
        for _ in 0..3 {
            assert!(cache.evict_one());
        }
        assert_eq!(request.longest_cache_hit(&mut cache, 4), 0);
    }

    #[test]
    fn failed_admission_releases_acquired_blocks_for_retry() {
        let mut scheduler = scheduler(16, true);
        scheduler.enqueue_sub_request(request(12, 1));
        scheduler.schedule().unwrap();
        scheduler.done_iteration(1.0);
        scheduler.usable_device_mem = 8;
        scheduler.enqueue_sub_request(request(12, 1));
        assert!(scheduler.schedule().is_err());
        assert_eq!(scheduler.queue.front().unwrap().shared_kv_tokens, 0);
        scheduler.usable_device_mem = 12;
        let batch = scheduler.schedule().unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 8));
        assert_eq!(scheduler.done_iteration(2.0).len(), 1);
        for _ in 0..3 {
            assert!(scheduler.prefix_cache.evict_one());
        }
    }

    #[test]
    fn missing_output_ids_preserve_only_the_known_prompt_blocks() {
        let mut request = RunningRequest::new(request(6, 8));
        let mut cache = PrefixCache::default();
        request.private_kv_tokens = 12;
        request.publish_shared_prefix(&mut cache, 4);
        assert_eq!(request.shared_kv_tokens, 4);
        assert_eq!(request.private_kv_tokens, 8);
        assert_eq!(request.block_hashes.len(), 1);
    }

    #[test]
    fn disabled_caching_or_missing_prompt_ids_creates_no_hashes() {
        for enabled in [false, true] {
            let mut scheduler = scheduler(4, enabled);
            let mut input = request(8, 1);
            if enabled {
                input.input_token_ids = None;
                input.output_token_ids = Some(vec![8]);
            }
            scheduler.enqueue_sub_request(input);
            scheduler.schedule().unwrap();
            scheduler.done_iteration(1.0);
            assert!(scheduler.queue.front().unwrap().block_hashes.is_empty());
            assert_eq!(scheduler.prefix_cache.bytes_used(4, 1), 0);
        }
    }

    #[test]
    fn zero_block_size_is_rejected() {
        let mut scheduler = scheduler(4, true);
        scheduler.block_size = 0;
        assert_eq!(
            scheduler
                .init(
                    0,
                    0,
                    1,
                    ParallelStrategy {
                        tp: 1,
                        pp: 1,
                        ep: 1
                    }
                )
                .unwrap_err()
                .to_string(),
            "block_size must be positive"
        );
    }
}
