use std::any::Any;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::engine::Graph;
use crate::model::Model;
use crate::module::{LogicalHandler, Module};
use crate::network_sim::NetworkSimulator;
use crate::system::{System, SystemEvent};
use crate::types::{
    NetworkDeviceID, NodeID, ParallelStrategy, Request, RequestResult, ResultSubRequest, SubRequest,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdDisaggregationSystem {
    prefill_instances: Vec<Box<dyn System>>,
    decode_instances: Vec<Box<dyn System>>,
    model: Box<dyn Model>,
    tp: u32,
    network_sim: Box<dyn NetworkSimulator>,
    link_bw: f64,
    link_latency: f64,

    #[serde(skip_deserializing, default)]
    prefill_device_ids: Vec<NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    decode_device_ids: Vec<NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    next_prefill_instance: usize,

    #[serde(skip_deserializing, default)]
    next_decode_instance: usize,

    #[serde(skip_deserializing, default)]
    pending_requests: HashMap<u64, PendingRequest>,

    #[serde(skip_deserializing, default)]
    results: Vec<RequestResult>,
}

crate::register_system!(PdDisaggregationSystem, "pd_disaggregation");

#[derive(Debug)]
struct PendingRequest {
    original: Request,
    prefill_request: Request,
    prefill_instance: usize,
    prefill_result: Option<(RequestResult, f64)>,
    decode_request: Option<Request>,
    decode_instance: Option<usize>,
    sink: Option<*mut dyn LogicalHandler>,
}

struct LlmTokenData<'a> {
    input_tokens: u32,
    output_tokens: u32,
    input_token_ids: Option<&'a [u32]>,
    output_token_ids: Option<&'a [u32]>,
}

#[derive(Debug)]
enum LogicalPayload {
    TransferComplete { request_id: u64 },
}

enum HandlerPayload {
    Internal(LogicalPayload),
    Event(SystemEvent),
}

fn parse_payload(payload: Box<dyn Any>) -> Result<HandlerPayload> {
    match payload.downcast::<LogicalPayload>() {
        Ok(payload) => Ok(HandlerPayload::Internal(*payload)),
        Err(payload) => match payload.downcast::<SystemEvent>() {
            Ok(payload) => Ok(HandlerPayload::Event(*payload)),
            Err(_) => Err(anyhow!(
                "pd_disaggregation received an unexpected logical payload"
            )),
        },
    }
}

impl PendingRequest {
    fn new(
        original: Request,
        prefill_request: Request,
        prefill_instance: usize,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Self {
        Self {
            original,
            prefill_request,
            prefill_instance,
            prefill_result: None,
            decode_request: None,
            decode_instance: None,
            sink,
        }
    }
}

impl PdDisaggregationSystem {
    fn single_llm_sub_request(request: &Request) -> Result<LlmTokenData<'_>> {
        let [sub_request] = request.sub_requests.as_slice() else {
            bail!("pd_disaggregation only supports requests with a single llm sub-request");
        };
        let SubRequest::Llm {
            input_tokens,
            output_tokens,
            input_token_ids,
            output_token_ids,
            ..
        } = sub_request
        else {
            bail!("pd_disaggregation only supports llm sub-requests");
        };
        Ok(LlmTokenData {
            input_tokens: *input_tokens,
            output_tokens: *output_tokens,
            input_token_ids: input_token_ids.as_deref(),
            output_token_ids: output_token_ids.as_deref(),
        })
    }

    fn single_llm_tokens(request: &Request) -> Result<(u32, u32)> {
        let tokens = Self::single_llm_sub_request(request)?;
        Ok((tokens.input_tokens, tokens.output_tokens))
    }

    fn build_llm_request(
        original: &Request,
        input_tokens: u32,
        output_tokens: u32,
        kv_tokens: u32,
        input_token_ids: Option<Vec<u32>>,
        output_token_ids: Option<Vec<u32>>,
    ) -> Request {
        Request {
            request_id: original.request_id,
            arrival_time: original.arrival_time,
            initial: vec![0],
            sub_requests: vec![SubRequest::Llm {
                input_tokens,
                output_tokens,
                known_tokens: Some(input_tokens),
                kv_tokens: Some(kv_tokens),
                next: Vec::new(),
                interval: None,
                model: None,
                input_token_ids,
                output_token_ids,
            }],
        }
    }

    fn build_prefill_request(request: &Request) -> Result<Request> {
        let tokens = Self::single_llm_sub_request(request)?;
        let mut prefill = Self::build_llm_request(
            request,
            tokens.input_tokens,
            tokens.output_tokens.min(1),
            0,
            tokens.input_token_ids.map(<[u32]>::to_vec),
            tokens.output_token_ids.map(<[u32]>::to_vec),
        );
        if let SubRequest::Llm {
            interval, model, ..
        } = &mut prefill.sub_requests[0]
        {
            *interval = request.sub_requests[0].interval();
            *model = request.sub_requests[0].model().map(str::to_owned);
        }
        Ok(prefill)
    }

    fn build_decode_request(
        original: &Request,
        prefill_request: &Request,
    ) -> Result<Option<Request>> {
        let tokens = Self::single_llm_sub_request(original)?;
        let carried_tokens = prefill_request
            .total_input_tokens()
            .saturating_add(prefill_request.total_output_tokens());
        let remaining_output = tokens
            .output_tokens
            .saturating_sub(prefill_request.total_output_tokens());
        if remaining_output == 0 {
            return Ok(None);
        }

        let carried_output = prefill_request.total_output_tokens() as usize;
        let input_token_ids = tokens.input_token_ids.map(|input_token_ids| {
            let mut combined = input_token_ids.to_vec();
            if let Some(output_token_ids) = tokens.output_token_ids {
                combined.extend(output_token_ids.iter().copied().take(carried_output));
            }
            combined
        });
        let output_token_ids = tokens.output_token_ids.map(|output_token_ids| {
            output_token_ids
                .iter()
                .copied()
                .skip(carried_output)
                .collect()
        });

        Ok(Some(Self::build_llm_request(
            original,
            carried_tokens,
            remaining_output,
            prefill_request.total_input_tokens(),
            input_token_ids,
            output_token_ids,
        )))
    }

    fn choose_prefill_instance(&mut self) -> Result<usize> {
        if self.prefill_instances.is_empty() {
            bail!("pd_disaggregation requires at least one prefill instance");
        }
        let index = self.next_prefill_instance % self.prefill_instances.len();
        self.next_prefill_instance = (index + 1) % self.prefill_instances.len();
        Ok(index)
    }

    fn choose_decode_instance(&mut self) -> Result<usize> {
        if self.decode_instances.is_empty() {
            bail!("pd_disaggregation requires at least one decode instance");
        }
        let index = self.next_decode_instance % self.decode_instances.len();
        self.next_decode_instance = (index + 1) % self.decode_instances.len();
        Ok(index)
    }

    fn merge_result(
        original: &Request,
        (prefill_result, prefill_finished_at): (RequestResult, f64),
        decode_result: Option<(RequestResult, f64)>,
    ) -> Result<RequestResult> {
        let (input_tokens, output_tokens) = Self::single_llm_tokens(original)?;
        let ResultSubRequest::Llm {
            token_latency: mut combined_token_latency,
            ..
        } = prefill_result
            .sub_requests
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("pd_disaggregation is missing prefill result sub-request"))?
        else {
            bail!("pd_disaggregation prefill result must contain one llm sub-request");
        };

        if let Some((decode_result, decode_finished_at)) = decode_result {
            let ResultSubRequest::Llm {
                token_latency: mut decode_token_latency,
                ..
            } = decode_result
                .sub_requests
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("pd_disaggregation is missing decode result sub-request"))?
            else {
                bail!("pd_disaggregation decode result must contain one llm sub-request");
            };
            // Recover the first decode token's time from completion and later ITLs.
            let later_latency: f64 = decode_token_latency.iter().skip(1).sum();
            if let Some(first) = decode_token_latency.first_mut() {
                *first = (decode_finished_at - prefill_finished_at - later_latency).max(0.0);
            }
            combined_token_latency.extend(decode_token_latency);
        }
        if combined_token_latency.len() != output_tokens as usize {
            bail!("pd_disaggregation completed with an incorrect output token count");
        }

        Ok(RequestResult {
            request_id: original.request_id,
            arrival_time: original.arrival_time,
            initial: vec![0],
            sub_requests: vec![ResultSubRequest::Llm {
                input_tokens,
                output_tokens,
                next: Vec::new(),
                token_latency: combined_token_latency,
                interval: original.sub_requests[0].interval(),
                model: original.sub_requests[0].model().map(str::to_owned),
            }],
        })
    }

    fn emit_final_result(
        &mut self,
        pending: PendingRequest,
        decode_result: Option<(RequestResult, f64)>,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let prefill_result = pending.prefill_result.ok_or_else(|| {
            anyhow!(
                "pd_disaggregation is missing prefill result for request {}",
                pending.original.request_id
            )
        })?;
        let result = Self::merge_result(&pending.original, prefill_result, decode_result)?;
        if let Some(sink) = pending.sink {
            let event = graph.add_logical_node(
                sink,
                SystemEvent::RequestCompleted {
                    request: pending.original,
                    result,
                },
            );
            graph.add_edge(current, event)?;
        } else {
            self.results.push(result);
        }
        Ok(())
    }

    fn handle_child_completion(
        &mut self,
        request: Request,
        result: RequestResult,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        let request_id = request.request_id;
        let prefill_request = self
            .pending_requests
            .get(&request_id)
            .map(|pending| pending.prefill_request.clone())
            .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;

        if request == prefill_request {
            let decode_request = {
                let pending = self
                    .pending_requests
                    .get(&request_id)
                    .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
                match &pending.decode_request {
                    Some(request) => Some(request.clone()),
                    None => {
                        Self::build_decode_request(&pending.original, &pending.prefill_request)?
                    }
                }
            };

            if let Some(decode_request) = decode_request {
                let decode_instance = self.choose_decode_instance()?;
                let prefill_instance = {
                    let pending = self.pending_requests.get(&request_id).ok_or_else(|| {
                        anyhow!("pd_disaggregation is missing request {request_id}")
                    })?;
                    pending.prefill_instance
                };

                let pending = self
                    .pending_requests
                    .get_mut(&request_id)
                    .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
                pending.prefill_result = Some((result, now));
                pending.decode_request = Some(decode_request.clone());
                pending.decode_instance = Some(decode_instance);

                let transfer_tokens = prefill_request.total_input_tokens();
                let transfer_bytes = u64::from(transfer_tokens)
                    .saturating_mul(self.model.kv_bytes_per_token_per_device());
                let transfer = graph.add_network_node(
                    self.network_sim_ptr(),
                    self.prefill_device_ids[prefill_instance],
                    self.decode_device_ids[decode_instance],
                    transfer_bytes,
                );
                graph.add_edge(current, transfer)?;
                let transfer_complete = graph.add_logical_node(
                    self as *mut Self as *mut dyn LogicalHandler,
                    LogicalPayload::TransferComplete { request_id },
                );
                graph.add_edge(transfer, transfer_complete)?;
                return Ok(0.0);
            }

            let pending = self
                .pending_requests
                .remove(&request_id)
                .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
            let mut pending = pending;
            pending.prefill_result = Some((result, now));
            self.emit_final_result(pending, None, current, graph)?;
            return Ok(0.0);
        }

        let expected_decode_request = self
            .pending_requests
            .get(&request_id)
            .and_then(|pending| pending.decode_request.clone())
            .ok_or_else(|| {
                anyhow!("pd_disaggregation received unexpected completion for request {request_id}")
            })?;
        if request != expected_decode_request {
            bail!(
                "pd_disaggregation received mismatched completion payload for request {request_id}"
            );
        }

        let pending = self
            .pending_requests
            .remove(&request_id)
            .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
        self.emit_final_result(pending, Some((result, now)), current, graph)?;
        Ok(0.0)
    }

    fn handle_transfer_complete(
        &mut self,
        request_id: u64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        let (decode_request, decode_instance) = {
            let pending = self
                .pending_requests
                .get(&request_id)
                .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
            (
                pending.decode_request.clone().ok_or_else(|| {
                    anyhow!("pd_disaggregation is missing decode request for request {request_id}")
                })?,
                pending.decode_instance.ok_or_else(|| {
                    anyhow!("pd_disaggregation is missing decode instance for request {request_id}")
                })?,
            )
        };

        let sink = self as *mut Self as *mut dyn LogicalHandler;
        self.decode_instances[decode_instance].add_request_arrival(
            decode_request,
            graph,
            Some(current),
            Some(sink),
        )?;
        Ok(0.0)
    }

    fn network_sim_ptr(&self) -> *mut dyn NetworkSimulator {
        self.network_sim.as_ref() as *const dyn NetworkSimulator as *mut dyn NetworkSimulator
    }
}

impl LogicalHandler for PdDisaggregationSystem {
    fn describe(&self, payload: &dyn Any) -> String {
        if let Some(value) = payload.downcast_ref::<LogicalPayload>() {
            format!("{}::{value:?}", std::any::type_name::<Self>())
        } else {
            std::any::type_name::<Self>().to_owned()
        }
    }

    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        match parse_payload(payload)? {
            HandlerPayload::Internal(LogicalPayload::TransferComplete { request_id }) => {
                self.handle_transfer_complete(request_id, current, graph)
            }
            HandlerPayload::Event(SystemEvent::RequestCompleted { request, result }) => {
                self.handle_child_completion(request, result, now, current, graph)
            }
        }
    }
}

impl Module for PdDisaggregationSystem {
    fn observe(&self) -> serde_json::Value {
        serde_json::json!({
            "prefill_instances": self.prefill_instances.iter().map(|s| s.observe()).collect::<Vec<_>>(),
            "decode_instances": self.decode_instances.iter().map(|s| s.observe()).collect::<Vec<_>>(),
        })
    }

    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        for instance in &mut self.prefill_instances {
            instance.resolve_path(base_dir)?;
        }
        for instance in &mut self.decode_instances {
            instance.resolve_path(base_dir)?;
        }
        self.model.resolve_path(base_dir)?;
        self.network_sim.resolve_path(base_dir)?;
        Ok(())
    }
}

impl System for PdDisaggregationSystem {
    fn instance_load(&self) -> crate::types::InstanceLoad {
        self.prefill_instances
            .iter()
            .chain(&self.decode_instances)
            .map(|system| system.instance_load())
            .sum()
    }

    fn init(&mut self) -> Result<()> {
        if self.prefill_instances.is_empty() {
            bail!("pd_disaggregation requires at least one prefill instance");
        }
        if self.decode_instances.is_empty() {
            bail!("pd_disaggregation requires at least one decode instance");
        }

        self.next_prefill_instance = 0;
        self.next_decode_instance = 0;
        self.pending_requests.clear();
        self.results.clear();
        self.prefill_device_ids.clear();
        self.decode_device_ids.clear();
        self.model.init(ParallelStrategy {
            tp: self.tp,
            pp: 1,
            ep: 1,
        })?;

        for instance in &mut self.prefill_instances {
            instance.init()?;
            self.prefill_device_ids.push(self.network_sim.add_device());
        }
        for instance in &mut self.decode_instances {
            instance.init()?;
            self.decode_device_ids.push(self.network_sim.add_device());
        }

        for &src in &self.prefill_device_ids {
            for &dst in &self.decode_device_ids {
                self.network_sim
                    .add_link(src, dst, self.link_bw, self.link_latency)?;
            }
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
        if request
            .sub_requests
            .iter()
            .any(|sub| sub.known_tokens().is_some() || sub.kv_tokens().is_some())
        {
            bail!("pd_disaggregation does not support external seeded progress");
        }
        if self.pending_requests.contains_key(&request.request_id) {
            bail!(
                "pd_disaggregation received duplicate request id {}",
                request.request_id
            );
        }

        let prefill_instance = self.choose_prefill_instance()?;
        let request_id = request.request_id;
        let prefill_request = Self::build_prefill_request(&request)?;
        let pending = PendingRequest::new(request, prefill_request.clone(), prefill_instance, sink);
        self.pending_requests.insert(request_id, pending);

        let sink = self as *mut Self as *mut dyn LogicalHandler;
        if let Err(err) = self.prefill_instances[prefill_instance].add_request_arrival(
            prefill_request,
            graph,
            parent,
            Some(sink),
        ) {
            self.pending_requests.remove(&request_id);
            return Err(err);
        }

        Ok(())
    }

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>> {
        if !self.pending_requests.is_empty() {
            bail!(
                "pd_disaggregation finished with {} incomplete requests",
                self.pending_requests.len()
            );
        }
        Ok(self.results)
    }
}

#[cfg(test)]
mod tests {
    use super::{PdDisaggregationSystem, Request, SubRequest};

    #[test]
    fn decode_handoff_excludes_sampled_token_from_kv() {
        let original = Request {
            request_id: 1,
            arrival_time: 2.0,
            initial: vec![0],
            sub_requests: vec![SubRequest::Llm {
                input_tokens: 16,
                output_tokens: 3,
                known_tokens: Some(16),
                kv_tokens: Some(0),
                next: vec![],
                interval: None,
                model: None,
                input_token_ids: Some((0..16).collect()),
                output_token_ids: Some(vec![20, 21, 22]),
            }],
        };
        let prefill = PdDisaggregationSystem::build_prefill_request(&original).unwrap();
        let decode = PdDisaggregationSystem::build_decode_request(&original, &prefill)
            .unwrap()
            .unwrap();
        assert_eq!(decode.total_input_tokens(), 17);
        assert_eq!(decode.total_output_tokens(), 2);
        assert_eq!(decode.sub_requests[0].known_tokens(), Some(17));
        assert_eq!(decode.sub_requests[0].kv_tokens(), Some(16));
        assert_eq!(decode.request_id, original.request_id);
        assert_eq!(decode.arrival_time, original.arrival_time);
        let tokens = PdDisaggregationSystem::single_llm_sub_request(&decode).unwrap();
        let expected_input = (0..16).chain([20]).collect::<Vec<_>>();
        assert_eq!(tokens.input_token_ids, Some(expected_input.as_slice()));
        assert_eq!(tokens.output_token_ids, Some([21, 22].as_slice()));
    }
}
