use serde_json::json;
use simthesizer_core::types::{Layer, ParallelStrategy, ReadySubRequest};
use simthesizer_core::{Model, Scheduler};

fn parallel(ep: u32) -> ParallelStrategy {
    ParallelStrategy { tp: 1, pp: 1, ep }
}

fn model(ep: u32) -> Box<dyn Model> {
    let mut model: Box<dyn Model> = toml::from_str("kind = 'qwen3_235b_a22b'").unwrap();
    model.init(parallel(ep)).unwrap();
    model
}

#[test]
fn fixed_checkpoint_builds_all_94_moe_layers_with_explicit_head_dimension() {
    let model = model(8);
    assert_eq!(model.num_blocks(), 94);
    assert_eq!(
        model.block_layers(),
        [
            Layer::LayerNorm {
                name: "layernorm".into(),
                hidden_size: 4096
            },
            Layer::Gemm {
                name: "qkv_projection".into(),
                m: 4096,
                n: 9216
            },
            Layer::LayerNorm {
                name: "qk_norm".into(),
                hidden_size: 128
            },
            Layer::Rope {
                name: "rope".into(),
                head_dim: 128,
                num_heads: 64
            },
            Layer::Attention {
                name: "attention".into(),
                num_q_heads: 64,
                num_kv_heads: 4,
                head_dim: 128,
            },
            Layer::Gemm {
                name: "o_projection".into(),
                m: 8192,
                n: 4096
            },
            Layer::LayerNorm {
                name: "layernorm".into(),
                hidden_size: 4096
            },
            Layer::Moe {
                name: "moe".into(),
                num_experts: 128,
                num_experts_per_token: 8,
                num_shared_experts_per_token: 0,
                dispatch_bytes_per_token: 8448,
                combine_bytes_per_token: 8192,
            },
        ]
    );
    assert_eq!(
        model.pre_block_layers(),
        [Layer::Embedding {
            name: "embedding".into(),
            num_embeddings: 151936,
            embedding_dim: 4096,
        }]
    );
    assert_eq!(
        model.post_block_layers(),
        [
            Layer::LayerNorm {
                name: "final_layernorm".into(),
                hidden_size: 4096
            },
            Layer::Gemm {
                name: "lm_head".into(),
                m: 4096,
                n: 151936
            },
            Layer::Sampler {
                name: "sampler".into(),
                vocab_size: 151936
            },
        ]
    );
}

#[test]
fn checkpoint_weights_and_kv_follow_expert_placement() {
    // BF16: 235,093,634,560 total parameters at EP=1. Only routed expert
    // matrices shard over EP; attention, router, embedding and head replicate.
    for (ep, weights, scheduler_kv) in [
        (1, 470_187_269_120, 192_512),
        (2, 243_090_873_344, 96_256),
        (8, 72_768_576_512, 24_064),
    ] {
        let model = model(ep);
        assert_eq!(model.weight_bytes_per_device(), weights, "EP={ep}");
        assert_eq!(model.scheduler_model_weight_bytes(), weights, "EP={ep}");
        // K/V x 4 heads x 128 dimensions x 94 layers x 2 bytes.
        assert_eq!(model.kv_bytes_per_token_per_device(), 192_512);
        assert_eq!(model.scheduler_kv_bytes_per_token(), scheduler_kv);
        assert_eq!(model.kv_bytes_per_request_per_device(), 0);
    }
}

#[test]
fn checkpoint_name_rejects_parameter_overrides_and_unsupported_tp() {
    for (field, value) in [
        ("num_hidden_layers", json!(2)),
        ("head_dim", json!(64)),
        ("num_experts_per_tok", json!(1)),
        ("shared_expert_intermediate_size", json!(4)),
        ("dtype", json!("fp8")),
    ] {
        let mut config = json!({"kind": "qwen3_235b_a22b"});
        config[field] = value;
        let error = serde_json::from_value::<Box<dyn Model>>(config)
            .err()
            .unwrap();
        assert!(error.to_string().contains("unknown field"), "{error}");
    }
    let mut model: Box<dyn Model> = toml::from_str("kind = 'qwen3_235b_a22b'").unwrap();
    let error = model
        .init(ParallelStrategy {
            tp: 2,
            pp: 1,
            ep: 1,
        })
        .unwrap_err();
    assert!(error.to_string().contains("only supports tp_size=1"));
}

#[test]
fn shared_attention_tp_preserves_complete_experts_and_embedding() {
    let reference = model(8);
    for (tp, weights, kv) in [
        (1, 72_768_576_512, 192_512),
        (2, 65_443_748_864, 96_256),
        (4, 61_781_335_040, 48_128),
    ] {
        let mut model: Box<dyn Model> = toml::from_str("kind = 'qwen3_235b_a22b'").unwrap();
        model
            .init_with_expert_tp(ParallelStrategy { tp, pp: 1, ep: 8 }, 1)
            .unwrap();
        assert_eq!(model.weight_bytes_per_device(), weights);
        // 16 full experts per rank alone occupy 56,774,098,944 bytes.
        assert!(model.weight_bytes_per_device() > 56_774_098_944);
        assert_eq!(model.kv_bytes_per_token_per_device(), kv);
        assert_eq!(model.scheduler_kv_bytes_per_token(), 24_064);
        assert_eq!(model.pre_block_layers(), reference.pre_block_layers());
        assert!(model.block_layers().contains(&Layer::Attention {
            name: "attention".into(),
            num_q_heads: 64 / u64::from(tp),
            num_kv_heads: 4 / u64::from(tp),
            head_dim: 128,
        }));
        assert_eq!(
            model
                .block_layers()
                .iter()
                .filter(|layer| matches!(layer, Layer::AllReduce { .. }))
                .count(),
            usize::from(tp > 1)
        );
        assert!(matches!(
            model.block_layers().last(),
            Some(Layer::Moe { .. })
        ));
        assert!(model.post_block_layers().contains(&Layer::Gemm {
            name: "lm_head".into(),
            m: 4096,
            n: 151936 / u64::from(tp),
        }));
    }
}

#[test]
fn independent_expert_tp_rejects_unsupported_deployments() {
    for (tp, pp, ep, expert_tp) in [
        (2, 1, 8, 2),
        (8, 1, 8, 1),
        (4, 1, 2, 1),
        (2, 2, 8, 1),
        (0, 1, 8, 1),
        (2, 1, 3, 1),
        (1, 1, 0, 1),
    ] {
        let mut model: Box<dyn Model> = toml::from_str("kind = 'qwen3_235b_a22b'").unwrap();
        assert!(model
            .init_with_expert_tp(ParallelStrategy { tp, pp, ep }, expert_tp)
            .is_err());
    }
}

#[test]
fn scheduler_reserves_full_checkpoint_weights_then_admits_and_releases_requests() {
    let model = model(8);
    let weights = 72_768_576_512_u64;
    let one_request = 4 * 24_064;
    for (capacity, fits) in [(weights + one_request, true), (weights - 1, false)] {
        let mut scheduler: Box<dyn Scheduler> = serde_json::from_value(json!({
            "kind": "chunked_prefill", "block_size": 4, "max_batch": 2,
            "max_num_batched_tokens": 8, "device_mem": capacity,
            "device_mem_utilization": 1.0, "enable_prefix_caching": false,
        }))
        .unwrap();
        scheduler
            .init(
                model.scheduler_model_weight_bytes(),
                model.scheduler_kv_bytes_per_request(),
                model.scheduler_kv_bytes_per_token(),
                parallel(8),
            )
            .unwrap();
        for id in 0..2 {
            scheduler.enqueue_sub_request(ReadySubRequest {
                request_id: id,
                subrequest_index: 0,
                completion_node_id: id,
                arrival_time: 0.0,
                input_tokens: 4,
                output_tokens: 1,
                known_tokens: 4,
                kv_tokens: 0,
                input_token_ids: None,
                output_token_ids: None,
            });
        }
        if fits {
            assert_eq!(scheduler.schedule().unwrap().len(), 1);
            assert_eq!(scheduler.done_iteration(1.0).len(), 1);
            assert_eq!(scheduler.schedule().unwrap().len(), 1);
            assert_eq!(scheduler.done_iteration(2.0).len(), 1);
            assert!(scheduler.schedule().unwrap().is_empty());
        } else {
            assert!(scheduler.schedule().is_err());
        }
    }
}
