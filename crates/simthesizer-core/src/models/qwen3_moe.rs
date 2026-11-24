use anyhow::{bail, Result};
use serde::Deserialize;

use crate::model::Model;
use crate::module::Module;
use crate::types::{Layer, ParallelStrategy};

const FP_BYTES: u64 = 2;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3MoeModel {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub moe_intermediate_size: u64,
    pub num_attention_heads: u64,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u64,
    pub num_experts: u64,
    pub num_experts_per_tok: u32,
    pub vocab_size: u64,

    #[serde(default)]
    pub decoder_sparse_step: u32,

    #[serde(default)]
    pub mlp_only_layers: Vec<u32>,

    #[serde(default)]
    pub shared_expert_intermediate_size: u64,

    #[serde(default)]
    pub head_dim: Option<u64>,

    #[serde(skip, default)]
    pre_block_layers: Vec<Layer>,

    #[serde(skip, default)]
    block_layers: Vec<Layer>,

    #[serde(skip, default)]
    post_block_layers: Vec<Layer>,

    #[serde(skip, default)]
    block_count: u32,

    #[serde(skip, default)]
    weight_bytes_per_device: u64,

    #[serde(skip, default)]
    kv_bytes_per_token_per_device: u64,

    #[serde(skip, default)]
    scheduler_kv_bytes_per_token: u64,
}

crate::register_model!(Qwen3MoeModel, "qwen3_moe");

impl Module for Qwen3MoeModel {}

impl Qwen3MoeModel {
    fn head_dim(&self) -> u64 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn tp_divisor(tp_size: u32) -> u64 {
        u64::from(tp_size.max(1))
    }

    fn ep_divisor(ep_size: u32) -> u64 {
        u64::from(ep_size.max(1))
    }

    fn scheduler_memory_divisor(ep_size: u32) -> u64 {
        Self::ep_divisor(ep_size)
    }

    fn tp_sync_bytes_per_token(&self) -> u64 {
        self.hidden_size * FP_BYTES
    }

    fn moe_dispatch_bytes_per_token(&self) -> u64 {
        self.hidden_size * FP_BYTES + self.num_experts * FP_BYTES
    }

    fn moe_combine_bytes_per_token(&self) -> u64 {
        self.hidden_size * FP_BYTES
    }

    fn local_num_attention_heads(&self, tp_size: u32) -> u64 {
        self.num_attention_heads / Self::tp_divisor(tp_size)
    }

    fn local_num_kv_heads(&self, tp_size: u32) -> u64 {
        let tp = Self::tp_divisor(tp_size);
        if self.num_key_value_heads >= tp {
            self.num_key_value_heads / tp
        } else {
            1
        }
    }

    fn kv_dim_per_device(&self, tp_size: u32) -> u64 {
        self.local_num_kv_heads(tp_size) * self.head_dim()
    }

    fn is_sparse_layer(&self, layer_idx: u32) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    fn num_shared_experts_per_token(&self) -> u32 {
        u32::from(self.shared_expert_intermediate_size > 0)
    }

    fn max_local_experts(&self, ep_size: u32) -> u64 {
        self.num_experts.div_ceil(Self::ep_divisor(ep_size))
    }

    fn build_pre_block_layers(&self) -> Vec<Layer> {
        vec![Layer::Embedding {
            name: "embedding".to_string(),
            num_embeddings: self.vocab_size,
            embedding_dim: self.hidden_size,
        }]
    }

    fn build_attention_prefix(&self, tp_size: u32) -> Vec<Layer> {
        let local_attention_heads = self.local_num_attention_heads(tp_size);
        let local_kv_heads = self.local_num_kv_heads(tp_size);
        let head_dim = self.head_dim();
        let attention_width = local_attention_heads * head_dim;

        let mut layers = vec![
            Layer::LayerNorm {
                name: "layernorm".to_string(),
                hidden_size: self.hidden_size,
            },
            Layer::Gemm {
                name: "qkv_projection".to_string(),
                m: self.hidden_size,
                n: (local_attention_heads + 2 * local_kv_heads) * head_dim,
            },
            Layer::LayerNorm {
                name: "qk_norm".to_string(),
                hidden_size: head_dim,
            },
            Layer::Rope {
                name: "rope".to_string(),
                head_dim,
                num_heads: local_attention_heads,
            },
            Layer::Attention {
                name: "attention".to_string(),
                num_q_heads: local_attention_heads,
                num_kv_heads: local_kv_heads,
                head_dim,
            },
            Layer::Gemm {
                name: "o_projection".to_string(),
                m: attention_width,
                n: self.hidden_size,
            },
        ];
        if tp_size > 1 {
            layers.push(Layer::AllReduce {
                name: "all_reduce".to_string(),
                bytes_per_token: self.tp_sync_bytes_per_token(),
            });
        }
        layers
    }

    fn build_dense_mlp_suffix(&self, tp_size: u32) -> Vec<Layer> {
        let tp = Self::tp_divisor(tp_size);
        let mut layers = vec![
            Layer::LayerNorm {
                name: "layernorm".to_string(),
                hidden_size: self.hidden_size,
            },
            Layer::Gemm {
                name: "ffn1".to_string(),
                m: self.hidden_size,
                n: 2 * (self.intermediate_size / tp),
            },
            Layer::Act {
                name: "act".to_string(),
                width: self.intermediate_size / tp,
            },
            Layer::Gemm {
                name: "ffn2".to_string(),
                m: self.intermediate_size / tp,
                n: self.hidden_size,
            },
        ];
        if tp_size > 1 {
            layers.push(Layer::AllReduce {
                name: "all_reduce".to_string(),
                bytes_per_token: self.tp_sync_bytes_per_token(),
            });
        }
        layers
    }

    fn build_sparse_mlp_suffix(&self, tp_size: u32) -> Vec<Layer> {
        let mut layers = vec![
            Layer::LayerNorm {
                name: "layernorm".to_string(),
                hidden_size: self.hidden_size,
            },
            Layer::Moe {
                name: "moe".to_string(),
                num_experts: self.num_experts,
                num_experts_per_token: self.num_experts_per_tok,
                num_shared_experts_per_token: self.num_shared_experts_per_token(),
                dispatch_bytes_per_token: self.moe_dispatch_bytes_per_token(),
                combine_bytes_per_token: self.moe_combine_bytes_per_token(),
            },
        ];
        if tp_size > 1 {
            layers.push(Layer::AllReduce {
                name: "all_reduce".to_string(),
                bytes_per_token: self.tp_sync_bytes_per_token(),
            });
        }
        layers
    }

    fn build_block_plans(&self, tp_size: u32, expert_tp: u32) -> Vec<Vec<Layer>> {
        let attention_prefix = self.build_attention_prefix(tp_size);
        (0..self.num_hidden_layers)
            .map(|block_idx| {
                let mut layers = attention_prefix.clone();
                if self.is_sparse_layer(block_idx) {
                    layers.extend(self.build_sparse_mlp_suffix(expert_tp));
                } else {
                    layers.extend(self.build_dense_mlp_suffix(tp_size));
                }
                layers
            })
            .collect()
    }

    fn build_repeated_block_plan(block_plans: &[Vec<Layer>]) -> (Vec<Layer>, u32) {
        let Some(first) = block_plans.first() else {
            return (Vec::new(), 0);
        };

        if block_plans.iter().all(|plan| plan == first) {
            (first.clone(), block_plans.len() as u32)
        } else {
            let total_layers = block_plans.iter().map(Vec::len).sum();
            let mut flattened = Vec::with_capacity(total_layers);
            for plan in block_plans {
                flattened.extend(plan.iter().cloned());
            }
            (flattened, 1)
        }
    }

    fn build_post_block_layers(&self, tp_size: u32) -> Vec<Layer> {
        let tp = Self::tp_divisor(tp_size);
        vec![
            Layer::LayerNorm {
                name: "final_layernorm".to_string(),
                hidden_size: self.hidden_size,
            },
            Layer::Gemm {
                name: "lm_head".to_string(),
                m: self.hidden_size,
                n: self.vocab_size / tp,
            },
            Layer::Sampler {
                name: "sampler".to_string(),
                vocab_size: self.vocab_size,
            },
        ]
    }

    fn attention_block_weight_elems(&self, tp_size: u32) -> u64 {
        let head_dim = self.head_dim();
        let local_attention_heads = self.local_num_attention_heads(tp_size);
        let local_kv_heads = self.local_num_kv_heads(tp_size);
        let attention_width = local_attention_heads * head_dim;

        let qkv = self.hidden_size * (local_attention_heads + 2 * local_kv_heads) * head_dim;
        let o_proj = attention_width * self.hidden_size;
        let qk_norm = 2 * head_dim;
        let layernorm = 2 * self.hidden_size;
        qkv + o_proj + qk_norm + layernorm
    }

    fn dense_mlp_weight_elems(&self, tp_size: u32) -> u64 {
        let tp = Self::tp_divisor(tp_size);
        3 * self.hidden_size * (self.intermediate_size / tp)
    }

    fn sparse_mlp_weight_elems(&self, tp_size: u32, ep_size: u32) -> u64 {
        let tp = Self::tp_divisor(tp_size);
        let local_experts = self.max_local_experts(ep_size);

        let gate = self.hidden_size * self.num_experts;
        let shared = if self.shared_expert_intermediate_size > 0 {
            self.hidden_size + 3 * self.hidden_size * (self.shared_expert_intermediate_size / tp)
        } else {
            0
        };
        let expert = local_experts * 3 * self.hidden_size * (self.moe_intermediate_size / tp);

        gate + shared + expert
    }

    fn compute_weight_bytes_per_device(
        &self,
        parallel: ParallelStrategy,
        expert_tp: u32,
        full_embedding: bool,
    ) -> u64 {
        let tp = Self::tp_divisor(parallel.tp);
        let embed = (self.vocab_size / if full_embedding { 1 } else { tp }) * self.hidden_size;
        let block = (0..self.num_hidden_layers)
            .map(|block_idx| {
                self.attention_block_weight_elems(parallel.tp)
                    + if self.is_sparse_layer(block_idx) {
                        self.sparse_mlp_weight_elems(expert_tp, parallel.ep)
                    } else {
                        self.dense_mlp_weight_elems(parallel.tp)
                    }
            })
            .sum::<u64>();
        let final_norm = self.hidden_size;
        let lm_head = self.hidden_size * (self.vocab_size / tp);
        FP_BYTES * (embed + block + final_norm + lm_head)
    }

    fn compute_kv_bytes_per_token_per_device(&self, tp_size: u32) -> u64 {
        2 * self.kv_dim_per_device(tp_size) * u64::from(self.num_hidden_layers) * FP_BYTES
    }

    fn compute_scheduler_kv_bytes_per_token(&self, parallel: ParallelStrategy) -> u64 {
        self.compute_kv_bytes_per_token_per_device(parallel.tp)
            .div_ceil(Self::scheduler_memory_divisor(parallel.ep))
    }

    fn validate(&self, parallel: ParallelStrategy) -> Result<()> {
        if parallel.tp != 1 {
            bail!("qwen3_moe only supports tp_size=1");
        }
        if parallel.ep == 0 {
            bail!("ep_size must be positive");
        }
        if self.hidden_size == 0 {
            bail!("hidden_size must be positive");
        }
        if self.intermediate_size == 0 {
            bail!("intermediate_size must be positive");
        }
        if self.moe_intermediate_size == 0 {
            bail!("moe_intermediate_size must be positive");
        }
        if self.num_attention_heads == 0 {
            bail!("num_attention_heads must be positive");
        }
        if self.num_key_value_heads == 0 {
            bail!("num_key_value_heads must be positive");
        }
        if self.num_hidden_layers == 0 {
            bail!("num_hidden_layers must be positive");
        }
        if self.num_experts == 0 {
            bail!("num_experts must be positive");
        }
        if self.num_experts_per_tok == 0 {
            bail!("num_experts_per_tok must be positive");
        }
        if self.vocab_size == 0 {
            bail!("vocab_size must be positive");
        }
        if self.decoder_sparse_step == 0 {
            bail!("decoder_sparse_step must be positive");
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            bail!("hidden_size must be divisible by num_attention_heads");
        }
        if let Some(head_dim) = self.head_dim {
            if head_dim == 0 {
                bail!("head_dim must be positive");
            }
        }
        if self.num_attention_heads < u64::from(parallel.tp)
            || !self
                .num_attention_heads
                .is_multiple_of(u64::from(parallel.tp))
        {
            bail!("num_attention_heads must be divisible by tp_size");
        }
        if self.num_key_value_heads >= u64::from(parallel.tp) {
            if !self
                .num_key_value_heads
                .is_multiple_of(u64::from(parallel.tp))
            {
                bail!("num_key_value_heads must be divisible by tp_size when kv heads are sharded");
            }
        } else if !u64::from(parallel.tp).is_multiple_of(self.num_key_value_heads) {
            bail!("tp_size must be divisible by num_key_value_heads when kv heads are replicated");
        }
        if self.num_experts_per_tok > self.num_experts as u32 {
            bail!("num_experts_per_tok must not exceed num_experts");
        }
        for &layer_idx in &self.mlp_only_layers {
            if layer_idx >= self.num_hidden_layers {
                bail!("mlp_only_layers contains out-of-range layer index {layer_idx}");
            }
        }
        Ok(())
    }

    fn initialize(&mut self, parallel: ParallelStrategy, expert_tp: u32, full_embedding: bool) {
        self.pre_block_layers = self.build_pre_block_layers();
        let block_plans = self.build_block_plans(parallel.tp, expert_tp);
        (self.block_layers, self.block_count) = Self::build_repeated_block_plan(&block_plans);
        self.post_block_layers = self.build_post_block_layers(parallel.tp);
        self.weight_bytes_per_device =
            self.compute_weight_bytes_per_device(parallel, expert_tp, full_embedding);
        self.kv_bytes_per_token_per_device =
            self.compute_kv_bytes_per_token_per_device(parallel.tp);
        self.scheduler_kv_bytes_per_token = if full_embedding {
            self.kv_bytes_per_token_per_device
                .div_ceil(u64::from(parallel.ep / parallel.tp))
        } else {
            self.compute_scheduler_kv_bytes_per_token(parallel)
        };
    }
}

impl Model for Qwen3MoeModel {
    fn init(&mut self, parallel: ParallelStrategy) -> Result<()> {
        self.validate(parallel)?;
        self.initialize(parallel, parallel.tp, false);
        Ok(())
    }

    fn init_with_expert_tp(&mut self, parallel: ParallelStrategy, expert_tp: u32) -> Result<()> {
        // Native profile runners retain their original TP=1 contract via init().
        self.validate(ParallelStrategy { tp: 1, ..parallel })?;
        anyhow::ensure!(expert_tp == 1, "Qwen shared supports expert_tp=1 only");
        anyhow::ensure!(parallel.pp == 1, "Qwen shared supports PP=1 only");
        anyhow::ensure!(
            matches!(parallel.tp, 1 | 2 | 4)
                && matches!(parallel.ep, 1 | 2 | 4 | 8)
                && parallel.ep.is_multiple_of(parallel.tp)
                && self.num_attention_heads.is_multiple_of(u64::from(parallel.tp))
                && self.num_key_value_heads.is_multiple_of(u64::from(parallel.tp))
                && self.vocab_size.is_multiple_of(u64::from(parallel.tp)),
            "Qwen shared requires Attention TP=1/2/4 dividing EP and all head/vocabulary dimensions"
        );
        self.initialize(parallel, expert_tp, true);
        Ok(())
    }

    fn weight_bytes_per_device(&self) -> u64 {
        self.weight_bytes_per_device
    }

    fn kv_bytes_per_token_per_device(&self) -> u64 {
        self.kv_bytes_per_token_per_device
    }

    fn scheduler_kv_bytes_per_token(&self) -> u64 {
        self.scheduler_kv_bytes_per_token
    }

    fn pre_block_layers(&self) -> &[Layer] {
        &self.pre_block_layers
    }

    fn block_layers(&self) -> &[Layer] {
        &self.block_layers
    }

    fn post_block_layers(&self) -> &[Layer] {
        &self.post_block_layers
    }

    fn num_blocks(&self) -> u32 {
        self.block_count
    }
}
