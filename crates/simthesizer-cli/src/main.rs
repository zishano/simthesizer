use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use simthesizer_core::types::Request;
use simthesizer_core::wall_profile::{self, scope, Section};
use simthesizer_core::{Engine, Graph, System};

mod progress;
mod write_jsonl;
mod write_trace;

use crate::write_jsonl::write_results_jsonl;

#[derive(Parser, Debug)]
struct Args {
    /// Print the native serving-policy contract without starting a simulation.
    #[arg(long, exclusive = true)]
    policy_capabilities: bool,
    #[arg(long, required_unless_present = "policy_capabilities")]
    config: Option<PathBuf>,
    /// Stream completed root-request counts to stderr.
    #[arg(long)]
    progress: bool,
    #[arg(long, required_unless_present = "policy_capabilities")]
    workload: Option<PathBuf>,
    #[arg(long, required_unless_present = "policy_capabilities")]
    output_jsonl: Option<PathBuf>,
    /// Aggregate decision counts and wall-clock overhead, written once at completion.
    #[arg(long)]
    output_policy_stats: Option<PathBuf>,
    /// Profile host wall time, independently of simulated service time.
    #[arg(long)]
    output_wall_profile: Option<PathBuf>,
    /// Component state at the engine clock, as JSONL.
    #[arg(long)]
    output_engine_trace: Option<PathBuf>,
    /// Complete dynamic graph event history, as JSONL.
    #[arg(long)]
    output_graph_trace: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    system: Box<dyn System>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.policy_capabilities {
        println!(
            "{}",
            serde_json::json!({
                "policy_api_version": simthesizer_core::policies::API_VERSION,
                "actions": ["route", "batch", "submit_kv", "wait"],
                "observation_features": [simthesizer_core::policies::OBSERVATION_FEATURE],
                "diagnostic_features": [simthesizer_core::policies::DIAGNOSTIC_FEATURE],
                "length_model": "known_trace", "scheduler_cost": "zero_simulated_time",
                "validates_actions": true, "checks_full_completion": true
            })
        );
        return Ok(());
    }
    let profile_path = args
        .output_wall_profile
        .as_deref()
        .map(output_path)
        .transpose()?;
    if let Some(path) = &profile_path {
        anyhow::ensure!(
            !path.exists(),
            "wall-profile output already exists: {}",
            path.display()
        );
        for output in [
            &args.output_jsonl,
            &args.output_policy_stats,
            &args.output_engine_trace,
            &args.output_graph_trace,
        ]
        .into_iter()
        .flatten()
        {
            anyhow::ensure!(
                path != &output_path(output)?,
                "wall profile and other output paths must differ"
            );
        }
        wall_profile::enable();
    }
    let result = run(args);
    if let Some(path) = profile_path {
        let profile_result = write_wall_profile(&path);
        if result.is_ok() {
            profile_result?;
        } else if let Err(error) = profile_result {
            eprintln!("failed to save partial wall profile: {error:#}");
        }
    }
    result
}

fn write_wall_profile(path: &Path) -> Result<()> {
    let snapshot = wall_profile::snapshot().context("wall profile is not enabled")?;
    let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
    serde_json::to_writer_pretty(file.as_file_mut(), &snapshot)?;
    file.persist_noclobber(path)?;
    Ok(())
}

fn run(mut args: Args) -> Result<()> {
    let output_jsonl = output_path(args.output_jsonl.as_ref().unwrap())?;
    args.output_engine_trace = args
        .output_engine_trace
        .as_deref()
        .map(output_path)
        .transpose()?;
    if let Some(path) = &args.output_engine_trace {
        anyhow::ensure!(
            !path.exists(),
            "trace output already exists: {}",
            path.display()
        );
        anyhow::ensure!(path != &output_jsonl, "trace and result paths must differ");
    }
    args.output_graph_trace = args
        .output_graph_trace
        .as_deref()
        .map(output_path)
        .transpose()?;
    if let Some(path) = &args.output_graph_trace {
        anyhow::ensure!(
            !path.exists(),
            "graph trace already exists: {}",
            path.display()
        );
        anyhow::ensure!(
            path != &output_jsonl && Some(path) != args.output_engine_trace.as_ref(),
            "graph trace, engine trace and result paths must differ"
        );
    }
    let policy_stats_path = args
        .output_policy_stats
        .as_deref()
        .map(output_path)
        .transpose()?;
    if let Some(path) = &policy_stats_path {
        anyhow::ensure!(!path.exists(), "policy statistics output already exists");
        anyhow::ensure!(
            path != &output_jsonl
                && Some(path) != args.output_engine_trace.as_ref()
                && Some(path) != args.output_graph_trace.as_ref(),
            "output paths must differ"
        );
    }
    let config_path = args.config.unwrap();
    let workload_path = args.workload.unwrap();
    let mut config = {
        let _wall = scope(Section::ConfigLoad);
        load_config(&config_path)?
    };
    let system_base_path = config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    {
        let _wall = scope(Section::SystemInit);
        config.system.resolve_path(&system_base_path)?;
        config.system.init()?;
    }
    let graph_file = args
        .output_graph_trace
        .as_ref()
        .map(|path| tempfile::NamedTempFile::new_in(path.parent().unwrap()))
        .transpose()?;
    let mut graph = if let Some(file) = &graph_file {
        Graph::with_trace(std::io::BufWriter::new(file.reopen()?))
    } else {
        Graph::new()
    };
    let total = {
        let _wall = scope(Section::WorkloadLoad);
        load_workload(config.system.as_mut(), &mut graph, &workload_path)?
    };
    let mut progress = progress::Progress::new(args.progress, total);
    progress.observe(0.0, config.system.as_ref())?;

    let engine = Engine::new(graph, config.system);
    let (results, statistics) = if let Some(path) = &args.output_engine_trace {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        let mut trace = write_trace::TraceWriter::new(std::io::BufWriter::new(file.as_file_mut()));
        let results = engine.run_with_summary(|time, system| {
            progress.observe(time, system)?;
            trace.observe(time, system.observe())
        })?;
        trace.finish()?;
        file.persist_noclobber(path)?;
        results
    } else if args.progress {
        engine.run_with_summary(|time, system| progress.observe(time, system))?
    } else {
        engine.run_with_summary(|_, _| Ok(()))?
    };

    let _wall = scope(Section::ResultWrite);
    write_results_jsonl(&output_jsonl, &results)?;
    if let Some(path) = policy_stats_path {
        let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
        serde_json::to_writer(file.as_file_mut(), &statistics)?;
        file.persist_noclobber(path)?;
    }
    if let Some(file) = graph_file {
        file.persist_noclobber(args.output_graph_trace.as_ref().unwrap())?;
    }

    Ok(())
}

fn output_path(path: &Path) -> Result<PathBuf> {
    let name = path.file_name().context("output path needs a filename")?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    Ok(parent.canonicalize()?.join(name))
}

fn load_config(config_path: &Path) -> Result<Config> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let document: toml::Value = toml::from_str(&raw)?;
    fn uses_rooflang(value: &toml::Value) -> bool {
        match value {
            toml::Value::Table(table) => {
                table.get("kind").and_then(toml::Value::as_str) == Some("rooflang")
                    || table.values().any(uses_rooflang)
            }
            toml::Value::Array(values) => values.iter().any(uses_rooflang),
            _ => false,
        }
    }
    if uses_rooflang(&document) {
        let system = document.get("system").context("missing system")?;
        match system.get("kind").and_then(toml::Value::as_str) {
            Some("single_instance") => {
                anyhow::ensure!(
                    system
                        .get("model_runner")
                        .and_then(|r| r.get("shared_execution"))
                        .and_then(toml::Value::as_bool)
                        != Some(true),
                    "shared Rooflang execution requires a P/D root"
                );
            }
            Some("pd_disaggregation") => {
                let tp = system
                    .get("tp")
                    .and_then(toml::Value::as_integer)
                    .context("Rooflang P/D requires parent tp")?;
                let mut identities = std::collections::HashSet::new();
                let topology: Option<simthesizer_core::systems::pd_disaggregation::KvTopology> =
                    system
                        .get("kv_topology")
                        .map(|value| value.clone().try_into())
                        .transpose()?;
                let mut hardware_topology = None;
                for pool in ["prefill_instances", "decode_instances"] {
                    let children = system
                        .get(pool)
                        .and_then(toml::Value::as_array)
                        .context("Rooflang P/D requires both instance pools")?;
                    anyhow::ensure!(!children.is_empty(), "Rooflang P/D pool is empty");
                    for (index, child) in children.iter().enumerate() {
                        anyhow::ensure!(child.get("model") == system.get("model"),
                            "Rooflang P/D parent and child model definitions must match for KV sizing");
                        anyhow::ensure!(
                            child.get("kind").and_then(toml::Value::as_str)
                                == Some("single_instance"),
                            "Rooflang P/D children must be single_instance systems"
                        );
                        let runner = child.get("model_runner").context("missing P/D runner")?;
                        anyhow::ensure!(
                            runner.get("kind").and_then(toml::Value::as_str) == Some("rooflang"),
                            "Rooflang P/D requires Rooflang runners in both pools"
                        );
                        let graph_model = runner
                            .get("graph_model")
                            .and_then(toml::Value::as_str)
                            .unwrap_or("llama");
                        anyhow::ensure!(
                            matches!(graph_model, "llama" | "qwen3_235b_a22b"),
                            "unsupported Rooflang graph model"
                        );
                        let qwen = graph_model == "qwen3_235b_a22b";
                        anyhow::ensure!(
                            qwen == (system
                                .get("model")
                                .and_then(|m| m.get("kind"))
                                .and_then(toml::Value::as_str)
                                == Some("qwen3_235b_a22b")),
                            "Rooflang graph model must match the native model kind"
                        );
                        if qwen {
                            anyhow::ensure!(
                                topology.is_some()
                                    && system
                                        .get("shared_execution")
                                        .and_then(toml::Value::as_bool)
                                        == Some(true)
                                    && runner
                                        .get("shared_execution")
                                        .and_then(toml::Value::as_bool)
                                        == Some(true),
                                "Qwen Rooflang requires shared P/D execution and a KV topology"
                            );
                        }
                        if let Some(topology) = &topology {
                            anyhow::ensure!(
                                tp == 1 && topology.shards == if qwen { 4 } else { 8 },
                                "KV topology requires parent TP=1 and the model's complete KV heads"
                            );
                            let hardware = runner.get("hardware").context("missing hardware")?;
                            let spec = hardware
                                .get("topology")
                                .context("missing shared hardware topology")?;
                            if let Some(previous) = hardware_topology {
                                anyhow::ensure!(
                                    spec == previous,
                                    "P/D hardware topology must be identical"
                                );
                            }
                            hardware_topology = Some(spec);
                            let replicas = if pool == "prefill_instances" {
                                &topology.prefill_groups
                            } else {
                                &topology.decode_groups
                            };
                            anyhow::ensure!(
                                replicas.len() == children.len(),
                                "KV pool dimensions mismatch"
                            );
                            let groups = &replicas[index];
                            let group: Vec<_> = groups.iter().flatten().copied().collect();
                            let gpu_ids = hardware
                                .get("gpu_ids")
                                .and_then(toml::Value::as_array)
                                .context("missing worker GPU IDs")?;
                            if qwen {
                                let attention_tp = runner
                                    .get("tp")
                                    .and_then(toml::Value::as_integer)
                                    .context("missing Qwen Attention TP")?;
                                anyhow::ensure!(
                                    matches!(attention_tp, 1 | 2 | 4)
                                        && matches!(group.len(), 1 | 2 | 4 | 8)
                                        && group.len().is_multiple_of(attention_tp as usize)
                                        && groups.len() == group.len() / attention_tp as usize
                                        && groups.iter().all(|g| g.len() == attention_tp as usize)
                                        && runner
                                            .get("ep")
                                            .and_then(toml::Value::as_integer)
                                            .unwrap_or(1)
                                            == group.len() as i64
                                        && runner.get("expert_tp").and_then(toml::Value::as_integer).unwrap_or(1) == 1,
                                    "Qwen requires EP 1/2/4/8, Attention TP 1/2/4 dividing EP, and expert_tp=1"
                                );
                                let expected: Vec<Vec<usize>> = (0..group.len())
                                    .collect::<Vec<_>>()
                                    .chunks(attention_tp as usize)
                                    .map(<[usize]>::to_vec)
                                    .collect();
                                if let Some(layout) = runner.get("attention_groups") {
                                    let layout: Vec<Vec<usize>> = layout.clone().try_into()?;
                                    anyhow::ensure!(layout == expected,
                                        "worker attention_groups must match the consecutive KV Attention TP groups");
                                } else {
                                    anyhow::ensure!(
                                        attention_tp == 1,
                                        "Qwen TP>1 requires explicit worker attention_groups"
                                    );
                                }
                            } else {
                                anyhow::ensure!(
                                    groups.len() == 1
                                        && runner.get("tp").and_then(toml::Value::as_integer) == Some(group.len() as i64)
                                        && runner.get("ep").and_then(toml::Value::as_integer).unwrap_or(1) == 1
                                        && runner.get("expert_tp").and_then(toml::Value::as_integer).unwrap_or(1) == 1
                                        && runner.get("attention_groups").is_none(),
                                    "KV group and worker TP/GPU count mismatch; Llama requires EP=1"
                                );
                            }
                            anyhow::ensure!(
                                hardware.get("gpu_count").and_then(toml::Value::as_integer)
                                    == Some(group.len() as i64)
                                    && gpu_ids.len() == group.len(),
                                "KV group and worker GPU count mismatch"
                            );
                            for (gpu, id) in group.iter().zip(gpu_ids) {
                                anyhow::ensure!(
                                    topology.devices.get(*gpu).map(String::as_str)
                                        == Some(
                                            id.as_str()
                                                .context("worker GPU ID must be a string")?
                                        ),
                                    "KV device and worker allocation mismatch"
                                );
                            }
                        } else {
                            anyhow::ensure!(
                                runner.get("tp").and_then(toml::Value::as_integer) == Some(tp),
                                "Rooflang P/D parent and child TP must match without KV topology"
                            );
                        }
                        let id = runner
                            .get("instance_id")
                            .and_then(toml::Value::as_str)
                            .context("Rooflang P/D requires an instance_id for every runner")?;
                        anyhow::ensure!(
                            !id.is_empty() && identities.insert(id),
                            "Rooflang P/D instance_id must be nonempty and unique"
                        );
                    }
                }
            }
            _ => anyhow::bail!(
                "Rooflang requires a single_instance root or independent P/D single_instance pools"
            ),
        }
    }
    toml::from_str(&raw).with_context(|| format!("failed to parse {}", config_path.display()))
}

fn load_workload(
    system: &mut dyn System,
    graph: &mut Graph,
    workload_path: &Path,
) -> Result<usize> {
    let file = std::fs::File::open(workload_path)
        .with_context(|| format!("failed to open {}", workload_path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut count = 0;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }

        let request: Request = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse workload row {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        request.validate().with_context(|| {
            format!(
                "invalid workload row {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        system.add_request_arrival(request, graph, None, None)?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use serde_json::{json, Value};

    fn model(qwen: bool) -> Value {
        if qwen {
            json!({"kind": "qwen3_235b_a22b"})
        } else {
            json!({
                "kind": "llama3_1", "hidden_size": 4096, "intermediate_size": 14336,
                "max_position_embeddings": 8192, "num_attention_heads": 32,
                "num_hidden_layers": 1, "num_key_value_heads": 8, "vocab_size": 128256
            })
        }
    }

    fn config(qwen: bool, p: usize, d: usize) -> Value {
        let devices: Vec<_> = (0..p + d).map(|i| format!("gpu{i}")).collect();
        let child = |stage: &str, start: usize, width: usize| {
            let mut runner = json!({
                "kind": "rooflang", "tp": if qwen { 1 } else { width },
                "instance_id": format!("{stage}0"), "stage": stage, "shared_execution": true,
                // load_config must validate without attempting to start this executable.
                "command": ["/nonexistent/config-test-worker"],
                "hardware": {"gpu_count": width, "gpu_ids": &devices[start..start + width],
                    "topology": {"clusters": [], "links": []}},
            });
            if qwen {
                runner["graph_model"] = json!("qwen3_235b_a22b");
                runner["ep"] = json!(width);
            }
            json!({
                "kind": "single_instance", "model": model(qwen), "model_runner": runner,
                "scheduler": {"kind": "chunked_prefill", "block_size": 16, "max_batch": 4,
                    "max_num_batched_tokens": 3, "device_mem": 144_000_000_000_u64,
                    "device_mem_utilization": 0.9, "enable_prefix_caching": false}
            })
        };
        let mut topology = json!({
            "devices": devices, "shards": if qwen { 4 } else { 8 },
            "prefill_gpus": [(0..p).collect::<Vec<_>>()],
            "decode_gpus": [(p..p + d).collect::<Vec<_>>()],
            "links": [], "routes": [],
        });
        if qwen {
            topology["attention_dp"] = json!(true);
        }
        json!({"system": {
            "kind": "pd_disaggregation", "shared_execution": true, "tp": 1,
            "model": model(qwen), "network_sim": {"kind": "flow"},
            "prefill_instances": [child("p", 0, p)], "decode_instances": [child("d", p, d)],
            "kv_topology": topology,
        }})
    }

    fn load(value: &Value) -> Result<Config> {
        let file = tempfile::NamedTempFile::new()?;
        std::fs::write(file.path(), toml::to_string(value)?)?;
        load_config(file.path())
    }

    #[test]
    fn accepts_qwen_unequal_ep_and_legacy_llama_tp_without_starting_a_worker() {
        for (p, d) in [(1, 2), (2, 4), (4, 8), (8, 1)] {
            load(&config(true, p, d)).unwrap();
            load(&config(false, p, d)).unwrap();
        }
    }

    fn attention_config(p: usize, ptp: usize, d: usize, dtp: usize) -> Value {
        let mut value = config(true, p, d);
        let topology = &mut value["system"]["kv_topology"];
        for field in ["prefill_gpus", "decode_gpus", "attention_dp"] {
            topology.as_object_mut().unwrap().remove(field);
        }
        topology["prefill_groups"] = json!([(0..p)
            .collect::<Vec<_>>()
            .chunks(ptp)
            .map(<[usize]>::to_vec)
            .collect::<Vec<_>>()]);
        topology["decode_groups"] = json!([(p..p + d)
            .collect::<Vec<_>>()
            .chunks(dtp)
            .map(<[usize]>::to_vec)
            .collect::<Vec<_>>()]);
        for (pool, ep, tp) in [("prefill_instances", p, ptp), ("decode_instances", d, dtp)] {
            let runner = &mut value["system"][pool][0]["model_runner"];
            runner["tp"] = json!(tp);
            runner["expert_tp"] = json!(1);
            runner["attention_groups"] = json!((0..ep)
                .collect::<Vec<_>>()
                .chunks(tp)
                .map(<[usize]>::to_vec)
                .collect::<Vec<_>>());
        }
        value
    }

    #[test]
    fn accepts_explicit_unequal_attention_tp_and_ep_groups() {
        for (p, ptp, d, dtp) in [(8, 2, 8, 4), (4, 4, 8, 2), (2, 2, 1, 1), (8, 1, 2, 2)] {
            load(&attention_config(p, ptp, d, dtp)).unwrap();
        }
    }

    #[test]
    fn rejects_invalid_expert_tp_or_attention_layout_before_workers() {
        let cases: &[fn(&mut Value)] = &[
            |v| v["system"]["prefill_instances"][0]["model_runner"]["expert_tp"] = json!(2),
            |v| v["system"]["prefill_instances"][0]["model_runner"]["tp"] = json!(8),
            |v| v["system"]["prefill_instances"][0]["model_runner"]["tp"] = json!(0),
            |v| {
                v["system"]["prefill_instances"][0]["model_runner"]["attention_groups"][0] =
                    json!([1, 0])
            },
            |v| v["system"]["prefill_instances"][0]["model_runner"]["attention_groups"] = json!([]),
            |v| {
                v["system"]["prefill_instances"][0]["model_runner"]
                    .as_object_mut()
                    .unwrap()
                    .remove("attention_groups");
            },
            |v| v["system"]["kv_topology"]["prefill_groups"][0][0] = json!([0, 2]),
            |v| v["system"]["kv_topology"]["prefill_groups"][0][0] = json!([0]),
            |v| v["system"]["kv_topology"]["prefill_groups"][0] = json!([]),
            |v| v["system"]["kv_topology"]["prefill_gpus"] = json!([[0, 1, 2, 3, 4, 5, 6, 7]]),
            |v| v["system"]["kv_topology"]["attention_dp"] = json!(true),
        ];
        for (index, change) in cases.iter().enumerate() {
            let mut value = attention_config(8, 2, 8, 4);
            change(&mut value);
            assert!(
                load(&value).is_err(),
                "invalid Attention TP layout case {index} passed"
            );
        }
    }

    #[test]
    fn rejects_qwen_model_kv_mode_and_parallelism_mismatches_at_load() {
        let cases: &[fn(&mut Value)] = &[
            |v| v["system"]["kv_topology"]["shards"] = json!(8),
            |v| v["system"]["kv_topology"]["attention_dp"] = json!(false),
            |v| {
                v["system"]["kv_topology"]
                    .as_object_mut()
                    .unwrap()
                    .remove("attention_dp");
            },
            |v| {
                v["system"].as_object_mut().unwrap().remove("kv_topology");
            },
            |v| v["system"]["tp"] = json!(2),
            |v| v["system"]["shared_execution"] = json!(false),
            |v| {
                v["system"]["prefill_instances"][0]["model_runner"]["shared_execution"] =
                    json!(false)
            },
            |v| v["system"]["prefill_instances"][0]["model_runner"]["tp"] = json!(4),
            |v| v["system"]["prefill_instances"][0]["model_runner"]["ep"] = json!(8),
            |v| {
                v["system"]["prefill_instances"][0]["model_runner"]["hardware"]["gpu_count"] =
                    json!(8)
            },
            |v| {
                v["system"]["prefill_instances"][0]["model_runner"]["hardware"]["gpu_ids"][0] =
                    json!("wrong-device")
            },
            |v| v["system"]["prefill_instances"][0]["model_runner"]["graph_model"] = json!("llama"),
            |v| {
                v["system"]["model"] = model(false);
                v["system"]["prefill_instances"][0]["model"] = model(false);
                v["system"]["decode_instances"][0]["model"] = model(false);
            },
        ];
        for (index, change) in cases.iter().enumerate() {
            let mut value = config(true, 4, 8);
            change(&mut value);
            assert!(load(&value).is_err(), "mismatch case {index} passed");
        }
        assert!(load(&config(true, 3, 8)).is_err());
    }

    #[test]
    fn llama_cannot_silently_use_attention_dp_ep_or_the_qwen_graph() {
        let cases: &[fn(&mut Value)] = &[
            |v| v["system"]["kv_topology"]["attention_dp"] = json!(true),
            |v| v["system"]["kv_topology"]["shards"] = json!(4),
            |v| v["system"]["prefill_instances"][0]["model_runner"]["tp"] = json!(1),
            |v| v["system"]["prefill_instances"][0]["model_runner"]["ep"] = json!(2),
            |v| {
                v["system"]["prefill_instances"][0]["model_runner"]["graph_model"] =
                    json!("qwen3_235b_a22b")
            },
        ];
        for (index, change) in cases.iter().enumerate() {
            let mut value = config(false, 2, 4);
            change(&mut value);
            assert!(load(&value).is_err(), "mismatch case {index} passed");
        }
    }
}
