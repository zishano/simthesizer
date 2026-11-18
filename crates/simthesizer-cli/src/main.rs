use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use simthesizer_core::types::Request;
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
    let mut args = Args::parse();
    if args.policy_capabilities {
        println!(
            "{}",
            serde_json::json!({
                "policy_api_version": simthesizer_core::policies::API_VERSION,
                "actions": ["route", "batch", "submit_kv", "wait"],
                "length_model": "known_trace", "scheduler_cost": "zero_simulated_time",
                "validates_actions": true, "checks_full_completion": true
            })
        );
        return Ok(());
    }
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
    let mut config = load_config(&config_path)?;
    let system_base_path = config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    config.system.resolve_path(&system_base_path)?;
    config.system.init()?;
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
    let total = load_workload(config.system.as_mut(), &mut graph, &workload_path)?;
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
                let topology = system.get("kv_topology");
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
                        if let Some(topology) = topology {
                            anyhow::ensure!(
                                tp == 1
                                    && topology.get("shards").and_then(toml::Value::as_integer)
                                        == Some(8),
                                "Rooflang KV topology requires parent TP=1 and eight KV heads"
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
                            let key = if pool == "prefill_instances" {
                                "prefill_gpus"
                            } else {
                                "decode_gpus"
                            };
                            let groups = topology
                                .get(key)
                                .and_then(toml::Value::as_array)
                                .context("missing KV GPU groups")?;
                            anyhow::ensure!(
                                groups.len() == children.len(),
                                "KV pool dimensions mismatch"
                            );
                            let group = groups[index].as_array().context("invalid KV GPU group")?;
                            let devices = topology
                                .get("devices")
                                .and_then(toml::Value::as_array)
                                .context("missing KV devices")?;
                            let gpu_ids = hardware
                                .get("gpu_ids")
                                .and_then(toml::Value::as_array)
                                .context("missing worker GPU IDs")?;
                            anyhow::ensure!(
                                runner.get("tp").and_then(toml::Value::as_integer)
                                    == Some(group.len() as i64)
                                    && hardware.get("gpu_count").and_then(toml::Value::as_integer)
                                        == Some(group.len() as i64)
                                    && gpu_ids.len() == group.len(),
                                "KV group and worker TP/GPU count mismatch"
                            );
                            for (gpu, id) in group.iter().zip(gpu_ids) {
                                let gpu = gpu.as_integer().context("invalid GPU index")?;
                                anyhow::ensure!(
                                    gpu >= 0 && devices.get(gpu as usize) == Some(id),
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
