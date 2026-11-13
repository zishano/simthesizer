use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use simthesizer_core::types::Request;
use simthesizer_core::{Engine, Graph, System};

mod write_jsonl;
mod write_trace;

use crate::write_jsonl::write_results_jsonl;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    workload: PathBuf,
    #[arg(long)]
    output_jsonl: PathBuf,
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
    args.output_jsonl = output_path(&args.output_jsonl)?;
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
        anyhow::ensure!(
            path != &args.output_jsonl,
            "trace and result paths must differ"
        );
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
            path != &args.output_jsonl && Some(path) != args.output_engine_trace.as_ref(),
            "graph trace, engine trace and result paths must differ"
        );
    }
    let config_path = args.config;
    let workload_path = args.workload;
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
    load_workload(config.system.as_mut(), &mut graph, &workload_path)?;

    let engine = Engine::new(graph, config.system);
    let results = if let Some(path) = &args.output_engine_trace {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        let mut trace = write_trace::TraceWriter::new(std::io::BufWriter::new(file.as_file_mut()));
        let results =
            engine.run_with_observer(|time, system| trace.observe(time, system.observe()))?;
        trace.finish()?;
        file.persist_noclobber(path)?;
        results
    } else {
        engine.run()?
    };

    write_results_jsonl(&args.output_jsonl, &results)?;
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
        anyhow::ensure!(document.get("system").and_then(|s|s.get("kind")).and_then(toml::Value::as_str) == Some("single_instance"),
            "Rooflang whole-iteration backend requires a single_instance root with exclusive resources");
    }
    toml::from_str(&raw).with_context(|| format!("failed to parse {}", config_path.display()))
}

fn load_workload(system: &mut dyn System, graph: &mut Graph, workload_path: &Path) -> Result<()> {
    let file = std::fs::File::open(workload_path)
        .with_context(|| format!("failed to open {}", workload_path.display()))?;
    let reader = std::io::BufReader::new(file);

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
    }
    Ok(())
}
