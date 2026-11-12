use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

#[test]
fn graph_history_captures_dynamic_dependencies_without_changing_results() {
    let dir = tempfile::tempdir().unwrap();
    let workload = dir.path().join("workload.jsonl");
    let request = json!({"request_id":42,"arrival_time":1.0,"initial":[0],"sub_requests":[
        {"kind":"llm","input_tokens":8,"output_tokens":2,"next":[1]},
        {"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":0.25,"next":[2]},
        {"kind":"llm","input_tokens":16,"output_tokens":1,"next":[]}]});
    std::fs::write(&workload, format!("{request}\n")).unwrap();
    let config = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/scenarios/1xrtx-pro-6000--llama3_1_8b--realistic/simthesizer.toml");
    let trace = dir.path().join("graph.jsonl");
    let state = dir.path().join("state.jsonl");
    let invoke = |output: &Path, graph: Option<&Path>| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_simthesizer"));
        cmd.arg("--config")
            .arg(&config)
            .arg("--workload")
            .arg(&workload)
            .arg("--output-jsonl")
            .arg(output);
        if let Some(path) = graph {
            cmd.arg("--output-graph-trace")
                .arg(path)
                .arg("--output-engine-trace")
                .arg(&state);
        }
        cmd.output().unwrap()
    };
    let off = dir.path().join("off.jsonl");
    let on = dir.path().join("on.jsonl");
    assert!(invoke(&off, None).status.success());
    let output = invoke(&on, Some(&trace));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read(&on).unwrap(), std::fs::read(&off).unwrap());
    let bytes = std::fs::read(&trace).unwrap();
    let rows: Vec<Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let mut created = HashSet::new();
    let mut finished = HashMap::new();
    let mut edges = Vec::new();
    let mut removed = 0;
    for (seq, row) in rows.iter().enumerate() {
        assert_eq!(row["seq"], seq);
        match row["event"].as_str().unwrap() {
            "node_create" => {
                assert!(created.insert(row["node"].as_u64().unwrap()));
                if !row["cause"].is_null() {
                    assert!(created.contains(&row["cause"].as_u64().unwrap()));
                }
            }
            "edge_add" => {
                edges.push((
                    row["parent"].as_u64().unwrap(),
                    row["child"].as_u64().unwrap(),
                ));
            }
            "edge_remove" => {
                removed += 1;
                let pair = (
                    row["parent"].as_u64().unwrap(),
                    row["child"].as_u64().unwrap(),
                );
                let i = edges.iter().position(|p| *p == pair).unwrap();
                edges.remove(i);
            }
            "node_complete" => {
                assert!(finished
                    .insert(
                        row["node"].as_u64().unwrap(),
                        (
                            row["start_s"].as_f64().unwrap(),
                            row["finish_s"].as_f64().unwrap()
                        )
                    )
                    .is_none());
            }
            _ => {}
        }
    }
    assert_eq!(created.len(), finished.len());
    assert!(removed >= 3);
    for (a, b) in edges {
        assert!(finished[&b].0 >= finished[&a].1);
    }
    assert_eq!(rows.last().unwrap()["event"], "run_complete");
    assert!(rows.iter().any(|r| r["details"]["label"]
        .as_str()
        .is_some_and(|s| s.contains("LlmSchedule"))));
    assert!(rows.iter().any(|r| r["details"]["label"]
        .as_str()
        .is_some_and(|s| s.contains("FastForward"))));
    assert!(!invoke(&on, Some(&trace)).status.success());
    assert_eq!(std::fs::read(&trace).unwrap(), bytes);
    // All export paths must be distinct, including normalized aliases.
    assert!(
        !invoke(&trace, Some(&dir.path().join("sub/../graph.jsonl")))
            .status
            .success()
    );
}
