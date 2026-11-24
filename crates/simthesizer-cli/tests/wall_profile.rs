use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Command;

fn config() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/scenarios/1xrtx-pro-6000--llama3_1_8b--realistic/simthesizer.toml")
}

fn assert_partition(profile: &Value) {
    assert_eq!(profile["schema_version"], 1);
    let wall = profile["wall_seconds"].as_f64().unwrap();
    assert!(wall > 0.0);
    let total: f64 = profile["sections"]
        .as_object()
        .unwrap()
        .values()
        .map(|section| {
            let exclusive = section["exclusive_seconds"].as_f64().unwrap();
            assert!(exclusive >= 0.0);
            assert!(exclusive <= section["inclusive_seconds"].as_f64().unwrap() + 1e-12);
            assert!(section["calls"].as_u64().unwrap() > 0);
            exclusive
        })
        .sum();
    assert!(
        total <= wall + 1e-12,
        "exclusive total {total} exceeds wall {wall}"
    );
}

#[test]
fn profiling_preserves_requests_events_and_progress_and_exports_live_partitions() {
    let root = tempfile::tempdir().unwrap();
    let workload = root.path().join("workload.jsonl");
    std::fs::write(
        &workload,
        (0..3)
            .map(|id| {
                json!({
                    "request_id":id, "arrival_time":id, "initial":[0],
                    "sub_requests":[{"kind":"tool_call","input_tokens":0,"output_tokens":0,
                                     "duration":0.25,"next":[]}]
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let mut progress = Vec::new();
    for enabled in [false, true] {
        let label = if enabled { "on" } else { "off" };
        let mut command = Command::new(env!("CARGO_BIN_EXE_simthesizer"));
        command
            .arg("--config")
            .arg(config())
            .arg("--workload")
            .arg(&workload)
            .arg("--output-jsonl")
            .arg(root.path().join(format!("{label}.jsonl")))
            .arg("--output-graph-trace")
            .arg(root.path().join(format!("{label}-graph.jsonl")))
            .arg("--progress");
        if enabled {
            command
                .arg("--output-wall-profile")
                .arg(root.path().join("wall-profile.json"));
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let events: Vec<Value> = String::from_utf8(output.stderr)
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("SIMTHESIZER_PROGRESS "))
            .map(|line| {
                let mut event: Value = serde_json::from_str(line).unwrap();
                let profile = event.as_object_mut().unwrap().remove("wall_profile");
                assert_eq!(profile.is_some(), enabled);
                if let Some(profile) = profile {
                    assert_partition(&profile);
                }
                event
            })
            .collect();
        progress.push(events);
    }
    assert_eq!(progress[0], progress[1]);
    for suffix in [".jsonl", "-graph.jsonl"] {
        assert_eq!(
            std::fs::read(root.path().join(format!("on{suffix}"))).unwrap(),
            std::fs::read(root.path().join(format!("off{suffix}"))).unwrap()
        );
    }
    let profile: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("wall-profile.json")).unwrap())
            .unwrap();
    assert_partition(&profile);
    for section in [
        "config_load",
        "system_init",
        "workload_load",
        "engine_other",
        "resource_admit",
        "resource_rates",
        "resource_advance",
        "event_dispatch",
        "result_write",
    ] {
        assert!(
            profile["sections"].get(section).is_some(),
            "missing {section}"
        );
    }
}

#[test]
fn profiler_refuses_existing_or_aliased_outputs() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("profile.json");
    std::fs::write(&path, b"preserve").unwrap();
    let result = root.path().join("result.jsonl");
    for profile in [&path, &result] {
        let output = Command::new(env!("CARGO_BIN_EXE_simthesizer"))
            .arg("--config")
            .arg(config())
            .arg("--workload")
            .arg(root.path().join("missing.jsonl"))
            .arg("--output-jsonl")
            .arg(&result)
            .arg("--output-wall-profile")
            .arg(profile)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!result.exists());
        assert_eq!(std::fs::read(&path).unwrap(), b"preserve");
    }
}

#[test]
fn failed_simulation_keeps_partial_profile_without_request_results() {
    let root = tempfile::tempdir().unwrap();
    let profile = root.path().join("profile.json");
    let result = root.path().join("result.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_simthesizer"))
        .arg("--config")
        .arg(config())
        .arg("--workload")
        .arg(root.path().join("missing.jsonl"))
        .arg("--output-jsonl")
        .arg(&result)
        .arg("--output-wall-profile")
        .arg(&profile)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!result.exists());
    let profile: Value = serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
    assert_partition(&profile);
    assert!(profile["sections"].get("workload_load").is_some());
}
