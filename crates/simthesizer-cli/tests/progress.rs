use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

#[test]
fn progress_counts_whole_requests_and_preserves_results_and_event_order() {
    let root = tempfile::tempdir().unwrap();
    let workload = root.path().join("workload.jsonl");
    let rows = (0..3).map(|i| json!({"request_id": i, "arrival_time": i as f64, "initial": [0],
        "sub_requests": [{"kind":"tool_call", "input_tokens":0,"output_tokens":0,"duration":0.25,"next":[1]},
        {"kind":"tool_call", "input_tokens":0,"output_tokens":0,"duration":0.5,"next":[]}]}).to_string()).collect::<Vec<_>>().join("\n");
    std::fs::write(&workload, rows).unwrap();
    let config = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/scenarios/1xrtx-pro-6000--llama3_1_8b--realistic/simthesizer.toml");
    for progress in [false, true] {
        let name = if progress { "on" } else { "off" };
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_simthesizer"));
        cmd.arg("--config")
            .arg(&config)
            .arg("--workload")
            .arg(&workload)
            .arg("--output-jsonl")
            .arg(root.path().join(format!("{name}.jsonl")))
            .arg("--output-graph-trace")
            .arg(root.path().join(format!("{name}-graph.jsonl")));
        if progress {
            cmd.arg("--progress");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let events = String::from_utf8(output.stderr)
            .unwrap()
            .lines()
            .filter_map(|s| s.strip_prefix("SIMTHESIZER_PROGRESS "))
            .map(|s| serde_json::from_str::<Value>(s).unwrap())
            .collect::<Vec<_>>();
        if progress {
            assert_eq!(
                events
                    .iter()
                    .map(|e| e["completed"].as_u64().unwrap())
                    .collect::<Vec<_>>(),
                vec![0, 1, 2, 3]
            );
            assert!(events.iter().all(|e| e["total"] == 3));
            for (index, event) in events.iter().skip(1).enumerate() {
                assert_eq!(event["time_s"].as_f64().unwrap(), index as f64 + 0.75);
            }
        } else {
            assert!(events.is_empty());
        }
    }
    for suffix in [".jsonl", "-graph.jsonl"] {
        assert_eq!(
            std::fs::read(root.path().join(format!("on{suffix}"))).unwrap(),
            std::fs::read(root.path().join(format!("off{suffix}"))).unwrap()
        );
    }
}
