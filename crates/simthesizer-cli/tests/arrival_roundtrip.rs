use std::path::Path;
use std::process::Command;

#[test]
fn workload_arrivals_round_trip_without_changing_binary_values() {
    // Decimal arrivals rejected by the Python evaluator in the m0.5 baseline.
    // std's parser is independent of serde_json's decimal conversion.
    let arrivals = [
        "3.9805331999999995",
        "9.970553200000001",
        "11.175233200000001",
        "11.777853200000001",
        "12.443013200000001",
        "13.325333200000001",
        "15.695973200000001",
        "18.220053200000002",
        "18.248033200000002",
        "0.0",
        "1e-20",
        "3.9805331999999995e1",
    ];
    let root = tempfile::tempdir().unwrap();
    let workload = root.path().join("workload.jsonl");
    let result = root.path().join("result.jsonl");
    let rows = arrivals.iter().enumerate().map(|(id, arrival)| {
        format!(r#"{{"request_id":{id},"arrival_time":{arrival},"initial":[0],"sub_requests":[{{"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":0.0,"next":[]}}]}}"#)
    }).collect::<Vec<_>>().join("\n");
    std::fs::write(&workload, rows).unwrap();
    let config = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/scenarios/1xrtx-pro-6000--llama3_1_8b--realistic/simthesizer.toml");
    let output = Command::new(env!("CARGO_BIN_EXE_simthesizer"))
        .arg("--config")
        .arg(config)
        .arg("--workload")
        .arg(&workload)
        .arg("--output-jsonl")
        .arg(&result)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let results = std::fs::read_to_string(result).unwrap();
    assert_eq!(results.lines().count(), arrivals.len());
    for (line, arrival) in results.lines().zip(arrivals) {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(
            row["arrival_time"].as_f64().unwrap().to_bits(),
            arrival.parse::<f64>().unwrap().to_bits(),
            "arrival changed: {arrival}; output: {line}"
        );
    }
}
