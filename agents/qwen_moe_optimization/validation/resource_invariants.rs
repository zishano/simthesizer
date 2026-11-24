//! Public resource contracts injected by the host, outside candidate ownership.
use std::collections::BTreeMap;

use simthesizer_core::resources::{Allocation, Demand, Resources, Work};

fn work(resources: &[(&str, f64)]) -> Work {
    Work {
        demands: vec![Demand {
            resources: resources
                .iter()
                .map(|(name, seconds)| (name.to_string(), *seconds))
                .collect(),
            latency: 0.0,
        }],
        ..Default::default()
    }
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

#[test]
fn shared_bandwidth_conserves_work_across_staggered_arrivals() {
    let mut resource = Resources::default();
    resource.submit(0, work(&[("link", 2.0)])).unwrap();
    resource.admit(0.0).unwrap();
    assert!(resource.advance(1.0).unwrap().is_empty());
    resource.submit(1, work(&[("link", 2.0)])).unwrap();
    resource.admit(1.0).unwrap();
    close(resource.next_time().unwrap(), 3.0);
    assert_eq!(resource.advance(3.0).unwrap(), vec![(0, 0.0)]);
    close(resource.next_time().unwrap(), 4.0);
    assert_eq!(resource.advance(4.0).unwrap(), vec![(1, 1.0)]);
    resource.check_drained().unwrap();
}

#[test]
fn independent_devices_overlap_but_one_stream_serializes() {
    let mut resource = Resources::default();
    resource.submit(0, work(&[("gpu0", 2.0)])).unwrap();
    resource.submit(1, work(&[("gpu1", 2.0)])).unwrap();
    assert_eq!(resource.admit(0.0).unwrap(), vec![0, 1]);
    assert_eq!(resource.advance(2.0).unwrap().len(), 2);
    for id in 2..4 {
        let mut operator = work(&[("gpu0", 2.0)]);
        operator.slots = vec!["stream".into()];
        resource.submit(id, operator).unwrap();
    }
    assert_eq!(resource.admit(2.0).unwrap(), vec![2]);
    resource.advance(4.0).unwrap();
    assert_eq!(resource.admit(4.0).unwrap(), vec![3]);
    resource.advance(6.0).unwrap();
    resource.check_drained().unwrap();
}

#[test]
fn copy_bottleneck_does_not_fabricate_hbm_contention() {
    let mut resource = Resources::default();
    resource
        .submit(0, work(&[("link", 10.0), ("hbm", 1.0)]))
        .unwrap();
    resource.submit(1, work(&[("hbm", 1.0)])).unwrap();
    resource.admit(0.0).unwrap();
    close(resource.next_time().unwrap(), 1.1);
    assert_eq!(resource.advance(1.1).unwrap(), vec![(1, 0.0)]);
    close(resource.next_time().unwrap(), 10.1);
    resource.advance(10.1).unwrap();
    resource.check_drained().unwrap();
}

#[test]
fn setup_latency_and_service_cost_are_both_charged() {
    let mut resource = Resources::default();
    let mut operator = work(&[("link", 2.0)]);
    operator.demands[0].latency = 1.0;
    resource.submit(0, operator).unwrap();
    resource.admit(0.0).unwrap();
    close(resource.next_time().unwrap(), 1.0);
    assert!(resource.advance(1.0).unwrap().is_empty());
    close(resource.next_time().unwrap(), 3.0);
    resource.advance(3.0).unwrap();
    resource.check_drained().unwrap();
}

#[test]
fn duplicate_reservations_count_once_and_capacity_waits_for_release() {
    let mut resource = Resources::default();
    resource.capacity("gpu".into(), 10).unwrap();
    let allocation = Allocation {
        id: "kv:first".into(),
        bytes: BTreeMap::from([("gpu".into(), 7)]),
    };
    let mut first = work(&[("copy", 2.0)]);
    first.reserve = vec![allocation.clone(), allocation];
    first.release = vec!["kv:first".into()];
    let second = Work {
        reserve: vec![Allocation {
            id: "kv:second".into(),
            bytes: BTreeMap::from([("gpu".into(), 7)]),
        }],
        release: vec!["kv:second".into()],
        ..Default::default()
    };
    resource.submit(0, first).unwrap();
    resource.submit(1, second).unwrap();
    assert_eq!(resource.admit(0.0).unwrap(), vec![0]);
    assert_eq!(resource.policy_memory()[0].allocated_bytes, 7);
    resource.advance(2.0).unwrap();
    assert_eq!(resource.policy_memory()[0].allocated_bytes, 0);
    assert_eq!(resource.admit(2.0).unwrap(), vec![1]);
    resource.advance(2.0).unwrap();
    resource.check_drained().unwrap();
}

#[test]
fn impossible_residency_and_negative_service_are_rejected() {
    let mut resource = Resources::default();
    resource.capacity("gpu".into(), 10).unwrap();
    let impossible = Work {
        reserve: vec![Allocation {
            id: "large".into(),
            bytes: BTreeMap::from([("gpu".into(), 11)]),
        }],
        ..Default::default()
    };
    assert!(resource.submit(0, impossible).is_err());
    assert!(resource.submit(1, work(&[("gpu", -1.0)])).is_err());
    assert!(resource.submit(2, work(&[("gpu", f64::NAN)])).is_err());
}
