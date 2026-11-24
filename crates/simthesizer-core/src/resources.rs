//! Fluid sharing and capacity reservations, driven only by the engine clock.
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;
use std::rc::Rc;

use crate::types::NodeID;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Demand {
    /// Seconds at full resource capacity for one unit of progress.
    pub resources: BTreeMap<String, f64>,
    #[serde(default)]
    pub latency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn work(resources: &[(&str, f64)]) -> Work {
        Work {
            demands: vec![Demand {
                resources: resources.iter().map(|(s, v)| (s.to_string(), *v)).collect(),
                latency: 0.0,
            }],
            ..Default::default()
        }
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }

    #[test]
    fn shared_work_keeps_progress_setup_and_rates_per_node() {
        let mut work = work(&[("hbm", 8.0)]);
        work.demands[0].latency = 1.0;
        let work = SharedWork::new(work);
        let mut resources = Resources::default();
        resources.submit_shared(0, work.clone()).unwrap();
        assert_eq!(resources.admit(0.0).unwrap(), vec![0]);
        resources.advance(1.0).unwrap();
        resources.advance(2.0).unwrap();
        resources.submit_shared(1, work.clone()).unwrap();
        let pressure = resources.policy_pressure();
        assert_eq!(pressure[0].active_remaining_service_s, 7.0);
        assert_eq!(pressure[0].waiting_service_s, 8.0);
        assert_eq!(resources.admit(2.0).unwrap(), vec![1]);
        assert!(std::ptr::eq(
            resources.active[&0].work.normalized(),
            resources.active[&1].work.normalized()
        ));
        assert_eq!(resources.active[&0].demands[0].progress, 0.125);
        assert_eq!(resources.active[&1].demands[0].progress, 0.0);
        assert_eq!(resources.active[&0].demands[0].latency, 0.0);
        assert_eq!(resources.active[&1].demands[0].latency, 1.0);
        let pressure = resources.policy_pressure();
        assert_eq!(pressure[0].active_remaining_service_s, 7.0);
        assert_eq!(pressure[0].setup_remaining_service_s, 8.0);
        resources.advance(3.0).unwrap();
        assert_eq!(resources.next_time(), Some(15.0));
        assert_eq!(resources.advance(15.0).unwrap(), vec![(0, 0.0)]);
        assert_eq!(resources.active[&1].demands[0].progress, 0.75);
        assert_eq!(resources.next_time(), Some(17.0));
        assert_eq!(resources.advance(17.0).unwrap(), vec![(1, 2.0)]);
        resources.check_drained().unwrap();
    }

    #[test]
    fn shared_work_preserves_slot_admission_allocation_release_and_domain_identity() {
        let mut work = work(&[("hbm", 2.0)]);
        work.slots = vec!["p0:compute:0".into()];
        work.reserve = vec![Allocation {
            id: "buffer".into(),
            bytes: BTreeMap::from([("gpu".into(), 9)]),
        }];
        work.release = vec!["buffer".into()];
        let work = SharedWork::new(work);
        let mut resources = Resources::default();
        resources.capacity("gpu".into(), 9).unwrap();
        resources.submit_shared(0, work.clone()).unwrap();
        resources.submit_shared(1, work.clone()).unwrap();
        assert_eq!(resources.admit(0.0).unwrap(), vec![0]);
        assert_eq!(resources.policy_memory()[0].allocated_bytes, 9);
        assert_eq!(resources.advance(2.0).unwrap(), vec![(0, 0.0)]);
        assert_eq!(resources.policy_memory()[0].allocated_bytes, 0);
        assert_eq!(resources.admit(2.0).unwrap(), vec![1]);
        assert_eq!(resources.policy_memory()[0].allocated_bytes, 9);
        let mut other = Resources::default();
        other.capacity("gpu".into(), 1).unwrap();
        assert!(other.submit_shared(2, work).is_err());
        assert!(other.waiting.is_empty() && other.active.is_empty());
        assert_eq!(resources.advance(4.0).unwrap(), vec![(1, 2.0)]);
        resources.check_drained().unwrap();
        other.check_drained().unwrap();
    }

    #[test]
    fn failed_shared_work_validation_does_not_prepare_or_queue_it() {
        let mut work = work(&[("hbm", 1.0)]);
        work.reserve = vec![Allocation {
            id: "buffer".into(),
            bytes: BTreeMap::from([("gpu".into(), 9)]),
        }];
        let work = SharedWork::new(work);
        let mut resources = Resources::default();
        assert!(resources.submit_shared(0, work.clone()).is_err());
        assert!(work.0.prepared.get().is_none());
        assert!(resources.waiting.is_empty());
        assert!(resources.resource_ids.is_empty());
        resources.capacity("gpu".into(), 9).unwrap();
        resources.submit_shared(0, work).unwrap();
        assert_eq!(resources.admit(0.0).unwrap(), vec![0]);
        resources.advance(1.0).unwrap();
        resources.release("buffer").unwrap();
        resources.check_drained().unwrap();
    }

    #[test]
    fn operator_details_remain_independent_when_work_is_shared() {
        let mut input = work(&[("hbm", 1.0)]);
        input.details = serde_json::json!({"name": "compute", "operator_id": 0});
        let shared = SharedWork::new(input);
        let other = shared.clone();
        let mut first = shared.into_work();
        first.details["operator_id"] = 1.into();
        assert_eq!(other.details["operator_id"], 0);
        assert_eq!(first.details["operator_id"], 1);
        // Once only one node owns the payload, reports can take it without copying.
        let demands = other.demands.as_ptr();
        let second = other.into_work();
        assert_eq!(second.demands.as_ptr(), demands);
        assert_eq!(second.details["operator_id"], 0);
    }

    // Historical progressive-filling implementation is an independent arithmetic
    // oracle for the compact workspace. Require bit equality, not a tolerance.
    fn reference_rates(resources: &Resources) -> BTreeMap<(NodeID, usize), f64> {
        // Progressive filling of normalized full-speed fractions. A slow network
        // copy consumes only its actual HBM rate; unused HBM remains available.
        let mut coefficients = BTreeMap::new();
        let mut bases = BTreeMap::new();
        let mut fractions = BTreeMap::new();
        let mut remaining = BTreeMap::new();
        for (&id, active) in &resources.active {
            for (i, d) in active.work.demands.iter().enumerate() {
                if active.demands[i].progress >= 1.0 || active.demands[i].latency > 0.0 {
                    continue;
                }
                let base = d.resources.values().copied().fold(0.0, f64::max);
                if base == 0.0 {
                    continue;
                }
                let c: BTreeMap<_, _> = d
                    .resources
                    .iter()
                    .map(|(r, v)| (r.clone(), v / base))
                    .collect();
                for r in c.keys() {
                    remaining.insert(r.clone(), 1.0);
                }
                coefficients.insert((id, i), c);
                bases.insert((id, i), base);
                fractions.insert((id, i), 0.0);
            }
        }
        let mut growing: BTreeSet<_> = coefficients.keys().copied().collect();
        while !growing.is_empty() {
            let mut usage = BTreeMap::<String, f64>::new();
            let mut delta = 1.0_f64;
            for key in &growing {
                delta = delta.min(1.0 - fractions[key]);
                for (r, c) in &coefficients[key] {
                    *usage.entry(r.clone()).or_default() += c;
                }
            }
            for (r, c) in &usage {
                delta = delta.min((remaining[r] / c).max(0.0));
            }
            for key in &growing {
                *fractions.get_mut(key).unwrap() += delta;
            }
            for (r, c) in usage {
                *remaining.get_mut(&r).unwrap() -= delta * c;
            }
            let old = growing.len();
            growing.retain(|key| {
                fractions[key] < 1.0 - 1e-12
                    && coefficients[key].keys().all(|r| remaining[r] > 1e-12)
            });
            assert!(growing.len() < old, "resource allocation made no progress");
        }
        fractions
            .into_iter()
            .map(|(key, fraction)| (key, fraction / bases[&key]))
            .collect()
    }

    fn reference_next_time(resources: &Resources) -> Option<f64> {
        let rates = reference_rates(resources);
        resources
            .active
            .iter()
            .map(|(&id, a)| {
                // Component completion changes contention even if the whole kernel remains active.
                a.work
                    .demands
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| a.demands[*i].progress < 1.0)
                    .map(|(i, d)| {
                        if a.demands[i].latency > 0.0 {
                            a.demands[i].latency
                        } else if d.resources.is_empty() {
                            0.0
                        } else {
                            (1.0 - a.demands[i].progress) / rates[&(id, i)]
                        }
                    })
                    .min_by(f64::total_cmp)
                    .unwrap_or(0.0)
            })
            .min_by(f64::total_cmp)
            .map(|dt| {
                let time = resources.clock + dt;
                if dt > 0.0 && time == resources.clock {
                    resources.clock.next_up()
                } else {
                    time
                }
            })
    }

    #[test]
    fn compact_filling_preserves_exact_rates() {
        fn assert_rates(resources: &mut Resources) {
            let expected = reference_rates(resources);
            let actual = resources.rates();
            assert_eq!(actual.len(), expected.len());
            for (key, value) in actual {
                assert_eq!(value.to_bits(), expected[&key].to_bits(), "{key:?}");
            }
            let expected_time = reference_next_time(resources).map(f64::to_bits);
            assert_eq!(resources.next_time().map(f64::to_bits), expected_time);
            assert_eq!(resources.next_time().map(f64::to_bits), expected_time);
            for ((id, index), expected) in expected {
                assert_eq!(
                    resources.active[&id].demands[index].rate.to_bits(),
                    expected.to_bits()
                );
            }
        }
        let mut seed = 42_u64;
        for _case in 0..128 {
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                seed >> 32
            };
            let mut resources = Resources::default();
            for id in 0..1 + next() % 12 {
                let mut demands = Vec::new();
                for _ in 0..1 + next() % 4 {
                    let mut coefficients = BTreeMap::new();
                    for resource in 0..6 {
                        if next() % 3 != 0 {
                            coefficients.insert(
                                format!("resource:{resource}"),
                                (1 + next() % 1000) as f64 / (1 + next() % 1024) as f64,
                            );
                        }
                    }
                    demands.push(Demand {
                        resources: coefficients,
                        latency: if next() % 4 == 0 { 0.25 } else { 0.0 },
                    });
                }
                resources
                    .submit(
                        id,
                        Work {
                            demands,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                resources.admit(0.0).unwrap();
                assert_rates(&mut resources);
            }
            while !resources.active.is_empty() {
                assert_rates(&mut resources);
                let now = resources.next_time().unwrap();
                resources.advance(now).unwrap();
            }
            resources.check_drained().unwrap();
        }
    }

    #[test]
    fn cached_boundaries_keep_zero_work_and_same_time_admissions() {
        let mut resources = Resources::default();
        assert_eq!(resources.next_time(), None);
        assert!(resources.advance(0.0).unwrap().is_empty());
        resources.capacity("gpu".into(), 10).unwrap();
        resources
            .resident(Allocation {
                id: "kv".into(),
                bytes: [("gpu".into(), 10)].into(),
            })
            .unwrap();
        let mut first = work(&[("gpu", 1.0)]);
        first.demands.push(Demand::default());
        first.slots.push("stream".into());
        resources.submit(0, first).unwrap();
        resources
            .submit(
                1,
                Work {
                    reserve: vec![Allocation {
                        id: "next".into(),
                        bytes: [("gpu".into(), 10)].into(),
                    }],
                    release: vec!["next".into()],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(resources.admit(0.0).unwrap(), vec![0]);
        assert_eq!(resources.next_time(), Some(0.0));
        assert_eq!(resources.next_time(), Some(0.0));
        assert!(resources.advance(0.0).unwrap().is_empty());
        assert_eq!(resources.next_time(), Some(1.0));
        assert!(resources.advance(0.0).unwrap().is_empty());
        resources.release("kv").unwrap();
        assert_eq!(resources.admit(0.0).unwrap(), vec![1]);
        assert_eq!(resources.next_time(), Some(0.0));
        assert_eq!(resources.advance(0.0).unwrap(), vec![(1, 0.0)]);
        assert_eq!(resources.next_time(), Some(1.0));
        resources.submit(2, work(&[("gpu", 1.0)])).unwrap();
        assert_eq!(resources.admit(0.0).unwrap(), vec![2]);
        assert_eq!(resources.next_time(), Some(2.0));
        assert!(resources.advance(0.0).unwrap().is_empty());
        assert!(resources.advance(0.5).unwrap().is_empty());
        assert_eq!(
            resources.next_time().map(f64::to_bits),
            reference_next_time(&resources).map(f64::to_bits)
        );
        assert!(resources.advance(0.5).unwrap().is_empty());
        assert_eq!(resources.advance(2.0).unwrap(), vec![(0, 0.0), (2, 0.0)]);
        assert_eq!(resources.next_time(), None);
        assert!(resources.advance(2.0).unwrap().is_empty());
        resources.check_drained().unwrap();
    }

    #[test]
    fn admission_indexes_preserve_shared_allocation_and_duplicate_slots() {
        let mut resources = Resources::default();
        resources.capacity("gpu".into(), 20).unwrap();
        let resident = Allocation {
            id: "weights:model".into(),
            bytes: [("gpu".into(), 3)].into(),
        };
        resources.resident(resident.clone()).unwrap();
        resources.resident(resident).unwrap();
        let allocation = Allocation {
            id: "kv".into(),
            bytes: [("gpu".into(), 7)].into(),
        };
        for id in 0..2 {
            let mut work = work(&[("gpu", 1.0)]);
            work.reserve = vec![allocation.clone(), allocation.clone()];
            work.slots = vec!["stream".into(), "stream".into()];
            if id == 1 {
                work.release.push("kv".into());
            }
            resources.submit(id, work).unwrap();
        }
        assert_eq!(resources.admit(0.0).unwrap(), vec![0]);
        assert_eq!(resources.policy_memory()[0].allocated_bytes, 10);
        assert_eq!(resources.occupied_slots.len(), 1);
        resources.advance(1.0).unwrap();
        assert!(resources.occupied_slots.is_empty());
        assert_eq!(resources.policy_memory()[0].allocated_bytes, 10);
        assert_eq!(resources.admit(1.0).unwrap(), vec![1]);
        assert_eq!(resources.policy_memory()[0].allocated_bytes, 10);
        resources.advance(2.0).unwrap();
        assert_eq!(resources.policy_memory()[0].allocated_bytes, 3);
        assert!(resources.occupied_slots.is_empty());
        resources.check_drained().unwrap();
    }

    #[test]
    fn staggered_sharing_updates_both_finish_times() {
        let mut r = Resources::default();
        r.submit(0, work(&[("link", 2.0)])).unwrap();
        r.admit(0.0).unwrap();
        assert!(r.advance(1.0).unwrap().is_empty());
        r.submit(1, work(&[("link", 2.0)])).unwrap();
        r.admit(1.0).unwrap();
        close(r.next_time().unwrap(), 3.0);
        assert_eq!(r.advance(3.0).unwrap(), vec![(0, 0.0)]);
        close(r.next_time().unwrap(), 4.0);
        assert_eq!(r.advance(4.0).unwrap(), vec![(1, 1.0)]);
        r.check_drained().unwrap();
    }

    #[test]
    fn disjoint_devices_overlap_and_streams_serialize() {
        let mut r = Resources::default();
        for id in 0..2 {
            r.submit(id, work(&[(if id == 0 { "gpu0" } else { "gpu1" }, 2.0)]))
                .unwrap();
        }
        assert_eq!(r.admit(0.0).unwrap(), vec![0, 1]);
        assert_eq!(r.advance(2.0).unwrap().len(), 2);
        for id in 2..4 {
            let mut w = work(&[("gpu0", 2.0)]);
            w.slots = vec!["stream".into()];
            r.submit(id, w).unwrap();
        }
        assert_eq!(r.admit(2.0).unwrap(), vec![2]);
        r.advance(4.0).unwrap();
        assert_eq!(r.admit(4.0).unwrap(), vec![3]);
        r.advance(6.0).unwrap();
        r.check_drained().unwrap();
    }

    #[test]
    fn slow_copy_does_not_reserve_half_of_hbm_bandwidth() {
        let mut r = Resources::default();
        r.submit(0, work(&[("link", 10.0), ("hbm", 1.0)])).unwrap();
        r.submit(1, work(&[("hbm", 1.0)])).unwrap();
        r.admit(0.0).unwrap();
        close(r.next_time().unwrap(), 1.1);
        assert_eq!(r.advance(1.1).unwrap(), vec![(1, 0.0)]);
        close(r.next_time().unwrap(), 10.1);
    }

    #[test]
    fn receive_reservation_waits_until_release_and_is_not_double_counted() {
        let mut r = Resources::default();
        r.capacity("gpu".into(), 10).unwrap();
        let a = Allocation {
            id: "kv:0:d".into(),
            bytes: BTreeMap::from([("gpu".into(), 7)]),
        };
        let mut first = work(&[("copy", 2.0)]);
        first.reserve = vec![a.clone(), a];
        first.release = vec!["kv:0:d".into()];
        let second = Work {
            reserve: vec![Allocation {
                id: "kv:1:d".into(),
                bytes: BTreeMap::from([("gpu".into(), 7)]),
            }],
            release: vec!["kv:1:d".into()],
            ..Default::default()
        };
        r.submit(0, first).unwrap();
        r.submit(1, second).unwrap();
        assert_eq!(r.admit(0.0).unwrap(), vec![0]);
        r.advance(2.0).unwrap();
        assert_eq!(r.admit(2.0).unwrap(), vec![1]);
        r.advance(2.0).unwrap();
        r.check_drained().unwrap();
        let impossible = Work {
            reserve: vec![Allocation {
                id: "large".into(),
                bytes: BTreeMap::from([("gpu".into(), 11)]),
            }],
            ..Default::default()
        };
        assert!(r
            .submit(2, impossible)
            .unwrap_err()
            .to_string()
            .contains("cannot fit"));
    }

    #[test]
    fn sub_ulp_latency_advances_representable_time() {
        let mut r = Resources::default();
        r.advance(1.0).unwrap();
        let mut w = work(&[("link", 1e-6)]);
        w.demands[0].latency = 1e-20;
        r.submit(0, w).unwrap();
        r.admit(1.0).unwrap();
        let next = r.next_time().unwrap();
        assert!(next > 1.0);
        r.advance(next).unwrap();
        let finish = r.next_time().unwrap();
        r.advance(finish).unwrap();
        // A rounded service boundary can need one final representable step.
        if let Some(last) = r.next_time() {
            r.advance(last).unwrap();
        }
        r.check_drained().unwrap();
        close(finish, 1.000001);
    }

    #[test]
    fn propagation_latency_does_not_consume_fabric_capacity() {
        let mut r = Resources::default();
        let mut w = work(&[("link", 2.0)]);
        w.demands[0].latency = 1.0;
        r.submit(0, w).unwrap();
        r.admit(0.0).unwrap();
        close(r.next_time().unwrap(), 1.0);
        r.advance(1.0).unwrap();
        close(r.next_time().unwrap(), 3.0);
        r.advance(3.0).unwrap();
        r.check_drained().unwrap();
    }

    #[test]
    fn policy_pressure_distinguishes_setup_service_and_waiting_work() {
        let mut r = Resources::default();
        let mut first = work(&[("link", 10.0), ("hbm", 1.0)]);
        first.demands[0].latency = 2.0;
        first.slots.push("copy".into());
        let mut second = work(&[("link", 4.0), ("hbm", 2.0)]);
        second.slots.push("copy".into());
        r.submit(0, first).unwrap();
        r.submit(1, second).unwrap();
        assert_eq!(r.admit(0.0).unwrap(), vec![0]);
        assert_eq!(
            r.policy_pressure(),
            vec![
                ResourcePressure {
                    resource: "hbm".into(),
                    waiting_service_s: 2.0,
                    setup_remaining_service_s: 1.0,
                    waiting_demands: 1,
                    setup_demands: 1,
                    ..Default::default()
                },
                ResourcePressure {
                    resource: "link".into(),
                    waiting_service_s: 4.0,
                    setup_remaining_service_s: 10.0,
                    waiting_demands: 1,
                    setup_demands: 1,
                    ..Default::default()
                },
            ]
        );
        let setup = r.policy_pressure();
        r.advance(1.0).unwrap();
        // Elapsed propagation does not reduce the service work still required.
        assert_eq!(r.policy_pressure(), setup);
        r.advance(2.0).unwrap();
        r.advance(7.0).unwrap();
        let pressure = r.policy_pressure();
        close(pressure[0].active_remaining_service_s, 0.5);
        close(pressure[1].active_remaining_service_s, 5.0);
        for resource in &pressure {
            assert_eq!(resource.active_demands, 1);
            assert_eq!(resource.waiting_demands, 1);
            assert_eq!(resource.setup_demands, 0);
            assert_eq!(resource.setup_remaining_service_s, 0.0);
        }
        assert_eq!(r.next_time(), Some(12.0));
        assert_eq!(r.advance(12.0).unwrap(), vec![(0, 0.0)]);
        assert_eq!(r.policy_pressure()[0].active_demands, 0);
        assert_eq!(r.admit(12.0).unwrap(), vec![1]);
        r.advance(16.0).unwrap();
        assert!(r.policy_pressure().is_empty());
        r.check_drained().unwrap();
    }

    #[test]
    fn policy_pressure_omits_completed_components_of_active_work() {
        let mut r = Resources::default();
        r.submit(
            0,
            Work {
                demands: vec![
                    work(&[("compute", 1.0)]).demands.remove(0),
                    work(&[("hbm", 4.0)]).demands.remove(0),
                ],
                ..Default::default()
            },
        )
        .unwrap();
        r.admit(0.0).unwrap();
        assert!(r.advance(1.0).unwrap().is_empty());
        assert_eq!(
            r.policy_pressure(),
            vec![ResourcePressure {
                resource: "hbm".into(),
                active_remaining_service_s: 3.0,
                active_demands: 1,
                ..Default::default()
            }]
        );
    }

    #[test]
    fn policy_memory_counts_unique_admitted_allocations_and_release() {
        let mut r = Resources::default();
        r.capacity("gpu".into(), 10).unwrap();
        r.capacity("idle".into(), 20).unwrap();
        r.resident(Allocation {
            id: "weights:model".into(),
            bytes: [("gpu".into(), 2)].into(),
        })
        .unwrap();
        let allocation = Allocation {
            id: "kv:0".into(),
            bytes: [("gpu".into(), 7)].into(),
        };
        let mut first = work(&[("link", 2.0)]);
        first.reserve = vec![allocation.clone(), allocation];
        first.release = vec!["kv:0".into()];
        r.submit(0, first).unwrap();
        r.submit(
            1,
            Work {
                reserve: vec![Allocation {
                    id: "kv:1".into(),
                    bytes: [("gpu".into(), 3)].into(),
                }],
                release: vec!["kv:1".into()],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.admit(0.0).unwrap(), vec![0]);
        assert_eq!(
            r.policy_memory(),
            vec![
                MemoryUsage {
                    device: "gpu".into(),
                    capacity_bytes: 10,
                    allocated_bytes: 9,
                },
                MemoryUsage {
                    device: "idle".into(),
                    capacity_bytes: 20,
                    allocated_bytes: 0,
                },
            ]
        );
        r.advance(2.0).unwrap();
        assert_eq!(r.policy_memory()[0].allocated_bytes, 2);
        assert_eq!(r.admit(2.0).unwrap(), vec![1]);
        assert_eq!(r.policy_memory()[0].allocated_bytes, 5);
        r.advance(2.0).unwrap();
        assert_eq!(r.policy_memory()[0].allocated_bytes, 2);
        r.check_drained().unwrap();
    }

    #[test]
    fn policy_snapshots_do_not_change_resource_execution() {
        let run = |observed: bool| {
            let mut r = Resources::default();
            for (id, demands, latency) in [
                (0, vec![("link", 10.0), ("hbm", 1.0)], 2.0),
                (1, vec![("hbm", 2.0)], 0.0),
                (2, vec![("compute", 3.0)], 1.0),
            ] {
                let mut submitted = work(&demands);
                submitted.demands[0].latency = latency;
                r.submit(id, submitted).unwrap();
            }
            let mut now = 0.0;
            let mut events = Vec::new();
            loop {
                let started = r.admit(now).unwrap();
                if observed {
                    assert_eq!(r.policy_pressure(), r.policy_pressure());
                    assert_eq!(r.policy_memory(), r.policy_memory());
                }
                let Some(next) = r.next_time() else {
                    break;
                };
                let finished = r.advance(next).unwrap();
                events.push((next, started, finished));
                now = next;
            }
            r.check_drained().unwrap();
            events
        };
        assert_eq!(run(false), run(true));
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Allocation {
    pub id: String,
    pub bytes: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Work {
    #[serde(default)]
    pub demands: Vec<Demand>,
    #[serde(default)]
    pub slots: Vec<String>,
    #[serde(default)]
    pub reserve: Vec<Allocation>,
    #[serde(default)]
    pub release: Vec<String>,
    #[serde(default)]
    pub details: Value,
}

/// Immutable input shared by graph nodes; execution state belongs to `Active`.
#[derive(Clone)]
pub(crate) struct SharedWork(Rc<WorkTemplate>);

struct WorkTemplate {
    work: Work,
    prepared: OnceCell<PreparedDemands>,
}

struct PreparedDemands {
    domain: Rc<()>,
    demands: Vec<NormalizedDemand>,
}

impl SharedWork {
    pub(crate) fn new(work: Work) -> Self {
        Self(Rc::new(WorkTemplate {
            work,
            prepared: OnceCell::new(),
        }))
    }

    pub(crate) fn into_work(self) -> Work {
        match Rc::try_unwrap(self.0) {
            Ok(template) => template.work,
            Err(shared) => shared.work.clone(),
        }
    }

    fn normalized(&self) -> &[NormalizedDemand] {
        &self.0.prepared.get().unwrap().demands
    }
}

impl Deref for SharedWork {
    type Target = Work;

    fn deref(&self) -> &Work {
        &self.0.work
    }
}

impl Serialize for SharedWork {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.work.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SharedWork {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Work::deserialize(deserializer).map(Self::new)
    }
}

/// Current work in full-capacity service seconds, not utilization or finish time.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ResourcePressure {
    pub resource: String,
    pub active_remaining_service_s: f64,
    pub waiting_service_s: f64,
    /// Service still required by admitted demands whose setup has not finished.
    /// This excludes setup latency itself and does not imply bandwidth use.
    pub setup_remaining_service_s: f64,
    pub active_demands: usize,
    pub waiting_demands: usize,
    pub setup_demands: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MemoryUsage {
    pub device: String,
    pub capacity_bytes: u64,
    pub allocated_bytes: u64,
}

struct Active {
    work: SharedWork,
    start: f64,
    demands: Vec<DemandState>,
}

struct DemandState {
    progress: f64,
    latency: f64,
    rate: f64,
}

struct NormalizedDemand {
    base: f64,
    coefficients: Vec<(usize, f64)>,
}

#[derive(Default)]
pub struct Resources {
    identity: Rc<()>,
    admission_dirty: bool,
    rates_dirty: bool,
    next_time_cache: Option<Option<f64>>,
    advance_pending: bool,
    resource_ids: BTreeMap<String, usize>,
    resource_order: Vec<usize>,
    capacities: BTreeMap<String, u64>,
    allocations: BTreeMap<String, Allocation>,
    allocated_bytes: BTreeMap<String, u64>,
    occupied_slots: BTreeSet<String>,
    waiting: BTreeMap<NodeID, SharedWork>,
    active: BTreeMap<NodeID, Active>,
    clock: f64,
}

impl Resources {
    /// Derive a stable, read-only summary only when a policy asks for it.
    pub fn policy_pressure(&self) -> Vec<ResourcePressure> {
        let mut pressure = BTreeMap::<&str, ResourcePressure>::new();
        for active in self.active.values() {
            for (index, demand) in active.work.demands.iter().enumerate() {
                let remaining = 1.0 - active.demands[index].progress;
                if remaining <= 0.0 {
                    continue;
                }
                for (resource, seconds) in &demand.resources {
                    let entry =
                        pressure
                            .entry(resource.as_str())
                            .or_insert_with(|| ResourcePressure {
                                resource: resource.clone(),
                                ..Default::default()
                            });
                    if active.demands[index].latency > 0.0 {
                        entry.setup_remaining_service_s += seconds * remaining;
                        entry.setup_demands += 1;
                    } else {
                        entry.active_remaining_service_s += seconds * remaining;
                        entry.active_demands += 1;
                    }
                }
            }
        }
        for work in self.waiting.values() {
            for demand in &work.demands {
                for (resource, seconds) in &demand.resources {
                    let entry =
                        pressure
                            .entry(resource.as_str())
                            .or_insert_with(|| ResourcePressure {
                                resource: resource.clone(),
                                ..Default::default()
                            });
                    entry.waiting_service_s += seconds;
                    entry.waiting_demands += 1;
                }
            }
        }
        pressure.into_values().collect()
    }

    /// Admitted allocations, including resident weights; waiting reservations
    /// consume no capacity until admission. Allocation identities stay private.
    pub fn policy_memory(&self) -> Vec<MemoryUsage> {
        self.capacities
            .iter()
            .map(|(device, capacity)| MemoryUsage {
                device: device.clone(),
                capacity_bytes: *capacity,
                allocated_bytes: self.allocated_bytes.get(device).copied().unwrap_or(0),
            })
            .collect()
    }

    pub fn resident(&mut self, allocation: Allocation) -> Result<()> {
        if let Some(old) = self.allocations.get(&allocation.id) {
            ensure!(old == &allocation, "conflicting resident allocation");
            return Ok(());
        }
        for (memory, bytes) in &allocation.bytes {
            let used = self.allocated_bytes.get(memory).copied().unwrap_or(0);
            ensure!(
                used.checked_add(*bytes)
                    .is_some_and(|v| v <= self.capacities[memory]),
                "resident weights exceed capacity of {memory}"
            );
        }
        self.insert_allocation(allocation);
        Ok(())
    }

    fn insert_allocation(&mut self, allocation: Allocation) {
        if self.allocations.contains_key(&allocation.id) {
            return;
        }
        for (memory, bytes) in &allocation.bytes {
            *self.allocated_bytes.entry(memory.clone()).or_default() += bytes;
        }
        self.allocations.insert(allocation.id.clone(), allocation);
        self.admission_dirty = true;
    }
    pub fn capacity(&mut self, name: String, bytes: u64) -> Result<()> {
        ensure!(bytes > 0, "zero memory capacity for {name}");
        if let Some(old) = self.capacities.insert(name.clone(), bytes) {
            ensure!(old == bytes, "conflicting capacity for {name}");
        }
        self.admission_dirty = true;
        Ok(())
    }

    pub fn submit(&mut self, id: NodeID, work: Work) -> Result<()> {
        self.submit_shared(id, SharedWork::new(work))
    }

    pub(crate) fn submit_shared(&mut self, id: NodeID, work: SharedWork) -> Result<()> {
        if let Some(prepared) = work.0.prepared.get() {
            ensure!(
                Rc::ptr_eq(&prepared.domain, &self.identity),
                "resource work belongs to another execution domain"
            );
        } else {
            self.validate_work(&work)?;
            let prepared = PreparedDemands {
                domain: self.identity.clone(),
                demands: self.normalize_demands(&work.demands),
            };
            // Only this single-threaded domain initializes the immutable template.
            let _ = work.0.prepared.set(prepared);
        }
        ensure!(
            self.waiting.insert(id, work).is_none(),
            "duplicate resource work"
        );
        self.admission_dirty = true;
        Ok(())
    }

    fn validate_work(&self, work: &Work) -> Result<()> {
        for demand in &work.demands {
            ensure!(
                demand.latency.is_finite() && demand.latency >= 0.0,
                "invalid latency"
            );
            ensure!(
                demand.resources.values().all(|v| v.is_finite() && *v > 0.0),
                "invalid resource demand"
            );
        }
        let mut own = BTreeMap::<String, u64>::new();
        let mut seen = BTreeSet::new();
        for allocation in &work.reserve {
            if !seen.insert(&allocation.id) {
                continue;
            }
            for (memory, bytes) in &allocation.bytes {
                let total = own.entry(memory.clone()).or_default();
                *total = total
                    .checked_add(*bytes)
                    .ok_or_else(|| anyhow::anyhow!("reservation overflow"))?;
                ensure!(
                    *total
                        <= *self
                            .capacities
                            .get(memory)
                            .ok_or_else(|| anyhow::anyhow!("unknown memory {memory}"))?,
                    "reservation cannot fit on {memory}"
                );
            }
        }
        Ok(())
    }

    pub fn release(&mut self, id: &str) -> Result<()> {
        let allocation = self
            .allocations
            .remove(id)
            .ok_or_else(|| anyhow::anyhow!("unknown allocation {id}"))?;
        for (memory, bytes) in allocation.bytes {
            *self.allocated_bytes.get_mut(&memory).unwrap() -= bytes;
        }
        self.admission_dirty = true;
        Ok(())
    }

    pub fn allocated(&self, id: &str) -> bool {
        self.allocations.contains_key(id)
    }

    pub fn snapshot(&self) -> Value {
        serde_json::json!({"capacity_bytes":self.capacities, "allocations": self.allocations, "active": self.active.len(), "waiting": self.waiting.len()})
    }

    pub fn admit(&mut self, now: f64) -> Result<Vec<NodeID>> {
        let mut started = Vec::new();
        if !self.admission_dirty {
            return Ok(started);
        }
        let ids: Vec<_> = self.waiting.keys().copied().collect();
        for id in ids {
            let work = &self.waiting[&id];
            if work.slots.iter().any(|s| self.occupied_slots.contains(s)) {
                continue;
            }
            let mut needed: BTreeMap<String, u64> = BTreeMap::new();
            let mut unique = BTreeMap::new();
            for allocation in &work.reserve {
                if let Some(old) = self
                    .allocations
                    .get(&allocation.id)
                    .or_else(|| unique.get(&allocation.id).copied())
                {
                    ensure!(
                        old == allocation,
                        "conflicting allocation {}",
                        allocation.id
                    );
                } else {
                    unique.insert(allocation.id.clone(), allocation);
                    for (m, b) in &allocation.bytes {
                        *needed.entry(m.clone()).or_default() += b;
                    }
                }
            }
            let fits = needed.iter().all(|(m, b)| {
                let used = self.allocated_bytes.get(m).copied().unwrap_or(0);
                used.checked_add(*b)
                    .is_some_and(|total| total <= self.capacities[m])
            });
            if !fits {
                continue;
            }
            let work = self.waiting.remove(&id).unwrap();
            for allocation in &work.reserve {
                self.insert_allocation(allocation.clone());
            }
            for slot in &work.slots {
                self.occupied_slots.insert(slot.clone());
            }
            let demands = work
                .demands
                .iter()
                .map(|d| DemandState {
                    progress: 0.0,
                    latency: d.latency,
                    rate: 0.0,
                })
                .collect();
            self.active.insert(
                id,
                Active {
                    work,
                    start: now,
                    demands,
                },
            );
            started.push(id);
            self.rates_dirty = true;
            self.next_time_cache = None;
        }
        self.admission_dirty = false;
        Ok(started)
    }

    fn normalize_demands(&mut self, demands: &[Demand]) -> Vec<NormalizedDemand> {
        let mut names_changed = false;
        let normalized = demands
            .iter()
            .map(|demand| {
                let base = demand.resources.values().copied().fold(0.0, f64::max);
                let coefficients = demand
                    .resources
                    .iter()
                    .map(|(name, value)| {
                        let id = if let Some(&id) = self.resource_ids.get(name) {
                            id
                        } else {
                            let id = self.resource_ids.len();
                            self.resource_ids.insert(name.clone(), id);
                            names_changed = true;
                            id
                        };
                        (id, value / base)
                    })
                    .collect();
                NormalizedDemand { base, coefficients }
            })
            .collect();
        if names_changed {
            self.resource_order = self.resource_ids.values().copied().collect();
        }
        normalized
    }

    fn rates(&self) -> Vec<((NodeID, usize), f64)> {
        // Demand order and lexicographic resource order match the historical
        // implementation. Immutable normalization is shared by a work template.
        struct Component<'a> {
            key: (NodeID, usize),
            demand: &'a NormalizedDemand,
            fraction: f64,
        }
        let mut components = Vec::new();
        let mut used = vec![false; self.resource_ids.len()];
        let mut shared_resources = false;
        for (&id, active) in &self.active {
            for (i, demand) in active.work.normalized().iter().enumerate() {
                if active.demands[i].progress >= 1.0
                    || active.demands[i].latency > 0.0
                    || demand.base == 0.0
                {
                    continue;
                }
                for &(resource, _) in &demand.coefficients {
                    shared_resources |= used[resource];
                    used[resource] = true;
                }
                components.push(Component {
                    key: (id, i),
                    demand,
                    fraction: 0.0,
                });
            }
        }
        if !shared_resources {
            // Disjoint resources reach fraction=1 in the first filling step.
            return components
                .into_iter()
                .map(|component| (component.key, 1.0 / component.demand.base))
                .collect();
        }
        let mut remaining = vec![1.0; self.resource_ids.len()];
        let mut usage = vec![0.0; self.resource_ids.len()];
        let mut growing: Vec<_> = (0..components.len()).collect();
        while !growing.is_empty() {
            usage.fill(0.0);
            let mut delta = 1.0_f64;
            for &index in &growing {
                let component = &components[index];
                delta = delta.min(1.0 - component.fraction);
                for &(resource, coefficient) in &component.demand.coefficients {
                    usage[resource] += coefficient;
                }
            }
            for &resource in &self.resource_order {
                let coefficient = usage[resource];
                if coefficient > 0.0 {
                    delta = delta.min((remaining[resource] / coefficient).max(0.0));
                }
            }
            for &index in &growing {
                components[index].fraction += delta;
            }
            for &resource in &self.resource_order {
                let coefficient = usage[resource];
                if coefficient > 0.0 {
                    remaining[resource] -= delta * coefficient;
                }
            }
            let old = growing.len();
            growing.retain(|&index| {
                let component = &components[index];
                component.fraction < 1.0 - 1e-12
                    && component
                        .demand
                        .coefficients
                        .iter()
                        .all(|&(resource, _)| remaining[resource] > 1e-12)
            });
            assert!(growing.len() < old, "resource allocation made no progress");
        }
        components
            .into_iter()
            .map(|component| (component.key, component.fraction / component.demand.base))
            .collect()
    }

    fn refresh_rates(&mut self) {
        if self.rates_dirty {
            let mut rates = self.rates().into_iter().peekable();
            for (&id, active) in &mut self.active {
                while rates.peek().is_some_and(|((node, _), _)| *node == id) {
                    let ((_, index), rate) = rates.next().unwrap();
                    active.demands[index].rate = rate;
                }
            }
            self.rates_dirty = false;
            self.advance_pending = true;
        }
    }

    pub fn next_time(&mut self) -> Option<f64> {
        if let Some(time) = self.next_time_cache {
            return time;
        }
        self.refresh_rates();
        let time = self.active.values().map(|a| {
            // Component completion changes contention even if the whole kernel remains active.
            a.work
                .demands
                .iter()
                .zip(&a.demands)
                .filter(|(_, state)| state.progress < 1.0)
                .map(|(d, state)| {
                    if state.latency > 0.0 {
                        state.latency
                    } else if d.resources.is_empty() {
                        0.0
                    } else {
                        (1.0 - state.progress) / state.rate
                    }
                })
                .min_by(f64::total_cmp)
                .unwrap_or(0.0)
        });
        let time = time.min_by(f64::total_cmp).map(|dt| {
            let time = self.clock + dt;
            if dt > 0.0 && time == self.clock {
                self.clock.next_up()
            } else {
                time
            }
        });
        self.next_time_cache = Some(time);
        time
    }

    pub fn advance(&mut self, now: f64) -> Result<Vec<(NodeID, f64)>> {
        ensure!(
            now >= self.clock && now.is_finite(),
            "resource clock went backwards"
        );
        if now.to_bits() == self.clock.to_bits() && !self.advance_pending && !self.rates_dirty {
            return Ok(Vec::new());
        }
        let dt = now - self.clock;
        self.refresh_rates();
        self.next_time_cache = None;
        for a in self.active.values_mut() {
            for (d, state) in a.work.demands.iter().zip(&mut a.demands) {
                if state.progress >= 1.0 {
                    continue;
                }
                if state.latency > 0.0 {
                    state.latency = (state.latency - dt).max(0.0);
                    self.rates_dirty |= state.latency == 0.0;
                } else {
                    state.progress = if d.resources.is_empty() {
                        1.0
                    } else {
                        (state.progress + dt * state.rate).min(1.0)
                    };
                    if 1.0 - state.progress < 1e-12 {
                        state.progress = 1.0;
                    }
                    self.rates_dirty |= state.progress == 1.0;
                }
            }
        }
        self.clock = now;
        self.advance_pending = false;
        let finished: Vec<_> = self
            .active
            .iter()
            .filter(|(_, a)| a.demands.iter().all(|state| state.progress >= 1.0))
            .map(|(&id, a)| (id, a.start))
            .collect();
        self.admission_dirty |= !finished.is_empty();
        for (id, _) in &finished {
            let a = self.active.remove(id).unwrap();
            for slot in &a.work.slots {
                self.occupied_slots.remove(slot);
            }
            for allocation in &a.work.release {
                self.release(allocation)?;
            }
        }
        Ok(finished)
    }

    pub fn check_drained(&self) -> Result<()> {
        if !self.waiting.is_empty() {
            bail!(
                "resource admission blocked: {} waiting nodes; insufficient memory or cyclic reservations",
                self.waiting.len()
            );
        }
        ensure!(self.active.is_empty(), "active resources remain");
        ensure!(
            self.allocations.keys().all(|k| k.starts_with("weights:")),
            "request or workspace allocations leaked"
        );
        Ok(())
    }
}
