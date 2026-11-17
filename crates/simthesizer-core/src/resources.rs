//! Fluid sharing and capacity reservations, driven only by the engine clock.
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

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

struct Active {
    work: Work,
    start: f64,
    progress: Vec<f64>,
    latency: Vec<f64>,
}

#[derive(Default)]
pub struct Resources {
    rates_dirty: bool,
    rates_cache: BTreeMap<(NodeID, usize), f64>,
    capacities: BTreeMap<String, u64>,
    allocations: BTreeMap<String, Allocation>,
    waiting: BTreeMap<NodeID, Work>,
    active: BTreeMap<NodeID, Active>,
    clock: f64,
}

impl Resources {
    pub fn resident(&mut self, allocation: Allocation) -> Result<()> {
        if let Some(old) = self.allocations.get(&allocation.id) {
            ensure!(old == &allocation, "conflicting resident allocation");
            return Ok(());
        }
        for (memory, bytes) in &allocation.bytes {
            let used: u64 = self
                .allocations
                .values()
                .map(|a| a.bytes.get(memory).copied().unwrap_or(0))
                .sum();
            ensure!(
                used.checked_add(*bytes)
                    .is_some_and(|v| v <= self.capacities[memory]),
                "resident weights exceed capacity of {memory}"
            );
        }
        self.allocations.insert(allocation.id.clone(), allocation);
        Ok(())
    }
    pub fn capacity(&mut self, name: String, bytes: u64) -> Result<()> {
        ensure!(bytes > 0, "zero memory capacity for {name}");
        if let Some(old) = self.capacities.insert(name.clone(), bytes) {
            ensure!(old == bytes, "conflicting capacity for {name}");
        }
        Ok(())
    }

    pub fn submit(&mut self, id: NodeID, work: Work) -> Result<()> {
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
        ensure!(
            self.waiting.insert(id, work).is_none(),
            "duplicate resource work"
        );
        Ok(())
    }

    pub fn release(&mut self, id: &str) -> Result<()> {
        ensure!(
            self.allocations.remove(id).is_some(),
            "unknown allocation {id}"
        );
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
        let ids: Vec<_> = self.waiting.keys().copied().collect();
        for id in ids {
            let work = &self.waiting[&id];
            if work
                .slots
                .iter()
                .any(|s| self.active.values().any(|a| a.work.slots.contains(s)))
            {
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
                let used: u64 = self
                    .allocations
                    .values()
                    .map(|a| a.bytes.get(m).copied().unwrap_or(0))
                    .sum();
                used.checked_add(*b)
                    .is_some_and(|total| total <= self.capacities[m])
            });
            if !fits {
                continue;
            }
            let work = self.waiting.remove(&id).unwrap();
            for allocation in &work.reserve {
                self.allocations
                    .insert(allocation.id.clone(), allocation.clone());
            }
            let progress = vec![0.0; work.demands.len()];
            let latency = work.demands.iter().map(|d| d.latency).collect();
            self.active.insert(
                id,
                Active {
                    work,
                    start: now,
                    progress,
                    latency,
                },
            );
            started.push(id);
            self.rates_dirty = true;
        }
        Ok(started)
    }

    fn rates(&self) -> BTreeMap<(NodeID, usize), f64> {
        // Progressive filling of normalized full-speed fractions. A slow network
        // copy consumes only its actual HBM rate; unused HBM remains available.
        let mut coefficients = BTreeMap::new();
        let mut bases = BTreeMap::new();
        let mut fractions = BTreeMap::new();
        let mut remaining = BTreeMap::new();
        for (&id, active) in &self.active {
            for (i, d) in active.work.demands.iter().enumerate() {
                if active.progress[i] >= 1.0 || active.latency[i] > 0.0 {
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

    fn refresh_rates(&mut self) {
        if self.rates_dirty {
            self.rates_cache = self.rates();
            self.rates_dirty = false;
        }
    }

    pub fn next_time(&mut self) -> Option<f64> {
        self.refresh_rates();
        let rates = &self.rates_cache;
        self.active
            .iter()
            .map(|(&id, a)| {
                // Component completion changes contention even if the whole kernel remains active.
                a.work
                    .demands
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| a.progress[*i] < 1.0)
                    .map(|(i, d)| {
                        if a.latency[i] > 0.0 {
                            a.latency[i]
                        } else if d.resources.is_empty() {
                            0.0
                        } else {
                            (1.0 - a.progress[i]) / rates[&(id, i)]
                        }
                    })
                    .min_by(f64::total_cmp)
                    .unwrap_or(0.0)
            })
            .min_by(f64::total_cmp)
            .map(|dt| {
                let time = self.clock + dt;
                if dt > 0.0 && time == self.clock {
                    self.clock.next_up()
                } else {
                    time
                }
            })
    }

    pub fn advance(&mut self, now: f64) -> Result<Vec<(NodeID, f64)>> {
        ensure!(
            now >= self.clock && now.is_finite(),
            "resource clock went backwards"
        );
        let dt = now - self.clock;
        self.refresh_rates();
        let rates = &self.rates_cache;
        for (&id, a) in &mut self.active {
            for (i, d) in a.work.demands.iter().enumerate() {
                if a.progress[i] >= 1.0 {
                    continue;
                }
                if a.latency[i] > 0.0 {
                    a.latency[i] = (a.latency[i] - dt).max(0.0);
                    self.rates_dirty |= a.latency[i] == 0.0;
                } else {
                    a.progress[i] = if d.resources.is_empty() {
                        1.0
                    } else {
                        (a.progress[i] + dt * rates[&(id, i)]).min(1.0)
                    };
                    if 1.0 - a.progress[i] < 1e-12 {
                        a.progress[i] = 1.0;
                    }
                    self.rates_dirty |= a.progress[i] == 1.0;
                }
            }
        }
        self.clock = now;
        let finished: Vec<_> = self
            .active
            .iter()
            .filter(|(_, a)| a.progress.iter().all(|p| *p >= 1.0))
            .map(|(&id, a)| (id, a.start))
            .collect();
        for (id, _) in &finished {
            let a = self.active.remove(id).unwrap();
            for allocation in a.work.release {
                self.release(&allocation)?;
            }
        }
        Ok(finished)
    }

    pub fn check_drained(&self) -> Result<()> {
        if !self.waiting.is_empty() {
            bail!("resource admission blocked: {} waiting nodes; insufficient memory or cyclic reservations", self.waiting.len());
        }
        ensure!(self.active.is_empty(), "active resources remain");
        ensure!(
            self.allocations.keys().all(|k| k.starts_with("weights:")),
            "request or workspace allocations leaked"
        );
        Ok(())
    }
}
