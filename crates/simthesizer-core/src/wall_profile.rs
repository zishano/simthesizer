//! Opt-in host wall time. These measurements never enter simulated decisions.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy)]
pub enum Section {
    ConfigLoad,
    SystemInit,
    WorkloadLoad,
    EngineOther,
    ResourceAdmit,
    ResourceRates,
    ResourceAdvance,
    EventDispatch,
    GraphBuild,
    WorkerRoundtrip,
    WorkerJsonDecode,
    ResultWrite,
}

const NAMES: [&str; 12] = [
    "config_load",
    "system_init",
    "workload_load",
    "engine_other",
    "resource_admit",
    "resource_rates",
    "resource_advance",
    "event_dispatch",
    "graph_build",
    "worker_roundtrip",
    "worker_json_decode",
    "result_write",
];

#[derive(Clone, Copy, Default)]
struct Measurement {
    calls: u64,
    inclusive: Duration,
    exclusive: Duration,
}

#[derive(Serialize)]
pub struct SectionSnapshot {
    pub calls: u64,
    pub inclusive_seconds: f64,
    pub exclusive_seconds: f64,
}

/// Python measurements subdivide worker_roundtrip; they are not extra host time.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct WorkerSection {
    pub calls: u64,
    pub seconds: f64,
}

#[derive(Serialize)]
pub struct Snapshot {
    pub schema_version: u32,
    pub wall_seconds: f64,
    pub sections: BTreeMap<&'static str, SectionSnapshot>,
    pub worker_sections: BTreeMap<String, WorkerSection>,
}

struct Frame {
    section: Section,
    started: Instant,
    children: Duration,
}

struct Profile {
    started: Instant,
    measurements: [Measurement; NAMES.len()],
    stack: Vec<Frame>,
    worker_sections: BTreeMap<String, WorkerSection>,
}

impl Profile {
    fn new(started: Instant) -> Self {
        Self {
            started,
            measurements: [Measurement::default(); NAMES.len()],
            stack: Vec::with_capacity(8),
            worker_sections: BTreeMap::new(),
        }
    }

    fn enter(&mut self, section: Section, now: Instant) {
        self.measurements[section as usize].calls += 1;
        self.stack.push(Frame {
            section,
            started: now,
            children: Duration::ZERO,
        });
    }

    fn leave(&mut self, now: Instant) {
        let frame = self.stack.pop().expect("wall-profile scope stack");
        let inclusive = now.duration_since(frame.started);
        let measurement = &mut self.measurements[frame.section as usize];
        measurement.inclusive += inclusive;
        measurement.exclusive += inclusive.saturating_sub(frame.children);
        if let Some(parent) = self.stack.last_mut() {
            parent.children += inclusive;
        }
    }

    fn snapshot(&self, now: Instant) -> Snapshot {
        let mut measurements = self.measurements;
        for (index, frame) in self.stack.iter().enumerate() {
            let inclusive = now.duration_since(frame.started);
            let active_child = self
                .stack
                .get(index + 1)
                .map_or(Duration::ZERO, |child| now.duration_since(child.started));
            let measurement = &mut measurements[frame.section as usize];
            measurement.inclusive += inclusive;
            measurement.exclusive += inclusive.saturating_sub(frame.children + active_child);
        }
        Snapshot {
            schema_version: 1,
            wall_seconds: now.duration_since(self.started).as_secs_f64(),
            sections: NAMES
                .iter()
                .zip(measurements)
                .filter(|(_, measurement)| measurement.calls > 0)
                .map(|(&name, measurement)| {
                    (
                        name,
                        SectionSnapshot {
                            calls: measurement.calls,
                            inclusive_seconds: measurement.inclusive.as_secs_f64(),
                            exclusive_seconds: measurement.exclusive.as_secs_f64(),
                        },
                    )
                })
                .collect(),
            worker_sections: self.worker_sections.clone(),
        }
    }
}

thread_local! {
    static PROFILE: RefCell<Option<Profile>> = const { RefCell::new(None) };
}

/// Start on the simulation thread before initialization. No environment mutation.
pub fn enable() {
    PROFILE.with(|profile| {
        let mut profile = profile.borrow_mut();
        assert!(profile.is_none(), "wall profiling already enabled");
        *profile = Some(Profile::new(Instant::now()));
    });
}

pub fn enabled() -> bool {
    PROFILE.with(|profile| profile.borrow().is_some())
}

/// Scopes must stay on their originating thread and be dropped in nested order.
#[must_use]
pub struct Scope {
    active: bool,
    _thread: PhantomData<Rc<()>>,
}

#[inline]
pub fn scope(section: Section) -> Scope {
    let active = PROFILE.with(|profile| {
        if let Some(profile) = profile.borrow_mut().as_mut() {
            profile.enter(section, Instant::now());
            true
        } else {
            false
        }
    });
    Scope {
        active,
        _thread: PhantomData,
    }
}

impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        if self.active {
            PROFILE.with(|profile| {
                profile.borrow_mut().as_mut().unwrap().leave(Instant::now());
            });
        }
    }
}

pub fn snapshot() -> Option<Snapshot> {
    PROFILE.with(|profile| {
        profile
            .borrow()
            .as_ref()
            .map(|profile| profile.snapshot(Instant::now()))
    })
}

pub fn add_worker_sections(sections: BTreeMap<String, WorkerSection>) {
    PROFILE.with(|profile| {
        if let Some(profile) = profile.borrow_mut().as_mut() {
            for (name, section) in sections {
                let total = profile.worker_sections.entry(name).or_default();
                total.calls += section.calls;
                total.seconds += section.seconds;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_and_active_scopes_partition_wall_time_without_double_counting() {
        let start = Instant::now();
        let at = |seconds| start + Duration::from_secs(seconds);
        let mut profile = Profile::new(start);
        profile.enter(Section::EngineOther, at(1));
        profile.enter(Section::GraphBuild, at(2));
        profile.enter(Section::WorkerRoundtrip, at(3));
        profile.leave(at(5));
        profile.enter(Section::WorkerJsonDecode, at(6));
        let active = profile.snapshot(at(7));
        assert_eq!(active.sections["engine_other"].exclusive_seconds, 1.0);
        assert_eq!(active.sections["graph_build"].exclusive_seconds, 2.0);
        assert_eq!(active.sections["worker_roundtrip"].exclusive_seconds, 2.0);
        assert_eq!(active.sections["worker_json_decode"].exclusive_seconds, 1.0);
        assert_eq!(active.sections["engine_other"].inclusive_seconds, 6.0);
        profile.leave(at(8));
        profile.leave(at(9));
        profile.leave(at(10));
        let complete = profile.snapshot(at(11));
        assert_eq!(complete.sections["engine_other"].exclusive_seconds, 2.0);
        assert_eq!(complete.sections["graph_build"].exclusive_seconds, 3.0);
        assert_eq!(
            complete.sections["worker_json_decode"].exclusive_seconds,
            2.0
        );
        assert_eq!(complete.sections["graph_build"].calls, 1);
        assert_eq!(
            complete
                .sections
                .values()
                .map(|s| s.exclusive_seconds)
                .sum::<f64>(),
            9.0
        );
        assert_eq!(complete.wall_seconds, 11.0);
    }

    #[test]
    fn measurements_default_to_disabled_on_an_independent_thread() {
        std::thread::spawn(|| {
            assert!(!enabled());
            let _scope = scope(Section::EngineOther);
            assert!(snapshot().is_none());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn completed_worker_sections_are_nested_telemetry() {
        std::thread::spawn(|| {
            enable();
            for _ in 0..2 {
                add_worker_sections(BTreeMap::from([(
                    "graph_build".into(),
                    WorkerSection {
                        calls: 1,
                        seconds: 2.0,
                    },
                )]));
            }
            let snapshot = snapshot().unwrap();
            assert_eq!(snapshot.worker_sections["graph_build"].calls, 2);
            assert_eq!(snapshot.worker_sections["graph_build"].seconds, 4.0);
            assert!(snapshot.sections.is_empty());
        })
        .join()
        .unwrap();
    }
}
