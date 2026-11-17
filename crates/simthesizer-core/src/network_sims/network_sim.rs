use std::any::Any;

use anyhow::Result;

use crate::module::Module;
use crate::types::{NetworkDeviceID, PollResult};

pub trait NetworkSimulator: Module {
    fn add_device(&mut self) -> NetworkDeviceID;

    fn add_link(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bandwidth: f64,
        latency: f64,
    ) -> Result<()>;

    /// Pin a route using link insertion indices, after all links have been added.
    fn set_route(
        &mut self,
        _src: NetworkDeviceID,
        _dst: NetworkDeviceID,
        _links: &[usize],
    ) -> Result<()> {
        anyhow::bail!("network simulator does not support fixed routes")
    }

    fn simulate(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
        now: f64,
        state: &mut Option<Box<dyn Any>>,
    ) -> Result<PollResult>;
}

crate::declare_registry! {
    registration = NetworkSimulatorRegistration,
    trait = crate::network_sim::NetworkSimulator,
    deserialize_fn = deserialize_network_simulator_impl,
    registry_name = "network_sim",
}

crate::impl_registry_deserialize! {
    trait = crate::network_sim::NetworkSimulator,
    deserialize_fn = crate::network_sim::deserialize_network_simulator_impl,
}

#[macro_export]
macro_rules! register_network_simulator {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::network_sim::NetworkSimulatorRegistration,
            trait = $crate::network_sim::NetworkSimulator,
            type = $ty,
            kind = $kind,
            registry_name = "network_sim",
        );
    };
}
