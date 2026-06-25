pub mod bbox;
pub mod connectors;
pub mod engine;
#[cfg(feature = "pyo3")]
pub mod engine_py;
mod octree;
#[cfg(feature = "pyo3")]
pub mod py_wrappers;
#[cfg(feature = "pyo3")]
use py_wrappers::*;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::engine::{AssemblyEngineConfig, EngineState, Part, Query};

#[cfg(feature = "pyo3")]
#[pymodule]
fn assembly_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVec3>()?;
    m.add_class::<PyQuat>()?;
    m.add_class::<Part>()?;
    m.add_class::<PyPlacement>()?;
    m.add_class::<Query>()?;
    m.add_class::<PyAssembledModel>()?;
    m.add_class::<AssemblyEngineConfig>()?;
    m.add_class::<PyAssemblyEngine>()?;
    m.add_class::<EngineState>()?;

    Ok(())
}
