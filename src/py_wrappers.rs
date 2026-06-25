use glam::{Quat, Vec3};
use pyo3::prelude::*;

use crate::engine::{
    AssembledModel, AssemblyEngine, AssemblyEngineConfig, EngineState, Part, Placement, Query,
};

#[pyclass]
#[derive(Copy, Clone, Debug)]
pub struct PyVec3 {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
    #[pyo3(get)]
    pub z: f32,
}

impl From<PyVec3> for Vec3 {
    fn from(val: PyVec3) -> Self {
        Vec3::new(val.x, val.y, val.z)
    }
}

impl From<Vec3> for PyVec3 {
    fn from(val: Vec3) -> Self {
        PyVec3 {
            x: val.x,
            y: val.y,
            z: val.z,
        }
    }
}

#[pymethods]
impl PyVec3 {
    #[new]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[pyclass]
#[derive(Copy, Clone, Debug)]
pub struct PyQuat {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
    #[pyo3(get)]
    pub z: f32,
    #[pyo3(get)]
    pub w: f32,
}

impl From<PyQuat> for Quat {
    fn from(val: PyQuat) -> Self {
        Quat::from_xyzw(val.x, val.y, val.z, val.w)
    }
}

impl From<Quat> for PyQuat {
    fn from(val: Quat) -> Self {
        PyQuat {
            x: val.x,
            y: val.y,
            z: val.z,
            w: val.w,
        }
    }
}

#[pymethods]
impl PyQuat {
    #[new]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct PyPlacement {
    #[pyo3(get)]
    pub part_index: usize,
    #[pyo3(get)]
    pub position: PyVec3,
    #[pyo3(get)]
    pub rotation: PyQuat,
}

impl From<PyPlacement> for Placement {
    fn from(value: PyPlacement) -> Self {
        Self {
            part_index: value.part_index,
            position: value.position.into(),
            rotation: value.rotation.into(),
        }
    }
}

impl From<Placement> for PyPlacement {
    fn from(value: Placement) -> Self {
        Self {
            part_index: value.part_index,
            position: value.position.into(),
            rotation: value.rotation.into(),
        }
    }
}

#[pymethods]
impl PyPlacement {
    #[new]
    pub fn new(part_index: usize, position: PyVec3, rotation: PyQuat) -> Self {
        Self {
            part_index,
            position,
            rotation,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyAssembledModel {
    #[pyo3(get)]
    pub placements: Vec<PyPlacement>,
}

impl From<PyAssembledModel> for AssembledModel {
    fn from(value: PyAssembledModel) -> Self {
        Self {
            placements: value.placements.iter().map(|&x| x.into()).collect(),
        }
    }
}

impl From<AssembledModel> for PyAssembledModel {
    fn from(value: AssembledModel) -> Self {
        Self {
            placements: value.placements.iter().map(|&x| x.into()).collect(),
        }
    }
}

#[pyclass]
pub struct PyAssemblyEngine {
    engine: AssemblyEngine,
}

#[pymethods]
impl PyAssemblyEngine {
    #[new]
    pub fn new(parts: Vec<Part>, config: &AssemblyEngineConfig) -> Self {
        Self {
            engine: AssemblyEngine::new(&parts, config),
        }
    }

    pub fn clear(&mut self) {
        self.engine.clear();
    }

    pub fn get_model(&self) -> PyAssembledModel {
        self.engine.get_model().clone().into()
    }

    pub fn load_model(&mut self, model: &PyAssembledModel) {
        self.engine.load_model(&model.clone().into());
    }

    pub fn get_parts(&self) -> Vec<Part> {
        self.engine.get_parts().to_vec()
    }

    pub fn add_placement(&mut self, placement: PyPlacement) {
        self.engine.add_placement(placement.into());
    }

    pub fn query(&mut self, query: &Query) -> Vec<PyPlacement> {
        self.engine.query(query).iter().map(|&x| x.into()).collect()
    }

    pub fn query_multi(&mut self, queries: Vec<Query>) -> Vec<Vec<PyPlacement>> {
        queries
            .iter()
            .map(|query| self.engine.query(query).iter().map(|&x| x.into()).collect())
            .collect()
    }

    pub fn query_exists(&mut self, part_id: Option<usize>, anchor_idx: Option<usize>) -> bool {
        !self
            .engine
            .query(&Query {
                part_id,
                anchor_idx,
                single: true,
            })
            .is_empty()
    }

    pub fn query_exists_multi(&mut self, queries: Vec<Query>) -> Vec<bool> {
        queries
            .iter()
            .map(|query| {
                !self
                    .engine
                    .query(&Query {
                        part_id: query.part_id,
                        anchor_idx: query.anchor_idx,
                        single: true,
                    })
                    .is_empty()
            })
            .collect()
    }

    pub fn get_state(&self) -> EngineState {
        self.engine.get_state()
    }

    pub fn load_state(&mut self, state: &EngineState) {
        self.engine.load_state(state);
    }
}
