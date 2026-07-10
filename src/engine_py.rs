#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::engine::{AssemblyEngineConfig, Part, Query};

#[pymethods]
impl Part {
    #[staticmethod]
    pub fn from_json(json_str: &str) -> Self {
        serde_json::from_str(json_str).unwrap()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

#[pymethods]
impl Query {
    #[new]
    fn new(part_id: Option<usize>, anchor_idx: Option<usize>, single: bool) -> Self {
        Self {
            part_id,
            anchor_idx,
            single,
        }
    }

    #[getter]
    fn part_id(&self) -> Option<usize> {
        self.part_id
    }

    #[getter]
    fn anchor_idx(&self) -> Option<usize> {
        self.anchor_idx
    }

    #[getter]
    fn single(&self) -> bool {
        self.single
    }
}

#[pymethods]
impl AssemblyEngineConfig {
    #[new]
    pub fn new_py(num_candidate_turns: u32, bar_increment_every: f32) -> Self {
        Self::new(num_candidate_turns, bar_increment_every)
    }

    /// The number of turns to consider when generating candidates.
    #[getter]
    pub fn num_candidate_turns(&self) -> u32 {
        self.num_candidate_turns
    }
    /// The number of increments on a bar to consider when generating candidates.
    #[getter]
    pub fn bar_increment_every(&self) -> f32 {
        self.bar_increment_every
    }
}
