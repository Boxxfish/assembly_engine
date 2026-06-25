use std::collections::HashSet;

use crate::bbox::{AABB, OBB};
use crate::connectors::{CLIP_LENGTH, Connector};
use crate::octree::Octree;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Indiciates an axis of rotational symmetry for a part.
///
/// The axis can be thought of a ray with direction `dir` and origin `origin`.
/// If two placements of the same part have identical rays, the angle between the placements
/// is checked to see if they are oriented symmetrically.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RotSym {
    /// Where the axis of rotation is centered around.
    pub origin: Vec3,
    /// The axis of rotation. This should be normalized to unit length.
    pub dir: Vec3,
    /// The number of symmetric orientations in a full turn (360 degrees).
    ///
    /// For example, if a part is rotationally symmetric after turning 45 degrees, this should be 4.
    ///
    /// If this is 1, this part effectively has no rotational symmetry around this axis.
    ///
    /// A value of 0 treats *every* rotation around this axis as symmetric.
    pub orientations: u8,
}

/// Represents a single part, in isolation.
///
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    pub bboxes: Vec<AABB>,
    pub connectors: Vec<Connector>,
    pub rot_syms: Vec<RotSym>,
}

/// Represents a placed part.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Placement {
    pub part_index: usize,
    pub position: Vec3,
    pub rotation: Quat,
}

impl Placement {
    /// Returns `true` if these two `Placement`s are approximately equal.
    ///
    /// `eps` determines the minimum similarity to be considered equal.
    pub fn approx_eq(&self, other: &Self, eps: f32) -> bool {
        self.part_index == other.part_index
            && self.position.abs_diff_eq(other.position, eps)
            && self.rotation.abs_diff_eq(other.rotation, eps)
    }

    /// Transforms a placement with the given transformation matrix.
    /// The matrix should only encode translation and rotation.
    pub fn transform(&self, mat: Mat4) -> Self {
        let (_, rot, _) = mat.to_scale_rotation_translation();
        let position = mat.transform_point3(self.position);
        let rotation = rot * self.rotation;
        Self {
            part_index: self.part_index,
            position,
            rotation,
        }
    }

    /// Rotates a placement with the given quaternion.
    ///
    /// The rotation will be done in global space.
    pub fn rotate(&self, rot: Quat) -> Self {
        let position = rot.mul_vec3(self.position);
        let rotation = rot * self.rotation;
        Self {
            part_index: self.part_index,
            position,
            rotation,
        }
    }
}

/// Represents a connection to another part.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Copy, Clone)]
pub struct ConnectedTo {
    pub placement_idx: usize,
    pub conn_idx: usize,
}

/// Constrains the placements returned by the query.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone, Default)]
pub struct Query {
    /// If present, limits placements to only those that use this part.
    pub part_id: Option<usize>,
    /// If present, limits placements to only those that are attached to this placement.
    pub anchor_idx: Option<usize>,
    /// If true, returns only the first placement found.
    /// This is faster than requesting all placements.
    pub single: bool,
}

/// Represents an assembled model, consisting of placed parts.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone)]
pub struct AssembledModel {
    pub placements: Vec<Placement>,
}

/// Configuration options for an `AssemblyEngine`.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyEngineConfig {
    /// The number of turns to consider when generating candidates.
    pub num_candidate_turns: u32,
    /// The number of increments on a bar to consider when generating candidates.
    pub bar_increment_every: f32,
}

impl Default for AssemblyEngineConfig {
    fn default() -> Self {
        Self {
            num_candidate_turns: 4,
            bar_increment_every: 1.,
        }
    }
}

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Clone)]
pub struct EngineState {
    model: AssembledModel,
    connections: Vec<Vec<Vec<ConnectedTo>>>,
    cached_part_bboxes: Vec<Vec<OBB>>,
    placement_aabbs: Vec<AABB>,
    bounds: AABB,
    octree: Octree,
    obbs_to_index: Vec<OBB>,
}

pub struct AssemblyEngine {
    model: AssembledModel,
    parts: Vec<Part>,
    config: AssemblyEngineConfig,
    /// A table recording which model placements are connected to with other model placements.
    /// Index this as `connections[placement_idx][connector_idx][connection_idx]`.
    connections: Vec<Vec<Vec<ConnectedTo>>>,
    /// Bounding boxes of parts in world space. Index as `cached_part_bboxes[placement_idx][bbox_idx]`.
    cached_part_bboxes: Vec<Vec<OBB>>,
    /// Broad-phase AABBs, one for each placement.
    pub placement_aabbs: Vec<AABB>,
    /// The extents of the current model.
    bounds: AABB,
    /// Octree for fine-grained collisions.
    pub octree: Octree,
    /// These OBBs should be added to the octree before querying.
    pub obbs_to_index: Vec<OBB>,
}

/// Minimum distance between two vectors or quaternions to be considered not equal.
const EQ_EPS: f32 = 1e-3;

/// Minimum size a candidate batch must be before we perform filtering.
const MIN_CANDIDATE_BATCH_SIZE: usize = 1;

impl AssemblyEngine {
    /// Creates a new instance of the engine.
    pub fn new(parts: &[Part], config: &AssemblyEngineConfig) -> Self {
        Self {
            model: AssembledModel {
                placements: Vec::new(),
            },
            parts: parts.to_vec(),
            config: config.clone(),
            connections: Vec::new(),
            cached_part_bboxes: Vec::new(),
            placement_aabbs: Vec::new(),
            bounds: AABB {
                center: Vec3::ZERO,
                half_sizes: Vec3::ZERO,
            },
            octree: Octree::new(AABB {
                center: Vec3::ZERO,
                half_sizes: Vec3::splat(50.),
            }),
            obbs_to_index: Vec::new(),
        }
    }

    /// Clears the current model.
    pub fn clear(&mut self) {
        self.model.placements.clear();
        self.connections.clear();
        self.placement_aabbs.clear();
        self.cached_part_bboxes.clear();
        self.bounds = AABB {
            center: Vec3::ZERO,
            half_sizes: Vec3::ZERO,
        };
        self.octree = Octree::new(AABB {
            center: Vec3::ZERO,
            half_sizes: Vec3::splat(50.),
        });
        self.obbs_to_index.clear();
    }

    /// Returns the current model.
    pub fn get_model(&self) -> &AssembledModel {
        &self.model
    }

    /// Loads the given model.
    pub fn load_model(&mut self, model: &AssembledModel) {
        self.model = model.clone();
    }

    /// Returns the engine's parts.
    pub fn get_parts(&self) -> &[Part] {
        &self.parts
    }

    /// Adds a part onto the model.
    ///
    /// Note that the placement itself is *not* checked for validity, e.g. it's possible to pass in a placement that
    /// causes a bounding box intersection. Generally, you should only pass in placements produced by `query()`.
    pub fn add_placement(&mut self, placement: Placement) {
        let c_placement = placement;
        let c_xform = Mat4::from_rotation_translation(c_placement.rotation, c_placement.position);

        // Connect this placement to other placements in the model
        let mut c_connections: Vec<_> = self.parts[placement.part_index]
            .connectors
            .iter()
            .map(|_| Vec::new())
            .collect();
        let c_idx = self.model.placements.len();
        let c_part = &self.parts[placement.part_index];
        if !self.model.placements.is_empty() {
            for m_idx in 0..self.model.placements.len() {
                let m_placement = &self.model.placements[m_idx];
                let m_xform =
                    Mat4::from_rotation_translation(m_placement.rotation, m_placement.position);
                let m_part = &self.parts[m_placement.part_index];
                for (m_conn_idx, m_conn) in m_part.connectors.iter().enumerate() {
                    match m_conn {
                        Connector::SnapConn(m_conn) => {
                            for (c_conn_idx, c_conn) in c_part.connectors.iter().enumerate() {
                                if let Connector::SnapConn(c_conn) = c_conn {
                                    if m_conn.is_a == c_conn.is_a {
                                        continue;
                                    }
                                    let m_conn_world_pos =
                                        m_xform.transform_point3(m_conn.position);
                                    let c_conn_world_pos =
                                        c_xform.transform_point3(c_conn.position);
                                    if !m_conn_world_pos.abs_diff_eq(c_conn_world_pos, EQ_EPS) {
                                        continue;
                                    }
                                    if self.connections[m_idx][m_conn_idx].is_empty()
                                        && c_connections[c_conn_idx].is_empty()
                                    {
                                        self.connections[m_idx][m_conn_idx].push(ConnectedTo {
                                            placement_idx: c_idx,
                                            conn_idx: c_conn_idx,
                                        });
                                        c_connections[c_conn_idx].push(ConnectedTo {
                                            placement_idx: m_idx,
                                            conn_idx: m_conn_idx,
                                        });
                                    }
                                }
                            }
                        }
                        Connector::ClipConn(m_conn) => {
                            for (c_conn_idx, c_conn) in c_part.connectors.iter().enumerate() {
                                if let Connector::BarConn(c_conn) = c_conn {
                                    let m_conn_world_pos = Mat4::from_rotation_translation(
                                        m_placement.rotation,
                                        m_placement.position,
                                    )
                                    .transform_point3(m_conn.position);
                                    let c_conn_world_start = Mat4::from_rotation_translation(
                                        c_placement.rotation,
                                        c_placement.position,
                                    )
                                    .transform_point3(c_conn.start);
                                    let c_conn_world_dir = Mat4::from_rotation_translation(
                                        c_placement.rotation,
                                        c_placement.position,
                                    )
                                    .transform_vector3(c_conn.rotation.mul_vec3(Vec3::Y));
                                    if is_parallel(
                                        m_conn_world_pos - c_conn_world_start,
                                        c_conn_world_dir,
                                    ) && (m_conn_world_pos - c_conn_world_start).length()
                                        <= c_conn.length
                                        && self.connections[m_idx][m_conn_idx].is_empty()
                                    {
                                        self.connections[m_idx][m_conn_idx].push(ConnectedTo {
                                            placement_idx: c_idx,
                                            conn_idx: c_conn_idx,
                                        });
                                        c_connections[c_conn_idx].push(ConnectedTo {
                                            placement_idx: m_idx,
                                            conn_idx: m_conn_idx,
                                        });
                                    }
                                }
                            }
                        }
                        Connector::BarConn(m_conn) => {
                            for (c_conn_idx, c_conn) in c_part.connectors.iter().enumerate() {
                                if let Connector::ClipConn(c_conn) = c_conn {
                                    let c_conn_world_pos = Mat4::from_rotation_translation(
                                        c_placement.rotation,
                                        c_placement.position,
                                    )
                                    .transform_point3(c_conn.position);
                                    let m_conn_world_start = Mat4::from_rotation_translation(
                                        m_placement.rotation,
                                        m_placement.position,
                                    )
                                    .transform_point3(m_conn.start);
                                    let m_conn_world_dir = Mat4::from_rotation_translation(
                                        m_placement.rotation,
                                        m_placement.position,
                                    )
                                    .transform_vector3(m_conn.rotation.mul_vec3(Vec3::Y));
                                    if is_parallel(
                                        c_conn_world_pos - m_conn_world_start,
                                        m_conn_world_dir,
                                    ) && (c_conn_world_pos - m_conn_world_start).length()
                                        <= m_conn.length
                                        && c_connections[c_conn_idx].is_empty()
                                    {
                                        self.connections[m_idx][m_conn_idx].push(ConnectedTo {
                                            placement_idx: c_idx,
                                            conn_idx: c_conn_idx,
                                        });
                                        c_connections[c_conn_idx].push(ConnectedTo {
                                            placement_idx: m_idx,
                                            conn_idx: m_conn_idx,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add placement to the model
        assert!(c_placement.part_index < self.parts.len());
        self.model.placements.push(c_placement);
        self.connections.push(c_connections);

        // Cache placement's transformed bboxes
        let mut c_bboxes = Vec::new();
        let mut c_aabb: Option<AABB> = None;
        for bbox in &c_part.bboxes {
            let bbox_to_m_placement = Mat4::from_translation(bbox.center);
            let bbox_to_world = c_xform * bbox_to_m_placement;
            let (_, bbox_rot, bbox_pos) = bbox_to_world.to_scale_rotation_translation();
            let obb = OBB {
                center: bbox_pos,
                half_sizes: bbox.half_sizes,
                rotation: bbox_rot,
            };
            c_bboxes.push(obb);
            if let Some(c_aabb) = &mut c_aabb {
                *c_aabb = c_aabb.expand(&obb.get_bounds_aabb());
            } else {
                c_aabb = Some(obb.get_bounds_aabb());
            }
        }
        self.cached_part_bboxes.push(c_bboxes.clone());

        // Insert AABB for the entire placement + update bounds
        if let Some(c_aabb) = c_aabb {
            self.placement_aabbs.push(c_aabb);
            self.bounds = self.bounds.expand(&c_aabb);
        } else {
            self.placement_aabbs.push(AABB {
                center: c_placement.position,
                half_sizes: Vec3::ZERO,
            });
        }

        self.obbs_to_index.extend(&c_bboxes);
    }

    /// Performs a query against the current model.
    ///
    /// This returns all potential placements that fulfill `query`'s criteria.
    pub fn query(&mut self, query: &Query) -> Vec<Placement> {
        // If the model is empty, return an empty list
        if self.model.placements.is_empty() {
            return Vec::new();
        }

        // If need be, reindex the octree before checking bounding boxes
        if !self.obbs_to_index.is_empty() {
            if !self.octree.contains_aabb(&self.bounds) {
                let bboxes = self.octree.bboxes.clone();
                self.octree = Octree::new(AABB {
                    center: self.bounds.center,
                    half_sizes: self.bounds.half_sizes * 2.,
                });
                for bbox in bboxes {
                    self.octree.add(bbox);
                }
            }
            for c_obb in &self.obbs_to_index {
                self.octree.add(*c_obb);
            }
            self.obbs_to_index.clear();
        }

        // Restrict the number of candidate parts if specified
        let c_part_ids = if let Some(part_id) = query.part_id {
            vec![part_id]
        } else {
            (0..self.parts.len()).collect()
        };

        // Find candidate placements
        let mut c_placements: Vec<Placement> = Vec::new();
        let mut c_placements_batch: Vec<Placement> = Vec::new();
        for c_part_id in c_part_ids {
            let c_part = &self.parts[c_part_id];

            // Iterate over each placement in the model
            let model_placements = query
                .anchor_idx
                .map(|x| vec![self.model.placements[x]])
                .unwrap_or(self.model.placements.clone());
            for m_placement in &model_placements {
                let m_part = &self.parts[m_placement.part_index];
                let m_placement_to_world =
                    Mat4::from_rotation_translation(m_placement.rotation, m_placement.position);

                // Iterate over every model connection in the model placement
                for m_conn in &m_part.connectors {
                    match m_conn {
                        Connector::SnapConn(m_conn) => {
                            let m_conn_to_placement =
                                Mat4::from_rotation_translation(m_conn.rotation, m_conn.position);
                            let m_conn_to_world = m_placement_to_world * m_conn_to_placement;

                            for c_conn in &c_part.connectors {
                                if let Connector::SnapConn(c_conn) = c_conn {
                                    if c_conn.is_a == m_conn.is_a {
                                        continue;
                                    }
                                    let c_conn_to_c_part = Mat4::from_rotation_translation(
                                        c_conn.rotation,
                                        c_conn.position,
                                    );
                                    for rot_idx in 0..self.config.num_candidate_turns {
                                        let turn_quat = Quat::from_rotation_y(
                                            std::f32::consts::TAU * rot_idx as f32
                                                / self.config.num_candidate_turns as f32,
                                        );
                                        let c_part_to_world = m_conn_to_world
                                            * Mat4::from_quat(turn_quat)
                                            * c_conn_to_c_part.inverse();
                                        let (_, rotation, position) =
                                            c_part_to_world.to_scale_rotation_translation();
                                        c_placements_batch.push(Placement {
                                            part_index: c_part_id,
                                            position,
                                            rotation,
                                        })
                                    }
                                }
                            }
                        }
                        Connector::ClipConn(m_conn) => {
                            let m_conn_to_placement =
                                Mat4::from_rotation_translation(m_conn.rotation, m_conn.position);
                            let m_conn_to_world = m_placement_to_world * m_conn_to_placement;

                            for c_conn in &c_part.connectors {
                                if let Connector::BarConn(c_conn) = c_conn {
                                    let clippable_len = c_conn.length - CLIP_LENGTH;
                                    let bar_dir = c_conn.rotation.mul_vec3(-Vec3::Y);
                                    let mut connect_positions =
                                        vec![c_conn.start + bar_dir * clippable_len];
                                    let num_positions_on_bar =
                                        (clippable_len / self.config.bar_increment_every).floor()
                                            as u32;
                                    for i in 0..num_positions_on_bar {
                                        connect_positions.push(
                                            c_conn.start
                                                + bar_dir
                                                    * self.config.bar_increment_every
                                                    * i as f32,
                                        );
                                    }
                                    for conn_pos in connect_positions {
                                        let c_conn_to_c_part = Mat4::from_rotation_translation(
                                            c_conn.rotation,
                                            conn_pos,
                                        );
                                        for rot_idx in 0..self.config.num_candidate_turns {
                                            let turn_quat = Quat::from_rotation_y(
                                                std::f32::consts::TAU * rot_idx as f32
                                                    / self.config.num_candidate_turns as f32,
                                            );
                                            let c_part_to_world = m_conn_to_world
                                                * Mat4::from_quat(turn_quat)
                                                * c_conn_to_c_part.inverse();
                                            let (_, rotation, position) =
                                                c_part_to_world.to_scale_rotation_translation();
                                            c_placements_batch.push(Placement {
                                                part_index: c_part_id,
                                                position,
                                                rotation,
                                            })
                                        }
                                    }
                                }
                            }
                        }
                        Connector::BarConn(m_conn) => {
                            let clippable_len = m_conn.length - CLIP_LENGTH;
                            let bar_dir = m_conn.rotation.mul_vec3(-Vec3::Y);
                            let mut connect_positions = vec![
                                m_conn.start + m_conn.rotation.mul_vec3(-Vec3::Y) * clippable_len,
                            ];
                            let num_positions_on_bar =
                                (clippable_len / self.config.bar_increment_every).floor() as u32;
                            for i in 0..num_positions_on_bar {
                                connect_positions.push(
                                    m_conn.start
                                        + bar_dir * self.config.bar_increment_every * i as f32,
                                );
                            }
                            for conn_pos in connect_positions {
                                let m_conn_to_placement =
                                    Mat4::from_rotation_translation(m_conn.rotation, conn_pos);
                                let m_conn_to_world = m_placement_to_world * m_conn_to_placement;

                                for c_conn in &c_part.connectors {
                                    if let Connector::ClipConn(c_conn) = c_conn {
                                        let c_conn_to_c_part = Mat4::from_rotation_translation(
                                            c_conn.rotation,
                                            c_conn.position,
                                        );
                                        for rot_idx in 0..self.config.num_candidate_turns {
                                            let turn_quat = Quat::from_rotation_y(
                                                std::f32::consts::TAU * rot_idx as f32
                                                    / self.config.num_candidate_turns as f32,
                                            );
                                            let c_part_to_world = m_conn_to_world
                                                * Mat4::from_quat(turn_quat)
                                                * c_conn_to_c_part.inverse();
                                            let (_, rotation, position) =
                                                c_part_to_world.to_scale_rotation_translation();
                                            c_placements_batch.push(Placement {
                                                part_index: c_part_id,
                                                position,
                                                rotation,
                                            })
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Filter batch
                if c_placements_batch.len() >= MIN_CANDIDATE_BATCH_SIZE {
                    let filtered = self.filter_placements(&c_placements_batch, query);

                    if query.single && !filtered.is_empty() {
                        return vec![filtered[0]];
                    }

                    c_placements.extend(&filtered);
                    c_placements_batch.clear();
                }
            }
        }

        // Filter remaining items
        let filtered = self.filter_placements(&c_placements_batch, query);
        if query.single && !filtered.is_empty() {
            return vec![filtered[0]];
        }
        c_placements.extend(&filtered);

        // If there are no candidate placements, immediately return
        if c_placements.is_empty() {
            return Vec::new();
        }

        // Filter out redundant placements
        let mut idxs_to_remove: Vec<_> = c_placements.iter().map(|_| false).collect();
        for c_idx_1 in 0..(c_placements.len() - 1) {
            if idxs_to_remove[c_idx_1] {
                continue;
            }
            for c_idx_2 in (c_idx_1 + 1)..c_placements.len() {
                if idxs_to_remove[c_idx_2] {
                    continue;
                }
                let c1 = &c_placements[c_idx_1];
                let c2 = &c_placements[c_idx_2];
                if c1.part_index == c2.part_index
                    && c1.position.abs_diff_eq(c2.position, EQ_EPS)
                    && c1.rotation.abs_diff_eq(c2.rotation, EQ_EPS)
                {
                    idxs_to_remove[c_idx_2] = true;
                }
            }
        }

        // Filter out rotationally symmetric placements
        for c_idx_1 in 0..(c_placements.len() - 1) {
            if idxs_to_remove[c_idx_1] {
                continue;
            }
            for c_idx_2 in (c_idx_1 + 1)..c_placements.len() {
                if idxs_to_remove[c_idx_2] {
                    continue;
                }
                let c1 = &c_placements[c_idx_1];
                let c2 = &c_placements[c_idx_2];
                if c1.part_index != c2.part_index {
                    continue;
                }

                let part = &self.parts[c1.part_index];
                for rot_sym in &part.rot_syms {
                    // Check that the rotational origins and directions line up in world space
                    let c1_xform = Mat4::from_rotation_translation(c1.rotation, c1.position);
                    let c1_rot_sym_origin = c1_xform.transform_point3(rot_sym.origin);
                    let c1_rot_sym_dir = c1_xform.transform_vector3(rot_sym.dir);

                    let c2_xform = Mat4::from_rotation_translation(c2.rotation, c2.position);
                    let c2_rot_sym_origin = c2_xform.transform_point3(rot_sym.origin);
                    let c2_rot_sym_dir = c2_xform.transform_vector3(rot_sym.dir);

                    if c1_rot_sym_origin.abs_diff_eq(c2_rot_sym_origin, EQ_EPS)
                        && c1_rot_sym_dir.abs_diff_eq(c2_rot_sym_dir, EQ_EPS)
                    {
                        if rot_sym.orientations == 0 {
                            idxs_to_remove[c_idx_2] = true;
                        } else {
                            // Check that the two placements are symmetrically oriented
                            let fwd = if rot_sym.dir.x < 0.9 {
                                Vec3::X
                            } else {
                                Vec3::Z
                            };
                            let c1_rot_sym_fwd = c1_xform.transform_vector3(fwd);
                            let c2_rot_sym_fwd = c2_xform.transform_vector3(fwd);
                            let angle = c1_rot_sym_fwd.angle_between(c2_rot_sym_fwd); // Guaranteed to not be 0 due to filtering out identical placements
                            let angle_increment =
                                (std::f32::consts::TAU / angle).round().min(u8::MAX as f32) as u8; // Goes from 2 to 256
                            if rot_sym.orientations % angle_increment == 0 {
                                idxs_to_remove[c_idx_2] = true;
                            }
                        }
                    }
                }
            }
        }
        for (idx, should_remove) in idxs_to_remove.iter().enumerate().rev() {
            if *should_remove {
                c_placements.remove(idx);
            }
        }

        c_placements
    }

    /// Filters placements to remove intersections.
    /// The octree must be indexed before calling this.
    fn filter_placements(&self, c_placements: &[Placement], query: &Query) -> Vec<Placement> {
        if c_placements.is_empty() {
            return Vec::new();
        }

        // Filter out placements that intersect bounding boxes
        if query.single {
            if let Some(c_placement) = c_placements
                .iter()
                .find(|c_placement| self.should_keep_placement(c_placement))
                .copied()
            {
                vec![c_placement]
            } else {
                Vec::new()
            }
        } else {
            c_placements
                .iter()
                .filter(|c_placement| self.should_keep_placement(c_placement))
                .copied()
                .collect()
        }
    }

    /// Returns true if a placement doesn't intersect with other bounding boxes.
    fn should_keep_placement(&self, c_placement: &Placement) -> bool {
        let c_part = &self.parts[c_placement.part_index];
        let c_placement_to_world =
            Mat4::from_rotation_translation(c_placement.rotation, c_placement.position);

        // Compute AABB for this candidate + all constituent OBBs
        let mut c_placement_aabb: Option<AABB> = None;
        let mut c_obbs = Vec::new();
        for c_bbox in &c_part.bboxes {
            let bbox_to_c_placement = Mat4::from_translation(c_bbox.center);
            let bbox_to_world = c_placement_to_world * bbox_to_c_placement;
            let (_, c_bbox_rot, c_bbox_pos) = bbox_to_world.to_scale_rotation_translation();
            let c_obb = OBB {
                center: c_bbox_pos,
                half_sizes: c_bbox.half_sizes,
                rotation: c_bbox_rot,
            };
            c_obbs.push(c_obb);
            if let Some(c_placement_aabb) = &mut c_placement_aabb {
                *c_placement_aabb = c_placement_aabb.expand(&c_obb.get_bounds_aabb());
            } else {
                c_placement_aabb = Some(c_obb.get_bounds_aabb());
            }
        }
        for c_obb in c_obbs {
            if self.octree.intersects(&c_obb) {
                return false;
            }
        }
        true
    }

    /// Given a model, returns all possible ways the model can be attached to the current model.
    pub fn query_model(&mut self, model: &AssembledModel) -> Vec<Mat4> {
        // Save the current model
        let old_model = self.model.clone();
        self.clear();

        // Load the new model into the engine to identify used up connections
        for placement in &model.placements {
            self.add_placement(*placement);
        }

        // Convert the model into a part
        let mut bboxes = Vec::new();
        let mut connectors = Vec::new();
        for (placement_idx, placement) in model.placements.iter().enumerate() {
            let part = &self.parts[placement.part_index];
            // This ends up creating larger AABBs due to OBBs that may be rotated, but it's probably worth it for now
            bboxes.extend(part.bboxes.iter().map(|x| {
                OBB {
                    center: x.center + placement.position,
                    half_sizes: x.half_sizes,
                    rotation: placement.rotation,
                }
                .get_bounds_aabb()
            }));
            let part_xform =
                Mat4::from_rotation_translation(placement.rotation, placement.position);
            for (conn_idx, connector) in part.connectors.iter().enumerate() {
                // Don't add snap connectors that are used up
                if let Connector::SnapConn(_) = connector
                    && !self.connections[placement_idx][conn_idx].is_empty()
                {
                    continue;
                }

                // Don't add clip connectors that are used up
                if let Connector::ClipConn(_) = connector
                    && !self.connections[placement_idx][conn_idx].is_empty()
                {
                    continue;
                }

                connectors.push(connector.transform(part_xform));
            }
        }
        let model_part = Part {
            bboxes,
            connectors,
            rot_syms: Vec::new(),
        };

        // Add back original model
        self.clear();
        self.add_model(&old_model, Mat4::IDENTITY);

        // Temporarily add the part and query for candidates
        self.parts.push(model_part);
        let candidates = self.query(&Query {
            part_id: Some(self.parts.len() - 1),
            anchor_idx: None,
            single: false,
        });
        self.parts.pop().unwrap();

        candidates
            .iter()
            .map(|x| Mat4::from_rotation_translation(x.rotation, x.position))
            .collect()
    }

    /// Joins a whole model to the current model.
    pub fn add_model(&mut self, model: &AssembledModel, transform: Mat4) {
        for placement in &model.placements {
            let xformed_placement = Placement {
                part_index: placement.part_index,
                position: transform.transform_point3(placement.position),
                rotation: (transform * Mat4::from_quat(placement.rotation))
                    .to_scale_rotation_translation()
                    .1,
            };
            self.add_placement(xformed_placement);
        }
    }

    /// Returns edges between connected placements in the current model.
    /// Each edge is bidirectional and is not repeated (e.g. (0, 1) also represents (1, 0), and (1, 0) will not appear).
    pub fn get_conn_edges(&self) -> Vec<(usize, usize)> {
        let mut edges = HashSet::new();
        for place_idx in 0..self.model.placements.len() {
            for conn in self.connections[place_idx].iter().flatten() {
                let edge = (place_idx, conn.placement_idx);
                if !(edges.contains(&edge) || edges.contains(&(edge.1, edge.0))) {
                    edges.insert(edge);
                }
            }
        }
        Vec::from_iter(edges)
    }

    /// Returns info needed to reconstruct the current state of the engine.
    /// This does not factor in things like the configuration or parts.
    pub fn get_state(&self) -> EngineState {
        EngineState {
            model: self.model.clone(),
            connections: self.connections.clone(),
            cached_part_bboxes: self.cached_part_bboxes.clone(),
            placement_aabbs: self.placement_aabbs.clone(),
            bounds: self.bounds,
            octree: self.octree.clone(),
            obbs_to_index: self.obbs_to_index.clone(),
        }
    }

    /// Reconstructs the engine from a save state.
    pub fn load_state(&mut self, state: &EngineState) {
        self.model = state.model.clone();
        self.connections = state.connections.clone();
        self.cached_part_bboxes = state.cached_part_bboxes.clone();
        self.placement_aabbs = state.placement_aabbs.clone();
        self.bounds = state.bounds;
        self.octree = state.octree.clone();
        self.obbs_to_index = state.obbs_to_index.clone();
    }
}

/// Returns true if two vectors are parallel.
fn is_parallel(v1: Vec3, v2: Vec3) -> bool {
    let cos_sim = v1.normalize().dot(v2.normalize());
    cos_sim >= (1. - EQ_EPS) || cos_sim <= -(1. - EQ_EPS)
}

#[cfg(test)]
mod tests {
    use crate::connectors::SnapConn;

    use super::*;

    /// Epsilon used when checking if vectors and quaternions are equal.
    const EQ_EPS: f32 = 1e-3;

    fn create_1x1() -> Part {
        // TODO: The Z and Y axes in this part must be swapped
        Part {
            bboxes: vec![
                // Piece sides
                AABB {
                    center: Vec3::new(-2., 0., 3.),
                    half_sizes: Vec3::new(0.5, 2.5, 3.),
                },
                AABB {
                    center: Vec3::new(2., 0., 3.),
                    half_sizes: Vec3::new(0.5, 2.5, 3.),
                },
                AABB {
                    center: Vec3::new(0., -2., 3.),
                    half_sizes: Vec3::new(2.5, 0.5, 3.),
                },
                AABB {
                    center: Vec3::new(0., 2., 3.),
                    half_sizes: Vec3::new(2.5, 0.5, 3.),
                },
                // Top plate
                AABB {
                    center: Vec3::new(0., 0., 5.5),
                    half_sizes: Vec3::new(2.5, 2.5, 0.5),
                },
                // Top stud
                AABB {
                    center: Vec3::new(0., 0., 6.5),
                    half_sizes: Vec3::new(1., 1., 0.5),
                },
            ],
            connectors: vec![
                // Top stud
                Connector::SnapConn(SnapConn {
                    position: Vec3::new(0., 0., 6.),
                    rotation: Quat::IDENTITY,
                    is_a: true,
                }),
                // Bottom anti-stud
                Connector::SnapConn(SnapConn {
                    position: Vec3::new(0., 0., 0.),
                    rotation: Quat::IDENTITY,
                    is_a: false,
                }),
            ],
            rot_syms: vec![RotSym {
                origin: Vec3::ZERO,
                dir: Vec3::Z,
                orientations: 4,
            }],
        }
    }

    #[test]
    fn add_single() {
        // Create engine
        let parts = &[create_1x1()];
        let mut engine = AssemblyEngine::new(parts, &AssemblyEngineConfig::default());

        // Add placement
        let placement = Placement {
            part_index: 0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        engine.add_placement(placement);

        // Check that part is there
        let model = engine.get_model();
        assert_eq!(model.placements.len(), 1);
        let new_placement = model.placements[0];
        assert_eq!(placement.part_index, new_placement.part_index);
        assert!(
            placement
                .position
                .abs_diff_eq(new_placement.position, EQ_EPS)
        );
        assert!(
            placement
                .rotation
                .abs_diff_eq(new_placement.rotation, EQ_EPS)
        );
    }

    #[test]
    fn add_to_existing() {
        // Create engine
        let parts = &[create_1x1()];
        let mut engine = AssemblyEngine::new(parts, &AssemblyEngineConfig::default());

        // Add placements
        let placement = Placement {
            part_index: 0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        engine.add_placement(placement);
        let candidates = engine.query(&Query::default());
        let placement = candidates[0];
        engine.add_placement(placement);

        // Check that new part is there
        let model = engine.get_model();
        assert_eq!(model.placements.len(), 2);
        let new_placement = model.placements[1];
        assert_eq!(placement.part_index, new_placement.part_index);
        assert!(
            placement
                .position
                .abs_diff_eq(new_placement.position, EQ_EPS)
        );
        assert!(
            placement
                .rotation
                .abs_diff_eq(new_placement.rotation, EQ_EPS)
        );
    }

    #[test]
    fn query_snap() {
        // Create engine
        let parts = &[create_1x1()];
        let mut engine = AssemblyEngine::new(parts, &AssemblyEngineConfig::default());

        // Add placement
        let placement = Placement {
            part_index: 0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        engine.add_placement(placement);

        // Check that placements are correct
        let placements = engine.query(&Query {
            part_id: Some(0),
            ..Default::default()
        });
        assert_eq!(placements.len(), 2);
        let expected = [
            Placement {
                part_index: 0,
                position: Vec3::new(0., 0., -6.),
                rotation: Quat::IDENTITY,
            },
            Placement {
                part_index: 0,
                position: Vec3::new(0., 0., 6.),
                rotation: Quat::IDENTITY,
            },
        ];
        for p in expected {
            assert!(placements.iter().any(|x| x.approx_eq(&p, EQ_EPS)))
        }
    }
}
