use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// A snap connector.
///
/// Two snap connectors can connect to each other if their `is_a` fields are not equal.
/// As a matter of convention, "plugs" should be type "A", and "sockets" should be type B.
///
/// Connectors must also lie on the same exact plane, i.e. their Y axes should be parallel.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SnapConn {
    pub position: Vec3,
    pub rotation: Quat,
    pub is_a: bool,
}

/// A clip connector.
///
/// A clip can connect to a single bar.
///
/// Connectors must lie on the same plane, but their Y axes can either be parallel or anti-parallel.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct ClipConn {
    pub position: Vec3,
    pub rotation: Quat,
}

pub const CLIP_LENGTH: f32 = 8.;

/// A bar connector.
///
/// A single bar can connect to multiple clip connectors.
///
/// Connectors must lie on the same plane, but their Y axes can either be parallel or anti-parallel.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct BarConn {
    pub start: Vec3,
    pub length: f32,
    pub rotation: Quat,
}

/// A connector for a part.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Connector {
    SnapConn(SnapConn),
    ClipConn(ClipConn),
    BarConn(BarConn),
}

impl Connector {
    /// Returns a transformed version of the connector.
    pub fn transform(&self, transform: Mat4) -> Self {
        match self {
            Connector::SnapConn(snap_conn) => Self::SnapConn(SnapConn {
                position: transform.transform_point3(snap_conn.position),
                rotation: (transform * Mat4::from_quat(snap_conn.rotation))
                    .to_scale_rotation_translation()
                    .1,
                is_a: snap_conn.is_a,
            }),
            Connector::ClipConn(clip_conn) => Self::ClipConn(ClipConn {
                position: transform.transform_point3(clip_conn.position),
                rotation: (transform * Mat4::from_quat(clip_conn.rotation))
                    .to_scale_rotation_translation()
                    .1,
            }),
            Connector::BarConn(bar_conn) => Self::BarConn(BarConn {
                start: transform.transform_point3(bar_conn.start),
                length: bar_conn.length,
                rotation: (transform * Mat4::from_quat(bar_conn.rotation))
                    .to_scale_rotation_translation()
                    .1,
            }),
        }
    }
}
