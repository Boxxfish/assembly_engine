use glam::{Mat4, Quat, Vec3, Vec4Swizzles};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// An axis-aligned bounding box.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct AABB {
    pub center: Vec3,
    pub half_sizes: Vec3,
}

impl AABB {
    /// An AABB at (0, 0, 0), with size (0, 0, 0).
    pub const ZERO: AABB = AABB {
        center: Vec3::ZERO,
        half_sizes: Vec3::ZERO,
    };

    /// Returns a new AABB containing both AABBs.
    pub fn expand(&self, other: &Self) -> Self {
        let min_aabb1 = self.center - self.half_sizes;
        let max_aabb1 = self.center + self.half_sizes;
        let min_aabb2 = other.center - other.half_sizes;
        let max_aabb2 = other.center + other.half_sizes;
        let min = min_aabb1.min(min_aabb2);
        let max = max_aabb1.max(max_aabb2);
        let half_sizes = (max - min) / 2.;
        let center = min + half_sizes;
        Self { center, half_sizes }
    }

    /// Returns `true` if these two AABBs intersect.
    pub fn intersects(&self, other: &Self) -> bool {
        let dist = (self.center - other.center).abs();
        let min_dist = self.half_sizes + other.half_sizes;
        dist.x <= min_dist.x && dist.y <= min_dist.y && dist.z <= min_dist.z
    }
    /// Returns true if the provided AABB is *completely* enclosed within this one.
    pub fn contains(&self, other: &Self) -> bool {
        let min_self = self.center - self.half_sizes;
        let max_self = self.center + self.half_sizes;
        let min_other = other.center - other.half_sizes;
        let max_other = other.center + other.half_sizes;
        min_self.cmplt(min_other).all() && max_self.cmpgt(max_other).all()
    }

    /// Returns the volume of the AABB.
    pub fn volume(&self) -> f32 {
        (self.half_sizes * 2.).length_squared()
    }

    /// Converts this AABB to an OBB.
    pub fn to_obb(&self) -> OBB {
        OBB {
            center: self.center,
            half_sizes: self.half_sizes,
            rotation: Quat::IDENTITY,
        }
    }
}

/// A oriented bounding box.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct OBB {
    pub center: Vec3,
    pub half_sizes: Vec3,
    pub rotation: Quat,
}

impl OBB {
    /// Returns `true` if these two OBBs intersect.
    pub fn intersects(&self, other: &Self) -> bool {
        // We use the SAT intersection test.

        // Get axes to check
        let self_rot_mat = Mat4::from_quat(self.rotation);
        let self_faces = [
            self_rot_mat.x_axis.xyz().normalize(),
            self_rot_mat.y_axis.xyz().normalize(),
            self_rot_mat.z_axis.xyz().normalize(),
        ];
        let other_rot_mat = Mat4::from_quat(other.rotation);
        let other_faces = [
            other_rot_mat.x_axis.xyz().normalize(),
            other_rot_mat.y_axis.xyz().normalize(),
            other_rot_mat.z_axis.xyz().normalize(),
        ];

        let mut axes = [self_faces, other_faces].concat();
        for face1 in self_faces {
            for face2 in other_faces {
                let axis = face1.cross(face2);
                if axis.length_squared() < 1e-6 {
                    continue;
                }
                axes.push(axis);
            }
        }

        // Get bounding box coords
        let self_bbox_coords = self.get_corners();
        let other_bbox_coords = other.get_corners();

        // Check if there is a separation
        for axis in axes {
            let self_points = self_bbox_coords.map(|x| OrderedFloat(x.dot(axis)));
            let self_min = self_points.iter().min().unwrap().0;
            let self_max = self_points.iter().max().unwrap().0;
            let self_half_size = (self_max - self_min) / 2.;
            let self_center = self_min + self_half_size;

            let other_points = other_bbox_coords.map(|x| OrderedFloat(x.dot(axis)));
            let other_min = other_points.iter().min().unwrap().0;
            let other_max = other_points.iter().max().unwrap().0;
            let other_half_size = (other_max - other_min) / 2.;
            let other_center = other_min + other_half_size;

            if (self_center - other_center).abs() > (self_half_size + other_half_size) {
                return false;
            }
        }

        true
    }

    /// Returns the coordinates of each corner point.
    pub fn get_corners(&self) -> [Vec3; 8] {
        let xform = Mat4::from_rotation_translation(self.rotation, self.center);
        [
            Vec3::new(-1., -1., -1.),
            Vec3::new(1., -1., -1.),
            Vec3::new(-1., 1., -1.),
            Vec3::new(1., 1., -1.),
            Vec3::new(-1., -1., 1.),
            Vec3::new(1., -1., 1.),
            Vec3::new(-1., 1., 1.),
            Vec3::new(1., 1., 1.),
        ]
        .map(|x| xform.transform_point3(x * self.half_sizes))
    }

    /// Returns the smallest AABB that contains this OBB.
    pub fn get_bounds_aabb(&self) -> AABB {
        let points = self.get_corners();
        let min_x = points.iter().min_by_key(|x| OrderedFloat(x.x)).unwrap().x;
        let min_y = points.iter().min_by_key(|x| OrderedFloat(x.y)).unwrap().y;
        let min_z = points.iter().min_by_key(|x| OrderedFloat(x.z)).unwrap().z;
        let max_x = points.iter().max_by_key(|x| OrderedFloat(x.x)).unwrap().x;
        let max_y = points.iter().max_by_key(|x| OrderedFloat(x.y)).unwrap().y;
        let max_z = points.iter().max_by_key(|x| OrderedFloat(x.z)).unwrap().z;
        let half_sizes = Vec3::new(
            (max_x - min_x) / 2.,
            (max_y - min_y) / 2.,
            (max_z - min_z) / 2.,
        );
        let center = Vec3::new(
            min_x + half_sizes.x,
            min_y + half_sizes.y,
            min_z + half_sizes.z,
        );
        AABB { center, half_sizes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_obb_intersect_center() {
        // Check that two OOBs in the same position intersect

        let center = Vec3::new(10., 4., -5.);
        let bbox1 = OBB {
            center,
            half_sizes: Vec3::new(4., 5., 2.),
            rotation: Quat::from_axis_angle(
                Vec3::new(2., 4., 3.).normalize(),
                std::f32::consts::PI * 1.2,
            ),
        };
        let bbox2 = OBB {
            center,
            half_sizes: Vec3::new(5., 2., 1.),
            rotation: Quat::from_axis_angle(Vec3::new(1., 6., 4.), std::f32::consts::PI * 0.3),
        };
        assert!(bbox1.intersects(&bbox2));
        assert!(bbox2.intersects(&bbox1));
    }

    #[test]
    fn check_obb_intersect_face() {
        // Check that bounding boxes positioned next to each other don't intersect

        let center = Vec3::new(10., 4., -5.);
        let bbox1 = OBB {
            center,
            half_sizes: Vec3::new(4., 5., 2.),
            rotation: Quat::IDENTITY,
        };

        let bbox2 = OBB {
            center,
            half_sizes: Vec3::new(5., 2., 1.),
            rotation: Quat::IDENTITY,
        };

        let in_offset = 0.2;

        let added_half_sizes = bbox1.half_sizes + bbox2.half_sizes;
        for dim_idx in 0..3 {
            for sign in [1., -1.] {
                // Check that moving a bbox inside is detected as an intersection
                let mut offset = Vec3::ZERO;
                offset[dim_idx] = sign * (added_half_sizes[dim_idx] - in_offset);
                let mut bbox2_copy = bbox2;
                bbox2_copy.center += offset;
                assert!(bbox1.intersects(&bbox2_copy));

                // Check that moving a bbox outside isn't detected as an intersection
                let mut offset = Vec3::ZERO;
                offset[dim_idx] = sign * (added_half_sizes[dim_idx] + in_offset);
                let mut bbox2_copy = bbox2;
                bbox2_copy.center += offset;
                assert!(!bbox1.intersects(&bbox2_copy));
            }
        }
    }
}
