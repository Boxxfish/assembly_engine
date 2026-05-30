use glam::Vec3;

use crate::bbox::{AABB, OBB};

/// An octree implementation.
///
/// Currently, this only supports checking if an item intersects with *any* item in the tree.
#[derive(Debug, Clone)]
pub struct Octree {
    pub root: OctreeNode,
    pub bboxes: Vec<OBB>,
}

impl Octree {
    pub fn new(volume: AABB) -> Self {
        let root = OctreeNode::new(volume);
        Self {
            root,
            bboxes: Vec::new(),
        }
    }

    pub fn add(&mut self, new_bbox: OBB) {
        let new_aabb = new_bbox.get_bounds_aabb();
        self.root.add(new_bbox, new_aabb);
        self.bboxes.push(new_bbox);
    }

    pub fn intersects(&self, new_bbox: &OBB) -> bool {
        let new_aabb = new_bbox.get_bounds_aabb();
        self.root.intersects(new_bbox, &new_aabb)
    }

    pub fn contains_aabb(&self, new_bbox: &AABB) -> bool {
        self.root.volume.contains(new_bbox)
    }
}

const MAX_ITEMS_PER_NODE: usize = 10;
const MIN_PARENT_VOLUME: f32 = 1.;

#[derive(Debug, Clone)]
pub struct OctreeNode {
    pub volume: AABB,
    pub children: Option<Vec<OctreeNode>>,
    /// Only contains items if `children` is `None`.
    pub items: Vec<OBB>,
}

impl OctreeNode {
    pub fn new(volume: AABB) -> Self {
        Self {
            volume,
            children: None,
            items: Vec::new(),
        }
    }

    pub fn add(&mut self, new_bbox: OBB, new_aabb: AABB) {
        if let Some(children) = &mut self.children {
            // If the new item intersects a child, add it to it
            for child in children {
                if child.volume.intersects(&new_aabb) {
                    child.add(new_bbox, new_aabb);
                }
            }
        } else {
            self.items.push(new_bbox);
            if self.items.len() + 1 == MAX_ITEMS_PER_NODE
                && self.volume.volume() > MIN_PARENT_VOLUME
            {
                // Subdivide this node
                let p_center = self.volume.center;
                let half_sizes = self.volume.half_sizes / 2.;
                let mut children = [
                    AABB {
                        center: p_center + Vec3::new(half_sizes.x, half_sizes.y, half_sizes.z),
                        half_sizes,
                    },
                    AABB {
                        center: p_center + Vec3::new(-half_sizes.x, half_sizes.y, half_sizes.z),
                        half_sizes,
                    },
                    AABB {
                        center: p_center + Vec3::new(half_sizes.x, -half_sizes.y, half_sizes.z),
                        half_sizes,
                    },
                    AABB {
                        center: p_center + Vec3::new(-half_sizes.x, -half_sizes.y, half_sizes.z),
                        half_sizes,
                    },
                    AABB {
                        center: p_center + Vec3::new(half_sizes.x, half_sizes.y, -half_sizes.z),
                        half_sizes,
                    },
                    AABB {
                        center: p_center + Vec3::new(-half_sizes.x, half_sizes.y, -half_sizes.z),
                        half_sizes,
                    },
                    AABB {
                        center: p_center + Vec3::new(half_sizes.x, -half_sizes.y, -half_sizes.z),
                        half_sizes,
                    },
                    AABB {
                        center: p_center + Vec3::new(-half_sizes.x, -half_sizes.y, -half_sizes.z),
                        half_sizes,
                    },
                ]
                .map(OctreeNode::new);
                for item in &self.items {
                    let item_aabb = item.get_bounds_aabb();
                    for child in &mut children {
                        if child.volume.intersects(&item_aabb) {
                            child.add(*item, item_aabb);
                        }
                    }
                }
                self.children = Some(children.to_vec());
                self.items = Vec::new();
            }
        }
    }

    pub fn intersects(&self, new_bbox: &OBB, new_aabb: &AABB) -> bool {
        if !self.volume.intersects(new_aabb) {
            return false;
        }
        if let Some(children) = &self.children {
            for child in children {
                if child.intersects(new_bbox, new_aabb) {
                    return true;
                }
            }
            false
        } else {
            for item in &self.items {
                // Check against course-grained AABB first
                if item.get_bounds_aabb().intersects(new_aabb) && item.intersects(new_bbox) {
                    return true;
                }
            }
            false
        }
    }
}
