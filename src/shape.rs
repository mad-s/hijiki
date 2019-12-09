//use crate::glm::*;
use bvh::aabb::Bounded;
use bvh::aabb::AABB;
use bvh::nalgebra::{Vector3,Vector4,Point3};

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub position: Point3<f32>,
    pub radius: f32,
}

impl Bounded for Sphere {
    fn aabb(&self) -> AABB {
        return AABB::with_bounds(
            self.position - Vector3::repeat(self.radius),
            self.position + Vector3::repeat(self.radius),
        );
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub origin: Point3<f32>,
    pad1: f32,
    pub edge1: Vector3<f32>,
    pad2: f32,
    pub edge2: Vector3<f32>,
    pad3: f32,
}

impl Quad {
    pub fn new(origin: Point3<f32>, edge1: Vector3<f32>, edge2: Vector3<f32>) -> Self {
        Quad {
            origin,
            pad1: 0.,
            edge1,
            pad2: 0.,
            edge2,
            pad3: 0.,
        }
    }
}

impl Bounded for Quad {
    fn aabb(&self) -> AABB {
        AABB::empty()
            .grow(&self.origin)
            .grow(&(self.origin+self.edge1))
            .grow(&(self.origin+self.edge2))
            .grow(&(self.origin+self.edge1+self.edge2))
    }
}
