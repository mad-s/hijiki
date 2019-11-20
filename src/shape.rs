use crate::glm::*;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub position_radius: Vec4,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal_offset: Vec4,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub origin: Vec3,
    pub edge1: Vec3,
    pub edge2: Vec3,
}
