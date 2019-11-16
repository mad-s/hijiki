#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
    x: f32,
    y: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Vec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

pub trait VectorSpace: Copy {
    fn dot(self, other: Self) -> f32;
    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
}

impl VectorSpace for Vec2 {
    fn dot(self, other: Self) -> f32 {
        self.x*other.x + self.y*other.y
    }
}
impl VectorSpace for Vec3 {
    fn dot(self, other: Self) -> f32 {
        self.x*other.x + self.y*other.y + self.z*other.z
    }
}
impl VectorSpace for Vec4 {
    fn dot(self, other: Self) -> f32 {
        self.x*other.x + self.y*other.y + self.z*other.z + self.w*other.w
    }
}

pub fn dot<T: VectorSpace>(a: T, b: T) -> f32 {
    a.dot(b)
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.y*b.z-a.z*b.y,
        y: a.z*b.x-a.x*b.z,
        z: a.x*b.y-a.y*b.x,
    }
}

pub fn vec2(x: f32, y: f32) -> Vec2 {
    Vec2 {
        x, y
    }
}

pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 {
        x, y, z
    }
}

pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4 {
        x, y, z, w
    }
}
