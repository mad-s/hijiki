#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

use std::ops::{Index,IndexMut};
macro_rules! index_impl {
    ($t:ty: $($ix:expr => $fname:ident),*; $len:expr) => {
        impl Index<usize> for $t {
            type Output = f32;
            fn index(&self, ix: usize) -> &f32 {
                match ix {
                    $(
                        $ix => &self.$fname,
                    )*
                    _ => panic!("index {} ouf of bounds for {}", ix, stringify!($t))
                }
            }
        }
        impl IndexMut<usize> for $t {
            fn index_mut(&mut self, ix: usize) -> &mut f32 {
                match ix {
                    $(
                        $ix => &mut self.$fname,
                    )*
                    _ => panic!("index {} ouf of bounds for {}", ix, stringify!($t))
                }
            }
        }

        impl $t {
            pub fn from_array(data: [f32; $len]) -> Self {
                Self {
                    $(
                        $fname: data[$ix]
                    ),*
                }
            }
        }
    }
}

index_impl!(Vec2: 0 => x, 1 => y; 2);
index_impl!(Vec3: 0 => x, 1 => y, 2 => z; 3);
index_impl!(Vec4: 0 => x, 1 => y, 2 => z, 3 => w; 4);

use std::ops::{Add,Sub,Mul,Div};
use std::cmp::{PartialOrd,Ordering,Ordering::*};
macro_rules! arith_impl {
    ($t: ty, $($fname:ident),*) => {
        impl $t {
            fn new($($fname: f32),*) -> Self {
                Self {
                    $($fname),*
                }
            }
        }
        impl Add for $t {
            type Output = $t;
            fn add(self, other: $t) -> $t {
                <$t>::new(
                    $(
                        self.$fname + other.$fname
                    ),*
                )
            }
        }
        impl Sub for $t {
            type Output = $t;
            fn sub(self, other: $t) -> $t {
                <$t>::new(
                    $(
                        self.$fname - other.$fname
                    ),*
                )
            }
        }
        impl Mul for $t {
            type Output = $t;
            fn mul(self, other: $t) -> $t {
                <$t>::new(
                    $(
                        self.$fname * other.$fname
                    ),*
                )
            }
        }

        // vector * scalar
        impl Mul<f32> for $t {
            type Output = $t;
            fn mul(self, fac: f32) -> $t {
                <$t>::new(
                    $(
                        self.$fname * fac
                    ),*
                )
            }
        }
        // scalar * vector
        impl Mul<$t> for f32 {
            type Output = $t;
            fn mul(self, vec: $t) -> $t {
                <$t>::new(
                    $(
                        self * vec.$fname
                    ),*
                )
            }
        }

        // vector / scalar
        impl Div<f32> for $t {
            type Output = $t;
            fn div(self, fac: f32) -> $t {
                <$t>::new(
                    $(
                        self.$fname / fac
                    ),*
                )
            }
        }

        impl PartialOrd for $t {
            fn partial_cmp(&self, other: &$t) -> Option<Ordering> {
                if self == other {
                    return Some(Equal)
                }
                if true $( && self.$fname < other.$fname)* {
                    return Some(Less)
                }
                if true $( && self.$fname > other.$fname)* {
                    return Some(Greater)
                }
                None
            }
        }
    }
}

arith_impl!(Vec2, x, y);
arith_impl!(Vec3, x, y, z);
arith_impl!(Vec4, x, y, z, w);


pub trait VectorSpace: Add + Mul<f32, Output=Self> + Div<f32, Output=Self> + Copy {
    fn dot(self, other: Self) -> f32;
    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn normalized(self) -> Self {
        self / self.length()
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

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Mat4 {
    pub data: [f32; 16], // column major
}
