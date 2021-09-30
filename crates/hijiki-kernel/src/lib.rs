#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::glam::*;

use hijiki_render::{
    add,
    Scene, Ray,
};


#[spirv(compute(threads(32, 32)))]
pub fn compute(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scene: &Scene,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rays: &[Ray],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [u32],
) {
    output[0] = add(scene.dummy, rays[0].dummy);

}