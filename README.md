# Hijiki

Hijiki is another physically-based renderer named after edible seaweed

It runs on GPU using [wgpu](https://github.com/gfx-rs/wgpu-rs) and GLSL compute
shaders.

![](https://user-images.githubusercontent.com/13462849/79961504-1cfcc380-8487-11ea-816d-7ece9acd5ccc.png)

## Features

 - high performance, outperforms a similar CPU renderer ([https://wjakob.github.io/nori/](https://wjakob.github.io/nori/)) even on integrated graphics
 - denoising using a bilateral filter (with feature buffers)
 - supports triangle meshes and analytic spheres
 - diffuse, mirror, dielectric and emissive materials
 - supports textures
