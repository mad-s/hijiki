extern crate wgpu;

extern crate shaderc;

extern crate rand;

extern crate winit;

extern crate tobj;

extern crate bvh;
use bvh::aabb::Bounded;
use bvh::aabb::AABB;
use bvh::nalgebra::Vector3;
use bvh::nalgebra::Point3;

use std::borrow::Borrow;

extern crate strum;
use strum::*;
#[macro_use]
extern crate strum_macros;

#[macro_use]
extern crate structopt;
use structopt::StructOpt;

mod glm;
use glm::*;

mod shape;
use shape::*;

#[derive(Debug, EnumDiscriminants)]
#[strum_discriminants(name(MaterialType))]
#[strum_discriminants(repr(u8))]
#[strum_discriminants(derive(EnumIter,AsRefStr))]
enum Material {
    Diffuse(DiffuseMaterial),
    DiffuseCBoard(DiffuseCheckerboardMaterial),
    Mirror(MirrorMaterial),
    Dielectric(DielectricMaterial),
    Emissive(EmitterMaterial),
}
const MATERIAL_TAG_SHIFT : u32 = 24; // top 8 bits are tag

#[derive(Debug,Clone,Copy)]
enum Shape {
    Sphere(Sphere),
    Quad(Quad),
    Triangle([u32;3]),
}

#[derive(Debug,Clone,Copy)]
struct Vertex {
    pos: Point3<f32>,
    u: f32,
    normal: Vector3<f32>,
    v: f32,
}


struct BVHShape<'a> {
    scene: &'a Scene,
    shape: Shape,
    node_index: usize,
}

impl<'a> bvh::aabb::Bounded for BVHShape<'a> {
    fn aabb(&self) -> AABB {
        match self.shape {
            Shape::Sphere(sphere) => sphere.aabb(),
            Shape::Quad(quad)     => quad.aabb(),
            Shape::Triangle([a,b,c])  => {
                AABB::empty()
                    .grow(&self.scene.vertices[a as usize].pos)
                    .grow(&self.scene.vertices[b as usize].pos)
                    .grow(&self.scene.vertices[c as usize].pos)
            }
        }
    }
}
impl<'a> bvh::bounding_hierarchy::BHShape for BVHShape<'a> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[repr(C,align(16))]
#[derive(Debug,Clone,Copy)]
struct CompiledBVHNode {
    aabb_min: Point3<f32>,
    shape_index: u32,
    aabb_max: Point3<f32>,
    exit_index: u32,
}


#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct DiffuseMaterial {
    color: Vec3,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct DiffuseCheckerboardMaterial {
    color1: Vector3<f32>,
    scale_u: f32,
    color2: Vector3<f32>,
    scale_v: f32
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct MirrorMaterial {
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct DielectricMaterial {
    extinction_eta: Vec4,
}

impl DielectricMaterial {
    fn clear(eta_ratio: f32) -> DielectricMaterial {
        DielectricMaterial {
            extinction_eta: vec4(0., 0., 0., eta_ratio),
        }
    }

    fn tinted(extinction: Vec3, eta_ratio: f32) -> DielectricMaterial {
        DielectricMaterial {
            extinction_eta: vec4(extinction[0], extinction[1], extinction[2], eta_ratio),
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct EmitterMaterial {
    power: Vec3,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct PortalMaterial {
    transform: Mat4,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct Camera {
    position: Vec4,
    rotation: Vec4,
    fov: f32,
}

#[derive(Debug)]
struct Scene {
    camera: Camera,

    objects: Vec<(Shape, usize)>,
    vertices: Vec<Vertex>,

    materials: Vec<Material>,
}

impl Scene {
    fn compile(self) -> CompiledScene {
        let mut spheres = vec![];
        let mut quads   = vec![];
        let mut triangles = vec![];
        let mut bvh_shapes = vec![];
        let mut shape_indices = vec![];

        for (shape, material) in self.objects.iter() {
            match shape {
                Shape::Sphere(sphere) => {
                    shape_indices.push(spheres.len());
                    spheres.push((sphere, *material))
                },
                Shape::Quad(quad)     => {
                    shape_indices.push(quads.len());
                    quads.push((quad,   *material))
                },
                Shape::Triangle(tri)     => {
                    shape_indices.push(triangles.len());
                    triangles.push((tri,   *material))
                },
            }
            bvh_shapes.push(BVHShape{shape: *shape, node_index: 0, scene: &self});
        }

        println!("Building BVH");
        let bvh = bvh::bvh::BVH::build(&mut bvh_shapes[..]);
        println!("Built BVH with {} nodes", bvh.nodes.len());
        //bvh.pretty_print();

        let mut indices = vec![usize::max_value(); bvh.nodes.len()];
        fn calculate_indices(current: usize, nodes: &[bvh::bvh::BVHNode], indices: &mut [usize], current_index: &mut usize) {
            let node = nodes[current];
            indices[current] = *current_index;
            *current_index += 1;
            if node.shape_index().is_none() {
                calculate_indices(node.child_l(), nodes, indices, current_index);
                calculate_indices(node.child_r(), nodes, indices, current_index);
            }
        }
        calculate_indices(0, &bvh.nodes, &mut indices, &mut 0);
        fn flatten_bvh(current: usize, aabb: AABB, skip_index: usize, nodes: &[bvh::bvh::BVHNode], flat: &mut Vec<CompiledBVHNode>, indices: &[usize]) {
            let node = nodes[current];
            let shape = node.shape_index();
            flat.push(CompiledBVHNode {
                aabb_min: aabb.min,
                shape_index: shape.map(|x| x as u32).unwrap_or(u32::max_value()),
                aabb_max: aabb.max,
                exit_index: skip_index as u32,
            });
            if shape.is_none() { // interior node, descend to children
                flatten_bvh(node.child_l(), node.child_l_aabb(), indices[node.child_r()], nodes, flat, indices);
                flatten_bvh(node.child_r(), node.child_r_aabb(), skip_index,              nodes, flat, indices);
            }
        };

        let mut flat = vec![];
        let root_aabb = bvh.nodes[0].child_l_aabb().join(&bvh.nodes[0].child_r_aabb());
        flatten_bvh(0, root_aabb, 1000000, &bvh.nodes, &mut flat, &indices[..]);
        // transform shape indices
        for node in &mut flat[..] {
            if node.shape_index == u32::max_value() {
                continue;
            }
            let offset = match bvh_shapes[node.shape_index as usize].shape {
                Shape::Sphere(_) => 0,
                Shape::Quad(_) => spheres.len(),
                Shape::Triangle(_) => spheres.len() + quads.len(),
            };
            node.shape_index = (shape_indices[node.shape_index as usize] + offset) as u32;
        }
        let bvh = flat;

        let mut diffuse    = vec![];
        let mut diffusecb  = vec![];
        let mut dielectric = vec![];
        let mut emissive   = vec![];

        let mut material_reprs : Vec<u32> = vec![];
        for mat in self.materials.into_iter() {
            let tag = MaterialType::from(&mat) as u8;
            let ix = match mat {
                Material::Diffuse(x) => {
                    diffuse.push(x);
                    diffuse.len()-1
                },
                Material::DiffuseCBoard(x) => {
                    diffusecb.push(x);
                    diffusecb.len()-1
                }
                Material::Mirror(MirrorMaterial{}) => {
                    0
                }, // no data needed
                Material::Dielectric(x) => {
                    dielectric.push(x);
                    dielectric.len()-1
                },
                Material::Emissive(x) => {
                    emissive.push(x);
                    emissive.len()-1
                }
            };
            material_reprs.push(((tag as u32) << MATERIAL_TAG_SHIFT) + ix as u32);
        }

        let mut materials = vec![];
        for &(_, mat) in &spheres {
            materials.push(material_reprs[mat]);
        }
        for &(_, mat) in &quads {
            materials.push(material_reprs[mat]);
        }
        for &(_, mat) in &triangles {
            materials.push(material_reprs[mat]);
        }

        let mut emitters = vec![];
        for (ix, mat) in materials.iter().enumerate() {
            if mat >> MATERIAL_TAG_SHIFT == MaterialType::Emissive as u8 as u32 {
                emitters.push(Emitter {
                    shape: ix as u32,
                    pdf: 0.,
                    cdf: 0.,
                    pad: 0.,
                })
            }
        };
        let num_emitters = emitters.len();
        let emitter_pdf = 1.0 / num_emitters as f32;
        let mut cdf = 0.;
        for emitter in &mut emitters {
            cdf += emitter_pdf;
            emitter.pdf = emitter_pdf;
            emitter.cdf = cdf;
        }

        let spheres = spheres.into_iter().map(|(x,_)| *x).collect::<Vec<_>>();
        let quads = quads.into_iter().map(|(x,_)| *x).collect::<Vec<_>>();
        let triangles = triangles.into_iter().map(|(x,_)| *x).collect::<Vec<_>>();
        let vertices = self.vertices;

        let bindings = [
            ("scene", std::mem::size_of::<SceneBufferInfo>()),
            ("bvh", std::mem::size_of_val(&bvh[..])),
            ("spheres", std::mem::size_of_val(&spheres[..])),
            ("quads", std::mem::size_of_val(&quads[..])),
            ("triangles", std::mem::size_of_val(&triangles[..])),
            ("vertices", std::mem::size_of_val(&vertices[..])),
            ("materials", std::mem::size_of_val(&materials[..])),
            ("emitters", std::mem::size_of_val(&emitters[..])),
            ("diffuse", std::mem::size_of_val(&diffuse[..])),
            ("diffusecb", std::mem::size_of_val(&diffusecb[..])),
            ("dielectric", std::mem::size_of_val(&dielectric[..])),
            ("emissive", std::mem::size_of_val(&emissive[..])),
        ].into_iter().scan((0u32,0u64), |&mut (ref mut index, ref mut offset), (name,size)| {
            let size = (size+BUFFER_ALIGNMENT-1)&!(BUFFER_ALIGNMENT-1);
            let size = size as wgpu::BufferAddress;
            let res = Some(BindingInfo {
                index: *index,
                name,
                offset: *offset,
                size
            });
            *offset += size;
            *index += 1;
            res
        }).collect::<Vec<BindingInfo>>();


        CompiledScene {
            camera: self.camera,
            bindings,
            spheres,
            quads,
            triangles,
            vertices,
            bvh,
            materials,
            emitters,
            diffuse,
            diffusecb,
            dielectric,
            emissive,
        }
    }
}

#[derive(Debug)]
struct BindingInfo {
    name: &'static str,
    index: u32,
    offset: wgpu::BufferAddress,
    size: wgpu::BufferAddress,
}

#[derive(Debug,Copy,Clone)]
struct Emitter {
    shape: u32,
    pdf: f32,
    cdf: f32,
    pad: f32,
}

#[derive(Debug)]
struct CompiledScene {
    camera: Camera,

    bindings: Vec<BindingInfo>,

    bvh: Vec<CompiledBVHNode>,

    spheres: Vec<Sphere>,
    quads: Vec<Quad>,
    triangles: Vec<[u32; 3]>, // flattened repr
    vertices: Vec<Vertex>,

    materials: Vec<u32>,

    emitters: Vec<Emitter>,

    diffuse: Vec<DiffuseMaterial>,
    diffusecb: Vec<DiffuseCheckerboardMaterial>,
    dielectric: Vec<DielectricMaterial>,
    emissive: Vec<EmitterMaterial>,
}


#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct SceneBufferInfo {
    camera: Camera,
    num_spheres: u32,
    num_quads: u32,
    num_triangles: u32,
    num_emitters: u32,
}


const BUFFER_ALIGNMENT: usize = 256;

impl Scene {
    fn from_obj<P: AsRef<std::path::Path>>(file: P) -> Self {
        let (models, materials) = tobj::load_obj(file.as_ref()).unwrap();

        let angle = -1.45f32.to_radians(); // look down a bit
        let rotation = vec4((0.5*angle).sin(), 0., 0., (0.5*angle).cos());

        let mut scene = Scene {
            camera: Camera {
                position: vec4(0., 0.91, 5.41, 0.0),
                rotation,
                fov: 27.7,
            },
            objects: vec![],
            materials: vec![],
            vertices: vec![],
        };


        for material in materials.iter() {
            if material.name.starts_with("light") {
                let power : Vec<f32> = material.unknown_param.get("Ke").unwrap().split(' ').map(|s| s.parse().unwrap()).collect();
                scene.materials.push(Material::Emissive(EmitterMaterial {
                    power: vec3(power[0], power[1], power[2]),
                }));
            } else if material.name.starts_with("glass") {
                scene.materials.push(Material::Dielectric(DielectricMaterial::clear(1.5)));
            } else if material.name.starts_with("mirror") {
                scene.materials.push(Material::Mirror(MirrorMaterial{}));
            }
            /*
            else if material.name.starts_with("floor") {
                scene.materials.push(Material::DiffuseCBoard(DiffuseCheckerboardMaterial {
                    color1: Vector3::from_element(0.8),
                    scale_u: 0.125,
                    color2: Vector3::from_element(0.2),
                    scale_v: 0.125,
                }));
            } 
             */
            else {
                scene.materials.push(Material::Diffuse(DiffuseMaterial {
                    color: Vec3::from_array(material.diffuse),
                }))
            }
        }

        for model in models.iter() {
            let vertex_offset = scene.vertices.len() as u32;

            let mesh = &model.mesh;

            for (i, pos) in mesh.positions.chunks(3).enumerate() {
                let uv = mesh.texcoords.get(2*i..2*(i+1)).unwrap_or(&[0., 0.]);
                let normal = mesh.normals.get(3*i..3*(i+1)).unwrap();
                scene.vertices.push(Vertex {
                    pos: Point3::new(pos[0], pos[1], pos[2]),
                    u: uv[0],
                    normal: Vector3::new(normal[0], normal[1], normal[2]),
                    v: uv[1],
                });
            }

            let material = match mesh.material_id {
                Some(x) => x,
                _ => continue,
            };

            let mut last = [0,0,0];
            for tri in mesh.indices.chunks(3) {
                let tri = [tri[0], tri[1], tri[2]];
                scene.objects.push((Shape::Triangle([
                    tri[0] + vertex_offset,
                    tri[1] + vertex_offset,
                    tri[2] + vertex_offset,
                ]), material));
                continue;

                if tri[0] == last[0] && tri[1] == last[2] { // recover triangulated quads
                    let a = last[0] as usize;
                    let b = last[1] as usize;
                    let c = last[2] as usize;
                    let d = tri[2] as usize;

                    let get_vertex_pos = |ix| {
                        Point3::new(
                            mesh.positions[3*ix],
                            mesh.positions[3*ix+1],
                            mesh.positions[3*ix+2],
                        )
                    };

                    let a = get_vertex_pos(a);
                    let b = get_vertex_pos(b);
                    let c = get_vertex_pos(c);
                    let d = get_vertex_pos(d);


                    let origin = a;
                    let edge1 = b-a;
                    let edge2 = d-a;

                    let error = (c-(origin+edge1+edge2)).norm();
                    if error < 0.01 {
                        // we have a quad!
                        scene.objects.push((Shape::Quad(Quad::new(
                            origin,
                            edge1,
                            edge2,
                        )), material));
                    }
                }
                last = tri;
            }
        }

        scene
    }

}

impl CompiledScene {

    /*
    fn subbuffer_sizes(&self) -> Vec<usize> {
        assert_eq!(
            self.spheres.len() + self.quads.len(),
            self.materials.len()
        );
        vec![
            std::mem::size_of::<SceneBufferInfo>(),
            self.bvh.len() * std::mem::size_of::<CompiledBVHNode>(),
            self.spheres.len() * std::mem::size_of::<Sphere>(),
            self.quads.len() * std::mem::size_of::<Quad>(),

            self.materials.len() * std::mem::size_of::<u32>(),
            self.diffuse.len() * std::mem::size_of::<DiffuseMaterial>(),
            //self.mirrors.len() * std::mem::size_of::<MirrorMaterial>(),
            self.dielectric.len() * std::mem::size_of::<DielectricMaterial>(),
            self.emitters.len() * std::mem::size_of::<EmitterMaterial>(),
            //self.portals.len() * std::mem::size_of::<PortalMaterial>(),
        ]
        .iter()
        .map(|size| (size + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1))
        .collect()
    }
    */

    fn write_to_buffer(&self, mut buffer: &mut [u8]) {
        assert_eq!(
            self.spheres.len() + self.quads.len() + self.triangles.len(),
            self.materials.len()
        );

        let info = SceneBufferInfo {
            camera: self.camera,
            num_spheres: self.spheres.len() as u32,
            num_quads: self.quads.len() as u32,
            num_triangles: self.triangles.len() as u32,
            num_emitters: self.emitters.len() as u32,
        };

        unsafe fn put<T: Copy>(buffer: &mut &mut [u8], data: &[T]) {
            let len = buffer.len();
            let size = (data.len() * std::mem::size_of::<T>() + BUFFER_ALIGNMENT - 1)
                & !(BUFFER_ALIGNMENT - 1); // align to buffer size
            assert!(len >= size);
            let ptr = buffer.as_mut_ptr() as *mut T;
            let target: &mut [T] = std::slice::from_raw_parts_mut(ptr, data.len());
            target.copy_from_slice(data);
            *buffer = std::slice::from_raw_parts_mut((ptr as *mut u8).add(size), len - size);
        }
        unsafe {
            put(&mut buffer, std::slice::from_ref(&info));
            put(&mut buffer, &self.bvh[..]);
            put(&mut buffer, &self.spheres[..]);
            put(&mut buffer, &self.quads[..]);
            put(&mut buffer, &self.triangles[..]);
            put(&mut buffer, &self.vertices[..]);
            put(&mut buffer, &self.materials[..]);

            put(&mut buffer, &self.emitters[..]);

            put(&mut buffer, &self.diffuse[..]);
            put(&mut buffer, &self.diffusecb[..]);
            //put(&mut buffer, &self.mirrors[..]);
            put(&mut buffer, &self.dielectric[..]);
            put(&mut buffer, &self.emissive[..]);
            //put(&mut buffer, &self.portals[..]);
        }

        assert!(buffer.is_empty()); // we filled the buffer
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ImageBlock {
    id: u32,
    seed: u32,
    origin: [u32; 2],
    dimension: [u32; 2],
    original_dimension: [u32; 2],
    sample_offset: Vec2,
}

struct ImageBlockGenerator {
    width: u32,
    height: u32,
    block_size: u32,
    num_samples: u32,
    id: u32,
    x: u32,
    y: u32,
    remaining_samples: u32,
    sample_offset: [f32; 2],
}

impl ImageBlockGenerator {
    fn new(width: u32, height: u32, block_size: u32, num_samples: u32) -> Self {
        assert!(block_size & 63 == 0);
        Self {
            width,
            height,
            block_size,
            num_samples,
            id: 0,
            x: 0,
            y: 0,
            remaining_samples: num_samples,
            sample_offset: rand::random(),
        }
    }
}

impl Iterator for ImageBlockGenerator {
    type Item = ImageBlock;
    fn next(&mut self) -> Option<ImageBlock> {
        if self.remaining_samples == 0 {
            return None;
        }

        // TODO: cool spiral pattern
        let x = self.x;
        let y = self.y;
        let w = self.block_size.min(self.width - x);
        let h = self.block_size.min(self.height - y);
        let id = self.id;

        self.id += 1;
        self.x += self.block_size;
        if self.x >= self.width {
            self.x = 0;
            self.y += self.block_size;
            if self.y >= self.height {
                self.y = 0;
                self.remaining_samples -= 1;
                self.sample_offset = rand::random();
            }
        }
        Some(ImageBlock {
            id,
            seed: rand::random(),
            origin: [x, y],
            dimension: [w, h],
            original_dimension: [self.width, self.height],
            sample_offset: Vec2::from_array(self.sample_offset),
        })
    }
}

struct GPU {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    compiler: shaderc::Compiler,
}

impl GPU {
    fn new() -> Self {
        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            backends: wgpu::BackendBit::PRIMARY,
        })
        .unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        });

        let compiler = shaderc::Compiler::new().unwrap();
        GPU {
            adapter,
            device,
            queue,
            compiler,
        }
    }

    //fn load_shader_from_file<'a, P: AsRef<std::path::Path>, I: IntoIterator<Item=&'a(&'a str, X)>, X: 'a+Borrow<str>>(
    fn load_shader_from_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        definitions: &[(&str, &str)],
    ) -> wgpu::ShaderModule {
        let path = path.as_ref();
        let glsl = std::fs::read_to_string(path).unwrap();

        let mut compile_options = shaderc::CompileOptions::new().unwrap();
        compile_options.set_include_callback(|file, _include_type, _source_file, _depth| {
            let file = std::path::Path::new("shader").join(file);
            let code = std::fs::read_to_string(&file).map_err(|e| e.to_string())?;
            Ok(shaderc::ResolvedInclude {
                resolved_name: file.to_string_lossy().into_owned(),
                content: code,
            })
        });
        for (name, value) in definitions {
            compile_options.add_macro_definition(name, Some(value.borrow()));
        }

        let compiled = self
            .compiler
            .compile_into_spirv(
                &glsl,
                shaderc::ShaderKind::InferFromSource,
                path.file_name().unwrap().to_str().unwrap(),
                "main",
                Some(&compile_options),
            )
            .unwrap_or_else(|err| match err {
                shaderc::Error::CompilationError(_, s) => panic!("{}", s),
                _ => panic!("{:?}", err),
            });

        self.device.create_shader_module(compiled.as_binary())
    }
}

struct IntegratorPipeline {
    shader_module: wgpu::ShaderModule,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

impl IntegratorPipeline {
    fn new(
        gpu: &mut GPU,
        scene: &CompiledScene,
        scene_buffer: &wgpu::Buffer,
        current_block: &wgpu::Buffer,
        output: &wgpu::TextureView,
        use_bvh: bool,
    ) -> Self {
        let mut definitions = vec![];
        for binding in &scene.bindings {
            definitions.push((
                    format!("BINDING_{}", binding.name.to_uppercase()),
                    format!("{}", binding.index+2)
                    ));
        }
        definitions.push(("USE_BVH".to_owned(), if use_bvh {"1".to_owned()} else {"0".to_owned()}));
        definitions.push(("MATERIAL_TAG_SHIFT".to_owned(), format!("{}", MATERIAL_TAG_SHIFT)));
        for material_type in MaterialType::iter() {
            definitions.push((
                    format!("MATERIAL_TAG_{}", material_type.as_ref().to_uppercase()),
                    format!("{}", material_type as u8),
                    ));
        }
        println!("{:#?}", definitions);
        let shader_module = gpu.load_shader_from_file("shader/render.glsl", &definitions.iter().map(|(a,b)| (a.as_ref(), b.as_ref())).collect::<Vec<(&str, &str)>>()[..]);
        let device = &mut gpu.device;

        //let scene_subbuffer_sizes = scene.subbuffer_sizes();
        let mut bindings = vec![
            wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,

                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
            wgpu::BindGroupLayoutBinding {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,

                ty: wgpu::BindingType::StorageTexture {
                    dimension: wgpu::TextureViewDimension::D2Array,
                },
            },
        ];
        for binding in &scene.bindings {
            bindings.push(wgpu::BindGroupLayoutBinding {
                binding: 2 + binding.index,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            });
        }
        /*
        for (i, _) in scene_subbuffer_sizes.iter().enumerate() {
            bindings.push(wgpu::BindGroupLayoutBinding {
                binding: 2 + i as u32,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            });
        }
        */
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &bindings[..],
        });

        let mut bindings = vec![
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: current_block,
                    range: 0..std::mem::size_of::<ImageBlock>() as wgpu::BufferAddress,
                },
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(output),
            },
        ];
        for binding in &scene.bindings {
            bindings.push(wgpu::Binding {
                binding: 2 + binding.index,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &scene_buffer,
                    range: binding.offset..binding.offset+binding.size,
                },
            });
        }
        /*
        let mut offs = 0;
        for (i, size) in scene_subbuffer_sizes.iter().enumerate() {
            let size = *size as wgpu::BufferAddress;
            bindings.push(wgpu::Binding {
                binding: 2 + i as u32,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &scene_buffer,
                    range: offs..offs + size,
                },
            });
            offs += size;
        }
        */
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &bindings[..],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout: &pipeline_layout,
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &shader_module,
                entry_point: "main",
            },
        });
        IntegratorPipeline {
            shader_module,
            bind_group,
            pipeline,
        }
    }

    fn run(&self, block: &ImageBlock, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch((block.dimension[0]+15)/16, (block.dimension[1]+15)/16, 1);
        drop(cpass);
    }
}

struct ReconstructionPipeline {
    shader_module: wgpu::ShaderModule,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    radius: u32,
}

impl ReconstructionPipeline {
    fn new(
        gpu: &mut GPU,
        current_block: &wgpu::Buffer,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        radius: u32,
        stddev: f32,
    ) -> Self {
        let shader_module = gpu.load_shader_from_file(
            "shader/reconstruction.glsl",
            &[
                ("RECONSTRUCTION_RADIUS", &format!("{}", radius)),
                ("RECONSTRUCTION_STDDEV", &format!("{}", stddev)),
            ],
        );
        let device = &mut gpu.device;
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,

                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                    },
                },
                // inputs
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,

                    ty: wgpu::BindingType::StorageTexture {
                        dimension: wgpu::TextureViewDimension::D2Array,
                    },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 2,
                    visibility: wgpu::ShaderStage::COMPUTE,

                    ty: wgpu::BindingType::StorageTexture {
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: current_block,
                        range: 0..std::mem::size_of::<ImageBlock>() as wgpu::BufferAddress,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(output),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout: &pipeline_layout,
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &shader_module,
                entry_point: "main",
            },
        });
        ReconstructionPipeline {
            shader_module,
            bind_group,
            pipeline,
            radius,
        }
    }

    fn run(&self, block: &ImageBlock, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        //cpass.dispatch((block.dimension[0]+15)/16, (block.dimension[1]+15)/16, 1);
        cpass.dispatch(
            (block.dimension[0] + 2*self.radius+15)/16,
            (block.dimension[1] + 2*self.radius+15)/16,
            1,
        );
        drop(cpass);
    }
}

struct PreviewPipeline {
    vertex_shader: wgpu::ShaderModule,
    fragment_shader: wgpu::ShaderModule,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    swap_chain: wgpu::SwapChain,
    evt_loop: winit::EventsLoop,
    window: winit::Window,
    surface: wgpu::Surface,
    width: u32,
    height: u32,
}

impl PreviewPipeline {
    fn new(gpu: &mut GPU, width: u32, height: u32, input: &wgpu::TextureView) -> Self {
        let vertex_shader = gpu.load_shader_from_file("shader/fsquad.glsl", &[]);
        let fragment_shader = gpu.load_shader_from_file("shader/preview.glsl", &[]);

        std::env::set_var("WINIT_HIDPI_FACTOR", "1");
        let evt_loop = winit::EventsLoop::new();
        let window = winit::WindowBuilder::new()
            .with_dimensions((800, 600).into())
            .with_resizable(false)
            .build(&evt_loop)
            .unwrap();
        let size = window.get_inner_size().unwrap().to_physical(window.get_hidpi_factor());
        let surface = wgpu::Surface::create(&window);

        let device = &mut gpu.device;
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::StorageTexture {
                    dimension: wgpu::TextureViewDimension::D2,
                },
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vertex_shader,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fragment_shader,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.,
                depth_bias_clamp: 0.,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
            color_states: &[wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });
        let mut swap_chain = device.create_swap_chain(&surface, &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width.round() as u32,
            height: size.height.round() as u32,
            present_mode: wgpu::PresentMode::Vsync,
        });

        PreviewPipeline {
            vertex_shader,
            fragment_shader,
            bind_group,
            pipeline,
            swap_chain,
            window,
            surface,
            evt_loop,
            width,
            height,
        }
    }

    fn update(&mut self, encoder: &mut wgpu::CommandEncoder) -> (bool,Option<wgpu::SwapChainOutput>) {
        let mut close = false;
        let mut refresh = false;
        self.evt_loop.poll_events(|event| {
            if let winit::Event::WindowEvent{event: winit::WindowEvent::CloseRequested, ..} = event {
                close = true;
            }
            if let winit::Event::WindowEvent{event: winit::WindowEvent::Refresh, ..} = event {
                refresh = true;
            }
        });

        let mut frame_keepalive = None;
        if true {
            let frame = self.swap_chain.get_next_texture();
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {r: 1., g: 0., b: 1., a: 1.},
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..4, 0..1);
            drop(rpass);
            frame_keepalive = Some(frame);
        }

        (close, frame_keepalive)
    }
}

struct Renderer {
    gpu: GPU,
    width: u32,
    height: u32,
    blocks: Vec<ImageBlock>,
    all_blocks: wgpu::Buffer,
    current_block: wgpu::Buffer,

    scene: CompiledScene,
    scene_buffer: wgpu::Buffer,

    intermediate_texture: wgpu::Texture,

    final_texture: wgpu::Texture,
    final_output_buffer: wgpu::Buffer,

    integrator_pipeline: IntegratorPipeline,
    reconstruction_pipeline: ReconstructionPipeline,
    preview_pipeline: PreviewPipeline,

    present_interval: u32, // how often to update preview
}

impl Renderer {
    fn new(scene: CompiledScene, generator: ImageBlockGenerator, present_interval: u32, use_bvh: bool) -> Self {
        let width = generator.width;
        let height = generator.height;

        let block_size = generator.block_size;

        let mut gpu = GPU::new();
        let device = &mut gpu.device;
        let blocks = generator.collect::<Vec<_>>();

        let all_blocks = device
            .create_buffer_mapped(blocks.len(), wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(&blocks[..]);
        let current_block = device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<ImageBlock>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

        //let scene_buffer_size = scene.subbuffer_sizes().iter().sum::<usize>();
        let scene_buffer_size = scene.bindings.iter().map(|binding| binding.size as usize).sum::<usize>();
        let mut scene_staging_buffer =
            device.create_buffer_mapped::<u8>(scene_buffer_size, wgpu::BufferUsage::COPY_SRC);
        scene.write_to_buffer(scene_staging_buffer.data);
        let scene_staging_buffer = scene_staging_buffer.finish();

        let scene_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: scene_buffer_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

        let intermediate_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: block_size,
                height: block_size,
                depth: 1,
            },
            array_layer_count: 3,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsage::STORAGE,
        });

        let final_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsage::STORAGE
                | wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST,
        });
        let row_size = (width + 255) & !255;
        let final_output_buffer = device
            .create_buffer_mapped(
                (row_size * height) as usize,
                wgpu::BufferUsage::COPY_SRC
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::MAP_READ,
            )
            .fill_from_slice(&vec![[0f32; 4]; (row_size * height) as usize]);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        encoder.copy_buffer_to_buffer(
            &scene_staging_buffer,
            0,
            &scene_buffer,
            0,
            scene_buffer_size as wgpu::BufferAddress,
        );
        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &final_output_buffer,
                offset: 0,
                row_pitch: row_size * std::mem::size_of::<[f32; 4]>() as u32,
                image_height: height,
            },
            wgpu::TextureCopyView {
                texture: &final_texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
        );
        // TODO: copy other data to gpu?
        gpu.queue.submit(&[encoder.finish()]);

        let integrator_pipeline = IntegratorPipeline::new(
            &mut gpu,
            &scene,
            &scene_buffer,
            &current_block,
            &intermediate_texture.create_default_view(),
            use_bvh,
        );
        let reconstruction_pipeline = ReconstructionPipeline::new(
            &mut gpu,
            &current_block,
            &intermediate_texture.create_default_view(),
            &final_texture.create_default_view(),
            2,   // radius
            0.5, // stddev
        );
        let preview_pipeline = PreviewPipeline::new(&mut gpu, width, height, &final_texture.create_default_view());

        Renderer {
            gpu,

            width,
            height,

            blocks,

            all_blocks,
            current_block,

            scene,
            scene_buffer,

            intermediate_texture,

            final_texture,
            final_output_buffer,

            integrator_pipeline,
            reconstruction_pipeline,
            preview_pipeline,

            present_interval,
        }
    }

    fn render(&mut self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        for block in self.blocks.iter() {
            encoder.copy_buffer_to_buffer(
                &self.all_blocks,
                (block.id as usize * std::mem::size_of::<ImageBlock>()) as wgpu::BufferAddress,
                &self.current_block,
                0,
                std::mem::size_of::<ImageBlock>() as wgpu::BufferAddress,
            );
            self.integrator_pipeline.run(&block, &mut encoder);
            self.reconstruction_pipeline.run(&block, &mut encoder);

            let mut frame_keepalive = None;
            let mut closed = false;

            if block.id % self.present_interval == 0 {
                self.preview_pipeline.window.set_title(&format!("{:3.3}% {}/{}", 100. * block.id as f32 / self.blocks.len() as f32, block.id, self.blocks.len()));
                let (a, b) = self.preview_pipeline.update(&mut encoder);
                closed = a;
                frame_keepalive = Some(b);
            }
            let next_encoder = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
            self.gpu
                .queue
                .submit(&[std::mem::replace(&mut encoder, next_encoder).finish()]);

            drop(frame_keepalive);
            if closed {
                break;
            }
        }
        self.gpu.queue.submit(&[encoder.finish()]);
    }

    fn save_image<P: AsRef<std::path::Path>>(&mut self, output_file: P) {
        let output_file = output_file.as_ref().to_owned();
        let width = self.width;
        let height = self.height;
        let row_size = (width + 255) & (!255);
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &self.final_texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
            },
            wgpu::BufferCopyView {
                buffer: &self.final_output_buffer,
                offset: 0,
                row_pitch: row_size * std::mem::size_of::<[f32; 4]>() as u32,
                image_height: height,
            },
            wgpu::Extent3d {
                width,
                height,
                depth: 1,
            },
        );
        self.gpu.queue.submit(&[encoder.finish()]);
        self.final_output_buffer.map_read_async(
            0,
            ((row_size * height) as usize * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
            move |result: wgpu::BufferMapAsyncResult<&[[f32; 4]]>| {
                if let Ok(map) = result {
                    let pixels = map
                        .data
                        .chunks(row_size as usize)
                        .flat_map(|row| &row[..width as usize])
                        .map(|&[r, g, b, n]| [r / n, g / n, b / n])
                        .collect::<Vec<[f32; 3]>>();

                    use openexr::frame_buffer::FrameBuffer;
                    use openexr::header::Header;
                    use openexr::output::ScanlineOutputFile;
                    use openexr::PixelType;

                    let mut file = std::fs::File::create(output_file).unwrap();
                    let mut output_file = ScanlineOutputFile::new(
                        &mut file,
                        Header::new()
                            .set_resolution(width, height)
                            .add_channel("R", PixelType::FLOAT)
                            .add_channel("G", PixelType::FLOAT)
                            .add_channel("B", PixelType::FLOAT),
                    )
                    .unwrap();
                    let mut fb = FrameBuffer::new(width, height);
                    fb.insert_channels(&["R", "G", "B"], &pixels);
                    output_file.write_pixels(&fb).unwrap();
                }
            },
        );
    }
}

#[derive(StructOpt)]
struct Opt {
    /// Add a mirror and glass sphere to the scene
    #[structopt(long)]
    put_cbox_spheres: bool,

    /// Use a BVH to optimize intersections
    #[structopt(long)]
    use_bvh: bool,

    /// Width of the image
    #[structopt(long,short, default_value="800")]
    width: u32,

    /// Height of the image
    #[structopt(long,short, default_value="600")]
    height: u32,

    /// How often to update preview during rendering
    #[structopt(long, default_value="128")]
    present_interval: u32,

    #[structopt(short,long,default_value="64")]
    sample_count: u32,

    #[structopt(short,long,default_value="/tmp/output.exr")]
    output_image: std::path::PathBuf,

    /// The scene (OBJ file) to render
    scene: std::path::PathBuf,
}


fn main() {
    let opt = Opt::from_args();

    let mut scene = Scene::from_obj(opt.scene);
    if opt.put_cbox_spheres {
        scene.materials.push(Material::Mirror(MirrorMaterial{}));
        //scene.materials.push(Material::Dielectric(DielectricMaterial::clear(1.5)));
        scene.materials.push(Material::DiffuseCBoard(DiffuseCheckerboardMaterial {
            color1: Vector3::new(1.0,0.4,0.7),
            scale_u: 0.1,
            color2: Vector3::new(0.4,0.7,1.0),
            scale_v: 0.2,
        }));
        //scene.materials.push(Material::Dielectric(DielectricMaterial::tinted(vec3(1., 0., 1.), 1.5)));
        scene.objects.push((Shape::Sphere(Sphere {
                    // mirror sphere
                    position: Point3::new(-0.421400, 0.332100, -0.280000),
                    radius: 0.3263,
                }), scene.materials.len()-2));
        scene.objects.push((Shape::Sphere(Sphere {
                    // glass sphere
                    position: Point3::new(0.445800, 0.332100, 0.376700),
                    radius: 0.3263,
                }), scene.materials.len()-1));
    }

    let block_generator = ImageBlockGenerator::new(opt.width, opt.height, 128, opt.sample_count);
    let mut renderer = Renderer::new(scene.compile(), block_generator, opt.present_interval, opt.use_bvh);
    let start = std::time::Instant::now();
    println!("Starting to render...");
    renderer.render();
    let render_time = std::time::Instant::now()-start;
    let ray_count = opt.width*opt.height*opt.sample_count;
    println!("Integrated {} rays in {:?} ({} rays/s)", ray_count, render_time, ray_count as f64 / render_time.as_secs_f64());
    renderer.save_image(opt.output_image);
}
