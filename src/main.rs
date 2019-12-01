extern crate wgpu;

extern crate shaderc;

extern crate rand;

mod glm;
use glm::*;

mod shape;
use shape::*;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct DiffuseMaterial {
    color: Vec3,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct MirrorMaterial {
    dummy: u8,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct DielectricMaterial {
    eta_ratio: f32,
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

struct Scene {
    camera: Camera,

    spheres: Vec<Sphere>,
    planes: Vec<Plane>,
    quads: Vec<Quad>,
    materials: Vec<u32>,

    diffuse: Vec<DiffuseMaterial>,
    mirrors: Vec<MirrorMaterial>,
    dielectric: Vec<DielectricMaterial>,
    emitters: Vec<EmitterMaterial>,
    portals: Vec<PortalMaterial>,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct SceneBufferInfo {
    camera: Camera,
    num_spheres: u32,
    num_planes: u32,
    num_quads: u32,

    num_diffuse: u32,
    num_mirrors: u32,
    num_dielectric: u32,
    num_emitters: u32,
    num_portals: u32,
}

const BUFFER_ALIGNMENT: usize = 256;

impl Scene {
    fn subbuffer_sizes(&self) -> Vec<usize> {
        assert_eq!(
            self.spheres.len() + self.planes.len() + self.quads.len(),
            self.materials.len()
        );
        vec![
            std::mem::size_of::<SceneBufferInfo>(),
            self.spheres.len() * std::mem::size_of::<Sphere>(),
            self.planes.len() * std::mem::size_of::<Plane>(),
            self.quads.len() * std::mem::size_of::<Quad>(),
            self.materials.len() * std::mem::size_of::<u32>(), // align to 16 bytes
            self.diffuse.len() * std::mem::size_of::<DiffuseMaterial>(),
            self.mirrors.len() * std::mem::size_of::<MirrorMaterial>(),
            self.dielectric.len() * std::mem::size_of::<DielectricMaterial>(),
            self.emitters.len() * std::mem::size_of::<EmitterMaterial>(),
            self.portals.len() * std::mem::size_of::<PortalMaterial>(),
        ]
        .iter()
        .map(|size| (size + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1))
        .collect()
    }

    fn write_to_buffer(&self, mut buffer: &mut [u8]) {
        assert_eq!(
            self.spheres.len() + self.planes.len() + self.quads.len(),
            self.materials.len()
        );

        let info = SceneBufferInfo {
            camera: self.camera,
            num_spheres: self.spheres.len() as u32,
            num_planes: self.planes.len() as u32,
            num_quads: self.quads.len() as u32,

            num_diffuse: self.diffuse.len() as u32,
            num_mirrors: self.mirrors.len() as u32,
            num_dielectric: self.dielectric.len() as u32,
            num_emitters: self.emitters.len() as u32,
            num_portals: self.portals.len() as u32,
        };

        println!("{} bytes left", buffer.len());
        unsafe fn put<T: Copy>(buffer: &mut &mut [u8], data: &[T]) {
            let len = buffer.len();
            let size = (data.len() * std::mem::size_of::<T>() + BUFFER_ALIGNMENT - 1)
                & !(BUFFER_ALIGNMENT - 1); // align to buffer size
            println!("putting {} bytes", size);
            assert!(len >= size);
            let ptr = buffer.as_mut_ptr() as *mut T;
            let target: &mut [T] = std::slice::from_raw_parts_mut(ptr, data.len());
            target.copy_from_slice(data);
            *buffer = std::slice::from_raw_parts_mut((ptr as *mut u8).add(size), len - size);
            println!("{} bytes left", buffer.len());
        }
        unsafe {
            put(&mut buffer, std::slice::from_ref(&info));
            put(&mut buffer, &self.spheres[..]);
            put(&mut buffer, &self.planes[..]);
            put(&mut buffer, &self.quads[..]);
            put(&mut buffer, &self.materials[..]);

            put(&mut buffer, &self.diffuse[..]);
            put(&mut buffer, &self.mirrors[..]);
            put(&mut buffer, &self.dielectric[..]);
            put(&mut buffer, &self.emitters[..]);
            put(&mut buffer, &self.portals[..]);
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
}

impl ImageBlockGenerator {
    fn new(width: u32, height: u32, block_size: u32, num_samples: u32) -> Self {
        assert!(block_size & 64 == 0);
        Self {
            width,
            height,
            block_size,
            num_samples,
            id: 0,
            x: 0,
            y: 0,
            remaining_samples: num_samples,
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
            }
        }
        Some(ImageBlock {
            id,
            seed: rand::random(),
            origin: [x, y],
            dimension: [w, h],
            original_dimension: [self.width, self.height],
            sample_offset: vec2(rand::random::<f32>(), rand::random::<f32>()),
        })
    }
    /*
    fn size_hint(&self) -> (usize, Option<usize>) {
        let num_x = (self.width  + self.block_size-1) / self.block_size;
        let num_y = (self.height + self.block_size-1) / self.block_size;
        let remaining = (self.num_samples * num_x * num_y - self.id) as usize;
        (remaining, Some(remaining))
    }
    */
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
            extensions: wgpu::Extensions::default(),
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
            compile_options.add_macro_definition(name, Some(value));
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
        scene: &Scene,
        scene_buffer: &wgpu::Buffer,
        current_block: &wgpu::Buffer,
        output: &wgpu::TextureView,
    ) -> Self {
        let shader_module = gpu.load_shader_from_file("shader/render.glsl", &[]);
        let device = &mut gpu.device;

        let scene_subbuffer_sizes = scene.subbuffer_sizes();
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
        cpass.dispatch(block.dimension[0], block.dimension[1], 1);
        //println!("{:?}", block);
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
        cpass.dispatch(block.dimension[0] + 2*self.radius, block.dimension[1] + 2*self.radius, 1);
        //println!("{:?}", block);
        drop(cpass);
    }
}

struct Renderer {
    gpu: GPU,
    width: u32,
    height: u32,
    blocks: Vec<ImageBlock>,
    all_blocks: wgpu::Buffer,
    current_block: wgpu::Buffer,

    scene: Scene,
    scene_buffer: wgpu::Buffer,

    intermediate_texture: wgpu::Texture,

    final_texture: wgpu::Texture,
    final_output_buffer: wgpu::Buffer,

    integrator_pipeline: IntegratorPipeline,
    reconstruction_pipeline: ReconstructionPipeline,
}

impl Renderer {
    fn new(scene: Scene, generator: ImageBlockGenerator) -> Self {
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

        let scene_buffer_size = scene.subbuffer_sizes().iter().sum::<usize>();
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
        );
        let reconstruction_pipeline = ReconstructionPipeline::new(
            &mut gpu,
            &current_block,
            &intermediate_texture.create_default_view(),
            &final_texture.create_default_view(),
            5,  // radius
            1., // stddev
        );

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
        }
    }

    fn render(&mut self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        for block in &self.blocks {
            encoder.copy_buffer_to_buffer(
                &self.all_blocks,
                (block.id as usize * std::mem::size_of::<ImageBlock>()) as wgpu::BufferAddress,
                &self.current_block,
                0,
                std::mem::size_of::<ImageBlock>() as wgpu::BufferAddress,
            );
            self.integrator_pipeline.run(&block, &mut encoder);
            self.reconstruction_pipeline.run(&block, &mut encoder);
            if block.id % 1 == 0 {
                let next_encoder = self
                    .gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
                self.gpu
                    .queue
                    .submit(&[std::mem::replace(&mut encoder, next_encoder).finish()]);
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
                    //println!("{:?}", &pixels);

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

fn main() {
    let scene = Scene {
        // cornell box
        camera: Camera {
            position: vec4(0., 0.91, 5.41, 0.0),
            rotation: vec4(0., 0., 0., 1.),
            fov: 27.7,
        },
        spheres: vec![
            Sphere {
                // mirror sphere
                position_radius: vec4(-0.421400, 0.332100, -0.280000, 0.3263),
            },
            Sphere {
                // glass sphere
                position_radius: vec4(0.445800, 0.332100, 0.376700, 0.3263),
            },
            /*Sphere {
                // light sphere
                position_radius: vec4(-0.6, 1.25, 0., 0.2),
                //position_radius: vec4(-0.1, 1.0, 0.376700, 0.3263),
            },*/
        ],
        planes: vec![
            Plane {
                normal_offset: vec4(
                    // right wall pointing to the left
                    -1., 0., 0., 1.,
                ),
            },
            Plane {
                normal_offset: vec4(
                    // left wall pointing to the right
                    1., 0., 0., 1.,
                ),
            },
            Plane {
                normal_offset: vec4(
                    // ceiling
                    0., -1.0, 0., 1.59,
                ),
            },
            Plane {
                normal_offset: vec4(
                    // floor
                    0., 1.0, 0., 0.,
                ),
            },
            Plane {
                normal_offset: vec4(
                    // back
                    0., 0., 1.0, 1.04,
                ),
            },
        ],
        quads: vec![Quad {
            origin: vec3(-0.24, 1.58, -0.22),
            edge1: vec3(0.24 + 0.23, 0., 0.),
            edge2: vec3(0., 0., 0.22 + 0.16),
        }],
        materials: vec![
            3, // mirror sphere
            4, // glass sphere,
            //5, // emissive sphere
            2, // right wall
            1, // left wall,
            0, // ceiling
            0, // floor
            0, // back
            5, // emissive quad
        ],

        diffuse: vec![
            DiffuseMaterial {
                color: vec3(0.725, 0.71, 0.68),
            }, // material 0: white diffuse
            DiffuseMaterial {
                color: vec3(0.630, 0.065, 0.05),
            }, // material 1: red diffuse
            DiffuseMaterial {
                color: vec3(0.161, 0.133, 0.427),
            }, // material 2: blue diffuse
        ],
        mirrors: vec![
            MirrorMaterial { dummy: 0 }, // material 3: perfect mirror
        ],
        dielectric: vec![DielectricMaterial {
            // material 4: glass
            eta_ratio: 1.5046,
        }],
        emitters: vec![
            EmitterMaterial {
                power: vec3(15., 15., 15.), // material 5: white emitter
            },
            EmitterMaterial {
                power: vec3(0., 0., 0.), // material 6: black hole
            },
        ],
        portals: vec![],
    };

    let block_generator = ImageBlockGenerator::new(800, 600, 128, 32);
    let mut renderer = Renderer::new(scene, block_generator);
    renderer.render();
    renderer.save_image("/tmp/output.exr");
}
