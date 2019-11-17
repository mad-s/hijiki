extern crate wgpu;

extern crate shaderc;

extern crate rand;

mod glm;
use glm::*;

fn load_shader(
    compiler: &mut shaderc::Compiler,
    device: &wgpu::Device,
    file: &std::path::Path,
) -> wgpu::ShaderModule {
    let glsl_metadata = std::fs::metadata(file).unwrap();

    let spv_file = file.with_extension("spv");

    let spv = (|| {
        // try to load cached file
        if let Ok(spv_metadata) = std::fs::metadata(&spv_file) {
            if let (Ok(glsl_mtime), Ok(spv_mtime)) =
                (glsl_metadata.modified(), spv_metadata.modified())
            {
                if spv_mtime >= glsl_mtime {
                    return wgpu::read_spirv(std::fs::File::open(spv_file).unwrap()).unwrap();
                }
            }
        }

        let glsl = std::fs::read_to_string(file).unwrap();
        let compiled = compiler
            .compile_into_spirv(
                &glsl,
                shaderc::ShaderKind::InferFromSource,
                file.file_name().unwrap().to_str().unwrap(),
                "main",
                None,
            )
            .unwrap_or_else(|err| match err {
                shaderc::Error::CompilationError(_, s) => panic!("{}", s),
                _ => panic!("{:?}", err),
            });

        std::fs::write(spv_file, compiled.as_binary_u8());
        compiled.as_binary().to_vec()
    })();

    //let reflect = spirv_reflect::ShaderModule::load_u8_data(&spv).unwrap();
    //println!("{:#?}", reflect.enumerate_input_variables(None));

    device.create_shader_module(&spv[..])
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct Sphere {
    position_radius: Vec4,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct Plane {
    normal_offset: Vec4,
}

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
struct Camera {
    position: Vec3,
    fov: f32,
}

struct Scene {
    camera: Camera,

    spheres: Vec<Sphere>,
    planes: Vec<Plane>,
    materials: Vec<u32>,

    diffuse: Vec<DiffuseMaterial>,
    mirrors: Vec<MirrorMaterial>,
    dielectric: Vec<DielectricMaterial>,
    emitters: Vec<EmitterMaterial>,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct SceneBufferInfo {
    camera: Camera,
    num_spheres: u32,
    num_planes: u32,

    num_diffuse: u32,
    num_mirrors: u32,
    num_dielectric: u32,
    num_emitters: u32,
}

const BUFFER_ALIGNMENT : usize = 256;

impl Scene {
    fn subbuffer_sizes(&self) -> Vec<usize> {
        assert_eq!(self.spheres.len() + self.planes.len(), self.materials.len());
        vec![
            std::mem::size_of::<SceneBufferInfo>(),
            self.spheres.len() * std::mem::size_of::<Sphere>(),
            self.planes.len() * std::mem::size_of::<Plane>(),
            self.materials.len() * std::mem::size_of::<u32>(), // align to 16 bytes
            self.diffuse.len() * std::mem::size_of::<DiffuseMaterial>(),
            self.mirrors.len() * std::mem::size_of::<MirrorMaterial>(),
            self.dielectric.len() * std::mem::size_of::<DielectricMaterial>(),
            self.emitters.len() * std::mem::size_of::<EmitterMaterial>(),
        ].iter().map(|size| (size + BUFFER_ALIGNMENT-1) & !(BUFFER_ALIGNMENT-1)).collect()
    }

    fn write_to_buffer(&self, mut buffer: &mut [u8]) {
        assert_eq!(self.spheres.len() + self.planes.len(), self.materials.len());

        let info = SceneBufferInfo {
            camera: self.camera,
            num_spheres: self.spheres.len() as u32,
            num_planes: self.planes.len() as u32,

            num_diffuse: self.diffuse.len() as u32,
            num_mirrors: self.mirrors.len() as u32,
            num_dielectric: self.dielectric.len() as u32,
            num_emitters: self.emitters.len() as u32,
        };

        println!("{} bytes left", buffer.len());
        unsafe fn put<T: Copy>(buffer: &mut &mut [u8], data: &[T]) {
            let len = buffer.len();
            let size = (data.len() * std::mem::size_of::<T>() + BUFFER_ALIGNMENT-1) & !(BUFFER_ALIGNMENT-1); // align to buffer size
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
            put(&mut buffer, &self.materials[..]);

            put(&mut buffer, &self.diffuse[..]);
            put(&mut buffer, &self.mirrors[..]);
            put(&mut buffer, &self.dielectric[..]);
            put(&mut buffer, &self.emitters[..]);
        }

        assert!(buffer.is_empty()); // we filled the buffer
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SampleInfo {
    id: u32,
    weight: f32,
    sample_offset: Vec2,
}

fn main() {
    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::Default,
        backends: wgpu::BackendBit::PRIMARY,
    })
    .unwrap();

    let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions::default(),
        limits: wgpu::Limits::default(),
    });

    let mut compiler = shaderc::Compiler::new().unwrap();

    let cs_module = load_shader(
        &mut compiler,
        &device,
        std::path::Path::new("shader/render.glsl"),
    );

    let scene = Scene {
        // cornell box
        camera: Camera {
            position: vec3(0., 0.91, 5.41),
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
            Sphere {
                // light sphere
                position_radius: vec4(0., 1.25, 0., 0.2),
                //position_radius: vec4(-0.1, 1.0, 0.376700, 0.3263),
            },
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
        materials: vec![
            3, // mirror sphere
            4, // glass sphere,
            5, // emissive sphere
            2, // right wall
            1, // left wall,
            0, // ceiling
            0, // floor
            0, // back
        ],

        diffuse: vec![
            DiffuseMaterial {
                color: vec3(0.5, 0.5, 0.5),
            }, // material 0: white diffuse
            DiffuseMaterial {
                color: vec3(0.5, 0.0, 0.0),
            }, // material 1: red diffuse
            DiffuseMaterial {
                color: vec3(0.0, 0.0, 0.5),
            }, // material 2: blue diffuse
        ],
        mirrors: vec![
            MirrorMaterial { dummy: 0 }, // material 3: perfect mirror
        ],
        dielectric: vec![DielectricMaterial {
            // material 4: glass
            eta_ratio: 1.5,
        }],
        emitters: vec![EmitterMaterial {
            power: vec3(10., 10., 10.), // material 5: white emitter
        }],
    };

    let width = 800;
    let height = 600;

    let fov_factor: f32 = (scene.camera.fov / 2.).to_radians().tan() / (height as f32 / 2.);

    let inputs: Vec<Vec4> = (0..height)
        .flat_map(move |y| {
            (0..width).map(move |x| {
                vec4(
                    (x as f32 - 0.5 * width as f32) * fov_factor,
                    -(y as f32 - 0.5 * height as f32) * fov_factor,
                    -1.0,
                    0.0,
                )
            })
        })
        .collect();
    let input_buffer_size = (inputs.len() * std::mem::size_of::<Vec4>()) as wgpu::BufferAddress;

    let num_samples = 512;
    let samples: Vec<SampleInfo> = (0..num_samples)
        .map(|i| SampleInfo {
            id: i,
            weight: 1. / num_samples as f32,
            sample_offset: vec2(
                fov_factor * (rand::random::<f32>() - 0.5),
                fov_factor * (rand::random::<f32>() - 0.5),
            ),
        })
        .collect();
    let sample_buffer_size =
        (samples.len() * std::mem::size_of::<SampleInfo>()) as wgpu::BufferAddress;

    /*
    let spheres: Vec<Sphere> = vec![
        // right wall
        Sphere {
            position: [1002., 0., 0., 0.],
            r: 1000.,
            diffuse: [0., 0., 0.5, 0.],
            emissive: [0., 0., 0., 0.],
            material: 0,
        },
        // left wall
        Sphere {
            position: [-1002., 0., 0., 0.],
            r: 1000.,
            diffuse: [0.5, 0., 0., 0.],
            emissive: [0., 0., 0., 0.],
            material: 0,
        },
        // back wall
        Sphere {
            position: [0., 0., 1004., 0.],
            r: 1000.,
            diffuse: [0.5, 0.5, 0.5, 0.],
            emissive: [0., 0., 0., 0.],
            material: 0,
        },
        // floor
        Sphere {
            position: [0., -1002., 0., 0.],
            r: 1000.,
            diffuse: [0.5, 0.5, 0.5, 0.],
            emissive: [0., 0., 0., 0.],
            material: 0,
        },
        // ceiling
        Sphere {
            position: [0., 1002., 0., 0.],
            r: 1000.,
            diffuse: [0.5, 0.5, 0.5, 0.],
            emissive: [0., 0., 0., 0.],
            material: 0,
        },
        // lamp
        Sphere {
            position: [-0.5, -1.5, 3., 0.],
            r: 0.5,
            diffuse: [0., 0., 0., 0.],
            emissive: [1., 1., 1., 0.],
            material: 0,
        },
    ];
    */
    //let scene_buffer_size = (spheres.len() * std::mem::size_of::<Sphere>()) as wgpu::BufferAddress;

    let input_staging_buffer = device
        .create_buffer_mapped(
            inputs.len(),
            wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        )
        .fill_from_slice(&inputs[..]);

    let input_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: input_buffer_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let output_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: input_buffer_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let sample_staging_buffer = device
        .create_buffer_mapped(samples.len(), wgpu::BufferUsage::COPY_SRC)
        .fill_from_slice(&samples[..]);

    let current_sample_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: std::mem::size_of::<SampleInfo>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let scene_subbuffer_sizes = scene.subbuffer_sizes();
    let scene_buffer_size = scene_subbuffer_sizes.iter().sum::<usize>() as wgpu::BufferAddress;
    let scene_staging_buffer =
        device.create_buffer_mapped::<u8>(scene_buffer_size as usize, wgpu::BufferUsage::COPY_SRC);
    scene.write_to_buffer(scene_staging_buffer.data);
    let scene_staging_buffer = scene_staging_buffer.finish();

    let scene_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: scene_buffer_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let mut bindings = vec![
        // input
        wgpu::BindGroupLayoutBinding {
            binding: 0,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
            },
        },
        // output
        wgpu::BindGroupLayoutBinding {
            binding: 1,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
            },
        },
        // sample
        wgpu::BindGroupLayoutBinding {
            binding: 2,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
            },
        },
    ];
    // scene info
    for (i, _subbuffer_size) in scene_subbuffer_sizes.iter().enumerate() {
        bindings.push(
            wgpu::BindGroupLayoutBinding {
                binding: 3+i as u32,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            }
        );
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &bindings[..],
    });

    let mut bindings = vec![
        // input
        wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::Buffer {
                buffer: &input_storage_buffer,
                range: 0..input_buffer_size,
            },
        },
        // output
        wgpu::Binding {
            binding: 1,
            resource: wgpu::BindingResource::Buffer {
                buffer: &output_storage_buffer,
                range: 0..input_buffer_size,
            },
        },
        // sample
        wgpu::Binding {
            binding: 2,
            resource: wgpu::BindingResource::Buffer {
                buffer: &current_sample_storage_buffer,
                range: 0..std::mem::size_of::<SampleInfo>() as wgpu::BufferAddress,
            },
        },
    ];

    let mut offs = 0;
    // scene info
    for (i, subbuffer_size) in scene_subbuffer_sizes.iter().enumerate() {
        bindings.push(
            wgpu::Binding {
                binding: 3+i as u32,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &scene_storage_buffer,
                    range: offs as wgpu::BufferAddress..(offs+subbuffer_size) as wgpu::BufferAddress,
                },
            }
        );
        offs += subbuffer_size;
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &bindings[..],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_buffer(
        &input_staging_buffer,
        0,
        &input_storage_buffer,
        0,
        input_buffer_size,
    );
    encoder.copy_buffer_to_buffer(
        &scene_staging_buffer,
        0,
        &scene_storage_buffer,
        0,
        scene_buffer_size,
    );
    queue.submit(&[encoder.finish()]);

    for sample_ix in 0..samples.len() {
        println!("sample {}", sample_ix);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        encoder.copy_buffer_to_buffer(
            &sample_staging_buffer,
            (sample_ix * std::mem::size_of::<SampleInfo>()) as wgpu::BufferAddress,
            &current_sample_storage_buffer,
            0,
            std::mem::size_of::<SampleInfo>() as wgpu::BufferAddress,
        );
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(inputs.len() as u32, 1, 1);
        }
        queue.submit(&[encoder.finish()]);
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_buffer(
        &output_storage_buffer,
        0,
        &input_staging_buffer,
        0,
        input_buffer_size,
    );
    queue.submit(&[encoder.finish()]);

    input_staging_buffer.map_read_async(
        0,
        input_buffer_size,
        // here we don't want to use vec4 because of OpenEXR
        move |result: wgpu::BufferMapAsyncResult<&[[f32; 4]]>| {
            if let Ok(mapping) = result {
                let pixels = mapping.data;

                use openexr::frame_buffer::FrameBuffer;
                use openexr::header::Header;
                use openexr::output::ScanlineOutputFile;
                use openexr::PixelType;

                let mut file = std::fs::File::create("/tmp/output.exr").unwrap();
                let mut output_file = ScanlineOutputFile::new(
                    &mut file,
                    Header::new()
                        .set_resolution(width as u32, height as u32)
                        .add_channel("R", PixelType::FLOAT)
                        .add_channel("G", PixelType::FLOAT)
                        .add_channel("B", PixelType::FLOAT),
                )
                .unwrap();
                let mut fb = FrameBuffer::new(width as u32, height as u32);
                fb.insert_channels(&["R", "G", "B"], pixels);
                output_file.write_pixels(&fb).unwrap();
            }
        },
    );
}
