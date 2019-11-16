extern crate wgpu;

extern crate shaderc;

extern crate rand;

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
    position: [f32; 4],
    diffuse: [f32; 4],
    emissive: [f32; 4],
    r: f32,
    material: i32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct SampleInfo {
    id: u32,
    weight: f32,
    sample_offset: [f32; 2],
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

    let width = 800;
    let height = 600;

    let inv_half_height: f32 = 1.0 / (height / 2) as f32;

    let inputs: Vec<[f32; 4]> = (0..height)
        .flat_map(move |y| {
            (0..width).map(move |x| {
                [
                    (x as f32 - 0.5 * width as f32) * inv_half_height,
                    -(y as f32 - 0.5 * height as f32) * inv_half_height,
                    1.0,
                    0.0,
                ]
            })
        })
        .collect();
    let input_buffer_size = (inputs.len() * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress;

    let num_samples = 512;
    let samples: Vec<SampleInfo> = (0..num_samples)
        .map(|i| SampleInfo {
            id: i,
            weight: 1./num_samples as f32,
            sample_offset: [
                inv_half_height * (rand::random::<f32>() - 0.5),
                inv_half_height * (rand::random::<f32>() - 0.5),
            ],
        })
        .collect();
    let sample_buffer_size = (samples.len() * std::mem::size_of::<SampleInfo>()) as wgpu::BufferAddress;

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
    let scene_buffer_size = (spheres.len() * std::mem::size_of::<Sphere>()) as wgpu::BufferAddress;

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

    let scene_staging_buffer = device
        .create_buffer_mapped(spheres.len(), wgpu::BufferUsage::COPY_SRC)
        .fill_from_slice(&spheres[..]);

    let scene_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: scene_buffer_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            // input
            wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
            // scene
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
            // output
            wgpu::BindGroupLayoutBinding {
                binding: 3,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[
            // input
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &input_storage_buffer,
                    range: 0..input_buffer_size,
                },
            },
            // scene
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &scene_storage_buffer,
                    range: 0..scene_buffer_size,
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
            // output
            wgpu::Binding {
                binding: 3,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &output_storage_buffer,
                    range: 0..input_buffer_size,
                },
            },
        ],
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
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
