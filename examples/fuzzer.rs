mod profile_with_puffin;

/*

Kind of an example, kind of a test - not good looking
Used for code coverage with https://github.com/mozilla/grcov

First time:
    rustup component add llvm-tools-preview

In a separate terminal:
    export RUSTFLAGS="-Cinstrument-coverage"
    cargo build --example fuzzer

Next:
    export LLVM_PROFILE_FILE="fuzzer-%p-%m.profraw"
    target/debug/examples/fuzzer


Also helpful to run with valgrind:
    cargo build --example fuzzer && valgrind target/debug/examples/fuzzer

*/
use {
    clap::Parser,
    log::debug,
    rand::{Rng, rng, seq::IndexedRandom},
    std::mem::size_of,
    vk_graph_prelude::*,
    vk_graph_window::{FrameContext, WindowBuilder, WindowError},
    vk_shader_macros::glsl,
};

type Operation = fn(&mut FrameContext, &mut HashPool);

static OPERATIONS: &[Operation] = &[
    record_pipeline_array_bind,
    record_pipeline_bindless,
    record_pipeline_no_op,
    record_graphic_bindless,
    record_graphic_load_store,
    record_graphic_msaa_depth_stencil,
    record_graphic_will_merge_subpass_input,
    record_graphic_will_merge_common_color1,
    record_graphic_will_merge_common_color2,
    record_graphic_will_merge_common_depth1,
    record_graphic_will_merge_common_depth2,
    record_graphic_will_merge_common_depth3,
    record_graphic_wont_merge,
    record_accel_struct_builds,
    record_transfer_graphic_multipass,
];

fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let mut rng = rng();

    let vk_graph = WindowBuilder::default().debug(true).build()?;
    let mut pool = HashPool::new(&vk_graph.device);

    let mut frame_count = 0;

    let args = Args::parse();

    vk_graph.run(|mut frame| {
        if frame_count == args.frame_count {
            *frame.will_exit = true;
            return;
        }

        frame_count += 1;

        // We are not testing the swapchain - so always clear it
        let clear_before: bool = rng.random();

        if clear_before {
            frame
                .render_graph
                .clear_color_image(frame.swapchain_image, [0f32; 4]);
        }

        for _ in 0..args.ops_per_frame {
            let op_fn = OPERATIONS.choose(&mut rng).unwrap();
            op_fn(&mut frame, &mut pool);
        }

        if !clear_before {
            frame
                .render_graph
                .clear_color_image(frame.swapchain_image, [0f32; 4]);
        }
    })?;

    debug!("OK");

    Ok(())
}

fn record_accel_struct_builds(frame: &mut FrameContext, pool: &mut HashPool) {
    const BLAS_COUNT: vk::DeviceSize = 32;

    // Vertex buffer for a triangle
    let vertex_buf = {
        let mut buf = pool
            .lease(BufferInfo::host_mem(
                36,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            ))
            .unwrap();

        // Vertex 1
        Buffer::copy_from_slice(&mut buf, 0, 0f32.to_ne_bytes());
        Buffer::copy_from_slice(&mut buf, 4, 0f32.to_ne_bytes());
        Buffer::copy_from_slice(&mut buf, 8, 0f32.to_ne_bytes());

        // Vertex 2
        Buffer::copy_from_slice(&mut buf, 12, 1f32.to_ne_bytes());
        Buffer::copy_from_slice(&mut buf, 16, 1f32.to_ne_bytes());
        Buffer::copy_from_slice(&mut buf, 20, 0f32.to_ne_bytes());

        // Vertex 3
        Buffer::copy_from_slice(&mut buf, 24, 2f32.to_ne_bytes());
        Buffer::copy_from_slice(&mut buf, 28, 0f32.to_ne_bytes());
        Buffer::copy_from_slice(&mut buf, 32, 0f32.to_ne_bytes());

        buf
    };

    // Index buffer for a single triangle
    let index_buf = {
        let mut buf = pool
            .lease(BufferInfo::host_mem(
                6,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            ))
            .unwrap();

        Buffer::copy_from_slice(&mut buf, 0, 0u16.to_ne_bytes());
        Buffer::copy_from_slice(&mut buf, 2, 1u16.to_ne_bytes());
        Buffer::copy_from_slice(&mut buf, 4, 2u16.to_ne_bytes());

        buf
    };

    let blas_geometry_info = AccelerationStructureGeometryInfo::blas([(
        AccelerationStructureGeometry {
            max_primitive_count: 1,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            geometry: AccelerationStructureGeometryData::Triangles {
                index_addr: DeviceOrHostAddress::DeviceAddress(Buffer::device_address(&index_buf)),
                index_type: vk::IndexType::UINT16,
                max_vertex: 3,
                transform_addr: None,
                vertex_addr: DeviceOrHostAddress::DeviceAddress(Buffer::device_address(
                    &vertex_buf,
                )),
                vertex_format: vk::Format::R32G32B32_SFLOAT,
                vertex_stride: 12,
            },
        },
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1),
    )]);
    let blas_size = AccelerationStructure::size_of(frame.device, &blas_geometry_info);
    let blas_info = AccelerationStructureInfo::blas(blas_size.create_size);

    let instance_len = size_of::<vk::AccelerationStructureInstanceKHR>() as vk::DeviceSize;
    let mut instance_buf = Buffer::create(
        frame.device,
        BufferInfo::host_mem(
            instance_len * BLAS_COUNT,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        ),
    )
    .unwrap();

    let accel_struct_scratch_offset_alignment = frame
        .device
        .physical_device
        .accel_struct_properties
        .as_ref()
        .unwrap()
        .min_accel_struct_scratch_offset_alignment
        as vk::DeviceSize;

    // Lease and bind a bunch of bottom-level acceleration structures and add to instance buffer
    let mut blas_nodes = Vec::with_capacity(BLAS_COUNT as _);
    for idx in 0..BLAS_COUNT {
        let blas = pool.lease(blas_info).unwrap();

        Buffer::copy_from_slice(
            &mut instance_buf,
            idx * instance_len,
            AccelerationStructure::instance_slice(&[vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: [
                        1.0, 0.0, 0.0, 0.0, //
                        0.0, 1.0, 0.0, 0.0, //
                        0.0, 0.0, 1.0, 0.0, //
                    ],
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as _,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: AccelerationStructure::device_address(&blas),
                },
            }]),
        );

        let blas_node = frame.render_graph.bind_node(blas);
        let scratch_buf = frame.render_graph.bind_node(
            pool.lease(
                BufferInfo::device_mem(
                    blas_size.build_size,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                )
                .to_builder()
                .alignment(accel_struct_scratch_offset_alignment),
            )
            .unwrap(),
        );

        blas_nodes.push((scratch_buf, blas_node));
    }

    // Lease and bind a single top-level acceleration structure
    let tlas_geometry_info = AccelerationStructureGeometryInfo::tlas([(
        AccelerationStructureGeometry {
            max_primitive_count: 1,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            geometry: AccelerationStructureGeometryData::Instances {
                array_of_pointers: false,
                addr: DeviceOrHostAddress::DeviceAddress(Buffer::device_address(&instance_buf)),
            },
        },
        vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1),
    )]);
    let instance_buf = frame.render_graph.bind_node(instance_buf);
    let tlas_size = AccelerationStructure::size_of(frame.device, &tlas_geometry_info);
    let tlas = pool
        .lease(AccelerationStructureInfo::tlas(tlas_size.create_size))
        .unwrap();
    let tlas_node = frame.render_graph.bind_node(tlas);
    let tlas_scratch_buf = frame.render_graph.bind_node(
        pool.lease(
            BufferInfo::device_mem(
                tlas_size.build_size,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .to_builder()
            .alignment(accel_struct_scratch_offset_alignment),
        )
        .unwrap(),
    );

    let index_node = frame.render_graph.bind_node(index_buf);
    let vertex_node = frame.render_graph.bind_node(vertex_buf);

    let pass = frame
        .render_graph
        .begin_cmd()
        .with_name("build acceleration structures");

    // TODO: AccessType for these is funky, should be access_node?
    let mut pass = pass.read_node(index_node).read_node(vertex_node);

    // TODO: Like this:
    for (scratch_buf, blas_node) in &blas_nodes {
        pass.access_node_mut(*scratch_buf, AccessType::AccelerationStructureBufferWrite);
        pass.access_node_mut(*blas_node, AccessType::AccelerationStructureBuildWrite);
    }

    // Ugly copy of the nodes that I want to figure out a way around while not being confusing
    let blas_nodes_copy = blas_nodes
        .iter()
        .map(|(_, blas_node)| *blas_node)
        .collect::<Vec<_>>();

    let mut pass = pass.record_acceleration(move |accel, bindings| {
        for (scratch_buf, blas_node) in blas_nodes {
            let scratch_data = Buffer::device_address(&bindings[scratch_buf]);
            accel.build_structure(&blas_geometry_info, blas_node, scratch_data);
        }
    });

    for blas_node in blas_nodes_copy {
        pass.access_node_mut(blas_node, AccessType::AccelerationStructureBuildRead);
    }

    pass.access_node_mut(instance_buf, AccessType::AccelerationStructureBuildRead);
    pass.access_node_mut(
        tlas_scratch_buf,
        AccessType::AccelerationStructureBufferWrite,
    );
    pass.access_node_mut(tlas_node, AccessType::AccelerationStructureBuildWrite);

    pass.record_acceleration(move |accel, bindings| {
        let scratch_data = Buffer::device_address(&bindings[tlas_scratch_buf]);
        accel.build_structure(&tlas_geometry_info, tlas_node, scratch_data);
    });
}

fn record_pipeline_array_bind(frame: &mut FrameContext, pool: &mut HashPool) {
    let pipeline = compute_pipeline(
        "array_bind",
        frame.device,
        ComputePipelineInfo::default(),
        Shader::new_compute(
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(compute)

                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

                layout(constant_id = 0) const uint LAYER_COUNT = 1;

                layout(push_constant) uniform PushConstants {
                    layout(offset = 0) float offset;
                } push_const;

                layout(set = 0, binding = 0) uniform sampler2D layer_images_sampler_llr[LAYER_COUNT];

                void main() {
                }
                "#
            )
            .as_slice(),
        )
        .specialization_info(SpecializationInfo::new(
            vec![vk::SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: 4,
            }],
            5u32.to_ne_bytes(),
        )),
    );

    let image_info = ImageInfo::image_2d(
        64,
        64,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
    );
    let images = [
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
    ];

    frame
        .render_graph
        .clear_color_image(images[0], [0f32; 4])
        .clear_color_image(images[1], [0f32; 4])
        .clear_color_image(images[2], [0f32; 4])
        .clear_color_image(images[3], [0f32; 4])
        .clear_color_image(images[4], [0f32; 4])
        .begin_cmd()
        .with_name("array-bind")
        .bind_pipeline(&pipeline)
        .read_descriptor((0, [0]), images[0])
        .read_descriptor((0, [1]), images[1])
        .read_descriptor((0, [2]), images[2])
        .read_descriptor((0, [3]), images[3])
        .read_descriptor((0, [4]), images[4])
        .record_pipeline(|compute, _| {
            compute
                .push_constants(0, &0f32.to_ne_bytes())
                .dispatch(64, 64, 1);
        });
}

fn record_pipeline_bindless(frame: &mut FrameContext, pool: &mut HashPool) {
    let pipeline = compute_pipeline(
        "bindless",
        frame.device,
        ComputePipelineInfo::default(),
        Shader::new_compute(
            glsl!(
                r#"
                #version 460 core
                #extension GL_EXT_nonuniform_qualifier : require
                #pragma shader_stage(compute)

                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

                layout(push_constant) uniform PushConstants {
                    layout(offset = 0) uint count;
                } push_const;

                layout(set = 0, binding = 0, rgba8) writeonly uniform image2D dst[];

                void main() {
                    for (uint idx = 0; idx < push_const.count; idx++) {
                        imageStore(
                            dst[idx],
                            ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y),
                            vec4(0)
                        );
                    }
                }
                "#
            )
            .as_slice(),
        ),
    );

    let image_info = ImageInfo::image_2d(
        64,
        64,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
    );
    let images = [
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
    ];

    frame
        .render_graph
        .begin_cmd()
        .with_name("compute-bindless")
        .bind_pipeline(&pipeline)
        .write_descriptor((0, [0]), images[0])
        .write_descriptor((0, [1]), images[1])
        .write_descriptor((0, [2]), images[2])
        .write_descriptor((0, [3]), images[3])
        .write_descriptor((0, [4]), images[4])
        .record_pipeline(|compute, _| {
            compute
                .push_constants(0, &5u32.to_ne_bytes())
                .dispatch(64, 64, 1);
        });
}

fn record_pipeline_no_op(frame: &mut FrameContext, _: &mut HashPool) {
    let pipeline = compute_pipeline(
        "no_op",
        frame.device,
        ComputePipelineInfo::default(),
        Shader::new_compute(
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(compute)

                void main() {
                }
                "#
            )
            .as_slice(),
        ),
    );
    frame
        .render_graph
        .begin_cmd()
        .with_name("no-op")
        .bind_pipeline(&pipeline)
        .record_pipeline(|compute, _| {
            compute.dispatch(1, 1, 1);
        });
}

fn record_graphic_bindless(frame: &mut FrameContext, pool: &mut HashPool) {
    let pipeline = graphic_vert_frag_pipeline(
        frame.device,
        GraphicPipelineInfo::default(),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(vertex)

            void main() {
            }
            "#
        )
        .as_slice(),
        glsl!(
            r#"
            #version 460 core
            #extension GL_EXT_nonuniform_qualifier : require
            #pragma shader_stage(fragment)

            layout(push_constant) uniform PushConstants {
                layout(offset = 0) uint count;
            } push_const;

            layout(set = 0, binding = 0) uniform sampler2D images_sampler_llr[];

            layout(location = 0) out vec4 color_out;

            void main() {
                for (uint idx = 0; idx < push_const.count; idx++) {
                    color_out += texture(
                        images_sampler_llr[idx],
                        vec2(0)
                    );
                }
            }
            "#
        )
        .as_slice(),
    );

    let image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
        ))
        .unwrap(),
    );
    let image_info = ImageInfo::image_2d(
        64,
        64,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_DST,
    );
    let images = [
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
        frame
            .render_graph
            .bind_node(pool.lease(image_info).unwrap()),
    ];

    frame
        .render_graph
        .clear_color_image(images[0], [0f32; 4])
        .clear_color_image(images[1], [0f32; 4])
        .clear_color_image(images[2], [0f32; 4])
        .clear_color_image(images[3], [0f32; 4])
        .clear_color_image(images[4], [0f32; 4])
        .begin_cmd()
        .with_name("graphic-bindless")
        .bind_pipeline(&pipeline)
        .read_descriptor((0, [0]), images[0])
        .read_descriptor((0, [1]), images[1])
        .read_descriptor((0, [2]), images[2])
        .read_descriptor((0, [3]), images[3])
        .read_descriptor((0, [4]), images[4])
        .clear_color(0, image)
        .store_color(0, image)
        .record_pipeline(|pipeline, _| {
            pipeline
                .push_constants(0, &5u32.to_ne_bytes())
                .draw(1, 1, 0, 0);
        });
}

fn record_graphic_load_store(frame: &mut FrameContext, _: &mut HashPool) {
    let pipeline = graphic_vert_frag_pipeline(
        frame.device,
        GraphicPipelineInfo::default(),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(vertex)

            void main() {
            }
            "#
        )
        .as_slice(),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(fragment)

            layout(location = 0) out vec4 color_out;

            void main() {
                color_out = vec4(0);
            }
            "#
        )
        .as_slice(),
    );

    frame
        .render_graph
        .begin_cmd()
        .with_name("load-store")
        .bind_pipeline(&pipeline)
        .load_color(0, frame.swapchain_image)
        .store_color(0, frame.swapchain_image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

fn record_graphic_msaa_depth_stencil(frame: &mut FrameContext, pool: &mut HashPool) {
    let sample_count = {
        let Vulkan10Limits {
            framebuffer_color_sample_counts,
            framebuffer_depth_sample_counts,
            framebuffer_stencil_sample_counts,
            ..
        } = frame.device.physical_device.properties_v1_0.limits;
        match framebuffer_color_sample_counts
            & framebuffer_depth_sample_counts
            & framebuffer_stencil_sample_counts
        {
            s if s.contains(vk::SampleCountFlags::TYPE_64) => SampleCount::Type64,
            s if s.contains(vk::SampleCountFlags::TYPE_32) => SampleCount::Type32,
            s if s.contains(vk::SampleCountFlags::TYPE_16) => SampleCount::Type16,
            s if s.contains(vk::SampleCountFlags::TYPE_8) => SampleCount::Type8,
            s if s.contains(vk::SampleCountFlags::TYPE_4) => SampleCount::Type4,
            s if s.contains(vk::SampleCountFlags::TYPE_2) => SampleCount::Type2,
            s if s.contains(vk::SampleCountFlags::TYPE_1) => SampleCount::Type1,
            _ => panic!("unsupported depth/stencil msaa"),
        }
    };
    let depth_stencil_format = {
        let mut best_format = None;
        for format in [
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D16_UNORM_S8_UINT,
            vk::Format::D32_SFLOAT_S8_UINT,
        ] {
            let format_props = frame.device.physical_device.image_format_properties(
                format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                vk::ImageCreateFlags::empty(),
            );

            if format_props.is_ok() {
                best_format = Some(format);
                break;
            }
        }

        best_format.expect("Unsupported depth/stencil format")
    };
    let depth_resolve_mode = {
        let mut best_mode = None;
        for (resolve_flags, resolve_mode) in [
            (vk::ResolveModeFlags::AVERAGE, ResolveMode::Average),
            (vk::ResolveModeFlags::SAMPLE_ZERO, ResolveMode::SampleZero),
        ] {
            if frame
                .device
                .physical_device
                .depth_stencil_resolve_properties
                .supported_depth_resolve_modes
                .contains(resolve_flags)
            {
                best_mode = Some(resolve_mode);
                break;
            }
        }

        best_mode.expect("Unsupported depth resolve mode")
    };

    let pipeline = graphic_vert_frag_pipeline(
        frame.device,
        GraphicPipelineInfoBuilder::default().samples(sample_count),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(vertex)

            const vec2 UV[3] = {
                vec2(-1, -1),
                vec2(-1, 1),
                vec2(1, 1),
            };

            void main() {
                gl_Position = vec4(UV[gl_VertexIndex], 0, 1);
            }
            "#
        )
        .as_slice(),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(fragment)

            layout(location = 0) out vec4 color_out;

            void main() {
                color_out = vec4(1);
            }
            "#
        )
        .as_slice(),
    );

    let swapchain_format = frame.render_graph.node_info(frame.swapchain_image).fmt;
    let msaa_color_image = frame.render_graph.bind_node(
        pool.lease(
            ImageInfo::image_2d(
                frame.width,
                frame.height,
                swapchain_format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            )
            .to_builder()
            .sample_count(sample_count),
        )
        .unwrap(),
    );
    let msaa_depth_stencil_image = frame.render_graph.bind_node(
        pool.lease(
            ImageInfo::image_2d(
                frame.width,
                frame.height,
                depth_stencil_format,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            )
            .to_builder()
            .sample_count(sample_count),
        )
        .unwrap(),
    );
    let depth_stencil_image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            frame.width,
            frame.height,
            depth_stencil_format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ))
        .unwrap(),
    );

    let depth_stencil_mode = DepthStencilMode {
        back: StencilMode::IGNORE,
        bounds_test: true,
        compare_op: vk::CompareOp::LESS_OR_EQUAL,
        depth_test: true,
        depth_write: true,
        front: StencilMode {
            fail_op: vk::StencilOp::ZERO,
            pass_op: vk::StencilOp::REPLACE,
            depth_fail_op: vk::StencilOp::ZERO,
            compare_op: vk::CompareOp::LESS_OR_EQUAL,
            compare_mask: 0xff,
            write_mask: 0xff,
            reference: 0x00,
        },
        min: 0.0.into(),
        max: 1.0.into(),
        stencil_test: true,
    };

    frame
        .render_graph
        .begin_cmd()
        .with_name("msaa-depth-stencil")
        .bind_pipeline(&pipeline)
        .set_depth_stencil(depth_stencil_mode)
        .clear_color(0, msaa_color_image)
        .clear_depth_stencil(msaa_depth_stencil_image)
        .resolve_color(0, 1, frame.swapchain_image)
        .resolve_depth_stencil(
            2,
            depth_stencil_image,
            Some(depth_resolve_mode),
            Some(ResolveMode::SampleZero),
        )
        .record_pipeline(|pipeline, _| {
            pipeline.draw(3, 1, 0, 0);
        });
}

fn record_graphic_will_merge_common_color1(frame: &mut FrameContext, pool: &mut HashPool) {
    let image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ))
        .unwrap(),
    );

    // Pass "a" stores color0 which "b" compatibly loads; so these two will get merged
    frame
        .render_graph
        .begin_cmd()
        .with_name("a")
        .bind_pipeline(graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                layout(location = 0) out vec4 color0;

                void main() {
                    color0 = vec4(0);
                }
                "#
            )
            .as_slice(),
        ))
        .store_color(0, image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("b")
        .bind_pipeline(&graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                layout(location = 0) out vec4 color0;

                void main() {
                    color0 = vec4(0);
                }
                "#
            )
            .as_slice(),
        ))
        .load_color(0, image)
        .store_color(0, image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

fn record_graphic_will_merge_common_color2(frame: &mut FrameContext, pool: &mut HashPool) {
    let image_0 = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ))
        .unwrap(),
    );
    let image_1 = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ))
        .unwrap(),
    );

    frame
        .render_graph
        .begin_cmd()
        .with_name("a")
        .bind_pipeline(graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                layout(location = 0) out vec4 color0;

                void main() {
                    color0 = vec4(0);
                }
                "#
            )
            .as_slice(),
        ))
        .store_color(0, image_0)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("b")
        .bind_pipeline(&graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                layout(location = 0) out vec4 color0;
                layout(location = 1) out vec4 color1;

                void main() {
                    color0 = vec4(0);
                    color1 = vec4(0);
                }
                "#
            )
            .as_slice(),
        ))
        .load_color(0, image_0)
        .store_color(0, image_0)
        .store_color(1, image_1)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("c")
        .bind_pipeline(&graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                layout(location = 0) out vec4 color0;

                void main() {
                    color0 = vec4(0);
                }
                "#
            )
            .as_slice(),
        ))
        .clear_color(0, image_0)
        .store_color(0, image_0)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

fn record_graphic_will_merge_common_depth1(frame: &mut FrameContext, pool: &mut HashPool) {
    let color_image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ))
        .unwrap(),
    );
    let depth_image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ))
        .unwrap(),
    );

    // Pass "a" stores color0+depth which "b" compatibly loads; so these two will get merged
    frame
        .render_graph
        .begin_cmd()
        .with_name("a")
        .bind_pipeline(graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                layout(location = 0) out vec4 color_out;

                void main() {
                    color_out = vec4(0);
                }
                "#
            )
            .as_slice(),
        ))
        .store_color(0, color_image)
        .store_depth_stencil(depth_image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("b")
        .bind_pipeline(graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                void main() {
                    gl_FragDepth = 0.0;
                }
                "#
            )
            .as_slice(),
        ))
        .load_depth_stencil(depth_image)
        .store_depth_stencil(depth_image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

fn record_graphic_will_merge_common_depth2(frame: &mut FrameContext, pool: &mut HashPool) {
    let color_image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ))
        .unwrap(),
    );
    let depth_image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ))
        .unwrap(),
    );

    // Pass "a" stores color0+depth which "b" compatibly loads; so these two will get merged
    frame
        .render_graph
        .begin_cmd()
        .with_name("a")
        .bind_pipeline(graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                void main() {
                    gl_FragDepth = 0.0;
                }
                "#
            )
            .as_slice(),
        ))
        .store_depth_stencil(depth_image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("b")
        .bind_pipeline(graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                layout(location = 0) out vec4 color_out;

                void main() {
                    color_out = vec4(0);
                }
                "#
            )
            .as_slice(),
        ))
        .store_color(0, color_image)
        .load_depth_stencil(depth_image)
        .store_depth_stencil(depth_image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

fn record_graphic_will_merge_common_depth3(frame: &mut FrameContext, pool: &mut HashPool) {
    let depth_image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ))
        .unwrap(),
    );

    frame
        .render_graph
        .begin_cmd()
        .with_name("a")
        .bind_pipeline(graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(                kind: frag,
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                void main() {
                    gl_FragDepth = 0.0;
                }
                "#
            )
            .as_slice(),
        ))
        .store_depth_stencil(depth_image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("b")
        .bind_pipeline(graphic_vert_frag_pipeline(
            frame.device,
            GraphicPipelineInfo::default(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(vertex)

                void main() { }
                "#
            )
            .as_slice(),
            glsl!(
                r#"
                #version 460 core
                #pragma shader_stage(fragment)

                void main() {
                    gl_FragDepth = 0.0;
                }
                "#
            )
            .as_slice(),
        ))
        .load_depth_stencil(depth_image)
        .store_depth_stencil(depth_image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

fn record_graphic_will_merge_subpass_input(frame: &mut FrameContext, pool: &mut HashPool) {
    let vertex = glsl!(
        r#"
        #version 460 core
        #pragma shader_stage(vertex)

        void main() {
        }
        "#
    )
    .as_slice();
    let pipeline_a = graphic_vert_frag_pipeline(
        frame.device,
        GraphicPipelineInfo::default(),
        vertex,
        glsl!(            kind: frag,
            r#"
            #version 460 core
            #pragma shader_stage(fragment)

            layout(location = 0) out vec4 color_out;

            void main() {
                color_out = vec4(0);
            }
            "#
        )
        .as_slice(),
    );
    let pipeline_b = graphic_vert_frag_pipeline(
        frame.device,
        GraphicPipelineInfo::default(),
        vertex,
        glsl!(
            kind: frag,
            r#"
            #version 460 core
            #pragma shader_stage(fragment)

            layout(input_attachment_index = 0, binding = 0) uniform subpassInput color_in;
            layout(location = 0) out vec4 color_out;

            void main() {
                color_out = subpassLoad(color_in);
            }
            "#
        )
        .as_slice(),
    );
    let image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::INPUT_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST,
        ))
        .unwrap(),
    );

    // Pass "a" stores color 0 which "b" compatibly inputs; so these two will get merged
    frame
        .render_graph
        .begin_cmd()
        .with_name("a")
        .bind_pipeline(&pipeline_a)
        .clear_color(0, image)
        .store_color(0, image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("b")
        .bind_pipeline(&pipeline_b)
        .store_color(0, image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

fn record_graphic_wont_merge(frame: &mut FrameContext, pool: &mut HashPool) {
    let pipeline = graphic_vert_frag_pipeline(
        frame.device,
        GraphicPipelineInfo::default(),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(vertex)

            void main() {
            }
            "#
        )
        .as_slice(),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(fragment)

            layout(location = 0) out vec4 color;

            void main() {
            }
            "#
        )
        .as_slice(),
    );

    let image = frame.render_graph.bind_node(
        pool.lease(ImageInfo::image_2d(
            256,
            256,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ))
        .unwrap(),
    );

    // These two passes have common writes but are otherwise regular - they won't get merged
    frame
        .render_graph
        .begin_cmd()
        .with_name("c")
        .bind_pipeline(&pipeline)
        .store_color(0, image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("d")
        .bind_pipeline(&pipeline)
        .store_color(0, image)
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

fn record_transfer_graphic_multipass(frame: &mut FrameContext, pool: &mut HashPool) {
    let pipeline = graphic_vert_frag_pipeline(
        frame.device,
        GraphicPipelineInfo::default(),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(vertex)

            void main() {
            }
            "#
        )
        .as_slice(),
        glsl!(
            r#"
            #version 460 core
            #pragma shader_stage(fragment)

            layout(binding = 0) uniform sampler2D my_sampler_lle;

            layout(location = 0) out vec4 color_out;

            void main() {
                color_out = texture(my_sampler_lle, vec2(0));
            }
            "#
        )
        .as_slice(),
    );
    let images = [
        frame.render_graph.bind_node(
            pool.lease(ImageInfo::image_2d(
                256,
                256,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ))
            .unwrap(),
        ),
        frame.render_graph.bind_node(
            pool.lease(ImageInfo::image_2d(
                256,
                256,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ))
            .unwrap(),
        ),
    ];

    frame.render_graph.clear_color_image(images[0], [0f32; 4]);
    frame.render_graph.clear_color_image(images[1], [0f32; 4]);

    // a and b should merge into one renderpass with two subpasses; however the use of images[1] in
    // b should have a pipeline barrier (on the clear we just did) before the pass starts.
    frame
        .render_graph
        .begin_cmd()
        .with_name("a")
        .bind_pipeline(&pipeline)
        .clear_color(0, frame.swapchain_image)
        .store_color(0, frame.swapchain_image)
        .read_descriptor(0, images[0])
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
    frame
        .render_graph
        .begin_cmd()
        .with_name("b")
        .bind_pipeline(&pipeline)
        .load_color(0, frame.swapchain_image)
        .store_color(0, frame.swapchain_image)
        .read_descriptor(0, images[1])
        .record_pipeline(|pipeline, _| {
            pipeline.draw(1, 1, 0, 0);
        });
}

// Below are convenience functions used to create test data

fn compute_pipeline(
    key: &'static str,
    device: &Device,
    info: impl Into<ComputePipelineInfo>,
    shader: impl Into<Shader>,
) -> ComputePipeline {
    use std::{cell::RefCell, collections::HashMap};

    thread_local! {
        static TLS: RefCell<HashMap<&'static str, ComputePipeline>> = Default::default();
    }

    TLS.with(|tls| {
        tls.borrow_mut()
            .entry(key)
            .or_insert_with(|| ComputePipeline::create(device, info, shader).unwrap())
            .clone()
    })
}

fn graphic_vert_frag_pipeline(
    device: &Device,
    info: impl Into<GraphicPipelineInfo>,
    vert_source: &'static [u32],
    frag_source: &'static [u32],
) -> GraphicPipeline {
    use std::{cell::RefCell, collections::HashMap};

    #[derive(Eq, Hash, PartialEq)]
    struct Key {
        info: GraphicPipelineInfo,
        vert_source: &'static [u32],
        frag_source: &'static [u32],
    }

    thread_local! {
        static TLS: RefCell<HashMap<Key, GraphicPipeline>> = Default::default();
    }

    let info = info.into();

    TLS.with(|tls| {
        tls.borrow_mut()
            .entry(Key {
                info,
                vert_source,
                frag_source,
            })
            .or_insert_with(move || {
                GraphicPipeline::create(
                    device,
                    info,
                    [
                        Shader::new_vertex(vert_source),
                        Shader::new_fragment(frag_source),
                    ],
                )
                .unwrap()
            })
            .clone()
    })
}

#[derive(Parser)]
struct Args {
    /// Count of fuzzing frames
    #[arg(long, default_value_t = 100)]
    frame_count: usize,

    /// Count of operations run per fuzzing frame
    #[arg(long, default_value_t = 16)]
    ops_per_frame: usize,
}
