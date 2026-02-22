use {
    bytemuck::{bytes_of, cast_slice, NoUninit},
    clap::Parser,
    glam::{Mat4, Quat, Vec3},
    pak::{
        anim::{Channel, Interpolation, Outputs},
        bitmap::BitmapFormat,
        model::{Joint, Vertex},
        Pak, PakBuf,
    },
    std::{
        cmp::Ordering,
        env::current_exe,
        iter::repeat_n,
        mem::{size_of, size_of_val},
        sync::Arc,
        time::{Duration, Instant},
    },
    vk_graph::{
        driver::{
            ash::vk,
            buffer::{Buffer, BufferInfo},
            device::Device,
            graphic::{DepthStencilMode, GraphicPipeline, GraphicPipelineInfoBuilder},
            image::{Image, ImageInfo},
            shader::Shader,
            AccessType, DriverError,
        },
        pool::{hash::HashPool, lazy::LazyPool, Pool as _},
        Graph,
    },
    vk_graph_window::{WindowBuilder, WindowError},
};

// This blog has a really good overview of what is happening here:
// https://vladh.net/game-engine-skeletal-animation
fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();

    let pak_path = current_exe().unwrap().parent().unwrap().join("res.pak");
    let mut pak = PakBuf::open(pak_path).unwrap();

    let args = Args::parse();
    let window = WindowBuilder::default().debug(args.debug).build()?;
    let device = &window.device;

    let pipeline = create_pipeline(device, &mut pak)?;
    let human_female = load_texture(device, &mut pak, "animated_characters_3/human_female")?;
    let human_male = load_texture(device, &mut pak, "animated_characters_3/human_male")?;
    let zombie_female = load_texture(device, &mut pak, "animated_characters_3/zombie_female")?;
    let zombie_male = load_texture(device, &mut pak, "animated_characters_3/zombie_male")?;
    let character = Model::load(device, &mut pak, "animated_characters_3/character_medium")?;
    let mut idle = Animation::load(&character, &mut pak, "animated_characters_3/idle")?;
    let mut run = Animation::load(&character, &mut pak, "animated_characters_3/run")?;

    let mut pool = LazyPool::new(device);
    let started = Instant::now();

    window.run(|frame| {
        let elapsed = (Instant::now() - started).as_secs_f32();

        let index_buf = frame.graph.bind_node(&character.index_buf);
        let vertex_buf = frame.graph.bind_node(&character.vertex_buf);
        let depth_image = frame.graph.bind_node(
            pool.lease(ImageInfo::image_2d(
                frame.width,
                frame.height,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .unwrap(),
        );

        let texture = frame
            .graph
            .bind_node(match (elapsed / 2.0).rem_euclid(4.0) {
                t if t < 1.0 => &human_female,
                t if t < 2.0 => &human_male,
                t if t < 3.0 => &zombie_female,
                _ => &zombie_male,
            });

        let camera_buf = frame.graph.bind_node({
            let position = Vec3::ONE * 3.0;
            let aspect_ratio = frame.render_aspect_ratio();
            let projection = Mat4::perspective_rh(45.0, aspect_ratio, 0.1, 100.0);
            let view = Mat4::look_at_rh(position, Vec3::Y * 2.0, -Vec3::Y);
            let mut buf = pool
                .lease(BufferInfo::host_mem(
                    size_of::<CameraUniform>() as _,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                ))
                .unwrap();

            Buffer::copy_from_slice(
                &mut buf,
                0,
                bytes_of(&CameraUniform {
                    projection,
                    view,
                    position,
                }),
            );

            buf
        });

        let animation_buf = frame.graph.bind_node({
            let animation = match (elapsed / 4.0).rem_euclid(2.0) {
                t if t < 1.0 => &mut run,
                _ => &mut idle,
            };
            let joints = animation.update(0.016);
            let mut buf = pool
                .lease(BufferInfo::host_mem(
                    size_of_val(joints) as _,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ))
                .unwrap();

            Buffer::copy_from_slice(&mut buf, 0, cast_slice(joints));

            buf
        });

        frame
            .graph
            .begin_cmd()
            .with_name("🦴")
            .bind_pipeline(&pipeline)
            .set_depth_stencil(DepthStencilMode::DEPTH_WRITE)
            .access_node(index_buf, AccessType::IndexBuffer)
            .access_node(vertex_buf, AccessType::VertexBuffer)
            .access_descriptor(0, camera_buf, AccessType::VertexShaderReadUniformBuffer)
            .access_descriptor(1, animation_buf, AccessType::VertexShaderReadOther)
            .read_descriptor(2, texture)
            .clear_color(0, frame.swapchain_image)
            .store_color(0, frame.swapchain_image)
            .clear_depth_stencil(depth_image)
            .record_pipeline(move |pipeline, _| {
                pipeline
                    .bind_index_buffer(index_buf, 0, vk::IndexType::UINT16)
                    .bind_vertex_buffer(0, vertex_buf, 0)
                    .push_constants(0, bytes_of(&Mat4::IDENTITY))
                    .draw_indexed(character.index_count, 1, 0, 0, 0);
            });
    })
}

fn create_pipeline(device: &Device, pak: &mut PakBuf) -> Result<GraphicPipeline, DriverError> {
    let vert_spirv = pak.read_blob("shader/animated_mesh_vert.spirv").unwrap();
    let frag_spirv = pak.read_blob("shader/mesh_frag.spirv").unwrap();

    GraphicPipeline::create(
        device,
        GraphicPipelineInfoBuilder::default().front_face(vk::FrontFace::CLOCKWISE),
        [
            Shader::new_vertex(vert_spirv.as_slice()),
            Shader::new_fragment(frag_spirv.as_slice()),
        ],
    )
}

fn load_texture(device: &Device, pak: &mut PakBuf, key: &str) -> Result<Arc<Image>, DriverError> {
    let bitmap = pak.read_bitmap(key).unwrap();

    assert_eq!(bitmap.format(), BitmapFormat::Rgba);
    assert_eq!(bitmap.width().count_ones(), 1);
    assert_eq!(bitmap.height().count_ones(), 1);

    // NOTE: This is the most basic way to load an image; you probably want to use something like
    // vk-graph-fx::ImageLoader instead!

    // We will stage the pixels in a host-accessible buffer
    let buffer = Arc::new(Buffer::create_from_slice(
        device,
        vk::BufferUsageFlags::TRANSFER_SRC,
        bitmap.pixels(),
    )?);
    let image = Arc::new(Image::create(
        device,
        ImageInfo::image_2d(
            bitmap.width(),
            bitmap.height(),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        ),
    )?);

    // Copy the host-accessible pixels into the device-only image
    let mut graph = Graph::default();
    let image_node = graph.bind_node(&image);
    let buffer_node = graph.bind_node(&buffer);
    graph.copy_buffer_to_image(buffer_node, image_node);
    graph.resolve().submit(&mut HashPool::new(device), 0, 0)?;

    Ok(image)
}

struct Animation {
    frame_joints: Vec<Mat4>,
    joints: Vec<Joint>,
    local_joints: Vec<Mat4>,
    channels: Vec<Channel>,
    time: u32,
    total_time: u32,
}

impl Animation {
    /// Note: Ignores specified channel interpolation and uses linear always!
    fn load(model: &Model, pak: &mut PakBuf, key: &str) -> Result<Self, DriverError> {
        let joints = model.joints.clone();
        let animation = pak.read_animation(key).unwrap();

        let total_time = animation
            .channels()
            .iter()
            .map(|channel| channel.inputs().last().copied().unwrap_or_default())
            .max()
            .unwrap_or_default();

        // TODO: Here is where you probably want to flatten out the channels into a constant
        // framerate animation for each joint - it would make it easier to run the update code
        // You might also want to do something like upload this to a buffer or texture! hint hint
        let channels = animation.channels().to_vec();

        // This demo only supports linear interpolation (not step or cubic)
        #[cfg(debug_assertions)]
        for channel in &channels {
            assert!(matches!(channel.interpolation(), Interpolation::Linear));
        }

        Ok(Animation {
            frame_joints: repeat_n(Mat4::IDENTITY, model.joints.len()).collect(),
            joints,
            local_joints: repeat_n(Mat4::IDENTITY, model.joints.len()).collect(),
            channels,
            time: 0,
            total_time,
        })
    }

    fn update(&mut self, dt: f32) -> &[Mat4] {
        self.time += Duration::from_secs_f32(dt).as_millis() as u32;
        self.time %= self.total_time;

        for transform in self.local_joints.iter_mut() {
            *transform = Mat4::IDENTITY;
        }

        for idx in 0..self.joints.len() {
            let joint = &self.joints[idx];
            let parent_transform = self.local_joints[joint.parent_index];

            // Look how much effort we're putting into finding the animation transform, you probably
            // want to store the animation in a format that makes more sense - but this is the basic
            // way you could do it
            let animation_transform = {
                let rotation = self
                    .channels
                    .iter()
                    .find(|channel| {
                        channel.target() == joint.name
                            && matches!(channel.outputs(), Outputs::Rotations(_))
                    })
                    .map(|channel| match channel.outputs() {
                        Outputs::Rotations(rotations) => (channel.inputs(), rotations),
                        _ => unreachable!(),
                    })
                    .map(|(inputs, rotations)| {
                        let (a, b, ab) = self.pick_weighted_keyframes(inputs);
                        let a = Quat::from_array(rotations[a]);
                        let b = Quat::from_array(rotations[b]);

                        Quat::slerp(a, b, ab)
                    })
                    .unwrap_or(Quat::IDENTITY);
                let scale = self
                    .channels
                    .iter()
                    .find(|channel| {
                        channel.target() == joint.name
                            && matches!(channel.outputs(), Outputs::Scales(_))
                    })
                    .map(|channel| match channel.outputs() {
                        Outputs::Scales(scales) => (channel.inputs(), scales),
                        _ => unreachable!(),
                    })
                    .map(|(inputs, scales)| {
                        let (a, b, ab) = self.pick_weighted_keyframes(inputs);
                        let a = Vec3::from_array(scales[a]);
                        let b = Vec3::from_array(scales[b]);

                        Vec3::lerp(a, b, ab)
                    })
                    .unwrap_or(Vec3::ONE);
                let translation = self
                    .channels
                    .iter()
                    .find(|channel| {
                        channel.target() == joint.name
                            && matches!(channel.outputs(), Outputs::Translations(_))
                    })
                    .map(|channel| match channel.outputs() {
                        Outputs::Translations(translations) => (channel.inputs(), translations),
                        _ => unreachable!(),
                    })
                    .map(|(inputs, translations)| {
                        let (a, b, ab) = self.pick_weighted_keyframes(inputs);
                        let a = Vec3::from_array(translations[a]);
                        let b = Vec3::from_array(translations[b]);

                        Vec3::lerp(a, b, ab)
                    })
                    .unwrap_or(Vec3::ZERO);

                Mat4::from_scale_rotation_translation(scale, rotation, translation)
            };

            // Uncomment to show how to manually target a bone (this twists the chest to the right)
            // let animation_transform = if joint.name.as_str() == "Chest" {
            //     self.joints[joint.parent_index].inverse_bind * joint.inverse_bind.inverse() * Mat4::from_rotation_y(90f32.to_radians())
            // } else {
            //     animation_transform
            // };

            self.local_joints[idx] = parent_transform * animation_transform;
            self.frame_joints[idx] =
                self.local_joints[idx] * Mat4::from_cols_array(&joint.inverse_bind);
        }

        &self.frame_joints
    }

    /// Given an array of keyframe times, returns the two keyframe indices and the weight factor
    /// to use when interpolating between them.
    fn pick_weighted_keyframes(&self, inputs: &[u32]) -> (usize, usize, f32) {
        let (idx_a, idx_b) = match inputs.binary_search(&self.time) {
            Err(idx) if idx == 0 || idx == inputs.len() => (inputs.len() - 1, 0),
            Err(idx) => (idx - 1, idx),
            Ok(idx) => (idx, idx),
        };
        let ab = match idx_a.cmp(&idx_b) {
            Ordering::Equal => 0.0,
            Ordering::Greater => self.time as f32 / inputs[idx_b] as f32,
            Ordering::Less => {
                (self.time - inputs[idx_a]) as f32 / (inputs[idx_b] - inputs[idx_a]) as f32
            }
        };

        (idx_a, idx_b, ab)
    }
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CameraUniform {
    projection: Mat4,
    view: Mat4,
    position: Vec3,
}

unsafe impl NoUninit for CameraUniform {}

struct Model {
    index_buf: Arc<Buffer>,
    index_count: u32,
    joints: Vec<Joint>,
    vertex_buf: Arc<Buffer>,
}

impl Model {
    fn load(device: &Device, pak: &mut PakBuf, key: &str) -> Result<Self, DriverError> {
        let model = pak.read_model(key).unwrap();

        // This obviously makes some assumptions about the input model!

        let mesh = model
            .meshes()
            .iter()
            .find(|mesh| mesh.skin().is_some())
            .unwrap();
        let joints = mesh.skin().unwrap().joints().to_vec();
        let parts = mesh.parts();

        assert_eq!(parts.len(), 1);

        let part = &parts[0];
        let lods = part.lods();

        assert_eq!(
            part.vertex(),
            Vertex::POSITION | Vertex::NORMAL | Vertex::TEXTURE0 | Vertex::JOINTS_WEIGHTS
        );
        assert!(!lods.is_empty());

        let lod = &lods[0];
        let indices = lod;

        assert!(indices.index_count() < u16::MAX as usize);

        let indices = indices.as_u16().unwrap();
        let index_data = cast_slice(&indices);
        let vertex_data = part.vertex_data();

        // Host-accessible staging buffers
        let index_staging_buf = Arc::new(Buffer::create_from_slice(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            index_data,
        )?);
        let vertex_staging_buf = Arc::new(Buffer::create_from_slice(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vertex_data,
        )?);

        // Device-only buffers
        let index_buf = Arc::new(Buffer::create(
            device,
            BufferInfo::device_mem(
                index_data.len() as _,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            ),
        )?);
        let vertex_buf = Arc::new(Buffer::create(
            device,
            BufferInfo::device_mem(
                vertex_data.len() as _,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            ),
        )?);

        // Copy the host-accessible staging buffers to device-only buffers
        let mut graph = Graph::default();
        let index_staging_buf_node = graph.bind_node(index_staging_buf);
        let vertex_staging_buf_node = graph.bind_node(vertex_staging_buf);
        let index_buf_node = graph.bind_node(&index_buf);
        let vertex_buf_node = graph.bind_node(&vertex_buf);
        graph.copy_buffer(index_staging_buf_node, index_buf_node);
        graph.copy_buffer(vertex_staging_buf_node, vertex_buf_node);
        graph.resolve().submit(&mut HashPool::new(device), 0, 0)?;

        Ok(Model {
            index_buf,
            index_count: indices.len() as _,
            joints,
            vertex_buf,
        })
    }
}
