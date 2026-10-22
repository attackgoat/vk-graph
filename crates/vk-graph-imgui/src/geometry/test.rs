use {
    super::*,
    crate::test::CONTEXT,
    imgui::{Condition, Context, Ui},
    vk_graph::{
        driver::device::{Device, DeviceInfo},
        pool::hash::HashPool,
    },
};

struct RejectPool(usize);

impl Pool<BufferInfo, Buffer> for RejectPool {
    fn resource(&mut self, _: BufferInfo) -> Result<Lease<Buffer>, DriverError> {
        self.0 += 1;
        Err(DriverError::Unsupported)
    }
}

fn draw_windows(ui: &Ui) {
    for index in 0..3 {
        ui.window(format!("geometry {index}"))
            .position([index as f32 * 220.0, 20.0], Condition::Always)
            .size([200.0, 200.0], Condition::Always)
            .build(|| {
                ui.text("separate draw list");
            });
    }
}

fn draw_windows_with_callbacks(ui: &Ui) {
    ui.get_background_draw_list().add_callback(|| {}).build();
    draw_windows(ui);
    ui.get_foreground_draw_list().add_callback(|| {}).build();
}

fn with_draw_data<R>(build: impl Fn(&Ui), run: impl FnOnce(&DrawData) -> R) -> R {
    let _context = CONTEXT.lock().unwrap();
    let mut context = Context::create();
    context.set_ini_filename(None);
    context.io_mut().display_size = [800.0, 600.0];
    context.io_mut().delta_time = 1.0 / 60.0;
    context.fonts().build_rgba32_texture();
    build(context.frame());
    context.render();
    build(context.frame());
    run(context.render())
}

#[test]
fn capacities_grow_geometrically_without_shrinking() {
    let mut capacity = GeometryCapacity::default();
    capacity
        .grow(GeometryOffsets {
            indices: 3,
            vertices: 21,
        })
        .unwrap();
    assert_eq!((capacity.index_bytes, capacity.vertex_bytes), (4, 32));
    capacity
        .grow(GeometryOffsets {
            indices: 1,
            vertices: 20,
        })
        .unwrap();
    assert_eq!((capacity.index_bytes, capacity.vertex_bytes), (4, 32));
    capacity
        .grow(GeometryOffsets {
            indices: 5,
            vertices: 40,
        })
        .unwrap();
    assert_eq!((capacity.index_bytes, capacity.vertex_bytes), (8, 64));
}

#[test]
fn list_offsets_preserve_local_indices_past_65535_total_vertices() {
    assert_eq!(size_of::<DrawIdx>(), 2);
    let mut end = GeometryOffsets::default();
    assert_eq!(
        end.push(60_000, 40_000).unwrap(),
        GeometryOffsets::default()
    );
    let second = end.push(60_000, 40_000).unwrap();
    assert_eq!(second.indices, 120_000);
    assert_eq!(second.vertices, 800_000);
    assert_eq!(end.vertices, 1_600_000);
}

#[test]
fn invalid_geometry_sizes_do_not_partially_update_capacity_or_offsets() {
    let mut offsets = GeometryOffsets::default();
    assert!(offsets.push(usize::MAX, 1).is_err());
    assert_eq!(offsets, GeometryOffsets::default());
    let mut capacity = GeometryCapacity::default();
    assert!(
        capacity
            .grow(GeometryOffsets {
                indices: 1,
                vertices: usize::MAX
            })
            .is_err()
    );
    assert_eq!((capacity.index_bytes, capacity.vertex_bytes), (0, 0));
}

#[test]
fn empty_frames_do_not_lease_zero_sized_buffers() {
    with_draw_data(
        |_| {},
        |data| {
            let mut pool = RejectPool(0);
            assert!(
                GeometryCapacity::default()
                    .lease(&mut pool, data)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(pool.0, 0);
        },
    );
}

#[test]
fn callback_only_frames_do_not_lease_geometry_buffers() {
    with_draw_data(
        |ui| {
            ui.get_background_draw_list().add_callback(|| {}).build();
        },
        |data| {
            assert_eq!(data.draw_lists_count(), 1);
            assert_eq!((data.total_idx_count, data.total_vtx_count), (0, 0));
            let mut pool = RejectPool(0);
            assert!(
                GeometryCapacity::default()
                    .lease(&mut pool, data)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(pool.0, 0);
        },
    );
}

#[test]
fn mixed_frames_skip_callback_only_lists() {
    with_draw_data(draw_windows_with_callbacks, |data| {
        assert!(data.draw_lists_count() >= 5);
        assert!(data.total_idx_count > 0);
        assert!(data.total_vtx_count > 0);
        assert_eq!(
            data.draw_lists()
                .filter(|list| draw_list_geometry(list).is_none())
                .count(),
            2
        );
        let mut pool = RejectPool(0);
        let mut capacity = GeometryCapacity::default();
        assert!(matches!(
            capacity.lease(&mut pool, data),
            Err(DriverError::Unsupported)
        ));
        assert_eq!(pool.0, 1);
        assert_eq!(
            capacity.index_bytes,
            (data.total_idx_count as vk::DeviceSize * size_of::<DrawIdx>() as vk::DeviceSize)
                .next_power_of_two()
        );
        assert_eq!(
            capacity.vertex_bytes,
            (data.total_vtx_count as vk::DeviceSize * VERTEX_STRIDE as vk::DeviceSize)
                .next_power_of_two()
        );
    });
}

#[test]
#[ignore = "requires Vulkan device"]
fn vulkan_batches_lists_and_keeps_outstanding_geometry_distinct() -> Result<(), DriverError> {
    struct InspectPool {
        pool: HashPool,
        requests: Vec<BufferInfo>,
    }
    impl Pool<BufferInfo, Buffer> for InspectPool {
        fn resource(&mut self, info: BufferInfo) -> Result<Lease<Buffer>, DriverError> {
            self.requests.push(info);
            self.pool.resource(info)
        }
    }
    with_draw_data(draw_windows_with_callbacks, |data| {
        assert!(data.draw_lists_count() >= 5);
        let device = Device::create(DeviceInfo::default())?;
        let mut pool = InspectPool {
            pool: HashPool::new(&device),
            requests: Vec::new(),
        };
        let mut capacity = GeometryCapacity::default();
        let first = capacity.lease(&mut pool, data)?.unwrap();
        assert_eq!(pool.requests.len(), 2);
        assert!(pool.requests.iter().all(|info| info.size.is_power_of_two()));
        let mut expected_indices = Vec::<u8>::new();
        let mut expected_vertices = Vec::<u8>::new();
        for (indices, vertices) in data.draw_lists().filter_map(draw_list_geometry) {
            expected_indices.extend_from_slice(cast_slice(indices));
            for vertex in vertices {
                expected_vertices.extend_from_slice(cast_slice(&vertex.pos));
                expected_vertices.extend_from_slice(cast_slice(&vertex.uv));
                expected_vertices.extend_from_slice(&vertex.col);
            }
        }
        assert_eq!(
            &Buffer::mapped_slice(&first.indices)[..expected_indices.len()],
            expected_indices
        );
        assert_eq!(
            &Buffer::mapped_slice(&first.vertices)[..expected_vertices.len()],
            expected_vertices
        );
        let second = capacity.lease(&mut pool, data)?.unwrap();
        assert_eq!(pool.requests.len(), 4);
        assert_ne!(first.indices.handle, second.indices.handle);
        assert_ne!(first.vertices.handle, second.vertices.handle);
        let handles = (second.indices.handle, second.vertices.handle);
        drop(first);
        drop(second);
        let reused = capacity.lease(&mut pool, data)?.unwrap();
        assert_eq!(pool.requests.len(), 6);
        assert_eq!((reused.indices.handle, reused.vertices.handle), handles);
        Ok(())
    })
}
