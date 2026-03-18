mod profile_with_puffin;

use {
    ash::vk,
    clap::Parser,
    vk_graph::{
        driver::image::ImageInfo,
        pool::{Pool as _, lazy::LazyPool},
    },
    vk_graph_egui::{Egui, egui},
    vk_graph_window::Window,
    winit::dpi::LogicalSize,
};

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    let args = Args::parse();
    let window = Window::builder()
        .debug(args.debug)
        .v_sync(false)
        .window(|window| window.with_inner_size(LogicalSize::new(1024, 768)))
        .build()?;
    let mut egui = Egui::new(&window.device, &window);

    let mut cache = LazyPool::new(&window.device);

    window.run(|frame| {
        let img = frame.graph.bind_resource(
            cache
                .lease_resource(ImageInfo::image_2d(
                    100,
                    100,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ))
                .unwrap(),
        );
        frame.graph.clear_color_image(img, [0., 1., 0., 1.]);
        frame
            .graph
            .clear_color_image(frame.swapchain_image, [0., 0., 0., 1.]);

        let id = egui.register_texture(img);

        egui.run(
            frame.window,
            frame.events,
            frame.swapchain_image,
            frame.graph,
            |ui| {
                egui::Window::new("Test")
                    .resizable(true)
                    .vscroll(true)
                    .default_size([400., 400.])
                    .show(ui, |ui| {
                        ui.add(egui::Button::new("Test"));
                        ui.add(egui::Link::new("Test"));
                        ui.add(egui::Image::new((id, egui::Vec2::new(50., 50.))));
                    });
            },
        );
    })?;

    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
