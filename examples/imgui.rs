mod profile_with_puffin;

use {
    clap::Parser,
    vk_graph_fx::*,
    vk_graph_imgui::{Condition, ImGui},
    vk_graph_prelude::*,
    vk_graph_window::{WindowBuilder, WindowError},
    winit::dpi::LogicalSize,
};

fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    // vk-graph things we need for this demo
    let args = Args::parse();
    let window = WindowBuilder::default()
        .debug(args.debug)
        .v_sync(false)
        .window(|window| window.with_inner_size(LogicalSize::new(1024, 768)))
        .build()?;
    let display = ComputePresenter::new(&window.device)?;
    let mut imgui = ImGui::new(&window.device);
    let mut pool = LazyPool::new(&window.device);

    // Some example state to make the demo more interesting
    let mut value = 0;
    let choices = ["test test this is 1", "test test this is 2"];

    window.run(|frame| {
        // Lease and clear an image as a stand-in for some real game or program output
        let app_image = frame.render_graph.bind_node(
            pool.lease(ImageInfo::image_2d(
                frame.width,
                frame.height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            ))
            .unwrap(),
        );
        frame
            .render_graph
            .clear_color_image_value(app_image, [0.2, 0.22, 0.2, 1.0]);

        // Use the draw function callback to do some fun meant-for-debug-mode GUI stuff
        let gui_image = imgui.draw(
            0.016,
            frame.events,
            frame.window,
            &mut pool,
            frame.render_graph,
            |ui, _, _| {
                ui.window("Hello world")
                    .position([10.0, 10.0], Condition::FirstUseEver)
                    .size([340.0, 250.0], Condition::FirstUseEver)
                    .build(|| {
                        ui.text_wrapped("Hello world!");
                        ui.text_wrapped("こんにちは世界！");
                        if ui.button(choices[value]) {
                            value += 1;
                            value %= 2;
                        }

                        ui.button("This...is...imgui-rs!");
                        ui.separator();
                        let mouse_pos = ui.io().mouse_pos;
                        ui.text(format!(
                            "Mouse Position: ({:.1},{:.1})",
                            mouse_pos[0], mouse_pos[1]
                        ));
                    });
            },
        );

        // Present "gui_image" on top of "app_image" onto "frame.swapchain"
        display.present_images(
            frame.render_graph,
            gui_image,
            app_image,
            frame.swapchain_image,
        );
    })
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
