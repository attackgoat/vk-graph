use vk_graph_window::{Window, WindowError};

/// This example requires a color graphics adapter.
fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();

    Window::new()?.run(|frame| {
        frame
            .graph
            .clear_color_image(frame.swapchain_image, [100u8, 149, 237]);
    })
}
