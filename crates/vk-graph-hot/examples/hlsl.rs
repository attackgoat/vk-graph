use {
    clap::Parser,
    std::path::PathBuf,
    vk_graph::{
        cmd::{LoadOp, StoreOp},
        driver::graphic::GraphicPipelineInfo,
    },
    vk_graph_hot::{HotGraphicPipeline, HotShader},
    vk_graph_window::{Window, WindowError},
};

/// This program draws a plasma animation to the swapchain - make changes to fill_image.hlsl or the
/// plasma.hlsl file it includes to see those changes update while the program is still running.
///
/// Run with RUST_LOG=info to get notification of shader compilations.
fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();

    let args = Args::parse();
    let window = Window::builder().debug(args.debug).build()?;

    // Create a graphic pipeline - the same as normal except for "Hot" prefixes and we provide the
    // shader source code path instead of the shader source code bytes
    let cargo_manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fill_image_path = cargo_manifest_dir.join("examples/res/fill_image.hlsl");
    let pipeline = HotGraphicPipeline::create(
        &window.device,
        GraphicPipelineInfo::default(),
        [
            HotShader::new_vertex(&fill_image_path).entry_name("vertex_main"),
            HotShader::new_fragment(&fill_image_path).entry_name("fragment_main"),
        ],
    )?;

    let mut frame_index: u32 = 0;

    window.run(|frame| {
        frame
            .graph
            .begin_cmd()
            .debug_name("neato colors")
            .bind_pipeline(&pipeline)
            .color_attachment_image(
                0,
                frame.swapchain_image,
                LoadOp::CLEAR_BLACK_ALPHA_ZERO,
                StoreOp::Store,
            )
            .record_cmd_buf(move |cmd_buf| {
                cmd_buf
                    .push_constants(0, &frame_index.to_ne_bytes())
                    .push_constants(4, &frame.width.to_ne_bytes())
                    .push_constants(8, &frame.height.to_ne_bytes())
                    .draw(3, 1, 0, 0);
            });

        frame_index += 1;
    })
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
