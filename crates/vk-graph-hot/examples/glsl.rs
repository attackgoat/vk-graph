use {
    clap::Parser,
    std::path::PathBuf,
    vk_graph_hot::{HotComputePipeline, HotShader},
    vk_graph_prelude::*,
    vk_graph_window::{Window, WindowError},
};

/// This program draws a noise signal to the swapchain - make changes to fill_image.comp or the
/// noise.glsl file it includes to see those changes update while the program is still running.
///
/// Run with RUST_LOG=info to get notification of shader compilations.
fn main() -> Result<(), WindowError> {
    pretty_env_logger::init();

    let args = Args::parse();
    let window = Window::builder().debug(args.debug).build()?;

    // Create a compute pipeline - the same as normal except for "Hot" prefixes and we provide the
    // shader source code path instead of the shader source code bytes
    let cargo_manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pipeline = HotComputePipeline::create(
        &window.device,
        ComputePipelineInfo::default(),
        HotShader::from_path(cargo_manifest_dir.join("examples/res/fill_image.comp")),
    )?;

    let mut frame_index: u32 = 0;

    window.run(|frame| {
        frame
            .graph
            .begin_cmd()
            .debug_name("make some noise")
            .bind_pipeline(&pipeline)
            .shader_resource_access(0, frame.swapchain_image, AccessType::ComputeShaderWrite)
            .record_cmd_buf(move |cmd_buf| {
                cmd_buf
                    .push_constants(0, &frame_index.to_ne_bytes())
                    .dispatch(frame.width, frame.height, 1);
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
