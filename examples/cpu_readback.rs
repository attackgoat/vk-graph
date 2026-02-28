use {clap::Parser, std::time::Instant, vk_graph_prelude::*};

/// Example demonstrating the steps to take when reading the results of buffer or image operations
/// on the CPU. These operations take time to submit and the GPU will execute them asynchronously.
fn main() -> Result<(), DriverError> {
    pretty_env_logger::init();

    // For this example we create a headless device, but the same thing works using a window
    let args = Args::parse();
    let device_info = DeviceInfoBuilder::default().debug(args.debug);
    let device = Device::new(device_info)?;

    let mut graph = Graph::default();

    let src_buf = graph.bind_resource(Buffer::create_from_slice(
        &device,
        vk::BufferUsageFlags::TRANSFER_SRC,
        &[1, 2, 3, 4],
    )?);
    let dst_buf = graph.bind_resource(Buffer::create_from_slice(
        &device,
        vk::BufferUsageFlags::TRANSFER_DST,
        &[0, 0, 0, 0],
    )?);

    // We are using the GPU to copy data, but the same thing works if you're executing a pipeline
    // such as a ComputePipeline to run some shader code which writes to a buffer or image. It is
    // important to note that dst_buf does not contain the new data until we submit this render
    // graph and wait on the result
    graph.copy_buffer(src_buf, dst_buf);

    // This line is optional - just bind a borrow of Arc<Buffer> or a leased buffer so you retain
    // the actual buffer for later use and you could then remove this line
    let dst_buf = graph.resource(dst_buf).clone();

    // Resolve and wait (or you can check has_executed without blocking) - alternatively you might
    // use device.queue_wait_idle(0) or device.device_wait_idle() - but those block on larger scopes
    let mut cmd_buf = graph.queue().submit(&mut HashPool::new(&device), 0, 0)?;

    println!("Has executed? {}", cmd_buf.has_executed()?);
    let started = Instant::now();

    cmd_buf.wait_until_executed()?;

    assert!(
        cmd_buf.has_executed()?,
        "We checked above - so this will always be true"
    );
    println!("Waited {}μs", (Instant::now() - started).as_micros());

    // It is now safe to read back what we did!
    println!("{:?}", Buffer::mapped_slice(&dst_buf));

    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
