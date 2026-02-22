use {clap::Parser, std::sync::Arc, vk_graph_prelude::*};

/// This example demonstrates resource aliasing. Aliasing is a memory-efficiency optimization that
/// may be used anywhere resources are leased and used in a render graph. Aliasing allows complex
/// graphs to require fewer individual resources.
///
/// The performance overhead of aliasing is an atomic load for each actively aliased item and one
/// check per active alias to see if it is compatible with the requested resource.
///
/// Acceleration structures, buffers and images may be "aliased" by different parts of any one or
/// more render graphs. The process involves wrapping any pool type (FifoPool, LazyPool, HashPool)
/// in an AliasPool container. AliasPool offers an alias(..) function which operates exactly the
/// same as a regular pool lease(..) except that the result is wrapped in an Arc<>.
///
/// AliasPool derefs to the base pool type and so leasing may be used normally too.
fn main() -> Result<(), DriverError> {
    pretty_env_logger::init();

    let args = Args::parse();
    let device_info = DeviceInfoBuilder::default().debug(args.debug);
    let device = Device::new(device_info)?;

    // We wrap HashPool in an AliasPool container to enable resource aliasing
    let mut pool = AliasPool::new(HashPool::new(&device));

    // This is the information we will use to alias image1 and image2
    let image_info = ImageInfo::image_2d(
        128,
        128,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
    );

    // Any two compatible images aliased from the same pool will be the same physical image
    let image1 = pool.alias(image_info)?;
    let image2 = pool.alias(image_info)?;
    assert!(Arc::ptr_eq(&image1, &image2));

    let mut graph = Graph::default();

    // Binding these images to any render graph will produce the same physical nodes
    let image1 = graph.bind_node(image1);
    let image2 = graph.bind_node(image2);
    assert_eq!(image1, image2);

    // Let's make up some different, yet compatible, image information:
    let image_info = ImageInfo::image_2d(
        image_info.width,
        image_info.height,
        image_info.fmt,
        vk::ImageUsageFlags::TRANSFER_DST,
    );

    // We alias the compatible information and still produce the same physical image and node
    let image3 = graph.bind_node(pool.alias(image_info)?);
    assert_eq!(image1, image3);

    // Using the same information for a new LEASE will generate an entirely different image!!
    let image4 = graph.bind_node(pool.lease(image_info)?);
    assert_ne!(image1, image4);

    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}
