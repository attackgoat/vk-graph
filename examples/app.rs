mod profile_with_puffin;

use {
    clap::Parser,
    log::error,
    std::sync::Arc,
    vk_graph::{
        RenderGraph,
        display::{Display, DisplayError, DisplayInfo},
        driver::{
            device::{Device, DeviceInfoBuilder},
            surface::Surface,
            swapchain::{Swapchain, SwapchainInfo},
        },
        pool::hash::HashPool,
    },
    winit::{
        application::ApplicationHandler,
        error::EventLoopError,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop},
        window::{Window, WindowId},
    },
};

fn main() -> Result<(), EventLoopError> {
    pretty_env_logger::init();
    profile_with_puffin::init();

    EventLoop::new()?.run_app(&mut Application::default())
}

#[derive(Default)]
struct Application(Option<Context>);

impl ApplicationHandler for Application {
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.0.as_ref().unwrap().window.request_redraw();
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("vk-graph");
        let window = event_loop.create_window(window_attributes).unwrap();

        let args = Args::parse();
        let device_info = DeviceInfoBuilder::default().debug(args.debug);
        let device = Arc::new(Device::create_display(device_info, &window).unwrap());

        let surface = Surface::create(&device, &window, &window).unwrap();
        let surface_formats = Surface::formats(&surface).unwrap();
        let surface_format = Surface::linear_or_default(&surface_formats);
        let window_size = window.inner_size();
        let swapchain = Swapchain::new(
            &device,
            surface,
            SwapchainInfo::new(window_size.width, window_size.height, surface_format),
        )
        .unwrap();

        let display_pool = HashPool::new(&device);
        let display = Display::new(&device, swapchain, DisplayInfo::default()).unwrap();

        self.0 = Some(Context {
            display,
            display_pool,
            window,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let context = self.0.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                let mut swapchain_info = context.display.swapchain.info;
                swapchain_info.width = size.width;
                swapchain_info.height = size.height;
                context.display.update_swapchain(swapchain_info);
            }
            WindowEvent::RedrawRequested => {
                if let Err(err) = context.draw() {
                    // This would be a good time to recreate the device or surface
                    error!("unable to draw window: {err}");

                    event_loop.exit();
                };

                profiling::finish_frame!();
            }
            _ => (),
        }
    }
}

#[derive(Parser)]
struct Args {
    /// Enable Vulkan SDK validation layers
    #[arg(long)]
    debug: bool,
}

struct Context {
    display: Display,
    display_pool: HashPool,
    window: Window,
}

impl Context {
    fn draw(&mut self) -> Result<(), DisplayError> {
        if let Some(swapchain_image) = self.display.acquire_next_image()? {
            let mut render_graph = RenderGraph::new();
            let swapchain_image = render_graph.bind_node(swapchain_image);

            // Rendering goes here!
            render_graph.clear_color_image_value(swapchain_image, [1.0, 0.0, 1.0]);

            self.window.pre_present_notify();
            self.display
                .present_image(&mut self.display_pool, render_graph, swapchain_image, 0)?;
        }

        Ok(())
    }
}
