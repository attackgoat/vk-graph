mod frame;

pub use self::frame::FrameContext;

use {
    log::{info, trace, warn},
    vk_graph::{
        driver::{
            ash::vk,
            device::{Device, DeviceInfo},
            surface::Surface,
            swapchain::{Swapchain, SwapchainInfo},
            DriverError,
        },
        graph::RenderGraph,
        pool::hash::HashPool,
        Display, DisplayError, DisplayInfoBuilder,
    },
    std::{error, fmt, sync::Arc},
    winit::{
        application::ApplicationHandler,
        error::EventLoopError,
        event::{DeviceEvent, DeviceId, Event, WindowEvent},
        event_loop::{ActiveEventLoop, EventLoop},
        monitor::MonitorHandle,
        window::{WindowAttributes, WindowId},
    },
};

/// Describes a screen mode for display.
#[derive(Clone, Copy, Debug)]
pub enum FullscreenMode {
    /// A display mode which retains other operating system windows behind the current window.
    Borderless,

    /// Seems to be the only way for stutter-free rendering on Nvidia + Win10.
    Exclusive,
}

// #[derive(Debug)]
pub struct Window {
    data: WindowData,
    pub device: Arc<Device>,
    event_loop: EventLoop<()>,
}

impl Window {
    pub fn new() -> Result<Self, WindowError> {
        Self::builder().build()
    }

    pub fn builder() -> WindowBuilder {
        WindowBuilder::default()
    }

    pub fn run<F>(self, draw_fn: F) -> Result<(), WindowError>
    where
        F: FnMut(FrameContext),
    {
        struct Application<F> {
            active_window: Option<ActiveWindow>,
            data: WindowData,
            device: Arc<Device>,
            draw_fn: F,
            error: Option<WindowError>,
            primary_monitor: Option<MonitorHandle>,
        }

        impl<F> Application<F> {
            fn create_display(
                &mut self,
                window: &winit::window::Window,
            ) -> Result<Display, DriverError> {
                let surface = Surface::create(&self.device, &window)?;
                let surface_formats = Surface::formats(&surface)?;
                let surface_format = self
                    .data
                    .surface_format_fn
                    .as_ref()
                    .map(|f| f(&surface_formats))
                    .unwrap_or_else(|| Surface::linear_or_default(&surface_formats));
                let window_size = window.inner_size();

                let mut swapchain_info =
                    SwapchainInfo::new(window_size.width, window_size.height, surface_format)
                        .to_builder();

                if let Some(image_count) = self.data.image_count {
                    swapchain_info = swapchain_info.desired_image_count(image_count);
                }

                if let Some(v_sync) = self.data.v_sync {
                    swapchain_info = if v_sync {
                        swapchain_info.present_modes(vec![
                            vk::PresentModeKHR::FIFO_RELAXED,
                            vk::PresentModeKHR::FIFO,
                        ])
                    } else {
                        swapchain_info.present_modes(vec![
                            vk::PresentModeKHR::MAILBOX,
                            vk::PresentModeKHR::IMMEDIATE,
                        ])
                    };
                }

                let swapchain = Swapchain::new(&self.device, surface, swapchain_info)?;
                let display = Display::new(
                    &self.device,
                    swapchain,
                    DisplayInfoBuilder::default().command_buffer_count(self.data.cmd_buf_count),
                )?;

                trace!("created display");

                Ok(display)
            }

            fn window_mode_attributes(
                &self,
                attributes: WindowAttributes,
                window_mode_override: Option<Option<FullscreenMode>>,
            ) -> WindowAttributes {
                match window_mode_override {
                    Some(Some(mode)) => {
                        let inner_size;
                        let attributes = attributes
                            .with_decorations(false)
                            .with_maximized(true)
                            .with_fullscreen(Some(match mode {
                                FullscreenMode::Borderless => {
                                    info!("Using borderless fullscreen");

                                    inner_size = None;

                                    winit::window::Fullscreen::Borderless(None)
                                }
                                FullscreenMode::Exclusive => {
                                    if let Some(video_mode) =
                                        self.primary_monitor.as_ref().and_then(|monitor| {
                                            let monitor_size = monitor.size();
                                            monitor.video_modes().find(|mode| {
                                                let mode_size = mode.size();

                                                // Don't pick a mode which has greater resolution than the monitor is
                                                // currently using: it causes a panic on x11 in winit
                                                mode_size.height <= monitor_size.height
                                                    && mode_size.width <= monitor_size.width
                                            })
                                        })
                                    {
                                        info!(
                                            "Using {}x{} {}bpp @ {}hz exclusive fullscreen",
                                            video_mode.size().width,
                                            video_mode.size().height,
                                            video_mode.bit_depth(),
                                            video_mode.refresh_rate_millihertz() / 1_000
                                        );

                                        inner_size = Some(video_mode.size());

                                        winit::window::Fullscreen::Exclusive(video_mode)
                                    } else {
                                        warn!("Using borderless fullscreen");

                                        inner_size = None;

                                        winit::window::Fullscreen::Borderless(None)
                                    }
                                }
                            }));

                        if let Some(inner_size) = inner_size
                            .or_else(|| self.primary_monitor.as_ref().map(|monitor| monitor.size()))
                        {
                            attributes.with_inner_size(inner_size)
                        } else {
                            attributes
                        }
                    }
                    Some(None) => attributes.with_fullscreen(None),
                    _ => attributes,
                }
            }
        }

        impl<F> ApplicationHandler for Application<F>
        where
            F: FnMut(FrameContext),
        {
            fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
                if let Some(ActiveWindow { window, .. }) = self.active_window.as_ref() {
                    window.request_redraw();
                }
            }

            fn device_event(
                &mut self,
                _event_loop: &ActiveEventLoop,
                device_id: DeviceId,
                event: DeviceEvent,
            ) {
                if let Some(ActiveWindow { events, .. }) = self.active_window.as_mut() {
                    events.push(Event::DeviceEvent { device_id, event });
                }
            }

            fn resumed(&mut self, event_loop: &ActiveEventLoop) {
                info!("Resumed");

                self.data.attributes = self.window_mode_attributes(
                    self.data.attributes.clone(),
                    self.data.window_mode_override,
                );

                let window = match event_loop.create_window(self.data.attributes.clone()) {
                    Err(err) => {
                        warn!("Unable to create window: {err}");

                        self.error = Some(EventLoopError::Os(err).into());
                        event_loop.exit();

                        return;
                    }
                    Ok(res) => res,
                };
                let display = match self.create_display(&window) {
                    Err(err) => {
                        warn!("Unable to create swapchain: {err}");

                        self.error = Some(err.into());
                        event_loop.exit();

                        return;
                    }
                    Ok(res) => res,
                };
                let display_pool = HashPool::new(&self.device);

                self.active_window = Some(ActiveWindow {
                    display,
                    display_pool,
                    display_resize: None,
                    events: vec![],
                    window,
                });
            }

            fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: ()) {
                if let Some(ActiveWindow { events, .. }) = self.active_window.as_mut() {
                    events.push(Event::UserEvent(event));
                }
            }

            fn window_event(
                &mut self,
                event_loop: &ActiveEventLoop,
                window_id: WindowId,
                event: WindowEvent,
            ) {
                if let Some(active_window) = self.active_window.as_mut() {
                    match &event {
                        WindowEvent::CloseRequested => {
                            info!("close requested");

                            event_loop.exit();
                        }
                        WindowEvent::RedrawRequested => {
                            let draw = active_window.draw(&self.device, &mut self.draw_fn);

                            profiling::finish_frame!();

                            if !draw.unwrap() {
                                event_loop.exit();
                            }
                        }
                        WindowEvent::Resized(size) => {
                            active_window.display_resize = Some((size.width, size.height));
                        }
                        _ => (),
                    }

                    active_window
                        .events
                        .push(Event::WindowEvent { window_id, event });
                }
            }
        }

        struct ActiveWindow {
            display: Display,
            display_pool: HashPool,
            display_resize: Option<(u32, u32)>,
            events: Vec<Event<()>>,
            window: winit::window::Window,
        }

        impl ActiveWindow {
            fn draw(
                &mut self,
                device: &Arc<Device>,
                mut f: impl FnMut(FrameContext),
            ) -> Result<bool, DisplayError> {
                if let Some((width, height)) = self.display_resize.take() {
                    let mut swapchain_info = self.display.swapchain_info();
                    swapchain_info.width = width;
                    swapchain_info.height = height;
                    self.display.set_swapchain_info(swapchain_info);
                }

                if let Some(swapchain_image) = self.display.acquire_next_image()? {
                    let mut render_graph = RenderGraph::new();
                    let swapchain_image = render_graph.bind_node(swapchain_image);
                    let swapchain_info = self.display.swapchain_info();

                    let mut will_exit = false;

                    trace!("drawing");

                    f(FrameContext {
                        device,
                        events: &self.events,
                        height: swapchain_info.height,
                        render_graph: &mut render_graph,
                        swapchain_image,
                        width: swapchain_info.width,
                        will_exit: &mut will_exit,
                        window: &self.window,
                    });

                    self.events.clear();

                    if will_exit {
                        info!("exit requested");

                        return Ok(false);
                    }

                    self.window.pre_present_notify();
                    self.display
                        .present_image(&mut self.display_pool, render_graph, swapchain_image, 0)
                        .inspect_err(|err| {
                            warn!("unable to present swapchain image: {err}");
                        })?;
                } else {
                    warn!("unable to acquire swapchain image");
                }

                self.window.request_redraw();

                Ok(true)
            }
        }

        let mut app = Application {
            active_window: None,
            data: self.data,
            device: self.device,
            draw_fn,
            error: None,
            primary_monitor: None,
        };

        self.event_loop.run_app(&mut app)?;

        if let Some(ActiveWindow {
            display, window, ..
        }) = app.active_window.take()
        {
            drop(display);
            drop(window);
        }

        info!("Window closed");

        if let Some(err) = app.error {
            Err(err)
        } else {
            Ok(())
        }
    }
}

impl AsRef<EventLoop<()>> for Window {
    fn as_ref(&self) -> &EventLoop<()> {
        &self.event_loop
    }
}

pub struct WindowBuilder {
    attributes: WindowAttributes,
    cmd_buf_count: usize,
    device_info: DeviceInfo,
    image_count: Option<u32>,
    surface_format_fn: Option<Box<dyn Fn(&[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR>>,
    v_sync: Option<bool>,
    window_mode_override: Option<Option<FullscreenMode>>,
}

impl WindowBuilder {
    pub fn build(self) -> Result<Window, WindowError> {
        let event_loop = EventLoop::new()?;
        let device = Arc::new(Device::create_display(self.device_info, &event_loop)?);

        Ok(Window {
            data: WindowData {
                attributes: self.attributes,
                cmd_buf_count: self.cmd_buf_count,
                image_count: self.image_count,
                surface_format_fn: self.surface_format_fn,
                v_sync: self.v_sync,
                window_mode_override: self.window_mode_override,
            },
            device,
            event_loop,
        })
    }

    /// Specifies the number of in-flight command buffers, which should be greater
    /// than or equal to the desired swapchain image count.
    ///
    /// More command buffers mean less time waiting for previously submitted frames to complete, but
    /// more memory in use.
    ///
    /// Generally a value of one or two greater than desired image count produces the smoothest
    /// animation.
    pub fn command_buffer_count(mut self, count: usize) -> Self {
        self.cmd_buf_count = count;
        self
    }

    /// Enables Vulkan graphics debugging layers.
    ///
    /// _NOTE:_ Any valdation warnings or errors will cause the current thread to park itself after
    /// describing the error using the `log` crate. This makes it easy to attach a debugger and see
    /// what is causing the issue directly.
    ///
    /// ## Platform-specific
    ///
    /// **macOS:** Has no effect.
    pub fn debug(mut self, enabled: bool) -> Self {
        self.device_info.debug = enabled;
        self
    }

    /// A function to select the desired swapchain surface image format.
    ///
    /// By default linear color space will be selected unless it is not available.
    pub fn desired_surface_format<F>(mut self, f: F) -> Self
    where
        F: 'static + Fn(&[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR,
    {
        self.surface_format_fn = Some(Box::new(f));
        self
    }

    /// The desired, but not guaranteed, number of images that will be in the created swapchain.
    ///
    /// More images introduces more display lag, but smoother animation.
    pub fn desired_image_count(mut self, count: u32) -> Self {
        self.image_count = Some(count);
        self
    }

    /// Sets up fullscreen mode. In addition, decorations are set to `false` and maximized is set to
    /// `true`.
    ///
    /// # Note
    ///
    /// There are additional options offered by `winit` which can be accessed using the `window`
    /// function.
    pub fn fullscreen_mode(mut self, mode: FullscreenMode) -> Self {
        self.window_mode_override = Some(Some(mode));
        self
    }

    /// When `true` specifies that the presentation engine does not wait for a vertical blanking
    /// period to update the current image, meaning this mode may result in visible tearing.
    ///
    /// # Note
    ///
    /// Applies only to exlcusive fullscreen mode.
    pub fn v_sync(mut self, enabled: bool) -> Self {
        self.v_sync = Some(enabled);
        self
    }

    /// Allows deeper customization of the window, if needed.
    pub fn window<WindowFn>(mut self, f: WindowFn) -> Self
    where
        WindowFn: FnOnce(WindowAttributes) -> WindowAttributes,
    {
        self.attributes = f(self.attributes);
        self
    }

    /// Sets up "windowed" mode, which is the opposite of fullscreen.
    ///
    /// # Note
    ///
    /// There are additional options offered by `winit` which can be accessed using the `window`
    /// function.
    pub fn window_mode(mut self) -> Self {
        self.window_mode_override = Some(None);
        self
    }
}

impl fmt::Debug for WindowBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowBuilder")
            .field("attributes", &self.attributes)
            .field("cmd_buffer_count", &self.cmd_buf_count)
            .field("device_info", &self.device_info)
            .field("image_count", &self.image_count)
            .field(
                "surface_format_fn",
                &self.surface_format_fn.as_ref().map(|_| ()),
            )
            .field("v_sync", &self.v_sync)
            .field("window_mode_override", &self.window_mode_override)
            .finish()
    }
}

impl Default for WindowBuilder {
    fn default() -> Self {
        Self {
            attributes: Default::default(),
            cmd_buf_count: 5,
            device_info: Default::default(),
            image_count: None,
            surface_format_fn: None,
            v_sync: None,
            window_mode_override: None,
        }
    }
}

struct WindowData {
    attributes: WindowAttributes,
    cmd_buf_count: usize,
    image_count: Option<u32>,
    surface_format_fn: Option<Box<dyn Fn(&[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR>>,
    v_sync: Option<bool>,
    window_mode_override: Option<Option<FullscreenMode>>,
}

#[derive(Debug)]
pub enum WindowError {
    Driver(DriverError),
    EventLoop(EventLoopError),
}

impl error::Error for WindowError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        Some(match self {
            Self::Driver(err) => err,
            Self::EventLoop(err) => err,
        })
    }
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => err.fmt(f),
            Self::EventLoop(err) => err.fmt(f),
        }
    }
}

impl From<DriverError> for WindowError {
    fn from(err: DriverError) -> Self {
        Self::Driver(err)
    }
}

impl From<EventLoopError> for WindowError {
    fn from(err: EventLoopError) -> Self {
        Self::EventLoop(err)
    }
}
