//! `winit` window, event loop, and swapchain helpers for `vk-graph`.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod frame;
pub mod graphchain;

pub use {self::frame::FrameContext, winit};

use {
    self::graphchain::{Graphchain, GraphchainError, GraphchainInfo},
    log::{error, info, trace, warn},
    std::{error, fmt, ops::Deref},
    vk_graph::{
        Graph,
        driver::{
            DriverError,
            ash::vk,
            device::{Device, DeviceInfo},
            surface::Surface,
        },
        pool::hash::HashPool,
    },
    winit::raw_window_handle::{DisplayHandle, HandleError, HasDisplayHandle},
    winit::{
        application::ApplicationHandler,
        error::EventLoopError,
        event::{DeviceEvent, DeviceId, Event, WindowEvent},
        event_loop::{ActiveEventLoop, EventLoop},
        monitor::MonitorHandle,
        window::{WindowAttributes, WindowId},
    },
};

/// A closure type for picking surface formats.
pub type SurfaceFormatFn = dyn Fn(&[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR;

fn create_graphchain(
    device: &Device,
    data: &WindowData,
    window: &winit::window::Window,
) -> Result<Graphchain, DriverError> {
    let surface = Surface::create(device, window, window)?;
    let surface_formats = Surface::formats(&surface)?;
    let surface_format = data
        .surface_format_fn
        .as_ref()
        .map(|f| f(&surface_formats))
        .unwrap_or_else(|| Surface::linear_or_default(&surface_formats));
    let window_size = window.inner_size();

    let mut graphchain_info =
        GraphchainInfo::new(window_size.width, window_size.height, surface_format)
            .into_builder()
            .frame_capacity(data.cmd_buf_count);

    if let Some(min_image_count) = data.min_image_count {
        graphchain_info = graphchain_info.min_image_count(min_image_count);
    }

    let v_sync = data.v_sync.unwrap_or_default();
    let present_modes = Surface::present_modes(&surface)?;
    if !present_modes.is_empty() {
        let best_modes = if v_sync {
            [vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO].as_slice()
        } else {
            [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE].as_slice()
        };

        graphchain_info = graphchain_info.present_mode(
            best_modes
                .iter()
                .copied()
                .find(|best| present_modes.contains(best))
                .or_else(|| {
                    warn!("requested present modes unsupported: {best_modes:?}");

                    present_modes.first().copied()
                })
                .ok_or_else(|| {
                    error!("display does not support presentation");

                    DriverError::Unsupported
                })?,
        );
    }

    let graphchain = Graphchain::new(surface, graphchain_info)?;

    trace!("created graphchain");

    Ok(graphchain)
}

/// Describes a screen mode for display.
#[derive(Clone, Copy, Debug)]
pub enum FullscreenMode {
    /// A display mode which retains other operating system windows behind the current window.
    Borderless,

    /// Seems to be the only way for stutter-free rendering on Nvidia + Win10.
    Exclusive,
}

/// A convenience wrapper that owns a `winit` event loop and a compatible `vk-graph` device.
#[read_only::embed]
pub struct Window {
    data: WindowData,

    /// A device which is compatible with this window.
    ///
    /// _Note:_ This field is read-only.
    #[readonly]
    pub device: Device,

    #[readonly]
    pub(self) event_loop: EventLoop<()>,
}

impl Deref for ReadOnlyWindow {
    type Target = EventLoop<()>;

    fn deref(&self) -> &Self::Target {
        &self.event_loop
    }
}

impl Window {
    /// Creates a window using the default [`WindowBuilder`] configuration.
    pub fn new() -> Result<Self, WindowError> {
        Self::builder().build()
    }

    /// Creates a builder for configuring a [`Window`] before construction.
    pub fn builder() -> WindowBuilder {
        Default::default()
    }

    /// Runs the application event loop and invokes `draw_fn` for each rendered frame.
    pub fn run<F>(self, draw_fn: F) -> Result<(), WindowError>
    where
        F: FnMut(FrameContext),
    {
        struct Application<F> {
            active_window: Option<ActiveWindow>,
            data: WindowData,
            device: Device,
            draw_fn: F,
            error: Option<WindowError>,
            primary_monitor: Option<MonitorHandle>,
        }

        impl<F> Application<F> {
            fn create_graphchain(
                &mut self,
                window: &winit::window::Window,
            ) -> Result<Graphchain, DriverError> {
                create_graphchain(&self.device, &self.data, window)
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

                                                /*
                                                Don't pick a mode with greater resolution than the
                                                monitor; it can panic on X11 in winit.
                                                */
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
                                        warn!("unsupported exclusive fullscreen mode");

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
            fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
                if event_loop.exiting() {
                    return;
                }

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
                        warn!("unable to create window: {err}");

                        self.error = Some(EventLoopError::Os(err).into());
                        event_loop.exit();

                        return;
                    }
                    Ok(res) => res,
                };
                let graphchain = match self.create_graphchain(&window) {
                    Err(err) => {
                        warn!("unable to create graphchain: {err}");

                        self.error = Some(err.into());
                        event_loop.exit();

                        return;
                    }
                    Ok(res) => res,
                };
                let swapchain_pool = HashPool::new(&self.device);

                self.active_window = Some(ActiveWindow {
                    graphchain: Some(graphchain),
                    swapchain_pool,
                    swapchain_resize: None,
                    events: vec![],
                    window,
                });
            }

            fn user_event(&mut self, event_loop: &ActiveEventLoop, event: ()) {
                info!("signal received, exiting event loop");

                event_loop.exit();

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
                if let Some(mut active_window) = self.active_window.take() {
                    match &event {
                        WindowEvent::CloseRequested => {
                            if event_loop.exiting() {
                                self.active_window = Some(active_window);

                                return;
                            }

                            info!("close requested");

                            event_loop.exit();
                            self.active_window = Some(active_window);
                        }
                        WindowEvent::RedrawRequested => {
                            if event_loop.exiting() {
                                self.active_window = Some(active_window);

                                return;
                            }

                            // Surface loss is recoverable here; other graphchain errors are fatal.
                            match active_window.draw(&self.device, &self.data, &mut self.draw_fn) {
                                Err(GraphchainError::SurfaceLost) => {
                                    warn!("surface lost; abandoning current frame");

                                    let _ = active_window.graphchain.take();

                                    active_window.window.request_redraw();
                                    self.active_window = Some(active_window);

                                    profiling::finish_frame!();

                                    return;
                                }
                                Err(err) => {
                                    self.error = Some(WindowError::Graphchain(err));
                                    event_loop.exit();
                                }
                                Ok(false) => {
                                    event_loop.exit();
                                }
                                Ok(true) => {}
                            }

                            profiling::finish_frame!();
                            self.active_window = Some(active_window);
                        }
                        WindowEvent::Resized(size) if size.width * size.height > 0 => {
                            active_window.swapchain_resize = Some((size.width, size.height));
                            self.active_window = Some(active_window);
                        }
                        _ => self.active_window = Some(active_window),
                    }

                    if let Some(active_window) = self.active_window.as_mut() {
                        active_window
                            .events
                            .push(Event::WindowEvent { window_id, event });
                    }
                }
            }
        }

        struct ActiveWindow {
            graphchain: Option<Graphchain>,
            swapchain_pool: HashPool,
            swapchain_resize: Option<(u32, u32)>,
            events: Vec<Event<()>>,
            window: winit::window::Window,
        }

        impl ActiveWindow {
            fn draw(
                &mut self,
                device: &Device,
                data: &WindowData,
                mut f: impl FnMut(FrameContext),
            ) -> Result<bool, GraphchainError> {
                if self.graphchain.is_none() {
                    self.graphchain = Some(create_graphchain(device, data, &self.window)?);
                }

                let graphchain = self.graphchain.as_mut().expect("missing graphchain");

                if let Some((width, height)) = self.swapchain_resize.take() {
                    if width == 0 || height == 0 {
                        self.swapchain_resize = Some((width, height));
                        self.events.clear();

                        return Ok(true);
                    }

                    let mut graphchain_info = graphchain.info;
                    graphchain_info.width = width;
                    graphchain_info.height = height;
                    graphchain.set_info(graphchain_info);
                }

                if let Some(swapchain_image) = graphchain.acquire_next_image()? {
                    let mut graph = Graph::default();
                    let swapchain_image = graph.bind_resource(swapchain_image);
                    let graphchain_info = graphchain.info;

                    let mut will_exit = false;

                    trace!("drawing");

                    f(FrameContext {
                        device,
                        events: &self.events,
                        height: graphchain_info.height,
                        graph: &mut graph,
                        swapchain_image,
                        width: graphchain_info.width,
                        will_exit: &mut will_exit,
                        window: &self.window,
                    });

                    self.events.clear();

                    if will_exit {
                        info!("exit requested");

                        return Ok(false);
                    }

                    self.window.pre_present_notify();
                    graphchain
                        .present_image(&mut self.swapchain_pool, graph, swapchain_image, 0)
                        .inspect_err(|err| {
                            warn!("unable to present graphchain image: {err}");
                        })?;
                } else {
                    warn!("unable to acquire graphchain image");
                }

                self.window.request_redraw();

                Ok(true)
            }
        }

        let mut app = Application {
            active_window: None,
            data: self.data,
            device: self.read_only.device,
            draw_fn,
            error: None,
            primary_monitor: None,
        };

        let proxy = self.read_only.event_loop.create_proxy();
        if let Err(e) = ctrlc::set_handler(move || {
            trace!("received SIGINT/SIGTERM");

            let _ = proxy.send_event(());
        }) {
            warn!("failed to set Ctrl-C handler: {e}");
        }

        self.read_only.event_loop.run_app(&mut app)?;

        if let Some(ActiveWindow {
            graphchain, window, ..
        }) = app.active_window.take()
        {
            drop(graphchain);
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

impl HasDisplayHandle for Window {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.event_loop.display_handle()
    }
}

/// Builder for configuring and constructing a [`Window`].
pub struct WindowBuilder {
    attributes: WindowAttributes,
    cmd_buf_count: usize,
    device_info: DeviceInfo,
    min_image_count: Option<u32>,
    surface_format_fn: Option<Box<SurfaceFormatFn>>,
    v_sync: Option<bool>,
    window_mode_override: Option<Option<FullscreenMode>>,
}

impl WindowBuilder {
    /// Builds the window, event loop, and compatible `vk-graph` device.
    pub fn build(self) -> Result<Window, WindowError> {
        let event_loop = EventLoop::new()?;
        let device = Device::try_from_display(&event_loop, self.device_info)?;

        Ok(Window {
            data: WindowData {
                attributes: self.attributes,
                cmd_buf_count: self.cmd_buf_count,
                min_image_count: self.min_image_count,
                surface_format_fn: self.surface_format_fn,
                v_sync: self.v_sync,
                window_mode_override: self.window_mode_override,
            },
            read_only: ReadOnlyWindow { device, event_loop },
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
    /// _NOTE:_ Validation errors will only park the current thread for debugger attach when the
    /// process is attached to an interactive terminal. Otherwise they continue after logging.
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

    /// The minimum number of presentable images that the application needs. The implementation will
    /// either create the swapchain with at least that many images, or it will fail to create the
    /// swapchain.
    ///
    /// More images introduce more display lag, but smoother animation.
    pub fn min_image_count(mut self, count: u32) -> Self {
        self.min_image_count = Some(count);
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

    /// When `true`, requests presentation synchronized to the display refresh.
    ///
    /// # Note
    ///
    /// Applies only to exclusive fullscreen mode.
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
            .field("min_image_count", &self.min_image_count)
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
            min_image_count: None,
            surface_format_fn: None,
            v_sync: None,
            window_mode_override: None,
        }
    }
}

struct WindowData {
    attributes: WindowAttributes,
    cmd_buf_count: usize,
    min_image_count: Option<u32>,
    surface_format_fn: Option<Box<SurfaceFormatFn>>,
    v_sync: Option<bool>,
    window_mode_override: Option<Option<FullscreenMode>>,
}

/// Errors produced while creating or running a [`Window`].
#[derive(Debug)]
pub enum WindowError {
    /// A Vulkan or `vk-graph` driver error occurred.
    Driver(DriverError),
    /// `winit` failed to create or run the event loop.
    EventLoop(EventLoopError),
    /// A window system integration or swapchain presentation error occurred.
    Graphchain(GraphchainError),
}

impl error::Error for WindowError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        Some(match self {
            Self::Driver(err) => err,
            Self::EventLoop(err) => err,
            Self::Graphchain(err) => err,
        })
    }
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => err.fmt(f),
            Self::EventLoop(err) => err.fmt(f),
            Self::Graphchain(err) => err.fmt(f),
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
