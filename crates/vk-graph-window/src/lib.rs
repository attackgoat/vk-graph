//! TODO

#![warn(missing_docs)]

mod frame;
pub mod swapchain;

pub use self::frame::FrameContext;

use {
    self::swapchain::{Swapchain, SwapchainError, SwapchainInfo},
    log::{error, info, trace, warn},
    std::{error, fmt},
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

/// TODO
// #[derive(Debug)]
pub struct Window {
    data: WindowData,

    /// TODO
    pub device: Device,

    event_loop: EventLoop<()>,
}

impl Window {
    /// TODO
    pub fn new() -> Result<Self, WindowError> {
        Self::builder().build()
    }

    /// TODO
    pub fn builder() -> WindowBuilder {
        WindowBuilder::default()
    }

    /// TODO
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
            fn create_swapchain(
                &mut self,
                window: &winit::window::Window,
            ) -> Result<Swapchain, DriverError> {
                let surface = Surface::create(&self.device, window, window)?;
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
                        .into_builder()
                        .command_buffer_count(self.data.cmd_buf_count);

                if let Some(min_image_count) = self.data.min_image_count {
                    swapchain_info = swapchain_info.min_image_count(min_image_count);
                }

                if let Some(v_sync) = self.data.v_sync {
                    let present_modes = Surface::present_modes(&surface)?;
                    if !present_modes.is_empty() {
                        let best_modes = if v_sync {
                            [vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO].as_slice()
                        } else {
                            [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE].as_slice()
                        };

                        swapchain_info = swapchain_info.present_mode(
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
                }

                let swapchain = Swapchain::new(surface, swapchain_info)?;

                trace!("created swapchain");

                Ok(swapchain)
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
                let swapchain = match self.create_swapchain(&window) {
                    Err(err) => {
                        warn!("Unable to create swapchain: {err}");

                        self.error = Some(err.into());
                        event_loop.exit();

                        return;
                    }
                    Ok(res) => res,
                };
                let swapchain_pool = HashPool::new(&self.device);

                self.active_window = Some(ActiveWindow {
                    swapchain,
                    swapchain_pool,
                    swapchain_resize: None,
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
                            active_window.swapchain_resize = Some((size.width, size.height));
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
            swapchain: Swapchain,
            swapchain_pool: HashPool,
            swapchain_resize: Option<(u32, u32)>,
            events: Vec<Event<()>>,
            window: winit::window::Window,
        }

        impl ActiveWindow {
            fn draw(
                &mut self,
                device: &Device,
                mut f: impl FnMut(FrameContext),
            ) -> Result<bool, SwapchainError> {
                if let Some((width, height)) = self.swapchain_resize.take() {
                    let mut swapchain_info = self.swapchain.info;
                    swapchain_info.width = width;
                    swapchain_info.height = height;
                    self.swapchain.set_info(swapchain_info);
                }

                if let Some(swapchain_image) = self.swapchain.acquire_next_image()? {
                    let mut graph = Graph::default();
                    let swapchain_image = graph.bind_resource(swapchain_image);
                    let swapchain_info = self.swapchain.info;

                    let mut will_exit = false;

                    trace!("drawing");

                    f(FrameContext {
                        device,
                        events: &self.events,
                        height: swapchain_info.height,
                        graph: &mut graph,
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
                    self.swapchain
                        .present_image(&mut self.swapchain_pool, graph, swapchain_image, 0)
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
            swapchain, window, ..
        }) = app.active_window.take()
        {
            drop(swapchain);
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

/// TODO
pub struct WindowBuilder {
    attributes: WindowAttributes,
    cmd_buf_count: usize,
    device_info: DeviceInfo,
    min_image_count: Option<u32>,
    surface_format_fn: Option<Box<dyn Fn(&[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR>>,
    v_sync: Option<bool>,
    window_mode_override: Option<Option<FullscreenMode>>,
}

impl WindowBuilder {
    /// TODO
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
    surface_format_fn: Option<Box<dyn Fn(&[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR>>,
    v_sync: Option<bool>,
    window_mode_override: Option<Option<FullscreenMode>>,
}

/// TODO
#[derive(Debug)]
pub enum WindowError {
    /// TODO
    Driver(DriverError),
    /// TODO
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
