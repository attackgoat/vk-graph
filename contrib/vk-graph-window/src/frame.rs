use {
    vk_graph::{driver::device::Device, node::SwapchainImageNode, Graph},
    winit::{dpi::PhysicalPosition, event::Event, window::Window},
};

/// Centers the mouse cursor within the window.
pub fn center_cursor(window: &Window) {
    let window_size = window.inner_size();
    let x = window_size.width / 2;
    let y = window_size.height / 2;
    set_cursor_position(window, x, y);
}

/// Sets the mouse cursor at the specified position within the window.
pub fn set_cursor_position(window: &Window, x: u32, y: u32) {
    let position = PhysicalPosition::new(x as i32, y as i32);
    window.set_cursor_position(position).unwrap_or_default();
}

/// A request to render a single frame to the provided render graph.
pub struct FrameContext<'a> {
    /// The device this frame belongs to.
    pub device: &'a Device,

    /// A slice of events that have occurred since the previous frame.
    pub events: &'a [Event<()>],

    /// The height, in pixels, of the current frame.
    pub height: u32,

    /// A render graph which rendering commands should be recorded into.
    ///
    /// Make sure to write to `swapchain_image` as part of this graph.
    pub render_graph: &'a mut Graph,

    /// A pre-bound image node for the swapchain image to be drawn.
    pub swapchain_image: SwapchainImageNode,

    /// A mutable `bool` which indicates if this frame should cause the program to exit.
    pub will_exit: &'a mut bool,

    /// The width, in pixels, of the current frame.
    pub width: u32,

    /// A borrow of the operating system window relating to this frame.
    pub window: &'a Window,
}

impl FrameContext<'_> {
    /// Causes the program to exit before displaying the current frame.
    pub fn exit(&mut self) {
        *self.will_exit = true;
    }

    /// Returns the frame width divided by the frame height.
    pub fn render_aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    /// Centers the mouse cursor within the window.
    pub fn center_cursor(&self) {
        center_cursor(self.window);
    }

    /// Sets the mouse cursor at the specified position within the window.
    pub fn set_cursor_position(&self, x: u32, y: u32) {
        set_cursor_position(self.window, x, y);
    }
}
