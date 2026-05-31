//! Initializer for puffin profiling of example code.
//!
//! This module is not required or recommended for production code! It is just safe "enough" to
//! be used for example purposes when an example is run like this:
//!
//! ```bash
//! cargo run --example hello_world --release --features profile-with-puffin
//! ```
//!
//! For more information see:
//! https://github.com/attackgoat/vk-graph/blob/master/examples/getting-started.md

#[cfg(feature = "profile-with-puffin")]
use {
    log::{Level::Info, info, log_enabled},
    std::mem::ManuallyDrop,
};

/// Initializes the `puffin_http` profiling crate.
///
/// Note that you will need to enable the `profile-with-puffin` feature and install `puffin_viewer`:
///
/// ```bash
/// cargo install puffin_viewer
/// ```
#[allow(unused)]
pub fn init() {
    #[cfg(feature = "profile-with-puffin")]
    {
        let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
        let puffin_server = puffin_http::Server::new(&server_addr).unwrap();

        // // We don't want to ever drop this server
        let _ = ManuallyDrop::new(puffin_server);

        const MESSAGE: &str = "Run this to view profiling data:  puffin_viewer";

        if log_enabled!(Info) {
            info!("{}", MESSAGE);
        } else {
            eprintln!("{}", MESSAGE);
        }

        puffin::set_scopes_on(true);
    }
}
