/*
    This example details some common debugging techniques you might find helpful
    when something goes wrong.

    I hope you enjoy this choose-your-own-debugger adventure!

    First you will want to read this:
    https://github.com/attackgoat/vk-graph/blob/master/examples/getting-started.md

    Enter your "name" to begin:
        cargo run --example debugger

    You see an error message up ahead:
        thread 'main' panicked at 'run with `RUST_LOG=trace` environment variable ...

    Note:
        In your own program you should init a compatible logger of some kind! Also note
        that in a release build a bunch of debug_assert! checks won't run and also the GPU
        is very likely to accept anything you give it and only just maybe crash or catch fire.

        All programs must be tested with Vulkan validation layers enabled and look for
        validity and synchronization errors.

    To continue, uncomment line 30.
*/
fn main() -> Result<(), vk_graph_window::WindowError> {
    use {log::debug, std::sync::Arc, vk_graph::prelude::*, vk_graph_window::Window};

    // 👋, 🌎!
    //pretty_env_logger::init();

    /*
        The code ahead is filled with a dangerous `panic` if you did not install the Vulkan SDK!

        You must now choose:
            - If you did not install the SDK, you must goto line 8, above.
            - If you have a recent SDK installed, you may advance the function pointer.
    */
    Window::builder().debug(true).build()?.run(|frame| {
        /*
            You have now entered the per-frame callback. Everything is happening *so* fast. We just
            executed line two of our program!

            Note:
                This callback runs each time the operating system requests a new window image and it
                expects you to render something to `frame.swapchain_image` using
                `frame.render_graph`. Note that this scope is infalliable. You may create additional
                images and graphs if you choose. You can resolve multiple render graphs per frame -
                but you only need to do that if you have a hot-section that is part of a VERY large
                graph.

                When something goes wrong, it is probably *not* during this frame closure. The
                reason is that during this scope nearly everything is deferred until frame
                resolution where we try to schedule the work and get it displayed on the screen.
                Typically, as here, we let vk-graph handle all graph resolution (no code or
                concerns here) - but it is valid to control the process manually, see the available
                functions in the API docs.

                There are a few cases that will cause issues, and a few of interest are shown below.

            You see a breakpoint approaching in the distance.

            For this next bit, you'll want to install a debugger such as gdb or:
                https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb
                See: https://askubuntu.com/questions/41629/after-upgrade-gdb-wont-attach-to-process

            Instructions for VS Code - your adventure continues!:

            - Setup your tasks.json file:
                {
                    "version": "0.2.0",
                    "configurations": [
                        {
                            "type": "lldb",
                            "request": "attach",
                            "name": "Attach",
                            "pid": "${command:pickMyProcess}"
                        }
                    ]
                }
            - Run `cargo run --example debugger`
            - You should see the PID in the console output
            - Enter the VS Code Debugger; click `[>] Attach (vk-graph)`
            - Enter the PID
            - In the call stack pane, select the first thread; pause it
            - You are now parked on a syscall
            - Walk up about 12 stack frames by scrolling down and selecting:
                `{closure#0} debugger.rs 110:21`
            - It's 🕓 to de-🐛!
        */

        /*
            Case #1:
                This will cause a validation error now (because this image is created here).

            You have followed the above directions and now have an active debug session looking at
            line 115. You try to step forward in vain. Comment out the second `image` binding to
            continue.

            It is left as an excerise to the reader to determine *what* might have gone wrong here.
        */
        #[allow(unused_variables)]
        let image = frame.render_graph.bind_node(
            Image::create(
                frame.device,
                ImageInfo::image_2d(
                    1024,
                    1024,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                ),
            )
            .unwrap(),
        );
        let image = frame.render_graph.bind_node(
            Image::create(
                frame.device,
                ImageInfo::image_2d(
                    u32::MAX,
                    u32::MAX,
                    vk::Format::UNDEFINED,
                    vk::ImageUsageFlags::default(),
                ),
            )
            .unwrap(),
        );

        // Note: This is just for example
        let compute_pipeline = Arc::new(
            ComputePipeline::create(
                frame.device,
                ComputePipelineInfo::default(),
                Shader::new_compute(
                    inline_spirv::inline_spirv!(
                        r#"
                        #version 460 core

                        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

                        layout(set = 0, binding = 42, rgba8) restrict readonly uniform image2D an_image;

                        void main() {/* TODO: 📈...💰! */}
                        "#,
                        comp
                    )
                    .as_slice(),
                ),
            )
            .unwrap(),
        );

        /*
            Case #2:
                We are about to record a compute pass which causes vk-graph to panic

            Note: You'll see a panic here:
                thread 'main' panicked at 'uninitialized swapchain image ...'

            This will cause a static assertion after this closure completes, but before it is
            called again. It happens in display.rs.

            Because this error does not cause validation layer messages, it does not hit the debug
            "breakpoint" we setup on line 39. You have two choices:
              - Read how pass_ref.rs lays out data which resolver.rs submits; debug it (you die)
              - Goto line 180 and fix the bug

            This is a valid thing to panic over because we expect that all frames will render
            something to the swapchain image. The operating system is nicely asking that we repaint
            the window, and so failing to do that means we're not a well-behaved program and so the
            display code decides to panic. We *should* render an error message at least.

            The simple fix is to write something - anything - to the swapchain. You may blit,
            compute, copy, render, store, transfer or any number of other things and that will
            signal it's OK to proceed without panicking.

            Here is a fixed line 180:
                .write_descriptor(42, frame.swapchain_image)
        */
        frame
            .render_graph
            .begin_pass("This doesn't look good...")
            .bind_pipeline(&compute_pipeline)
            .write_descriptor(42, image)
            .record_compute(|compute, _| {
                compute.dispatch(1024, 1024, 1);
            });

        // Growing tired of your advenutes, you signal that it is time to close the window and exit
        *frame.will_exit = true;
    })?;

    debug!("GAME OVER");

    Ok(())

    /*
        The game is over - OR IS IT?!

        We never actually wrote to the swapchain image! AH!

        It turns out the image binding in the shader was set to read-only AND we didn't provide any
        implementation in the main function either. In this case the image is noise, and *nothing*
        complained.

        Where to next? Fire up RenderDoc, capture a frame and have fun! But beware - RenderDoc does
        a replay of the capture it created; and it resubmits things ever so slightly differently at
        times - you most likely will NOT see any synchronization issues in RenderDoc if you DO see
        them in vk-graph.

        If you ever get stuck, switch between `vkconfig` settings of API dump and synchronization;
        those usually say exactly what is going wrong, and usually you need to use multiple layers
        together or not, depending on the order they "crash" in which may obscure the root cause.

        THANK YOU FOR PLAYING!
    */
}
