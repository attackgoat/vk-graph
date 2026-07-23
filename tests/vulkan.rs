use std::{env, ffi::OsString, process::Command, thread, time::Duration};

const EXAMPLE_TIMEOUT: Duration = Duration::from_secs(30);

fn run_example(target_name: &str, extra_args: &[&str]) {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let mut child = Command::new(cargo)
        .args(["run", "--quiet", "--example", target_name, "--"])
        .args(extra_args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .spawn()
        .unwrap_or_else(|err| panic!("unable to run {target_name} example: {err}"));

    let started = std::time::Instant::now();
    loop {
        if let Some(status) = child
            .try_wait()
            .unwrap_or_else(|err| panic!("unable to poll {target_name} example: {err}"))
        {
            assert!(
                status.success(),
                "run error: example `{target_name}` exited with status {:?}",
                status.code(),
            );
            return;
        }

        if started.elapsed() >= EXAMPLE_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            panic!("example `{target_name}` exceeded {EXAMPLE_TIMEOUT:?}");
        }

        thread::sleep(Duration::from_millis(100));
    }
}

#[test]
#[ignore = "requires Vulkan device"]
fn vulkan_cpu_readback() {
    run_example("cpu_readback", &[]);
}
