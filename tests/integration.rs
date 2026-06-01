use std::{env, ffi::OsString, process::Command};

fn run_cpu_readback_example(extra_args: &[&str]) {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let output = Command::new(cargo)
        .args(["run", "--quiet", "--example", "cpu_readback", "--"])
        .args(extra_args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("unable to run cpu_readback example");

    assert!(
        output.status.success(),
        "cpu_readback example failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
#[ignore = "requires a working Vulkan device"]
fn live_device_cpu_readback() {
    run_cpu_readback_example(&[]);
}
