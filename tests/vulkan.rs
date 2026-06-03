use std::{env, ffi::OsString, process::Command};

fn run_example(target_name: &str, extra_args: &[&str]) {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let output = Command::new(cargo)
        .args(["run", "--quiet", "--example", target_name, "--"])
        .args(extra_args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap_or_else(|err| panic!("unable to run {target_name} example: {err}"));

    assert!(
        output.status.success(),
        "run error: example `{target_name}` exited with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
#[ignore = "requires Vulkan device"]
fn vulkan_cpu_readback() {
    run_example("cpu_readback", &[]);
}
