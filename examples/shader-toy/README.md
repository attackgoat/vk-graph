<img alt="Preview" src="../../.github/img/shader-toy.png">

# Shadertoy Example

This example uses computational fluid dynamics to create an effect like spilled paint. The original
shader code comes from [Florian Berger](https://www.shadertoy.com/view/MsGSRd) and is attached to a
permissive [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) license.

The implementation stays close to the original Shadertoy usage. For production code, a compute
pipeline would usually be a better fit for this kind of effect. The example also keeps the unused
descriptor bindings and push constant ranges that standard Shadertoy-style pipelines expect.

## Details

See the `build.rs` script: it packs the example images into a `.pak` file. This makes the example
easier to run from different working directories. It also pre-compiles the shader code from GLSL to
SPIR-V.

### Adding/Changing files

The `pak.toml` file references the images used in this example using a glob; see line 6 in that file.

If you add a file reference directly to `pak.toml`, the build script will pick up and pack the new
file. If the glob should pick up new files instead, ask the build script to look again:

```bash
touch res/pak.toml
```

Now try building again and the newly added files should be packed and have bindings generated in the
Rust code. If any of those files change, the build script will automatically re-pack things.
