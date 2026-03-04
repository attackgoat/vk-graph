# User Contributions to vk-graph

These subdirectories contain additions, changes, and other things you might find useful while
using _vk-graph_. These user-provided contributions are not guaranteed to work and are untested.

## [`.vscode/`](.vscode/)

Configuration files for users of _[Visual Studio Code](https://code.visualstudio.com/)_. Copy the
`.vscode/` directory into the root _vk-graph_ project directory in order to enable build and debug
configurations.

**_NOTE:_** Requires installation of the
_[CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)_ extension for
debugging.

### [`rel-mgmt/`](rel-mgmt/)

A script which exercises all test cases and build conditions which must succeed prior to merging new
code into the main branch.

### [`vk-graph-egui/`](vk-graph-egui/README.md)

Renderer for [egui](https://github.com/emilk/egui); a simple, fast, and highly portable immediate
mode GUI library.

### [`vk-graph-fx/`](vk-graph-fx/README.md)

Pre-defined effects and tools built using _vk-graph_ features. Generally anything that requires
shaders or other physical data which shouldn't be part of the main library.

### [`vk-graph-hot/`](vk-graph-hot/README.md)

Adds a hot-reload feature to compute, graphic and ray-trace shader pipelines.

### [`vk-graph-imgui/`](vk-graph-imgui/README.md)

Renderer for [Dear ImGui](https://github.com/imgui-rs/imgui-rs). Provides a graphical user interface
useful for debug purposes.
